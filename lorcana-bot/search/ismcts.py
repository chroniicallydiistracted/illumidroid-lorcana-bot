"""Phase 1C — B-ISMCTS planner (PUCT, depth-limited, neural leaves).

Spine of the search per architecture doc §3. Phase 1 scope:
  * PUCT selection over information-set nodes (search/node.py)
  * depth-limited; leaves are evaluated by the value head (never rolled to
    terminal) unless the game actually ends
  * progressive widening over the factored action space
  * determinization: Phase 1 uses N=1 over the *realized* world (the engine's
    true state). The `n_determinizations` seam and uniform world-resampling are
    where Phase 2's belief-weighted determinization plugs in.

Tree branching uses the bridge's server-side snapshot/restore: we snapshot the
root once, then each simulation restores it and walks down by executing actions
on the live engine (cheap, the tree is shallow).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from engine.bridge import LorcanaEngine
from search.node import InfoSetNode
from search.determinize import sample_worlds

# evaluator: obs -> (priors over obs.legal [n], scalar value in [-1,1] from
#                    the obs actor's perspective)
Evaluator = Callable[[dict], tuple[np.ndarray, float]]


def value_for(player: Optional[str], leaf: InfoSetNode, leaf_net_value: float) -> float:
    """Leaf value expressed from `player`'s perspective (2-player zero-sum)."""
    if leaf.terminal:
        if leaf.winner is None or player is None:
            return 0.0
        return 1.0 if str(leaf.winner) == str(player) else -1.0
    # non-terminal leaf: net value is from the leaf actor's perspective
    if player is None or leaf.actor is None:
        return leaf_net_value
    return leaf_net_value if str(player) == str(leaf.actor) else -leaf_net_value


@dataclass
class SearchConfig:
    simulations: int = 64
    c_puct: float = 1.5
    depth_limit: int = 8          # decision plies before falling back to leaf value
    pw_c: float = 2.0
    pw_alpha: float = 0.5
    fpu: float = 0.0              # first-play urgency for unvisited actions
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.0    # >0 only at self-play root
    n_determinizations: int = 1   # Phase 1: realized world only (Phase 2 hook)
    temperature: float = 1.0


@dataclass
class SearchResult:
    pi: np.ndarray                # visit-count policy over root.legal
    value: float                  # root value estimate (root actor perspective)
    root: InfoSetNode
    sims: int = 0
    stats: dict = field(default_factory=dict)


class BISMCTS:
    def __init__(self, engine: LorcanaEngine, evaluator: Evaluator,
                 config: SearchConfig | None = None, rng: np.random.Generator | None = None):
        self.engine = engine
        self.evaluator = evaluator
        self.cfg = config or SearchConfig()
        self.rng = rng or np.random.default_rng()

    def _evaluate_and_expand(self, node: InfoSetNode) -> float:
        """Expand a leaf with priors+value. Returns net value (leaf actor POV)."""
        if node.terminal or node.n_actions == 0:
            node.expanded = True
            node.leaf_value = 0.0
            return 0.0
        priors, value = self.evaluator(node.obs)
        node.expand(np.asarray(priors, dtype=np.float32), float(value))
        return float(value)

    def _add_root_noise(self, root: InfoSetNode) -> None:
        if self.cfg.dirichlet_eps <= 0 or root.n_actions <= 1:
            return
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * root.n_actions)
        root.P = (1 - self.cfg.dirichlet_eps) * root.P + self.cfg.dirichlet_eps * noise.astype(np.float32)

    def run(self, root_obs: dict) -> SearchResult:
        cfg = self.cfg
        root = InfoSetNode(root_obs)
        if root.terminal or root.n_actions == 0:
            return SearchResult(pi=np.zeros(root.n_actions, np.float32), value=0.0, root=root)

        self._evaluate_and_expand(root)
        self._add_root_noise(root)

        root_snap = self.engine.snapshot()
        sims_done = 0
        try:
            for _ in range(cfg.simulations):
                self._simulate(root, root_snap)
                sims_done += 1
        finally:
            # leave the engine ON the root decision state so the caller can
            # step the chosen action from the correct position.
            self.engine.restore(root_snap)
            self.engine.drop_snapshot(root_snap)

        pi = root.visit_policy(cfg.temperature)
        # root value from root actor perspective = visit-weighted child Q
        q = root.q_values()
        value = float((pi * q).sum()) if root.total_visits() > 0 else root.leaf_value
        return SearchResult(pi=pi, value=value, root=root, sims=sims_done,
                            stats={"root_visits": root.total_visits(),
                                   "n_actions": root.n_actions})

    def run_belief(self, root_obs: dict, belief_evaluator,
                   n_worlds: int = 8, sims_per_world: int | None = None,
                   proposal: str = "belief") -> SearchResult:
        """Belief-guided importance-weighted determinization search (Phase 2).

        Samples N opponent-hand worlds from the belief, runs a Phase-1 PUCT
        search in each determinized world, and pools the root statistics across
        worlds weighted by importance weights ρ_i. Correct because
        determinization only changes the opponent's HIDDEN cards, so the acting
        player's filtered view — and hence the root legal-action set — is
        identical across all worlds, letting us pool by action index.
        """
        cfg = self.cfg
        shared = InfoSetNode(root_obs)
        if shared.terminal or shared.n_actions == 0:
            return SearchResult(pi=np.zeros(shared.n_actions, np.float32), value=0.0, root=shared)
        self._evaluate_and_expand(shared)  # priors over root actions (for reference)

        self_id = root_obs.get("self")
        hand_count = root_obs.get("players", {}).get("opp", {}).get("handCount", 0)
        pool_ids, probs = belief_evaluator(root_obs)
        worlds = sample_worlds(pool_ids, probs, hand_count, n_worlds, proposal, self.rng)

        sub_sims = sims_per_world or cfg.simulations
        sub_cfg = replace(cfg, simulations=sub_sims, dirichlet_eps=0.0)
        sub = BISMCTS(self.engine, self.evaluator, sub_cfg, self.rng)

        true_root = self.engine.snapshot()
        pooled_worlds = 0
        total_w = 0.0
        try:
            for k, world in enumerate(worlds):
                self.engine.restore(true_root)
                world_obs = self.engine.determinize(self_id, world.hand_ids, seed=f"w{k}")
                # determinization must not change the acting player's options
                if len(world_obs.get("legal", [])) != shared.n_actions:
                    continue
                res = sub.run(world_obs)
                shared.N += world.weight * res.root.N
                shared.W += world.weight * res.root.W
                pooled_worlds += 1
                total_w += world.weight
        finally:
            self.engine.restore(true_root)
            self.engine.drop_snapshot(true_root)

        pi = shared.visit_policy(cfg.temperature)
        q = shared.q_values()
        value = float((pi * q).sum()) if shared.total_visits() > 0 else shared.leaf_value
        return SearchResult(pi=pi, value=value, root=shared,
                            sims=pooled_worlds * sub_sims,
                            stats={"n_worlds": n_worlds, "pooled_worlds": pooled_worlds,
                                   "total_weight": total_w, "proposal": proposal,
                                   "root_visits": shared.total_visits()})

    def _simulate(self, root: InfoSetNode, root_snap: int) -> None:
        """One PUCT simulation. Phase A: walk the in-memory tree (NO engine) to
        find the descent path to a not-yet-expanded leaf. Phase B: if a new leaf
        must be created, execute the whole path in ONE in-process `run_paths`
        IPC to materialize its observation, then evaluate + back up. Sims that
        re-hit an existing terminal / depth-limited node touch the engine zero
        times. This replaces the old per-node `engine.step` round-trips."""
        cfg = self.cfg
        path: list[tuple[InfoSetNode, int]] = []
        keys: list[str] = []
        node = root
        depth = 0
        parent: Optional[InfoSetNode] = None
        leaf_action = -1
        create = False

        while True:
            if node.terminal or node.n_actions == 0:
                leaf, leaf_v = node, 0.0
                break
            if depth >= cfg.depth_limit:
                leaf, leaf_v = node, node.leaf_value      # depth cut -> stored net value
                break
            a = node.select(cfg.c_puct, cfg.pw_c, cfg.pw_alpha, cfg.fpu)
            path.append((node, a))
            keys.append(node.legal[a]["stableKey"])
            child = node.children.get(a)
            if child is None:                              # unexpanded -> create here
                parent, leaf_action, create = node, a, True
                break
            node = child
            depth += 1

        if create:
            leaf_obs = self.engine.run_paths(root_snap, [keys])[0]
            leaf = InfoSetNode(leaf_obs)
            parent.children[leaf_action] = leaf
            leaf_v = self._evaluate_and_expand(leaf)

        # backup: each node gets the leaf value from *its own* actor's POV
        for n, a in path:
            v = value_for(n.actor, leaf, leaf_v)
            n.N[a] += 1.0
            n.W[a] += v
