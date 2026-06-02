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
    batch_size: int = 1           # >1 = virtual-loss wave batching (batched GPU inference)
    assert_root_invariant: bool = False  # full-ISMCTS: every root world must keep the root key


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
        self._invalid_leaves = 0   # exact-execution mismatches seen by search (diagnostic)

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
        if self.cfg.batch_size > 1:
            return self.run_batched(root_obs)
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

    def run_pimc_diagnostic(self, root_obs: dict, belief_evaluator,
                            n_worlds: int = 8, sims_per_world: int | None = None,
                            proposal: str = "belief", tracker=None) -> SearchResult:
        """Diagnostic-only root-pooled PIMC search. Never use for training labels.

        Samples N opponent-hand worlds from the belief, runs a Phase-1 PUCT
        search in each determinized world, and pools the root statistics across
        worlds weighted by importance weights ρ_i. This legacy path has known
        Tier-A correctness gaps: it is hand-only and creates separate per-world
        perfect-information trees, which permits strategy fusion. It remains
        callable only for diagnostics while the full shared-tree path is remediated.

        If `tracker` (a search.belief_filter.BeliefTracker) is given, worlds are
        drawn from a persistent SIR particle filter conditioned on accumulated
        public evidence; otherwise they're sampled fresh from the belief each ply.
        """
        cfg = self.cfg
        shared = InfoSetNode(root_obs)
        if shared.terminal or shared.n_actions == 0:
            return SearchResult(pi=np.zeros(shared.n_actions, np.float32), value=0.0, root=shared)
        self._evaluate_and_expand(shared)  # priors over root actions (for reference)

        self_id = root_obs.get("self")
        hand_count = root_obs.get("players", {}).get("opp", {}).get("handCount", 0)
        pool_ids, probs = belief_evaluator(root_obs)
        if tracker is not None:
            worlds = tracker.worlds(pool_ids, probs, hand_count, n_worlds)
        else:
            worlds = sample_worlds(pool_ids, probs, hand_count, n_worlds, proposal, self.rng)

        sub_sims = sims_per_world or cfg.simulations
        # keep the configured root Dirichlet noise: each world's root selection is
        # perturbed for exploration, so the pooled visit-policy training target is
        # explored (forcing eps=0 here silently disabled self-play exploration).
        sub_cfg = replace(cfg, simulations=sub_sims)
        sub = BISMCTS(self.engine, self.evaluator, sub_cfg, self.rng)

        # pool by EXACT stable-key, not positional index: determinization must not
        # reorder/relabel the acting player's options, or visits land on wrong actions.
        shared_keys = [a["stableKey"] for a in shared.legal]
        shared_index = {k: i for i, k in enumerate(shared_keys)}
        shared_keyset = set(shared_keys)

        true_root = self.engine.snapshot()
        pooled_worlds = 0
        total_w = 0.0
        try:
            for k, world in enumerate(worlds):
                self.engine.restore(true_root)
                world_obs = self.engine.determinize(self_id, world.hand_ids, seed=f"w{k}")
                world_keys = [a["stableKey"] for a in world_obs.get("legal", [])]
                # require the identical option set (determinization touches only
                # the opponent's hidden cards, so the actor's keys must be unchanged)
                if set(world_keys) != shared_keyset:
                    continue
                res = sub.run(world_obs)
                for j, wk in enumerate(world_keys):
                    i = shared_index[wk]
                    shared.N[i] += world.weight * res.root.N[j]
                    shared.W[i] += world.weight * res.root.W[j]
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

    def run_infoset(self, root_obs: dict, belief_evaluator, tracker,
                    total_simulations: int | None = None) -> SearchResult:
        """Tier-A #2 — full ISMCTS over ONE shared info-set table (replaces the
        root-pooled per-world PIMC of `run_pimc_diagnostic`). Samples ONE posterior world per
        simulation from the seat's persistent `tracker`, traverses the sampled world
        adaptively, and shares node-edge statistics by actor-filtered `infoSetKey` —
        removing strategy-fusion bias. `total_simulations` is the TOTAL sim budget
        (one world each), NOT n_worlds × sims_per_world.

        DIAGNOSTIC-ONLY until Tier-A Phases 3/7/6 land: the lane currently uses the
        HAND-ONLY determinization RPC (it discards the world's opponent inkwell/deck and
        self deck), so it is NOT a valid clean-label search yet — clean-label belief
        sample writing stays blocked by the Phase 0 guard. This entry point is for
        diagnostics/tests until full-world determinization and Phase 6 activation land.

        Returns a `SearchResult` with `pi` aligned to `root_obs["legal"]` so callers
        are unchanged. Requires the bridge `step_exact` op + a leak-free Python key.
        """
        from dataclasses import replace
        from search.infoset_tree import run_infoset as _run_infoset
        from search.engine_sim import EngineSimulator

        legal = root_obs.get("legal", [])
        if root_obs.get("done") or not legal:
            return SearchResult(pi=np.zeros(len(legal), np.float32), value=0.0,
                                root=InfoSetNode(root_obs))

        # seed the seat's posterior for the root, then sample one world per simulation
        self_id = root_obs.get("self")
        hand_count = root_obs.get("players", {}).get("opp", {}).get("handCount", 0)
        pool_ids, probs = belief_evaluator(root_obs)
        n_seed = max(tracker.n_particles, 1)
        tracker.worlds(pool_ids, probs, hand_count, n_seed)   # (re)seed/condition the SIR filter

        def sampler(rng):
            w = tracker.sample_world(rng)
            if w is None:                          # empty pool -> trivial all-deck world
                from search.determinize import World
                return World()
            return w

        def evaluator(obs):
            priors, value = self.evaluator(obs)
            lg = obs.get("legal", [])
            d = {str(a.get("actionId", a.get("stableKey"))): float(priors[i])
                 for i, a in enumerate(lg) if i < len(priors)}
            return d, value

        root_snap = self.engine.snapshot()
        # DIAGNOSTIC-ONLY until Phases 3/7: the lane uses the HAND-ONLY determinization
        # RPC (opponent inkwell/deck + self deck are discarded), so this is not yet a
        # valid clean-label search. Opt into the gated hand-only path explicitly; the
        # Phase 0 guard still blocks belief-guided sample WRITING upstream.
        sim = EngineSimulator(self.engine, root_snap, self_id, hand_only_diagnostic=True)
        cfg = replace(self.cfg, simulations=int(total_simulations or self.cfg.simulations))
        # the actor-filtered key is engine-independent of the hidden world, so all root
        # determinizations must yield the root key — assert it to catch any leak.
        cfg = replace(cfg, assert_root_invariant=True)
        # the root obs needs a key to register the root node
        from search.infoset import info_set_key
        root_obs = {**root_obs, "infoSetKey": info_set_key(root_obs, ())}
        try:
            res = _run_infoset(root_obs, sim, sampler, evaluator, cfg, self.rng)
        finally:
            self.engine.restore(root_snap)
            self.engine.drop_snapshot(root_snap)

        # adapt to SearchResult (legacy InfoSetNode root for caller compatibility)
        node = InfoSetNode(root_obs)
        return SearchResult(pi=res.pi, value=res.value, root=node, sims=res.sims,
                            stats={**res.stats, "mode": "infoset"})

    def run_batched(self, root_obs: dict) -> SearchResult:
        """Wave-batched PUCT (batched GPU inference). Each wave: descend
        `batch_size` sims with virtual loss so they fan out to different leaves,
        execute all their paths in ONE `run_paths` IPC, evaluate all leaves in
        ONE batched net forward (GPU), then back up. Falls back to stored values
        for sims that re-hit terminal/depth nodes (no engine/net touch)."""
        cfg = self.cfg
        root = InfoSetNode(root_obs)
        if root.terminal or root.n_actions == 0:
            return SearchResult(pi=np.zeros(root.n_actions, np.float32), value=0.0, root=root)
        self._evaluate_and_expand(root)
        self._add_root_noise(root)

        root_snap = self.engine.snapshot()
        sims_done = 0
        max_batch = 0
        try:
            while sims_done < cfg.simulations:
                wave = min(cfg.batch_size, cfg.simulations - sims_done)
                pending = []  # (path, keys, parent, leaf_action)
                for _ in range(wave):
                    path, keys, create, parent, la, leaf_node, leaf_v = self._descend_vloss(root)
                    if create:
                        pending.append((path, keys, parent, la))
                    else:  # terminal / depth re-hit: known value, no engine/net
                        self._backup(path, leaf_node, leaf_v)
                        sims_done += 1
                if not pending:
                    continue
                # ONE IPC: execute every descent path in-process -> leaf observations
                leaf_obs_list = self.engine.run_paths(root_snap, [p[1] for p in pending])
                # materialize leaf nodes (dedup collisions where sims hit the same new leaf)
                created: dict[tuple, InfoSetNode] = {}
                to_eval: list[tuple[InfoSetNode, int]] = []
                leaves: list[InfoSetNode] = []
                for i, (path, keys, parent, la) in enumerate(pending):
                    leaf = parent.children.get(la) or created.get((id(parent), la))
                    if leaf is None:
                        leaf = InfoSetNode(leaf_obs_list[i])
                        parent.children[la] = leaf
                        created[(id(parent), la)] = leaf
                        if leaf_obs_list[i].get("invalidPath"):
                            # exact-execution failed on this path: distrust the leaf,
                            # mark it a dead neutral node (not re-expanded), value 0.
                            leaf.terminal = True
                            leaf.expanded = True
                            leaf.leaf_value = 0.0
                            self._invalid_leaves += 1
                        elif leaf.terminal or leaf.n_actions == 0:
                            leaf.expanded = True
                            leaf.leaf_value = 0.0
                        else:
                            to_eval.append((leaf, i))
                    leaves.append(leaf)
                # ONE batched forward (GPU) for all unique non-terminal leaves
                if to_eval:
                    evals = self.evaluator.batch_eval([leaf_obs_list[i] for _, i in to_eval])
                    for (leaf, _), (priors, value) in zip(to_eval, evals):
                        leaf.expand(np.asarray(priors, np.float32), float(value))
                    max_batch = max(max_batch, len(to_eval))
                for (path, keys, parent, la), leaf in zip(pending, leaves):
                    self._backup(path, leaf, leaf.leaf_value)
                    sims_done += 1
        finally:
            self.engine.restore(root_snap)
            self.engine.drop_snapshot(root_snap)

        pi = root.visit_policy(cfg.temperature)
        q = root.q_values()
        value = float((pi * q).sum()) if root.total_visits() > 0 else root.leaf_value
        return SearchResult(pi=pi, value=value, root=root, sims=sims_done,
                            stats={"root_visits": root.total_visits(), "n_actions": root.n_actions,
                                   "max_eval_batch": max_batch, "batch_size": cfg.batch_size})

    def _descend_vloss(self, root: InfoSetNode):
        """Tree-walk applying virtual loss; return the path + whether a new leaf
        must be created (and where), or the terminal/depth node + its value."""
        cfg = self.cfg
        path: list[tuple[InfoSetNode, int]] = []
        keys: list[str] = []
        node = root
        depth = 0
        while True:
            if node.terminal or node.n_actions == 0:
                return path, keys, False, None, -1, node, 0.0
            if depth >= cfg.depth_limit:
                return path, keys, False, None, -1, node, node.leaf_value
            a = node.select(cfg.c_puct, cfg.pw_c, cfg.pw_alpha, cfg.fpu)
            node.add_vloss(a)
            path.append((node, a))
            keys.append(node.legal[a]["stableKey"])
            child = node.children.get(a)
            if child is None:
                return path, keys, True, node, a, None, 0.0
            node = child
            depth += 1

    def _backup(self, path, leaf: InfoSetNode, leaf_v: float) -> None:
        """Remove virtual loss and apply the real visit/value along the path."""
        for n, a in path:
            n.remove_vloss(a)
            v = value_for(n.actor, leaf, leaf_v)
            n.N[a] += 1.0
            n.W[a] += v

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
            if leaf_obs.get("invalidPath"):
                # exact-execution failed somewhere on this path: do NOT trust the
                # leaf (it's a different line than the tree chose). Mark it a dead
                # neutral node so it isn't re-expanded; back up a neutral value.
                leaf.terminal = True
                leaf.expanded = True
                leaf.leaf_value = leaf_v = 0.0
                self._invalid_leaves += 1
            else:
                leaf_v = self._evaluate_and_expand(leaf)

        # backup: each node gets the leaf value from *its own* actor's POV
        for n, a in path:
            v = value_for(n.actor, leaf, leaf_v)
            n.N[a] += 1.0
            n.W[a] += v
