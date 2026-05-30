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

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from engine.bridge import LorcanaEngine
from search.node import InfoSetNode

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
                self.engine.restore(root_snap)
                self._simulate(root)
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

    def _simulate(self, root: InfoSetNode) -> None:
        cfg = self.cfg
        path: list[tuple[InfoSetNode, int]] = []
        node = root
        depth = 0

        while True:
            if node.terminal or node.n_actions == 0:
                leaf, leaf_v = node, 0.0
                break
            if depth >= cfg.depth_limit:
                leaf, leaf_v = node, node.leaf_value  # depth cut -> stored net value
                break

            a = node.select(cfg.c_puct, cfg.pw_c, cfg.pw_alpha, cfg.fpu)
            path.append((node, a))

            key = node.legal[a]["stableKey"]
            result = self.engine.step(key)
            child_obs = result["obs"]

            child = node.children.get(a)
            if child is None:
                child = InfoSetNode(child_obs)
                node.children[a] = child

            if not child.expanded:
                leaf_v = self._evaluate_and_expand(child)
                leaf = child
                break

            node = child
            depth += 1

        # backup: each node gets the leaf value from *its own* actor's POV
        for n, a in path:
            v = value_for(n.actor, leaf, leaf_v)
            n.N[a] += 1.0
            n.W[a] += v
