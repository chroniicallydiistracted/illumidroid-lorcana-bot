"""Tier-A #2 — full Information-Set MCTS (shared tree, root-sampled worlds).

Replaces the determinized-UCT / PIMC ensemble in `BISMCTS.run_belief` (which built
a SEPARATE perfect-information tree per sampled world and pooled only root
statistics — the strategy-fusion bug Cowling et al. 2012 warn about) with ONE
`InfoSetTable` per root decision:

  1. sample one hidden world per simulation (from the seat's posterior; ρ == 1);
  2. traverse the sampled world adaptively, resolving the next node by the
     actor-filtered `infoSetKey` after every transition;
  3. reuse the same shared node across simulations AND worlds (deeper statistics
     are shared, so the actor cannot learn a per-world contingent plan it could not
     execute without distinguishing the worlds);
  4. restrict selection to the actions available in the current sampled world, and
     track an availability count per edge separately from visits
     (availability-aware PUCT — Cowling et al.'s subset-armed UCB adapted to PUCT).

The planner is simulator-agnostic: a `LaneSimulator` supplies world-sampled
observations carrying `infoSetKey`, `actor`, `terminal`/`winner`, and a `legal`
list of `{actionId, ...}`; an `evaluator(obs) -> ({actionId: prior}, value)` gives
priors + a leaf value. This lets the algorithm be proven on a synthetic
imperfect-information fixture (spec §13.3) and driven against the real engine by an
adapter, with no hidden state in the planner itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# shared node-edge statistics
# ---------------------------------------------------------------------------
@dataclass
class EdgeStats:
    action_id: str
    descriptor: dict = field(default_factory=dict)
    prior_sum: float = 0.0
    prior_samples: int = 0
    visits: float = 0.0
    value_sum: float = 0.0
    availability: float = 0.0       # how often this action was AVAILABLE at a backup
    virtual_loss: float = 0.0
    virtual_availability: float = 0.0
    invalid_execs: int = 0

    @property
    def prior(self) -> float:
        return self.prior_sum / self.prior_samples if self.prior_samples else 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass
class SharedInfoSetNode:
    key: str
    actor: Optional[str]
    exemplar_obs: dict
    terminal: bool = False
    winner: Optional[str] = None
    edges: dict[str, EdgeStats] = field(default_factory=dict)
    expanded: bool = False
    leaf_value: float = 0.0

    def ensure_edges(self, legal: list[dict]) -> None:
        """Register every currently-observed action id in the node union."""
        for a in legal:
            aid = str(a.get("actionId", a.get("stableKey")))
            if aid not in self.edges:
                self.edges[aid] = EdgeStats(action_id=aid, descriptor=a)

    def add_prior(self, aid: str, p: float) -> None:
        e = self.edges.get(aid)
        if e is None:
            e = self.edges[aid] = EdgeStats(action_id=aid)
        e.prior_sum += float(p)
        e.prior_samples += 1

    def successful_visits(self) -> float:
        return float(sum(e.visits for e in self.edges.values()))

    # -- availability-aware PUCT + progressive widening over the available subset --
    def select(self, available_ids: list[str], cfg) -> str:
        avail = [a for a in available_ids if a in self.edges]
        if not avail:
            avail = list(available_ids)
            for a in avail:
                self.edges.setdefault(a, EdgeStats(action_id=a))
        if len(avail) == 1:
            return avail[0]
        # renormalize priors across the CURRENT available subset (subsets vary by world)
        psum = sum(max(self.edges[a].prior, 0.0) for a in avail) or 1.0
        # progressive widening: open ⌈pw_c·(visits+1)^pw_alpha⌉ best-prior actions
        succ = sum(self.edges[a].visits for a in avail)
        allowed = max(1, int(math.ceil(cfg.pw_c * (succ + 1.0) ** cfg.pw_alpha)))
        ordered = sorted(avail, key=lambda a: -self.edges[a].prior)
        open_ids = ordered[:min(allowed, len(ordered))]
        best_a, best_score = open_ids[0], -1e18
        for a in open_ids:
            e = self.edges[a]
            n_eff = e.visits + e.virtual_loss
            a_eff = e.availability + e.virtual_availability
            q = ((e.value_sum - e.virtual_loss) / n_eff) if n_eff > 0 else cfg.fpu
            u = cfg.c_puct * (e.prior / psum) * math.sqrt(max(a_eff, 1.0)) / (1.0 + n_eff)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a


class InfoSetTable:
    def __init__(self) -> None:
        self.nodes: dict[str, SharedInfoSetNode] = {}
        self.transposition_hits = 0

    def get_or_create(self, obs: dict) -> tuple[SharedInfoSetNode, bool]:
        key = obs["infoSetKey"]
        node = self.nodes.get(key)
        if node is None:
            node = SharedInfoSetNode(
                key=key, actor=obs.get("actor"), exemplar_obs=obs,
                terminal=bool(obs.get("done")), winner=obs.get("winner"))
            node.ensure_edges(obs.get("legal", []))
            self.nodes[key] = node
            return node, True
        self.transposition_hits += 1
        node.ensure_edges(obs.get("legal", []))
        return node, False


# ---------------------------------------------------------------------------
# simulator protocol
# ---------------------------------------------------------------------------
class LaneSimulator(Protocol):
    def begin_lane(self, world) -> dict: ...      # restore root + determinize world -> obs
    def step(self, action_id: str) -> dict: ...   # execute action; -> next obs (may set "invalid")
    def end(self) -> None: ...                     # release lane / restore real root


Evaluator = Callable[[dict], "tuple[dict, float]"]   # obs -> ({action_id: prior}, value)


# ---------------------------------------------------------------------------
# value (2-player zero-sum, expressed from a given node-actor's perspective)
# ---------------------------------------------------------------------------
def value_for(node_actor: Optional[str], leaf_obs: dict, leaf_value: float) -> float:
    if leaf_obs.get("done"):
        w = leaf_obs.get("winner")
        if w is None or node_actor is None:
            return 0.0
        return 1.0 if str(w) == str(node_actor) else -1.0
    leaf_actor = leaf_obs.get("actor")
    if node_actor is None or leaf_actor is None:
        return leaf_value
    return leaf_value if str(node_actor) == str(leaf_actor) else -leaf_value


@dataclass
class InfoSetResult:
    pi: np.ndarray                  # visit-count policy aligned to root_obs["legal"] order
    value: float
    table: InfoSetTable
    root: SharedInfoSetNode
    sims: int = 0
    stats: dict = field(default_factory=dict)


def _expand(node: SharedInfoSetNode, obs: dict, evaluator: Evaluator) -> float:
    if node.terminal or not obs.get("legal"):
        node.expanded = True
        node.leaf_value = 0.0
        return 0.0
    priors, value = evaluator(obs)
    node.ensure_edges(obs.get("legal", []))
    for aid, p in priors.items():
        node.add_prior(aid, p)
    node.leaf_value = float(value)
    node.expanded = True
    return float(value)


def _add_root_noise(node: SharedInfoSetNode, cfg, rng: np.random.Generator) -> None:
    if cfg.dirichlet_eps <= 0 or len(node.edges) <= 1:
        return
    aids = list(node.edges.keys())
    noise = rng.dirichlet([cfg.dirichlet_alpha] * len(aids))
    for aid, nz in zip(aids, noise):
        e = node.edges[aid]
        # blend noise into the running prior MEAN (priors are keyed by actionId)
        mean = e.prior
        e.prior_sum = (1 - cfg.dirichlet_eps) * mean + cfg.dirichlet_eps * float(nz)
        e.prior_samples = 1


def run_infoset(root_obs: dict, sim: LaneSimulator, world_sampler: Callable,
                evaluator: Evaluator, cfg, rng: np.random.Generator,
                ) -> InfoSetResult:
    """One full-ISMCTS search over a single shared `InfoSetTable` (spec §4/§10).

    `cfg.simulations` is the TOTAL number of root-sampled simulations (one posterior
    world each) — NOT n_worlds × sims_per_world.
    """
    table = InfoSetTable()
    root, _ = table.get_or_create(root_obs)
    if root.terminal or not root_obs.get("legal"):
        return InfoSetResult(pi=np.zeros(len(root_obs.get("legal", [])), np.float32),
                             value=0.0, table=table, root=root)
    _expand(root, root_obs, evaluator)
    _add_root_noise(root, cfg, rng)

    sims = 0
    invalid = 0
    max_depth = 0
    try:
        for _ in range(cfg.simulations):
            world = world_sampler(rng)
            obs = sim.begin_lane(world)
            # root determinizations MUST resolve to the same root info set + action ids
            # (else hidden info leaked into the key) — assert in debug builds.
            if obs.get("infoSetKey") != root.key and cfg.assert_root_invariant:
                raise AssertionError("root determinization changed the info-set key (leak)")
            path: list[tuple[SharedInfoSetNode, str, list[str]]] = []
            depth = 0
            leaf_obs = obs
            quarantine = False
            while True:
                node, created = table.get_or_create(obs)
                leaf_obs = obs
                if node.terminal:
                    if not node.expanded:
                        _expand(node, obs, evaluator)
                    break
                if created:
                    _expand(node, obs, evaluator)
                    break
                if depth >= cfg.depth_limit:
                    if not node.expanded:
                        _expand(node, obs, evaluator)
                    break
                available = [str(a.get("actionId", a.get("stableKey")))
                             for a in obs.get("legal", [])]
                action_id = node.select(available, cfg)
                path.append((node, action_id, available))
                obs = sim.step(action_id)
                depth += 1
                if obs.get("invalid"):
                    # exact-execution mismatch: quarantine this simulation — remove any
                    # pending virtual stats and DO NOT pollute shared node statistics.
                    node.edges[action_id].invalid_execs += 1
                    quarantine = True
                    break
            max_depth = max(max_depth, depth)
            if quarantine:
                invalid += 1
                continue
            leaf_value = node.leaf_value if not node.terminal else 0.0
            # backup along the path; availability increments every available sibling
            for n, sel, avail in path:
                for aid in avail:
                    n.edges[aid].availability += 1.0
                e = n.edges[sel]
                e.visits += 1.0
                e.value_sum += value_for(n.actor, leaf_obs, leaf_value)
            sims += 1
    finally:
        sim.end()

    # root policy aligned to the REAL root legal order, by actionId (never index)
    legal = root_obs.get("legal", [])
    counts = np.array([root.edges[str(a.get("actionId", a.get("stableKey")))].visits
                       if str(a.get("actionId", a.get("stableKey"))) in root.edges else 0.0
                       for a in legal], dtype=np.float64)
    if cfg.temperature <= 1e-6 and counts.sum() > 0:
        pi = np.zeros(len(legal), np.float32); pi[int(counts.argmax())] = 1.0
    elif counts.sum() > 0:
        c = counts ** (1.0 / cfg.temperature)
        pi = (c / c.sum()).astype(np.float32)
    else:
        pi = (np.ones(len(legal), np.float32) / max(len(legal), 1))
    qvals = np.array([root.edges[str(a.get("actionId", a.get("stableKey")))].q
                      if str(a.get("actionId", a.get("stableKey"))) in root.edges else 0.0
                      for a in legal], dtype=np.float64)
    value = float((pi * qvals).sum()) if counts.sum() > 0 else root.leaf_value
    return InfoSetResult(
        pi=pi, value=value, table=table, root=root, sims=sims,
        stats={"unique_infosets": len(table.nodes),
               "transposition_hits": table.transposition_hits,
               "invalid_sims": invalid, "max_depth": max_depth,
               "avg_actions": float(np.mean([len(n.edges) for n in table.nodes.values()]))})
