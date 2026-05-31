"""Phase 1C — information-set node with pooled visit/value statistics.

Values are stored *per node from that node's own acting player's perspective*
(2-player zero-sum), so PUCT at each node maximizes the right player's equity.
Backup flips sign by whether a node's actor matches the player a leaf value is
expressed for (see ismcts.value_for).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class InfoSetNode:
    __slots__ = (
        "obs", "actor", "legal", "n_actions", "terminal", "winner",
        "P", "N", "W", "vloss", "children", "expanded", "leaf_value",
    )

    def __init__(self, obs: dict) -> None:
        self.obs = obs
        self.actor: Optional[str] = obs.get("actor")
        self.legal: list[dict] = obs.get("legal", [])
        self.n_actions = len(self.legal)
        self.terminal = bool(obs.get("done"))
        self.winner = obs.get("winner")
        self.P = np.zeros(self.n_actions, dtype=np.float32)
        self.N = np.zeros(self.n_actions, dtype=np.float32)
        self.W = np.zeros(self.n_actions, dtype=np.float32)
        # virtual loss: temporary per-edge penalty so concurrent in-wave sims
        # diverge to different leaves (each counts as a pending loss of -1).
        self.vloss = np.zeros(self.n_actions, dtype=np.float32)
        self.children: dict[int, "InfoSetNode"] = {}
        self.expanded = False
        self.leaf_value: float = 0.0

    def add_vloss(self, a: int) -> None:
        self.vloss[a] += 1.0

    def remove_vloss(self, a: int) -> None:
        self.vloss[a] = max(0.0, self.vloss[a] - 1.0)

    def expand(self, priors: np.ndarray, leaf_value: float) -> None:
        n = self.n_actions
        self.P = priors[:n].astype(np.float32) if len(priors) >= n else np.ones(n, np.float32) / max(n, 1)
        s = self.P.sum()
        if s > 0:
            self.P /= s
        self.leaf_value = leaf_value
        self.expanded = True

    def total_visits(self) -> float:
        return float(self.N.sum())

    def q_values(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(self.N > 0, self.W / np.maximum(self.N, 1e-8), 0.0)
        return q

    def select(self, c_puct: float, pw_c: float, pw_alpha: float,
               fpu: float = 0.0) -> int:
        """PUCT with progressive widening over actions ranked by prior.
        Uses *effective* stats (N + virtual loss; W − virtual loss) so that,
        within a batched wave, already-selected edges look worse and the next
        sim diverges to a different leaf."""
        n = self.n_actions
        if n == 1:
            return 0
        n_eff = self.N + self.vloss
        total = float(n_eff.sum())
        allowed = max(1, int(math.ceil(pw_c * (total + 1.0) ** pw_alpha)))
        allowed = min(allowed, n)
        order = np.argsort(-self.P)  # by prior desc
        open_actions = order[:allowed]

        sqrt_total = math.sqrt(total + 1.0)
        best_a, best_score = int(open_actions[0]), -1e18
        for a in open_actions:
            ne = n_eff[a]
            qa = ((self.W[a] - self.vloss[a]) / ne) if ne > 0 else fpu
            u = c_puct * self.P[a] * sqrt_total / (1.0 + ne)
            score = qa + u
            if score > best_score:
                best_score, best_a = score, int(a)
        return best_a

    def visit_policy(self, temperature: float = 1.0) -> np.ndarray:
        """Search-improved policy = normalized visit counts (the AZ target)."""
        n = self.n_actions
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if temperature <= 1e-6:
            pi = np.zeros(n, dtype=np.float32)
            pi[int(np.argmax(self.N))] = 1.0
            return pi
        counts = self.N ** (1.0 / temperature)
        s = counts.sum()
        if s <= 0:
            return np.ones(n, dtype=np.float32) / n
        return (counts / s).astype(np.float32)
