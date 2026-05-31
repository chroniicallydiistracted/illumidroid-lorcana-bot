"""Phase 2 — Bayesian belief filtering between turns (architecture doc §2.4).

A sequential-importance-resampling (SIR) particle filter over hypotheses for the
opponent's hidden hand. The neural belief net is the *proposal* (it seeds the
particles); each observed opponent action is the *correction* (particles are
reweighted by how likely that action is under their world, then resampled when
the effective sample size collapses). This keeps the belief sharp as hidden
information is revealed across turns.

Engine-free and pluggable: the caller supplies a `likelihood_fn(hand_id_set) ->
float` (e.g. the policy net's probability of the observed action in the world
determinized to that hand, or a legality/consistency proxy).
"""

from __future__ import annotations

import numpy as np

from search.determinize import sample_worlds, World


class ParticleFilter:
    def __init__(self, pool_ids: list[str], hand_count: int,
                 rng: np.random.Generator | None = None) -> None:
        self.pool_ids = list(pool_ids)
        self.hand_count = int(hand_count)
        self.rng = rng or np.random.default_rng()
        self.particles: list[frozenset[str]] = []
        self.weights: np.ndarray = np.zeros(0, dtype=np.float64)

    # -- lifecycle ------------------------------------------------------------
    def seed_from_belief(self, probs: np.ndarray, n_particles: int) -> "ParticleFilter":
        worlds = sample_worlds(self.pool_ids, probs, self.hand_count,
                               n_particles, proposal="belief", rng=self.rng)
        self.particles = [frozenset(w.hand_ids) for w in worlds]
        self.weights = np.array([w.weight for w in worlds], dtype=np.float64)
        self._normalize()
        return self

    def _normalize(self) -> None:
        s = self.weights.sum()
        if s <= 0:
            self.weights = np.full(len(self.particles), 1.0 / max(len(self.particles), 1))
        else:
            self.weights = self.weights / s

    # -- the Bayesian update --------------------------------------------------
    def reweight(self, likelihood_fn) -> None:
        """Multiply each particle weight by L(observed action | particle world)."""
        like = np.array([max(0.0, float(likelihood_fn(set(p)))) for p in self.particles],
                        dtype=np.float64)
        self.weights = self.weights * like
        if self.weights.sum() <= 0:  # all particles inconsistent -> reset to likelihood
            self.weights = like
        self._normalize()

    def ess(self) -> float:
        """Effective sample size = 1 / Σ w_i²."""
        if len(self.weights) == 0:
            return 0.0
        return float(1.0 / np.square(self.weights).sum())

    def maybe_resample(self, threshold: float = 0.5) -> bool:
        """SIR resample when ESS drops below threshold·K. Returns whether it ran."""
        k = len(self.particles)
        if k == 0 or self.ess() >= threshold * k:
            return False
        idx = self.rng.choice(k, size=k, replace=True, p=self.weights)
        self.particles = [self.particles[i] for i in idx]
        self.weights = np.full(k, 1.0 / k)
        return True

    # -- outputs --------------------------------------------------------------
    def marginal(self) -> dict[str, float]:
        """Per-card P(in opp hand) = weighted fraction of particles containing it."""
        out = {cid: 0.0 for cid in self.pool_ids}
        for w, p in zip(self.weights, self.particles):
            for cid in p:
                out[cid] += float(w)
        return out

    def marginal_vector(self) -> np.ndarray:
        m = self.marginal()
        return np.array([m[cid] for cid in self.pool_ids], dtype=np.float32)

    def to_worlds(self, n_worlds: int | None = None) -> list[World]:
        """Expose the (resampled) particles as determinization worlds for search."""
        n = n_worlds or len(self.particles)
        if not self.particles:
            return []
        idx = self.rng.choice(len(self.particles), size=n, replace=True, p=self.weights)
        return [World(hand_ids=list(self.particles[i]), weight=1.0) for i in idx]
