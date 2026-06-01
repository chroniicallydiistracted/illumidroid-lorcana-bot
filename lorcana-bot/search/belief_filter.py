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
        return [World(opponent_hand_ids=tuple(self.particles[i]), weight=1.0, rho=1.0)
                for i in idx]

    def sample_world(self, rng: np.random.Generator | None = None) -> World | None:
        """Draw ONE world from the current posterior (with replacement). This is the
        full-ISMCTS contract (#4/§5.2): one world per simulation, ρ == 1 because the
        particle weights already encode the posterior."""
        if not self.particles:
            return None
        r = rng or self.rng
        i = int(r.choice(len(self.particles), p=self.weights))
        return World(opponent_hand_ids=tuple(self.particles[i]), weight=1.0, rho=1.0)


class BeliefTracker:
    """Per-game SIR belief over the opponent's hidden hand, integrated into the
    active self-play loop (Phase 2 §2.4). Each of our decisions:
      * neural belief net = proposal; on the first decision (or whenever the
        opponent's hidden pool changes because cards were revealed/played) we
        (re)seed particles from it;
      * otherwise we REWEIGHT the carried particles by the current belief and
        resample (SIR) — conditioning on accumulated public evidence, so the
        determinization worlds are temporally coherent instead of i.i.d. each ply.
    `worlds()` returns determinization worlds drawn from the maintained particle
    set for `run_belief`. (Basic SIR; full action-likelihood ReBeL is future.)
    """

    def __init__(self, n_particles: int = 64, ess_threshold: float = 0.5,
                 rng: np.random.Generator | None = None) -> None:
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.rng = rng or np.random.default_rng()
        self.pf: ParticleFilter | None = None
        self.pool_set: frozenset[str] = frozenset()
        self.hand_count: int = -1
        self.reseeds = 0
        self.updates = 0
        self.action_updates = 0    # observed-opponent-action corrections applied

    def worlds(self, pool_ids: list[str], probs: np.ndarray, hand_count: int,
               n_worlds: int) -> list[World]:
        pset = frozenset(pool_ids)
        prob_of = {cid: float(p) for cid, p in zip(pool_ids, probs)}
        # reseed on a new pool OR a changed hand size: after a draw the pool
        # (hand+deck) is unchanged but the hand count grows, and the filter's hand
        # cardinality is fixed at construction — stale worlds would keep the old size.
        if (self.pf is None or pset != self.pool_set
                or hand_count != self.hand_count or not self.pf.particles):
            # (re)seed from the neural belief proposal — new game or revealed cards
            self.pf = ParticleFilter(pool_ids, hand_count, rng=self.rng)
            self.pf.seed_from_belief(np.asarray(probs, dtype=np.float64), self.n_particles)
            self.pool_set = pset
            self.hand_count = hand_count
            self.reseeds += 1
        else:
            # correction: reweight carried particles by current belief, resample
            def like(hand_set):
                v = 1.0
                for cid in hand_set:
                    v *= max(prob_of.get(cid, 1e-6), 1e-6)
                return v
            self.pf.reweight(like)
            self.pf.maybe_resample(self.ess_threshold)
            self.updates += 1
        return self.pf.to_worlds(n_worlds)

    def observe_opponent_action(self, likelihood_fn) -> None:
        """#4 action-likelihood correction. After we OBSERVE an opponent action,
        reweight each particle by `P(observed action | that world)` and SIR-resample.
        This conditions the posterior on what the opponent actually did, not just on
        the current neural belief output (the SIR `correction` step, §2.4/§5.2).
        `likelihood_fn(hand_id_set) -> float` is supplied by the caller (e.g. the
        policy net's probability of the observed action in the determinized world)."""
        if self.pf is None or not self.pf.particles:
            return
        self.pf.reweight(likelihood_fn)
        self.pf.maybe_resample(self.ess_threshold)
        self.action_updates += 1

    def sample_world(self, rng: np.random.Generator | None = None):
        """Draw ONE world from the maintained posterior (full-ISMCTS, one per sim)."""
        return self.pf.sample_world(rng) if self.pf is not None else None

    def ess(self) -> float:
        """Effective sample size of the current particle set (posterior sharpness)."""
        return self.pf.ess() if self.pf is not None else 0.0
