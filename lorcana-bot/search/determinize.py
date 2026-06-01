"""Phase 2 — belief-sampled determinization with importance weights.

A determinization fixes the opponent's hidden hand to a concrete subset of their
known pool. We sample N such worlds from the belief and attach importance
weights `ρ_i = b(w_i)/q(w_i)` (architecture doc §2.4):

  * proposal `q = "belief"`  -> sample ∝ belief (ρ ≈ 1, the usual ISMCTS choice)
  * proposal `q = "uniform"` -> ρ ∝ b, reweighting toward belief-likely worlds

`b(subset)` uses an independent-Bernoulli model over the pool restricted to the
known hand size. Pure / engine-free so the weighting math is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    hand_ids: list[str]      # opponent instance ids assigned to hand
    weight: float            # normalized importance weight (Σ weights = n_worlds)


def _gumbel_topk(logp: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample k distinct indices ∝ exp(logp) without replacement (Gumbel-top-k)."""
    g = rng.gumbel(size=logp.shape)
    return np.argpartition(-(logp + g), kth=min(k, len(logp) - 1))[:k]


def _log_bernoulli_subset(probs: np.ndarray, chosen: np.ndarray) -> float:
    """log b(subset) under independent Bernoulli: Π p_in · Π (1-p_out)."""
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    mask = np.zeros(len(p), dtype=bool)
    mask[chosen] = True
    return float(np.log(p[mask]).sum() + np.log(1 - p[~mask]).sum())


def sample_worlds(pool_ids: list[str], probs: np.ndarray, hand_count: int,
                  n_worlds: int, proposal: str = "belief",
                  rng: np.random.Generator | None = None) -> list[World]:
    """Return N belief-sampled opponent-hand worlds with importance weights."""
    rng = rng or np.random.default_rng()
    if n_worlds <= 0:
        return []
    n = len(pool_ids)
    hand_count = max(0, min(int(hand_count), n))
    if n == 0 or hand_count == 0:
        return [World(hand_ids=[], weight=1.0) for _ in range(n_worlds)]

    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    logp = np.log(probs)
    chosen_sets: list[np.ndarray] = []
    log_w = np.zeros(n_worlds, dtype=np.float64)

    uniform_logq = 0.0  # constant for uniform proposal (drops out after normalization)
    for i in range(n_worlds):
        if proposal == "uniform":
            chosen = rng.choice(n, size=hand_count, replace=False)
            logq = uniform_logq
        else:  # belief proposal
            chosen = _gumbel_topk(logp, hand_count, rng)
            # approximate q ≈ b (sampling ∝ belief) -> ρ ≈ 1
            logq = _log_bernoulli_subset(probs, chosen)
        logb = _log_bernoulli_subset(probs, chosen)
        chosen_sets.append(chosen)
        log_w[i] = logb - logq

    # self-normalized importance weights, scaled so Σ = n_worlds
    log_w -= log_w.max()
    w = np.exp(log_w)
    w = w / w.sum() * n_worlds
    return [World(hand_ids=[pool_ids[j] for j in chosen_sets[i]], weight=float(w[i]))
            for i in range(n_worlds)]
