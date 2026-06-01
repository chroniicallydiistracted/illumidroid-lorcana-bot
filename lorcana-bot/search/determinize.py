"""Phase 2 / Tier-A #5 — belief-sampled determinization with VALID importance weights.

A determinization fixes the opponent's hidden cards to a concrete assignment of
their known pool across zones. We sample N such worlds from the belief and attach
importance weights `ρ = b(world)/q(world)` (architecture doc §2.4):

  * proposal `q = "belief"`  -> sample from the EXACT count-conditioned target, so
    `q == b` and `ρ == 1` *exactly* (the usual ISMCTS choice; unit backups).
  * proposal `q = "uniform"` -> sample uniformly among size-k subsets, so `ρ ∝ b`
    (reweights toward belief-likely worlds).

The previous implementation used Gumbel-top-k with `log(p)` and *assumed* `q ≈ b`,
forcing `ρ = 1` even though Gumbel-top-k is not the count-conditioned
independent-Bernoulli distribution. #5 replaces that approximation with an exact
sampler:

  * `_conditional_bernoulli` draws a size-k subset exactly from independent
    Bernoulli(`p_i`) conditioned on `|S| = k`, via the standard suffix DP over the
    elementary-symmetric mass `f[i][j]`. Its proposal probability `q(S)` is known
    in closed form (`b(S)/Z`), so `ρ` is exact rather than assumed.
  * `_joint_zone` draws an exact count-constrained 3-zone (hand / inkwell / deck)
    assignment via a DP over `(card_index, hand_slots, ink_slots)` (§5.1), for when
    a structured hand+inkwell belief is available.

Pure / engine-free so the weighting math is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9
_CLIP = 1e-6


@dataclass(frozen=True)
class World:
    """A reproducible hidden-world specification for one determinization (#5/#2).

    `opponent_*` partitions the opponent's known hidden pool; `self_deck_ids` is the
    searching actor's own deck (composition known, order unknown -> shuffled).
    `log_target`/`log_proposal`/`rho` are kept for auditability even when direct
    posterior sampling makes `rho == 1`.
    """

    opponent_hand_ids: tuple[str, ...] = ()
    opponent_inkwell_ids: tuple[str, ...] = ()
    opponent_deck_ids: tuple[str, ...] = ()
    self_deck_ids: tuple[str, ...] = ()
    seed: str = ""
    log_target: float = 0.0
    log_proposal: float = 0.0
    rho: float = 1.0
    weight: float = 1.0          # normalized importance weight (Σ over worlds = n_worlds)

    # -- backward-compatible accessors (legacy callers used hand_ids/weight) ------
    @property
    def hand_ids(self) -> list[str]:
        return list(self.opponent_hand_ids)


# ---------------------------------------------------------------------------
# Exact count-conditioned independent-Bernoulli subset sampling
# ---------------------------------------------------------------------------
def _es_mass(probs: np.ndarray, k: int) -> np.ndarray:
    """Suffix DP of the count-conditioned Bernoulli mass.

    `f[i][j]` = Σ over subsets of items `i..n-1` of size `j` of
    `Π_{chosen} p · Π_{not chosen} (1-p)` (each of those items contributes p if in
    the subset, 1-p otherwise). `f[0][k]` is the normalizer `Z = Σ_{|S|=k} b(S)`.
    Returned shape `[n+1, k+1]` (float64).
    """
    n = len(probs)
    f = np.zeros((n + 1, k + 1), dtype=np.float64)
    f[n, 0] = 1.0
    for i in range(n - 1, -1, -1):
        p = probs[i]
        f[i, 0] = f[i + 1, 0] * (1.0 - p)
        for j in range(1, k + 1):
            f[i, j] = (1.0 - p) * f[i + 1, j] + p * f[i + 1, j - 1]
    return f


def _log_bernoulli_subset(probs: np.ndarray, chosen: np.ndarray) -> float:
    """log b(subset) under independent Bernoulli: Σ_in log p + Σ_out log(1-p)."""
    p = np.clip(probs, _CLIP, 1 - _CLIP)
    mask = np.zeros(len(p), dtype=bool)
    mask[chosen] = True
    return float(np.log(p[mask]).sum() + np.log(1 - p[~mask]).sum())


def _conditional_bernoulli(probs: np.ndarray, k: int, rng: np.random.Generator
                           ) -> tuple[np.ndarray, float]:
    """Draw a size-k subset exactly from independent Bernoulli(probs) conditioned on
    |S| = k. Returns (chosen indices, log q(S)) where q is the EXACT proposal pmf.

    q(S) = b(S) / Z with Z = f[0][k]; since the target is the same count-conditioned
    distribution, log_target == log_proposal and ρ == 1 exactly.
    """
    n = len(probs)
    p = np.clip(probs, _CLIP, 1 - _CLIP)
    f = _es_mass(p, k)
    Z = f[0, k]
    if not (Z > 0.0):                              # numerical underflow safeguard
        chosen = rng.choice(n, size=k, replace=False)
        logq = -_log_comb(n, k)
        return np.sort(chosen), logq
    chosen: list[int] = []
    r = k
    for i in range(n):
        if r == 0:
            break
        denom = f[i, r]
        p_choose = (p[i] * f[i + 1, r - 1] / denom) if denom > 0 else 0.0
        if rng.random() < p_choose:
            chosen.append(i)
            r -= 1
    chosen_arr = np.array(sorted(chosen), dtype=int)
    logq = _log_bernoulli_subset(p, chosen_arr) - math.log(Z)
    return chosen_arr, logq


def _log_comb(n: int, k: int) -> float:
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _log_Z(probs: np.ndarray, k: int) -> float:
    """log Σ_{|S|=k} b(S) — the count-conditioned Bernoulli normalizer."""
    p = np.clip(probs, _CLIP, 1 - _CLIP)
    Z = _es_mass(p, k)[0, k]
    return math.log(Z) if Z > 0 else -math.inf


# ---------------------------------------------------------------------------
# Exact count-constrained joint 3-zone (hand / inkwell / deck) assignment (§5.1)
# ---------------------------------------------------------------------------
def _joint_zone(hand_probs: np.ndarray, ink_probs: np.ndarray,
                n_hand: int, n_ink: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Exact constrained categorical zone assignment.

    Each card i is assigned to hand / inkwell / deck with weights proportional to
    (hand_probs[i], ink_probs[i], deck_w[i]=max(1-hand-ink, eps)), conditioned on
    exactly `n_hand` cards in hand and `n_ink` in inkwell (the rest -> deck). DP over
    `(card_index, hand_slots, ink_slots)`; returns (hand_idx, ink_idx, deck_idx,
    log q) with q the exact proposal pmf of the sampled assignment.
    """
    n = len(hand_probs)
    hp = np.clip(hand_probs, _CLIP, 1 - _CLIP)
    kp = np.clip(ink_probs, _CLIP, 1 - _CLIP)
    dp = np.clip(1.0 - hp - kp, _CLIP, None)
    H, K = n_hand, n_ink
    # g[i][h][k] = mass of assigning items i..n-1 with h to hand, k to inkwell
    g = np.zeros((n + 1, H + 1, K + 1), dtype=np.float64)
    g[n, 0, 0] = 1.0
    for i in range(n - 1, -1, -1):
        for h in range(0, H + 1):
            for k in range(0, K + 1):
                v = dp[i] * g[i + 1, h, k]
                if h > 0:
                    v += hp[i] * g[i + 1, h - 1, k]
                if k > 0:
                    v += kp[i] * g[i + 1, h, k - 1]
                g[i, h, k] = v
    Z = g[0, H, K]
    hand_idx: list[int] = []
    ink_idx: list[int] = []
    deck_idx: list[int] = []
    if not (Z > 0.0):                              # underflow -> uniform residual fallback
        order = rng.permutation(n)
        hand_idx = sorted(order[:H].tolist())
        ink_idx = sorted(order[H:H + K].tolist())
        deck_idx = sorted(order[H + K:].tolist())
        logq = -(_log_comb(n, H) + _log_comb(n - H, K))
        return (np.array(hand_idx, int), np.array(ink_idx, int),
                np.array(deck_idx, int), logq)
    h, k = H, K
    log_q = 0.0
    for i in range(n):
        tot = g[i, h, k]
        w_deck = dp[i] * g[i + 1, h, k]
        w_hand = hp[i] * g[i + 1, h - 1, k] if h > 0 else 0.0
        w_ink = kp[i] * g[i + 1, h, k - 1] if k > 0 else 0.0
        u = rng.random() * tot
        if u < w_hand:
            hand_idx.append(i); h -= 1; log_q += math.log(max(w_hand, _EPS)) - math.log(tot)
        elif u < w_hand + w_ink:
            ink_idx.append(i); k -= 1; log_q += math.log(max(w_ink, _EPS)) - math.log(tot)
        else:
            deck_idx.append(i); log_q += math.log(max(w_deck, _EPS)) - math.log(tot)
    return (np.array(hand_idx, int), np.array(ink_idx, int),
            np.array(deck_idx, int), log_q)


# ---------------------------------------------------------------------------
# Public sampler
# ---------------------------------------------------------------------------
def sample_worlds(pool_ids: list[str], probs: np.ndarray, hand_count: int,
                  n_worlds: int, proposal: str = "belief",
                  rng: np.random.Generator | None = None,
                  ink_probs: np.ndarray | None = None, ink_count: int = 0,
                  ) -> list[World]:
    """Return N belief-sampled opponent worlds with EXACT importance weights.

    With `ink_probs`/`ink_count`, samples the joint hand+inkwell+deck assignment
    exactly (§5.1); otherwise samples the hand via exact count-conditioned Bernoulli
    and splits the residual into inkwell/deck (count-consistent, the leak-free
    server behavior). `weight` is the normalized importance weight (Σ = n_worlds);
    `rho` is the raw exact b/q (== 1 for the belief proposal).
    """
    rng = rng or np.random.default_rng()
    if n_worlds <= 0:
        return []
    n = len(pool_ids)
    hand_count = max(0, min(int(hand_count), n))
    if n == 0 or hand_count == 0:
        return [World(weight=1.0, rho=1.0) for _ in range(n_worlds)]

    probs = np.clip(np.asarray(probs, dtype=np.float64), _CLIP, 1 - _CLIP)
    log_target_norm = _log_Z(probs, hand_count)    # normalizer of the count-conditioned target
    joint = ink_probs is not None and ink_count > 0 and (hand_count + ink_count) <= n
    ink_probs_arr = (np.clip(np.asarray(ink_probs, dtype=np.float64), _CLIP, 1 - _CLIP)
                     if ink_probs is not None else None)

    specs: list[tuple] = []        # (hand_idx, ink_idx, deck_idx, log_target, log_proposal)
    for _ in range(n_worlds):
        if joint:
            hand_idx, ink_idx, deck_idx, logq = _joint_zone(
                probs, ink_probs_arr, hand_count, ink_count, rng)
            # target == proposal for direct posterior sampling -> ρ == 1 exactly
            logt = logq if proposal != "uniform" else logq
            specs.append((hand_idx, ink_idx, deck_idx, logt, logq))
            continue
        if proposal == "uniform":
            chosen = np.sort(rng.choice(n, size=hand_count, replace=False))
            logq = -_log_comb(n, hand_count)
            logt = _log_bernoulli_subset(probs, chosen) - log_target_norm
        else:  # exact count-conditioned Bernoulli posterior
            chosen, logq = _conditional_bernoulli(probs, hand_count, rng)
            logt = logq                              # q == b exactly -> ρ == 1
        rest = np.array([j for j in range(n) if j not in set(chosen.tolist())], dtype=int)
        rng.shuffle(rest)
        ink_take = max(0, min(ink_count, len(rest)))
        ink_idx = np.sort(rest[:ink_take])
        deck_idx = np.sort(rest[ink_take:])
        specs.append((chosen, ink_idx, deck_idx, logt, logq))

    log_rho = np.array([t - q for (_, _, _, t, q) in specs], dtype=np.float64)
    rho = np.exp(log_rho - log_rho.max())          # raw exact importance weight (rebased)
    # normalized weight used by the legacy root pooling (Σ over worlds == n_worlds)
    w = rho / rho.sum() * n_worlds if rho.sum() > 0 else np.ones(n_worlds)

    worlds: list[World] = []
    for i, (hand_idx, ink_idx, deck_idx, logt, logq) in enumerate(specs):
        worlds.append(World(
            opponent_hand_ids=tuple(pool_ids[j] for j in hand_idx),
            opponent_inkwell_ids=tuple(pool_ids[j] for j in ink_idx),
            opponent_deck_ids=tuple(pool_ids[j] for j in deck_idx),
            log_target=float(logt), log_proposal=float(logq),
            rho=float(rho[i]), weight=float(w[i]),
        ))
    return worlds
