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

import hashlib
import math
import numbers
from dataclasses import dataclass

import numpy as np


def _default_base_seed(pool_ids: list[str], probs, hand_count: int, n_worlds: int,
                       proposal: str, ink_probs, ink_count: int) -> str:
    """Deterministic base seed derived from ALL sampler inputs — pool ids, the belief
    `probs` AND `ink_probs`, counts, and proposal — so two identical calls reproduce
    identical world seeds while a different belief (e.g. reversed `probs`) changes them.

    This is content-derived, NOT context-derived: it does not encode the caller's game/
    seat/decision/sim/RNG identity, so it cannot by itself disambiguate two decisions
    that happen to share inputs. Callers that need true run-level reproducibility pass an
    explicit `base_seed` (Phase 12 threads the real `game:seat:decision:sim` context)."""
    pb = np.asarray(probs, dtype=np.float64).round(9).tobytes() if probs is not None else b""
    ib = np.asarray(ink_probs, dtype=np.float64).round(9).tobytes() if ink_probs is not None else b""
    h = hashlib.blake2b(digest_size=8)
    h.update(repr((list(pool_ids), int(hand_count), int(n_worlds), proposal, int(ink_count))).encode())
    h.update(pb)
    h.update(ib)
    return f"sw:{h.hexdigest()}"

_CLIP = 1e-6
_PROPOSALS = frozenset({"belief", "uniform"})


class WorldContractError(ValueError):
    """A `World` violates the canonical full-hidden-zone determinization contract."""


def _require_sampler_count(value, key: str, *, maximum: int | None = None) -> int:
    """Read a sampler count without coercion or clamping."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise WorldContractError(f"{key} must be a non-negative int, got {type(value).__name__} {value!r}")
    count = int(value)
    if count < 0:
        raise WorldContractError(f"{key} must be non-negative, got {count}")
    if maximum is not None and count > maximum:
        raise WorldContractError(f"{key} {count} exceeds maximum {maximum}")
    return count


def _validate_proposal(proposal: str) -> str:
    if not isinstance(proposal, str) or proposal not in _PROPOSALS:
        raise WorldContractError(f"proposal must be one of {sorted(_PROPOSALS)}, got {proposal!r}")
    return proposal


def _validate_pool_ids(pool_ids) -> list[str]:
    if not isinstance(pool_ids, (list, tuple)):
        raise WorldContractError("pool_ids must be list-like")
    ids = list(pool_ids)
    if any(not isinstance(cid, str) or not cid.strip() for cid in ids):
        raise WorldContractError("pool_ids must contain non-empty string ids")
    if len(ids) != len(set(ids)):
        raise WorldContractError("pool_ids must not contain duplicate ids")
    return ids


def _validate_probability_vector(values, key: str, *, size: int | None = None) -> np.ndarray:
    try:
        probs = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WorldContractError(f"{key} must be a numeric probability vector") from exc
    if probs.ndim != 1:
        raise WorldContractError(f"{key} must be one-dimensional, got shape {probs.shape}")
    if size is not None and len(probs) != size:
        raise WorldContractError(f"{key} length {len(probs)} != pool size {size}")
    if not np.all(np.isfinite(probs)):
        raise WorldContractError(f"{key} must contain only finite probabilities")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise WorldContractError(f"{key} probabilities must be in [0, 1]")
    return np.clip(probs, _CLIP, 1 - _CLIP)


def _require_public_count(d, key: str, who: str) -> int:
    """STRICT public-count read (fail-closed): the key MUST be present, non-None, and a
    NON-NEGATIVE INTEGER. `bool`, `float`, `str`, and list/tuple are all rejected (no
    coercion) — a malformed counter cannot witness a leak-free partition."""
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise WorldContractError(f"{who} is missing public count '{key}' (fail-closed)")
    v = d[key]
    # bool is a subclass of int; exclude it explicitly. Accept Python/NumPy integers only.
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise WorldContractError(
            f"{who} public count '{key}' must be a non-negative int, "
            f"got {type(v).__name__} {v!r} (fail-closed)")
    if int(v) < 0:
        raise WorldContractError(f"{who} public count '{key}' must be non-negative, got {v} (fail-closed)")
    return int(v)


def _strict_zone_ids(hidden: dict, key: str) -> list[str]:
    """STRICT hidden-zone read (fail-closed): the zone MUST be present + list-like, and every
    entry MUST be a non-empty string id — a dict entry MUST carry a non-empty string `id`
    (a missing `id` raises `WorldContractError`, never a raw `KeyError`)."""
    if key not in hidden or not isinstance(hidden[key], (list, tuple)):
        raise WorldContractError(f"hidden zone '{key}' is missing or not list-like (fail-closed)")
    out: list[str] = []
    for entry in hidden[key]:
        if isinstance(entry, dict):
            if "id" not in entry:
                raise WorldContractError(f"hidden zone '{key}' entry is missing 'id' (fail-closed)")
            cid = entry["id"]
        else:
            cid = entry
        if not isinstance(cid, str) or not cid.strip():
            raise WorldContractError(
                f"hidden zone '{key}' has a non-string/empty id {cid!r} (fail-closed)")
        out.append(cid)
    return out


def _witness_pool(obs: dict) -> dict:
    """STRICT opponent hidden-pool WITNESS (fail-closed). Validates the SOURCE evidence in
    `obs["hidden"]` BEFORE anything partitions it (Phase 2 consumes this as a hard boundary):
      * `players`/`opp` present; all three opponent public counts present non-negative ints;
      * each hidden zone present, list-like, with non-empty STRING ids;
      * each hidden zone CARDINALITY equals its public count (no contradictory witness);
      * NO id duplicated across hand/inkwell/deck (no duplicate witness).
    Returns {hand_ids, inkwell_ids, deck_ids, hand_count, inkwell_count, deck_count}."""
    players = obs.get("players")
    if not isinstance(players, dict):
        raise WorldContractError("obs.players is missing or not a dict (fail-closed)")
    opp = players.get("opp")
    if not isinstance(opp, dict):
        raise WorldContractError("obs.players.opp is missing (fail-closed)")
    hidden = obs.get("hidden")
    if not isinstance(hidden, dict):
        raise WorldContractError("obs has no hidden opponent zones (fail-closed)")
    hand = _strict_zone_ids(hidden, "hand")
    ink = _strict_zone_ids(hidden, "inkwell")
    deck = _strict_zone_ids(hidden, "deck")
    hc = _require_public_count(opp, "handCount", "opp")
    kc = _require_public_count(opp, "inkwell", "opp")
    dc = _require_public_count(opp, "deckCount", "opp")
    if len(hand) != hc:
        raise WorldContractError(f"hidden hand witness count {len(hand)} != opp.handCount {hc} (fail-closed)")
    if len(ink) != kc:
        raise WorldContractError(f"hidden inkwell witness count {len(ink)} != opp.inkwell {kc} (fail-closed)")
    if len(deck) != dc:
        raise WorldContractError(f"hidden deck witness count {len(deck)} != opp.deckCount {dc} (fail-closed)")
    allids = hand + ink + deck
    if len(allids) != len(set(allids)):
        raise WorldContractError("duplicate id across hidden hand/inkwell/deck witness (fail-closed)")
    return {"hand_ids": hand, "inkwell_ids": ink, "deck_ids": deck,
            "hand_count": hc, "inkwell_count": kc, "deck_count": dc}


@dataclass(frozen=True)
class World:
    """The single canonical full-hidden-zone determinization contract (Tier-A Phase 1).

    `opponent_*` exactly partitions the opponent's known hidden pool (hand + inkwell +
    deck); `self_deck_ids` is the searching actor's own deck ORDER and is the one
    optional field — `None` means "not supplied" (the server may realize the order from
    `seed`), `()` means "supplied and empty". `seed` is part of the contract (canonical
    here in Phase 1; Phase 12 verifies reproducibility end-to-end). `log_target`/
    `log_proposal`/`rho` are kept for auditability even when posterior sampling makes
    `rho == 1`.

    These search-only identities must NEVER enter the policy/value trunk.
    """

    opponent_hand_ids: tuple[str, ...] = ()
    opponent_inkwell_ids: tuple[str, ...] = ()
    opponent_deck_ids: tuple[str, ...] = ()
    self_deck_ids: tuple[str, ...] | None = None     # None = unsupplied; () = supplied-empty
    seed: str = ""
    log_target: float = 0.0
    log_proposal: float = 0.0
    rho: float = 1.0
    weight: float = 1.0          # normalized importance weight (Σ over worlds = n_worlds)

    # -- backward-compatible accessors (legacy callers used hand_ids/weight) ------
    @property
    def hand_ids(self) -> list[str]:
        return list(self.opponent_hand_ids)

    # -- canonical hidden-pool exposure for the search-only sampler --------------
    @staticmethod
    def opponent_hidden_pool(obs: dict) -> dict:
        """The COMPLETE opponent hidden pool the search-only sampler partitions:
        per-zone instance ids (hand/inkwell/deck) + the public counts. Sourced from
        `obs["hidden"]` (search-only) + `obs["players"]["opp"]`; never fed to the trunk.

        STRICT/fail-closed: delegates to `_witness_pool`, so it raises `WorldContractError`
        on missing `players`/`opp`/`hidden`, a missing/non-integer public counter, a missing
        or malformed hidden zone, a NON-STRING/empty id, a zone cardinality that disagrees
        with its public count, or a duplicate id across zones. A downstream sampler (Phase 2)
        must never receive a silently empty or incoherent pool."""
        return _witness_pool(obs)

    # -- contract validation -----------------------------------------------------
    def validate_against_obs(self, obs: dict, *, require_seed: bool = True,
                             require_self_deck: bool = False) -> "World":
        """Prove this world is a legal full partition of the opponent's hidden pool for
        `obs` (raises `WorldContractError` on any violation; returns self on success).

        Checks (Phase 1 contract, STRICT/fail-closed): `seed` is a STRING (and non-empty for
        clean-label); the SOURCE hidden pool is validated by `_witness_pool` (well-formed
        string ids, zone cardinalities == public counts, no duplicate ids); the world's
        per-zone counts match those public counts; the world has no internal duplicates; the
        world partition equals the validated witness pool; `self_deck_ids` (when supplied)
        requires `self.deckCount` present and matching. Any missing/None/malformed field is
        REJECTED — an incomplete or incoherent observation cannot witness a leak-free
        partition.
        """
        hand, ink, deck = self.opponent_hand_ids, self.opponent_inkwell_ids, self.opponent_deck_ids

        # seed is declared `str` — enforce the type always; require non-empty for clean-label.
        if not isinstance(self.seed, str):
            raise WorldContractError(
                f"World.seed must be a string, got {type(self.seed).__name__} (fail-closed)")
        if require_seed and not self.seed.strip():
            raise WorldContractError("World.seed is empty/whitespace; clean-label ISMCTS requires a seed")

        # STRICT witness validation of the SOURCE hidden pool (counts, ids, cardinalities, dups)
        pool_info = _witness_pool(obs)
        hc, kc, dc = pool_info["hand_count"], pool_info["inkwell_count"], pool_info["deck_count"]

        # the world's per-zone partition must match the public counts
        if len(hand) != hc:
            raise WorldContractError(f"opponent_hand_ids count {len(hand)} != opp.handCount {hc}")
        if len(ink) != kc:
            raise WorldContractError(f"opponent_inkwell_ids count {len(ink)} != opp.inkwell {kc}")
        if len(deck) != dc:
            raise WorldContractError(f"opponent_deck_ids count {len(deck)} != opp.deckCount {dc}")
        # no duplicate ids across the world's three zones
        allids = list(hand) + list(ink) + list(deck)
        if len(allids) != len(set(allids)):
            raise WorldContractError("duplicate id across opponent hand/inkwell/deck")
        # the world partition must equal the validated (dedup-checked) witness pool exactly
        pool = set(pool_info["hand_ids"] + pool_info["inkwell_ids"] + pool_info["deck_ids"])
        if set(allids) != pool:
            missing = pool - set(allids)
            extra = set(allids) - pool
            raise WorldContractError(
                f"partition != hidden pool (missing={sorted(missing)}, extra={sorted(extra)})")
        # self deck: when an order is supplied (None == unsupplied), self.deckCount MUST be
        # present and match
        if self.self_deck_ids is not None:
            sdc = _require_public_count(obs["players"].get("self"), "deckCount", "self")
            if len(self.self_deck_ids) != sdc:
                raise WorldContractError(
                    f"self_deck_ids count {len(self.self_deck_ids)} != self.deckCount {sdc}")
        elif require_self_deck:
            raise WorldContractError("self_deck_ids is unsupplied but required")
        return self


# ---------------------------------------------------------------------------
# Exact count-conditioned independent-Bernoulli subset sampling
# ---------------------------------------------------------------------------
def _log_es_mass(probs: np.ndarray, k: int) -> np.ndarray:
    """Log-domain suffix DP of the count-conditioned Bernoulli mass.

    `exp(f[i][j])` = Σ over subsets of items `i..n-1` of size `j` of
    `Π_{chosen} p · Π_{not chosen} (1-p)` (each of those items contributes p if in
    the subset, 1-p otherwise). `f[0][k]` is `log Z`, where
    `Z = Σ_{|S|=k} b(S)`. Keeping the DP in log space prevents the underflow that
    would otherwise silently change a low-mass target into a uniform proposal.
    """
    p = _validate_probability_vector(probs, "probs")
    n = len(p)
    k = _require_sampler_count(k, "k", maximum=n)
    log_p = np.log(p)
    log_not_p = np.log1p(-p)
    f = np.full((n + 1, k + 1), -math.inf, dtype=np.float64)
    f[n, 0] = 0.0
    for i in range(n - 1, -1, -1):
        f[i, 0] = log_not_p[i] + f[i + 1, 0]
        for j in range(1, k + 1):
            f[i, j] = np.logaddexp(
                log_not_p[i] + f[i + 1, j],
                log_p[i] + f[i + 1, j - 1],
            )
    return f


def _log_bernoulli_subset(probs: np.ndarray, chosen: np.ndarray) -> float:
    """log b(subset) under independent Bernoulli: Σ_in log p + Σ_out log(1-p)."""
    p = _validate_probability_vector(probs, "probs")
    mask = np.zeros(len(p), dtype=bool)
    mask[chosen] = True
    return float(np.log(p[mask]).sum() + np.log1p(-p[~mask]).sum())


def _conditional_bernoulli(probs: np.ndarray, k: int, rng: np.random.Generator
                           ) -> tuple[np.ndarray, float]:
    """Draw a size-k subset exactly from independent Bernoulli(probs) conditioned on
    |S| = k. Returns (chosen indices, log q(S)) where q is the EXACT proposal pmf.

    q(S) = b(S) / Z with Z = f[0][k]; since the target is the same count-conditioned
    distribution, log_target == log_proposal and ρ == 1 exactly.
    """
    p = _validate_probability_vector(probs, "probs")
    n = len(p)
    k = _require_sampler_count(k, "k", maximum=n)
    f = _log_es_mass(p, k)
    log_z = float(f[0, k])
    if not math.isfinite(log_z):
        raise WorldContractError("count-conditioned Bernoulli normalizer is not finite")
    chosen: list[int] = []
    r = k
    for i in range(n):
        if r == 0:
            break
        if n - i == r:
            chosen.extend(range(i, n))
            r = 0
            break
        log_choose = math.log(p[i]) + f[i + 1, r - 1]
        p_choose = math.exp(min(0.0, log_choose - f[i, r]))
        if rng.random() < p_choose:
            chosen.append(i)
            r -= 1
    if r != 0:
        raise WorldContractError("count-conditioned Bernoulli sampler did not fill its slots")
    chosen_arr = np.array(sorted(chosen), dtype=int)
    logq = _log_bernoulli_subset(p, chosen_arr) - log_z
    return chosen_arr, logq


def _log_comb(n: int, k: int) -> float:
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _log_Z(probs: np.ndarray, k: int) -> float:
    """log Σ_{|S|=k} b(S) — the count-conditioned Bernoulli normalizer."""
    return float(_log_es_mass(probs, k)[0, k])


# ---------------------------------------------------------------------------
# Exact count-constrained joint 3-zone (hand / inkwell / deck) assignment (§5.1)
# ---------------------------------------------------------------------------
def _joint_zone(hand_probs: np.ndarray, ink_probs: np.ndarray,
                n_hand: int, n_ink: int, rng: np.random.Generator,
                proposal: str = "belief",
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Exact count-constrained categorical zone assignment over hand / inkwell / deck.

    Each card i is assigned with weights proportional to (hand_probs[i], ink_probs[i],
    deck_w[i]=max(1-hand-ink, eps)), conditioned on exactly `n_hand` in hand and `n_ink`
    in inkwell (rest -> deck). A DP over `(card_index, hand_slots, ink_slots)` gives the
    normalizer Z, so the STRUCTURED target probability of any assignment is known exactly:
    `b(A) = (Π_z weight_z(A)) / Z`.

    Returns `(hand_idx, ink_idx, deck_idx, log_target, log_proposal)`:
      * `proposal="belief"` samples FROM the structured target, so `log_proposal ==
        log_target` and `rho == 1` exactly;
      * `proposal="uniform"` samples a UNIFORM zone assignment, so `log_proposal` is the
        uniform pmf `1 / (C(n,H)·C(n-H,K))` and `log_target` is the structured probability
        — `rho = exp(log_target - log_proposal)` then varies and reweights toward the belief.
    """
    proposal = _validate_proposal(proposal)
    hp = _validate_probability_vector(hand_probs, "hand_probs")
    n = len(hp)
    kp = _validate_probability_vector(ink_probs, "ink_probs", size=n)
    dp = np.clip(1.0 - hp - kp, _CLIP, None)
    H = _require_sampler_count(n_hand, "n_hand", maximum=n)
    K = _require_sampler_count(n_ink, "n_ink", maximum=n - H)
    log_hp, log_kp, log_dp = np.log(hp), np.log(kp), np.log(dp)
    # exp(g[i][h][k]) = mass of assigning items i..n-1 with h hand and k ink slots.
    g = np.full((n + 1, H + 1, K + 1), -math.inf, dtype=np.float64)
    g[n, 0, 0] = 0.0
    for i in range(n - 1, -1, -1):
        for h in range(0, H + 1):
            for k in range(0, K + 1):
                v = log_dp[i] + g[i + 1, h, k]
                if h > 0:
                    v = np.logaddexp(v, log_hp[i] + g[i + 1, h - 1, k])
                if k > 0:
                    v = np.logaddexp(v, log_kp[i] + g[i + 1, h, k - 1])
                g[i, h, k] = v
    log_z = float(g[0, H, K])
    if not math.isfinite(log_z):
        raise WorldContractError("structured-zone normalizer is not finite")
    uniform_logp = -(_log_comb(n, H) + _log_comb(n - H, K))

    def _log_target(hand_idx, ink_idx, deck_idx) -> float:
        s = (sum(log_hp[i] for i in hand_idx)
             + sum(log_kp[i] for i in ink_idx)
             + sum(log_dp[i] for i in deck_idx))
        return s - log_z

    if proposal == "uniform":
        order = rng.permutation(n)
        hand_idx = sorted(order[:H].tolist())
        ink_idx = sorted(order[H:H + K].tolist())
        deck_idx = sorted(order[H + K:].tolist())
        return (np.array(hand_idx, int), np.array(ink_idx, int), np.array(deck_idx, int),
                _log_target(hand_idx, ink_idx, deck_idx), uniform_logp)

    # belief: sample from the structured target -> proposal == target (rho == 1)
    h, k = H, K
    hand_idx: list[int] = []
    ink_idx: list[int] = []
    deck_idx: list[int] = []
    for i in range(n):
        log_weights = np.array([
            log_hp[i] + g[i + 1, h - 1, k] if h > 0 else -math.inf,
            log_kp[i] + g[i + 1, h, k - 1] if k > 0 else -math.inf,
            log_dp[i] + g[i + 1, h, k],
        ])
        max_weight = float(log_weights.max())
        if not math.isfinite(max_weight):
            raise WorldContractError("structured-zone sampler reached an impossible state")
        weights = np.exp(log_weights - max_weight)
        weights /= weights.sum()
        u = rng.random()
        if u < weights[0]:
            hand_idx.append(i); h -= 1
        elif u < weights[0] + weights[1]:
            ink_idx.append(i); k -= 1
        else:
            deck_idx.append(i)
    if h != 0 or k != 0:
        raise WorldContractError("structured-zone sampler did not fill its slots")
    log_target = _log_target(hand_idx, ink_idx, deck_idx)
    return (np.array(hand_idx, int), np.array(ink_idx, int), np.array(deck_idx, int),
            log_target, log_target)


# ---------------------------------------------------------------------------
# Public sampler
# ---------------------------------------------------------------------------
def sample_worlds(pool_ids: list[str], probs: np.ndarray, hand_count: int,
                  n_worlds: int, proposal: str = "belief",
                  rng: np.random.Generator | None = None,
                  ink_probs: np.ndarray | None = None, ink_count: int = 0,
                  base_seed: str | None = None,
                  ) -> list[World]:
    """Return N belief-sampled opponent worlds with EXACT importance weights.

    With `ink_probs`/`ink_count`, samples the joint hand+inkwell+deck assignment
    exactly (§5.1); otherwise samples the hand via exact count-conditioned Bernoulli
    and splits the residual into inkwell/deck (count-consistent, the leak-free
    server behavior). `weight` is the normalized importance weight (Σ = n_worlds);
    `rho` is the raw exact b/q (== 1 for the belief proposal).

    Every returned `World` carries a deterministic non-empty `seed` (Phase 1 canonical
    contract): `f"{base}:particle={i}"`, where `base` is `base_seed` or a stable hash
    of the sampler inputs. Identical inputs (incl. the RNG seed) reproduce identical
    worlds and seeds.
    """
    rng = rng or np.random.default_rng()
    pool_ids = _validate_pool_ids(pool_ids)
    n = len(pool_ids)
    proposal = _validate_proposal(proposal)
    n_worlds = _require_sampler_count(n_worlds, "n_worlds")
    hand_count = _require_sampler_count(hand_count, "hand_count", maximum=n)
    ink_count = _require_sampler_count(ink_count, "ink_count", maximum=n - hand_count)
    probs = _validate_probability_vector(probs, "probs", size=n)
    ink_probs_arr = (_validate_probability_vector(ink_probs, "ink_probs", size=n)
                     if ink_probs is not None else None)
    if base_seed is not None and (not isinstance(base_seed, str) or not base_seed.strip()):
        raise WorldContractError("base_seed must be a non-empty string when supplied")
    base = base_seed if base_seed is not None else _default_base_seed(
        pool_ids, probs, hand_count, n_worlds, proposal, ink_probs_arr, ink_count)
    if n_worlds == 0:
        return []
    # Only a genuinely EMPTY hidden pool yields empty opponent zones. A zero hand_count with
    # a non-empty pool must still partition inkwell + deck (Phase 2 fix).
    if n == 0:
        return [World(weight=1.0, rho=1.0, seed=f"{base}:particle={i}")
                for i in range(n_worlds)]

    log_target_norm = _log_Z(probs, hand_count)    # normalizer of the count-conditioned target
    joint = ink_probs_arr is not None

    specs: list[tuple] = []        # (hand_idx, ink_idx, deck_idx, log_target, log_proposal)
    for _ in range(n_worlds):
        if joint:
            hand_idx, ink_idx, deck_idx, logt, logq = _joint_zone(
                probs, ink_probs_arr, hand_count, ink_count, rng, proposal=proposal)
            specs.append((hand_idx, ink_idx, deck_idx, logt, logq))
            continue
        if proposal == "uniform":
            chosen = (np.sort(rng.choice(n, size=hand_count, replace=False))
                      if hand_count > 0 else np.array([], dtype=int))
            logq = -_log_comb(n, hand_count)         # uniform proposal pmf over size-k subsets
            logt = _log_bernoulli_subset(probs, chosen) - log_target_norm   # structured target
        else:  # exact count-conditioned Bernoulli posterior -> q == b exactly
            chosen, logq = _conditional_bernoulli(probs, hand_count, rng)
            logt = logq
        rest = np.array([j for j in range(n) if j not in set(chosen.tolist())], dtype=int)
        rng.shuffle(rest)
        ink_idx = np.sort(rest[:ink_count])
        deck_idx = np.sort(rest[ink_count:])
        # The residual hand-complement split is sampled uniformly. Keep that factor in
        # both audit logs: it cancels in rho, but belongs to each full-zone world pmf.
        residual_logp = -_log_comb(n - hand_count, ink_count)
        logt += residual_logp
        logq += residual_logp
        specs.append((chosen, ink_idx, deck_idx, logt, logq))

    # rho = RAW exact importance weight b/q (NEVER rebased) — == 1 for the belief proposal.
    log_rho = np.array([t - q for (_, _, _, t, q) in specs], dtype=np.float64)
    if not np.all(np.isfinite(log_rho)):
        raise WorldContractError("importance log-ratios must be finite")
    rho_raw = np.exp(log_rho)
    if not np.all(np.isfinite(rho_raw)):
        raise WorldContractError("importance ratios are not representable as finite floats")
    # weight = SEPARATE normalized/log-stabilized weight for pooling (Σ over worlds == n_worlds).
    stable = np.exp(log_rho - log_rho.max())
    stable_sum = stable.sum()
    if not math.isfinite(float(stable_sum)) or not stable_sum > 0:
        raise WorldContractError("normalized importance weights are not finite")
    weight = stable / stable_sum * n_worlds

    worlds: list[World] = []
    for i, (hand_idx, ink_idx, deck_idx, logt, logq) in enumerate(specs):
        worlds.append(World(
            opponent_hand_ids=tuple(pool_ids[j] for j in hand_idx),
            opponent_inkwell_ids=tuple(pool_ids[j] for j in ink_idx),
            opponent_deck_ids=tuple(pool_ids[j] for j in deck_idx),
            seed=f"{base}:particle={i}",
            log_target=float(logt), log_proposal=float(logq),
            rho=float(rho_raw[i]), weight=float(weight[i]),
        ))
    return worlds
