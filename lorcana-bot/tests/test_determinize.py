"""Phase 2 — determinization sampling + Bayesian particle filter (engine-free)."""

import itertools
import math
from collections import Counter

import numpy as np

from search.determinize import (
    sample_worlds, World,
    _conditional_bernoulli, _joint_zone, _log_bernoulli_subset, _log_Z, _log_comb,
)
from search.belief_filter import ParticleFilter, BeliefTracker


def test_sample_worlds_hand_size_and_weight_normalization():
    pool = [f"c{i}" for i in range(8)]
    probs = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1])
    ws = sample_worlds(pool, probs, 3, 30, "belief", np.random.default_rng(0))
    assert len(ws) == 30
    assert all(len(w.hand_ids) == 3 for w in ws)
    assert all(len(set(w.hand_ids)) == 3 for w in ws)          # distinct ids
    assert abs(sum(w.weight for w in ws) - 30) < 1e-6          # Σ weights = n_worlds


def test_belief_proposal_concentrates_on_high_prob_cards():
    pool = [f"c{i}" for i in range(8)]
    probs = np.array([0.95, 0.9, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05])
    ws = sample_worlds(pool, probs, 3, 300, "belief", np.random.default_rng(1))
    cnt = Counter(i for w in ws for i in w.hand_ids)
    hi = sum(cnt[f"c{i}"] for i in range(3))
    lo = sum(cnt[f"c{i}"] for i in range(3, 8))
    assert hi > 3 * lo  # the three likely cards dominate


def test_uniform_proposal_weights_vary_belief_proposal_uniform():
    pool = [f"c{i}" for i in range(6)]
    probs = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    belief = sample_worlds(pool, probs, 2, 40, "belief", np.random.default_rng(2))
    uniform = sample_worlds(pool, probs, 2, 40, "uniform", np.random.default_rng(2))
    bw = np.array([w.weight for w in belief])
    uw = np.array([w.weight for w in uniform])
    assert np.allclose(bw, 1.0, atol=1e-6)        # belief proposal -> ρ ≈ 1
    assert uw.std() > 1e-3                          # uniform proposal -> ρ varies (reweights)


def test_particle_filter_seed_matches_base_rate():
    pool = [f"c{i}" for i in range(10)]
    probs = np.array([0.8, 0.8, 0.8] + [0.6 / 7] * 7)  # ~3 in hand, front-loaded
    pf = ParticleFilter(pool, hand_count=3, rng=np.random.default_rng(0))
    pf.seed_from_belief(probs, 300)
    m = pf.marginal_vector()
    assert abs(m.sum() - 3) < 0.6                   # marginal mass ≈ hand size
    assert m[:3].mean() > m[3:].mean()              # high-prob cards more present


def test_particle_filter_reweight_sharpens_toward_observation():
    pool = [f"c{i}" for i in range(8)]
    probs = np.full(8, 3.0 / 8)
    pf = ParticleFilter(pool, 3, np.random.default_rng(0)).seed_from_belief(probs, 400)
    before = pf.marginal()["c0"]
    pf.reweight(lambda hs: 1.0 if "c0" in hs else 0.02)  # evidence: c0 is in hand
    after = pf.marginal()["c0"]
    assert after > before + 0.1


def test_particle_filter_resample_preserves_count_and_lifts_ess():
    pool = [f"c{i}" for i in range(8)]
    probs = np.full(8, 3.0 / 8)
    pf = ParticleFilter(pool, 3, np.random.default_rng(0)).seed_from_belief(probs, 200)
    pf.reweight(lambda hs: 1.0 if "c0" in hs else 1e-6)  # collapse weights
    ess_before = pf.ess()
    ran = pf.maybe_resample(0.5)
    assert ran
    assert len(pf.particles) == 200
    assert pf.ess() > ess_before                    # weights uniform after SIR


def test_importance_weighting_belief_inert_uniform_reweights():
    # #12 PROOF: belief proposal -> rho == 1 (inert by design; samples already
    # belief-distributed). uniform proposal -> rho varies AND correlates with the
    # belief (worlds holding high-prob cards get more weight).
    pool = [f"c{i}" for i in range(8)]
    probs = np.array([0.95, 0.9, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05])
    bel = sample_worlds(pool, probs, 3, 200, "belief", np.random.default_rng(3))
    assert np.allclose([w.weight for w in bel], 1.0, atol=1e-9)   # ρ == 1 exactly
    uni = sample_worlds(pool, probs, 3, 400, "uniform", np.random.default_rng(4))
    w = np.array([x.weight for x in uni])
    assert w.std() > 1e-3                                          # ρ varies
    hi = {"c0", "c1", "c2"}
    n_hi = np.array([len(hi & set(x.hand_ids)) for x in uni])      # high-prob cards held
    assert np.corrcoef(n_hi, w)[0, 1] > 0.2                        # ρ tracks belief
    # worlds richer in high-belief cards carry strictly more weight on average
    assert w[n_hi >= 2].mean() > w[n_hi <= 1].mean()


def test_pfsp_draws_counted_as_half():
    """#8: draws must score 0.5 for both sides, not as a win for the second id."""
    from training.league import League
    lg = League()
    lg.elo["a"] = lg.elo["b"] = 1000.0      # _update_elo needs ratings present
    lg.record_result("a", "b", None)        # a single draw
    assert abs(lg.win_prob("a", "b") - 0.5) < 1e-9
    assert abs(lg.win_prob("b", "a") - 0.5) < 1e-9
    lg.record_result("a", "b", "a")         # then a wins one
    pa, pb = lg.win_prob("a", "b"), lg.win_prob("b", "a")
    assert pa > 0.5 > pb and abs((pa + pb) - 1.0) < 1e-9   # complementary, draw split evenly


def test_stuck_step_guard():
    """The stuck-game guard aborts the infinite no-progress loop (repeated failed
    steps on a frozen state) but never on healthy successful play."""
    from engine.serialization import stuck_step
    # frozen + failing -> abort on the 3rd consecutive
    s = m = 0
    s, m, ab = stuck_step(s, m, True, False); assert not ab
    s, m, ab = stuck_step(s, m, True, False); assert not ab
    s, m, ab = stuck_step(s, m, True, False); assert ab
    # a single advancing step (state changed, success) resets the counters
    s, m, ab = stuck_step(s, m, False, True); assert s == 0 and m == 0 and not ab
    # healthy play (always advancing) never aborts
    s = m = 0
    for _ in range(50):
        s, m, ab = stuck_step(s, m, False, True); assert not ab
    # backstop: frozen but "successful" for 12 steps still aborts
    s = m = 0
    ab = False
    for _ in range(12):
        s, m, ab = stuck_step(s, m, True, True)
    assert ab


def _exact_conditional_marginals(probs, k):
    """Brute-force P(card i in hand) under independent-Bernoulli conditioned on |S|=k."""
    n = len(probs)
    p = np.asarray(probs, float)
    num = np.zeros(n)
    Z = 0.0
    for combo in itertools.combinations(range(n), k):
        mask = np.zeros(n, bool); mask[list(combo)] = True
        w = float(np.prod(p[mask]) * np.prod(1 - p[~mask]))
        Z += w
        num[list(combo)] += w
    return num / Z


def test_5_conditional_bernoulli_marginals_match_exact_target():
    """#5: the exact count-conditioned sampler's empirical hand marginals match the
    brute-force independent-Bernoulli-conditioned-on-|S|=k target (not Gumbel-top-k)."""
    probs = np.array([0.7, 0.55, 0.4, 0.3, 0.2, 0.1])
    k = 3
    target = _exact_conditional_marginals(probs, k)
    rng = np.random.default_rng(0)
    cnt = np.zeros(len(probs))
    N = 40000
    for _ in range(N):
        chosen, _ = _conditional_bernoulli(probs, k, rng)
        assert len(chosen) == k and len(set(chosen.tolist())) == k
        cnt[chosen] += 1
    emp = cnt / N
    assert np.max(np.abs(emp - target)) < 0.02            # matches exact target
    assert abs(emp.sum() - k) < 1e-9                       # every sample has exactly k


def test_5_conditional_bernoulli_proposal_pmf_is_exact_rho_one():
    """#5: q(S) returned by the sampler equals the true count-conditioned pmf, so the
    belief proposal carries ρ == 1 EXACTLY (the old code only assumed this)."""
    probs = np.array([0.8, 0.6, 0.45, 0.25, 0.15])
    k = 2
    logZ = _log_Z(probs, k)
    rng = np.random.default_rng(1)
    for _ in range(200):
        chosen, logq = _conditional_bernoulli(probs, k, rng)
        # exact target pmf b(S)/Z for a count-conditioned subset
        logt = _log_bernoulli_subset(probs, chosen) - logZ
        assert abs(logq - logt) < 1e-9                     # q == target -> ρ == 1
    # and the closed-form proposal pmf is a proper distribution (sums to 1)
    total = 0.0
    for combo in itertools.combinations(range(len(probs)), k):
        total += math.exp(_log_bernoulli_subset(probs, np.array(combo)) - logZ)
    assert abs(total - 1.0) < 1e-9


def test_5_world_spec_partitions_pool_and_satisfies_counts():
    """#5: every sampled World partitions the pool exactly (no dup/loss) and meets the
    public hand+inkwell counts."""
    pool = [f"c{i}" for i in range(12)]
    probs = np.linspace(0.1, 0.9, 12)
    ws = sample_worlds(pool, probs, 4, 50, "belief", np.random.default_rng(2), ink_count=2)
    for w in ws:
        assert len(w.opponent_hand_ids) == 4
        assert len(w.opponent_inkwell_ids) == 2
        union = list(w.opponent_hand_ids) + list(w.opponent_inkwell_ids) + list(w.opponent_deck_ids)
        assert sorted(union) == sorted(pool)               # exact partition, no leak/dup
        assert abs(w.rho - 1.0) < 1e-9                      # belief proposal -> ρ == 1


def test_5_joint_zone_marginals_match_brute_force():
    """#5/§5.1: the exact joint 3-zone DP sampler reproduces the brute-force
    count-constrained categorical marginals for hand and inkwell."""
    hp = np.array([0.6, 0.4, 0.3, 0.2, 0.1])
    kp = np.array([0.2, 0.3, 0.3, 0.2, 0.2])
    H, K = 2, 1
    n = len(hp)
    dp = np.clip(1 - hp - kp, 1e-9, None)
    # brute-force target marginals
    hnum = np.zeros(n); knum = np.zeros(n); Z = 0.0
    for hand in itertools.combinations(range(n), H):
        rest = [i for i in range(n) if i not in hand]
        for ink in itertools.combinations(rest, K):
            w = 1.0
            inkset = set(ink); handset = set(hand)
            for i in range(n):
                w *= hp[i] if i in handset else kp[i] if i in inkset else dp[i]
            Z += w
            for i in hand: hnum[i] += w
            for i in ink: knum[i] += w
    h_target, k_target = hnum / Z, knum / Z
    rng = np.random.default_rng(3)
    hc = np.zeros(n); kc = np.zeros(n); N = 40000
    for _ in range(N):
        hand_idx, ink_idx, deck_idx, _ = _joint_zone(hp, kp, H, K, rng)
        assert len(hand_idx) == H and len(ink_idx) == K and len(deck_idx) == n - H - K
        assert sorted(hand_idx.tolist() + ink_idx.tolist() + deck_idx.tolist()) == list(range(n))
        hc[hand_idx] += 1; kc[ink_idx] += 1
    assert np.max(np.abs(hc / N - h_target)) < 0.02
    assert np.max(np.abs(kc / N - k_target)) < 0.02


def test_5_uniform_proposal_rho_tracks_target_and_normalizes():
    """#5: uniform proposal yields varying ρ that correlates with the belief, and the
    normalized weights sum to n_worlds (pooling contract preserved)."""
    pool = [f"c{i}" for i in range(8)]
    probs = np.array([0.95, 0.9, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05])
    ws = sample_worlds(pool, probs, 3, 400, "uniform", np.random.default_rng(4))
    rho = np.array([w.rho for w in ws])
    weight = np.array([w.weight for w in ws])
    assert rho.std() > 1e-3                                # ρ varies under uniform proposal
    assert abs(weight.sum() - len(ws)) < 1e-6              # Σ normalized weights == n_worlds
    hi = {"c0", "c1", "c2"}
    n_hi = np.array([len(hi & set(w.opponent_hand_ids)) for w in ws])
    assert np.corrcoef(n_hi, rho)[0, 1] > 0.2              # ρ tracks the belief


def test_5_same_seed_reproduces_worlds():
    """#5: identical RNG seed reproduces the sampled worlds exactly (auditable)."""
    pool = [f"c{i}" for i in range(10)]
    probs = np.linspace(0.1, 0.8, 10)
    a = sample_worlds(pool, probs, 3, 20, "belief", np.random.default_rng(7))
    b = sample_worlds(pool, probs, 3, 20, "belief", np.random.default_rng(7))
    assert [w.opponent_hand_ids for w in a] == [w.opponent_hand_ids for w in b]
    assert [w.opponent_deck_ids for w in a] == [w.opponent_deck_ids for w in b]


def test_4_tracker_sample_world_and_action_likelihood():
    """#4: BeliefTracker exposes sample_world (one posterior world / sim) and an
    observed-action correction that conditions on what the opponent actually did."""
    pool = [f"c{i}" for i in range(8)]
    probs = np.full(8, 3.0 / 8)
    t = BeliefTracker(n_particles=400, rng=np.random.default_rng(0))
    t.worlds(pool, probs, 3, 8)                            # seed the posterior
    w = t.sample_world(np.random.default_rng(1))
    assert isinstance(w, World) and len(w.opponent_hand_ids) == 3
    before = t.pf.marginal()["c0"]
    # observe an opponent action only consistent with worlds holding c0
    t.observe_opponent_action(lambda hs: 1.0 if "c0" in hs else 0.02)
    after = t.pf.marginal()["c0"]
    assert after > before + 0.1 and t.action_updates == 1
    assert t.ess() > 0.0


def test_belief_tracker_seeds_reweights_reseeds():
    # #9 PROOF: BeliefTracker maintains a persistent SIR filter across decisions,
    # reseeds when the opponent's pool changes, reweights+resamples otherwise.
    from search.belief_filter import BeliefTracker
    pool = [f"c{i}" for i in range(10)]
    probs = np.array([0.8, 0.8, 0.8] + [0.06] * 7)
    t = BeliefTracker(n_particles=200, rng=np.random.default_rng(0))
    w1 = t.worlds(pool, probs, 3, 8)
    assert len(w1) == 8 and t.reseeds == 1 and t.updates == 0      # first decision -> seed
    assert t.pf is not None and abs(t.pf.marginal_vector().sum() - 3) < 0.6
    # same pool next decision -> reweight + resample (no reseed)
    t.worlds(pool, probs, 3, 8)
    assert t.reseeds == 1 and t.updates == 1
    # opponent revealed a card (pool shrinks) -> reseed
    t.worlds(pool[1:], probs[1:], 3, 8)
    assert t.reseeds == 2
    # #3: hand size grew after a draw (same pool) -> reseed (stale cardinality fixed)
    t.worlds(pool[1:], probs[1:], 4, 8)
    assert t.reseeds == 3
    assert all(len(w.hand_ids) == 4 for w in t.pf.to_worlds(8))
