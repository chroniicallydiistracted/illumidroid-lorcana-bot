"""Phase 2 — determinization sampling + Bayesian particle filter (engine-free)."""

from collections import Counter

import numpy as np

from search.determinize import sample_worlds
from search.belief_filter import ParticleFilter


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
