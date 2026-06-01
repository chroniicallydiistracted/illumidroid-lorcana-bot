"""Phase 2 — engine-backed: determinize op + belief-guided ISMCTS."""

import pytest

from engine.bridge import LorcanaEngine
from network.model import LorcanaNet
from search.ismcts import BISMCTS, SearchConfig
from search.evaluator import NetEvaluator, BeliefEvaluator


@pytest.fixture
def midgame():
    e = LorcanaEngine()
    obs = e.reset("p2-search")
    for _ in range(8):
        if obs["done"]:
            break
        obs = e.step_auto("best")["obs"]
    yield e, obs
    e.close()


def test_observation_exposes_hidden_truth(midgame):
    _, obs = midgame
    h = obs["hidden"]
    assert len(h["hand"]) == obs["players"]["opp"]["handCount"]
    assert all("id" in c and "def" in c for c in h["hand"])


def test_determinize_repartitions_and_restores(midgame):
    eng, obs = midgame
    if obs["done"]:
        pytest.skip("ended")
    self_id = obs["self"]
    pool = [c["id"] for c in obs["hidden"]["hand"]] + [c["id"] for c in obs["hidden"]["deck"]]
    hand_count = obs["players"]["opp"]["handCount"]
    before_keys = [a["stableKey"] for a in obs["legal"]]

    root = eng.snapshot()
    # put a deterministic different subset into the opponent's hand
    world = eng.determinize(self_id, pool[:hand_count], seed="w0")
    assert world["players"]["opp"]["handCount"] == hand_count
    # acting player's legal set is unchanged by opponent-hidden repartition
    assert [a["stableKey"] for a in world["legal"]] == before_keys
    # still steppable
    assert eng.step_auto("best")["success"]
    # restore the true root
    back = eng.restore(root)
    eng.drop_snapshot(root)
    assert [a["stableKey"] for a in back["legal"]] == before_keys


def test_run_belief_pools_and_restores(midgame):
    eng, obs = midgame
    if obs["done"]:
        pytest.skip("ended")
    net = LorcanaNet(d_model=32, n_layers=2)
    mcts = BISMCTS(eng, NetEvaluator(net), SearchConfig(simulations=6, depth_limit=3))
    before = [a["stableKey"] for a in obs["legal"]]
    res = mcts.run_belief(obs, BeliefEvaluator(net), n_worlds=4, sims_per_world=5)
    assert len(res.pi) == len(obs["legal"])
    assert res.pi.sum() == pytest.approx(1.0, abs=1e-4)
    assert res.stats["pooled_worlds"] >= 1
    assert res.stats["root_visits"] == pytest.approx(res.stats["pooled_worlds"] * 5 * 1.0, rel=0.5)
    # engine left on the true root
    assert [a["stableKey"] for a in eng.observe()["legal"]] == before


def test_run_infoset_full_ismcts_real_engine(midgame):
    """Tier-A #2: run_infoset drives one shared info-set tree over root-sampled worlds
    against the real engine, returns a root policy aligned to the real legal order,
    and restores the true root. (step_exact must not pollute the public history.)"""
    import numpy as np
    from search.belief_filter import BeliefTracker
    eng, obs = midgame
    if obs["done"] or len(obs["legal"]) < 2:
        pytest.skip("no branching decision")
    net = LorcanaNet(d_model=32, n_layers=2)
    mcts = BISMCTS(eng, NetEvaluator(net), SearchConfig(simulations=12, depth_limit=3),
                   rng=np.random.default_rng(0))
    tracker = BeliefTracker(n_particles=16, rng=np.random.default_rng(0))
    before = [a["stableKey"] for a in obs["legal"]]
    res = mcts.run_infoset(obs, BeliefEvaluator(net), tracker, total_simulations=12)
    assert res.pi.shape == (len(before),)
    assert abs(float(res.pi.sum()) - 1.0) < 1e-4
    assert res.stats.get("mode") == "infoset"
    assert res.stats.get("invalid_sims", 0) == 0      # exact execution, no quarantine
    assert res.stats.get("unique_infosets", 0) >= 1
    # engine is back on the true root (same legal set), and history was NOT polluted
    now = eng.observe()
    assert [a["stableKey"] for a in now["legal"]] == before


def test_run_infoset_root_key_is_world_invariant(midgame):
    """#2/§8.6 + leak: every root determinization must resolve to the SAME info-set
    key (the key reads only actor-visible info), so run_infoset's assert_root_invariant
    holds — a different key would mean hidden info leaked into the key."""
    import numpy as np
    from search.infoset import info_set_key
    eng, obs = midgame
    if obs["done"]:
        pytest.skip("ended")
    self_id = obs["self"]
    pool = [c["id"] for c in obs["hidden"]["hand"]] + [c["id"] for c in obs["hidden"]["deck"]]
    hc = obs["players"]["opp"]["handCount"]
    root = eng.snapshot()
    k_true = info_set_key(obs, ())
    rng = np.random.default_rng(1)
    try:
        for s in range(5):
            sub = list(rng.permutation(len(pool))[:hc])
            world = eng.determinize(self_id, [pool[i] for i in sub], seed=f"w{s}")
            assert info_set_key(world, ()) == k_true   # world-invariant -> leak-free key
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)
