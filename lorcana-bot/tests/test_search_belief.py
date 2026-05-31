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
