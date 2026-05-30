"""Phase 1C — ISMCTS sanity: visit policy, root restoration, value backup."""

import json

import numpy as np
import pytest

from engine.bridge import LorcanaEngine
from search.ismcts import BISMCTS, SearchConfig, value_for
from search.node import InfoSetNode
from search.evaluator import UniformEvaluator, NetEvaluator
from network.model import LorcanaNet


def _keys(obs):
    return [a["stableKey"] for a in obs["legal"]]


@pytest.fixture
def engine():
    e = LorcanaEngine()
    yield e
    e.close()


def test_value_for_zero_sum():
    leaf = InfoSetNode({"done": True, "winner": "player_one", "legal": [], "actor": None})
    assert value_for("player_one", leaf, 0.0) == 1.0
    assert value_for("player_two", leaf, 0.0) == -1.0
    leaf2 = InfoSetNode({"done": False, "winner": None, "legal": [{"stableKey": "x"}],
                         "actor": "player_one"})
    assert value_for("player_one", leaf2, 0.7) == pytest.approx(0.7)
    assert value_for("player_two", leaf2, 0.7) == pytest.approx(-0.7)


def test_search_returns_valid_policy_and_restores_root(engine):
    obs = engine.reset("s-test")
    for _ in range(4):
        if obs["done"]:
            break
        obs = engine.step_auto("best")["obs"]
    if obs["done"]:
        pytest.skip("ended during warmup")

    before_keys = _keys(obs)
    cfg = SearchConfig(simulations=24, depth_limit=4)
    mcts = BISMCTS(engine, UniformEvaluator(), cfg)
    res = mcts.run(obs)

    # valid visit-count policy over the root legal set
    assert len(res.pi) == len(obs["legal"])
    assert res.pi.sum() == pytest.approx(1.0, abs=1e-4)
    assert res.stats["root_visits"] == cfg.simulations or res.root.n_actions == 1

    # engine must be back on the root decision state
    after = engine.observe()
    assert _keys(after) == before_keys


def test_net_evaluator_search_runs(engine):
    obs = engine.reset("s-net")
    for _ in range(3):
        if obs["done"]:
            break
        obs = engine.step_auto("best")["obs"]
    if obs["done"]:
        pytest.skip("ended during warmup")
    net = LorcanaNet(d_model=32, n_layers=2)
    res = BISMCTS(engine, NetEvaluator(net), SearchConfig(simulations=16, depth_limit=3)).run(obs)
    assert res.pi.sum() == pytest.approx(1.0, abs=1e-4)
    assert 0 <= int(res.pi.argmax()) < len(obs["legal"])


def test_terminal_root_returns_empty(engine):
    # craft by playing a full random game to the end, then search at terminal
    obs = engine.reset("s-term")
    steps = 0
    rng = np.random.default_rng(0)
    while not obs["done"] and steps < 400:
        a = int(rng.integers(len(obs["legal"])))
        obs = engine.step(obs["legal"][a]["stableKey"])["obs"]
        steps += 1
    if not obs["done"]:
        pytest.skip("did not terminate within cap")
    res = BISMCTS(engine, UniformEvaluator(), SearchConfig(simulations=4)).run(obs)
    assert len(res.pi) == 0 or res.pi.sum() == 0.0
