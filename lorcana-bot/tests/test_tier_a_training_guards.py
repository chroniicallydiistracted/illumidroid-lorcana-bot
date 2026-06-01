"""Phase-0 fail-closed guards for Tier-A belief-search remediation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from search.ismcts import BISMCTS, SearchConfig
from training import parallel
from training import run as training_run
from training import selfplay
from training.exploitability import exploitability_proxy
from training.league import NetPlayer
from training.league_train import run_league
from training.tier_a_guard import (
    TierARemediationIncompleteError,
    require_tier_a_clean_label_belief_training_ready,
)


class _NeverTouched:
    def __getattr__(self, name):
        raise AssertionError(f"guard did not fail before touching {name}")


class _Queue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)


def test_clean_label_belief_training_guard_fails_closed():
    with pytest.raises(TierARemediationIncompleteError, match="Tier-A"):
        require_tier_a_clean_label_belief_training_ready(
            use_belief=True, context="test")
    require_tier_a_clean_label_belief_training_ready(use_belief=False, context="test")


def test_legacy_pimc_is_explicitly_diagnostic_only():
    assert not hasattr(BISMCTS, "run_belief")
    assert callable(BISMCTS.run_pimc_diagnostic)


def test_primary_selfplay_entrypoint_blocks_belief_labels_before_engine_use():
    with pytest.raises(TierARemediationIncompleteError):
        training_run.selfplay_game(
            _NeverTouched(), _NeverTouched(), SearchConfig(), "seed",
            np.random.default_rng(0), _NeverTouched(), [], True, 2)


def test_legacy_selfplay_entrypoint_blocks_belief_labels_before_engine_use():
    with pytest.raises(TierARemediationIncompleteError):
        selfplay.play_game(
            _NeverTouched(), _NeverTouched(), SearchConfig(), "seed",
            rng=np.random.default_rng(0), use_belief=True)


def test_legacy_selfplay_iteration_blocks_before_engine_startup():
    with pytest.raises(TierARemediationIncompleteError):
        selfplay.run_iteration(
            _NeverTouched(), SearchConfig(), 1, "seed",
            _NeverTouched(), _NeverTouched(), use_belief=True)


def test_parallel_parent_blocks_belief_labels_before_spawning_workers():
    with pytest.raises(TierARemediationIncompleteError):
        parallel.generate_round_parallel(
            "unused.pt", 2, {"use_belief": True}, _NeverTouched(),
            {"sims": 0, "decisions": 0, "games": 0}, 0)


def test_parallel_worker_blocks_belief_labels_before_loading_checkpoint():
    q = _Queue()
    parallel._worker({"wid": 3, "use_belief": True}, q)
    assert q.messages[0][0:2] == ("error", 3)
    assert "Tier-A" in q.messages[0][2]
    assert q.messages[-1] == ("done", 3)


def test_league_recording_player_blocks_belief_labels():
    with pytest.raises(TierARemediationIncompleteError):
        NetPlayer("learner", _NeverTouched(), record=True, use_belief=True)


def test_league_non_recording_belief_player_remains_available_for_diagnostics():
    player = NetPlayer("diagnostic", _NeverTouched(), record=False, use_belief=True)
    assert player.use_belief and not player.record


def test_league_and_exploiter_trainers_block_before_engine_or_net_use():
    with pytest.raises(TierARemediationIncompleteError):
        run_league(use_belief=True)
    with pytest.raises(TierARemediationIncompleteError):
        exploitability_proxy(
            _NeverTouched(), _NeverTouched(), lambda: _NeverTouched(),
            SearchConfig(), use_belief=True)


def test_sample_writing_loops_do_not_call_diagnostic_pimc():
    for fn in (training_run.selfplay_game, selfplay.play_game, parallel._worker):
        source = inspect.getsource(fn)
        assert "run_pimc_diagnostic" not in source
        assert "run_belief" not in source
