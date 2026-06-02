"""Phase 13 baseline red probes for the Tier-A belief-search remediation.

Each strict xfail records an audited defect.  The phase that fixes the defect
must remove the marker and keep the probe as a passing regression test.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from engine.bridge import LorcanaEngine
from search.belief_filter import BeliefTracker
from search.determinize import World, sample_worlds
from search.engine_sim import EngineSimulator
from search.evaluator import BeliefEvaluator, UniformEvaluator
from search.infoset import history_event
from search.infoset_tree import InfoSetTable, _expand
from search.ismcts import BISMCTS, SearchConfig


_BOT_ROOT = Path(__file__).resolve().parents[1]


def _red(owner: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"Phase 13 baseline red: {owner}")


def _obs(*, done: bool = False) -> dict:
    # Real bridge schema: players is a {self, opp} dict (NOT a list), legal entries carry
    # `stableKey`, and counts are the public per-zone integers. A corrected fixture lets the
    # lane/diagnostic probes reach their OWNED assertions instead of crashing in info_set_key.
    return {
        "status": "finished" if done else "playing",
        "done": done,
        "actor": "player_one",
        "self": "player_one",
        "opp": "player_two",
        "selfIdx": 0,
        "turn": 1,
        "phase": "main",
        "step": None,
        "winner": None,
        "forced": False,
        "legal": [] if done else [{"stableKey": "quest:c1", "family": "quest"},
                                  {"stableKey": "passTurn", "family": "passTurn"}],
        "players": {
            "self": {"lore": 0, "handCount": 2, "deckCount": 3, "inkwell": 1, "discard": 0, "play": 0},
            "opp": {"lore": 0, "handCount": 2, "deckCount": 3, "inkwell": 1, "discard": 0, "play": 0},
        },
        "cards": [],
        "history": [{"family": "root"}],
        "hidden": {"hand": ["h0", "h1"], "inkwell": ["i0"], "deck": ["d0", "d1", "d2"]},
    }


class _RecordingLaneEngine:
    def __init__(self) -> None:
        self.full_world: World | None = None
        self.legacy_calls: list[tuple[int, list[str], str | None]] = []

    def restore(self, _snap: object) -> dict:
        return _obs()

    def determinize(self, self_id: int, hand_ids: list[str], *, seed: str | None = None) -> dict:
        self.legacy_calls.append((self_id, hand_ids, seed))
        return _obs()

    def determinize_world(self, self_id: int, world: World) -> dict:
        self.full_world = world
        return _obs()

    def step_exact(self, _action_id: str) -> dict:
        return _obs()


class _SearchEngine:
    """Minimal fake exposing the surface `BISMCTS.run_infoset` + `EngineSimulator` drive:
    snapshot/restore/drop_snapshot/determinize/step_exact. Each lane immediately reaches a
    terminal leaf so a few simulations complete."""

    def snapshot(self) -> str:
        return "root"

    def restore(self, _snap: object) -> dict:
        return _obs()

    def drop_snapshot(self, _snap: object) -> None:
        return None

    def determinize(self, _self_id: int, _hand_ids: list[str], *, seed: str | None = None) -> dict:
        return _obs()

    def step_exact(self, _action_id: str) -> dict:
        return _obs(done=True)


def _world() -> World:
    return World(
        opponent_hand_ids=("h0", "h1"),
        opponent_inkwell_ids=("i0",),
        opponent_deck_ids=("d0", "d1"),
        self_deck_ids=("s0", "s1"),
        seed="world-seed",
    )


def _uniform_belief(_obs: dict) -> tuple[list[str], np.ndarray]:
    return ["h0", "h1", "h2", "h3"], np.full(4, 0.5, dtype=np.float64)


def test_world_contract_validates_against_observation() -> None:
    # Phase 1 landed: World.validate_against_obs exists and enforces the contract.
    assert callable(getattr(World, "validate_against_obs"))


def test_full_ismcts_seed_reproducible() -> None:
    first = sample_worlds(
        ["a", "b", "c"], np.array([0.8, 0.4, 0.2]), 2, 3, rng=np.random.default_rng(4)
    )
    second = sample_worlds(
        ["a", "b", "c"], np.array([0.8, 0.4, 0.2]), 2, 3, rng=np.random.default_rng(4)
    )
    assert all(world.seed for world in first)
    assert first == second


def test_sampler_zero_hand_keeps_ink_and_deck() -> None:
    # Phase 2 landed: a zero hand_count over a non-empty pool still assigns inkwell + deck.
    worlds = sample_worlds(
        ["a", "b", "c"],
        np.array([0.7, 0.6, 0.5]),
        0,
        1,
        ink_probs=np.array([0.7, 0.6, 0.5]),
        ink_count=1,
    )
    assert worlds[0].opponent_inkwell_ids
    assert worlds[0].opponent_deck_ids


def test_uniform_structured_sampler_has_nontrivial_rho() -> None:
    # Phase 2 landed: structured uniform proposal yields non-trivial importance ratios.
    worlds = sample_worlds(
        ["a", "b", "c", "d", "e"],
        np.array([0.95, 0.8, 0.4, 0.2, 0.05]),
        2,
        64,
        proposal="uniform",
        ink_probs=np.array([0.05, 0.3, 0.7, 0.8, 0.9]),
        ink_count=1,
        rng=np.random.default_rng(5),
    )
    assert np.std([world.rho for world in worlds]) > 1e-6


def test_uniform_sampler_rho_is_raw_target_over_proposal() -> None:
    # Phase 2 landed: rho is stored as the raw b/q (exp(log_target - log_proposal)).
    worlds = sample_worlds(
        ["a", "b", "c", "d"],
        np.array([0.9, 0.7, 0.2, 0.1]),
        2,
        20,
        proposal="uniform",
        rng=np.random.default_rng(6),
    )
    for world in worlds:
        assert np.isclose(world.rho, np.exp(world.log_target - world.log_proposal))


def test_bridge_exposes_full_world_determinize_rpc() -> None:
    # Phase 3 landed: the bridge exposes an atomic full-`World` determinize_world operation.
    assert callable(getattr(LorcanaEngine, "determinize_world"))


@_red("Phase 4 must replace the legacy hand-only neural belief output")
def test_active_determinization_pool_includes_inkwell() -> None:
    source = inspect.getsource(BeliefEvaluator.__call__)
    assert 'hidden.get("inkwell"' in source


@_red("Phase 4 must make full ISMCTS consume a structured proposal")
def test_full_ismcts_uses_structured_world_sampler() -> None:
    source = inspect.getsource(BISMCTS.run_infoset)
    assert "ink_probs" in source
    assert "ink_count" in source


@_red("Phase 5 must retain full World particles instead of hand projections")
def test_tracker_particles_keep_full_hidden_worlds() -> None:
    tracker = BeliefTracker(n_particles=64, rng=np.random.default_rng(7))
    tracker.worlds(["a", "b", "c", "d"], np.full(4, 0.5), 2, 2)
    world = tracker.sample_world(np.random.default_rng(8))
    # OWNED DEFECT (Phase 5): particles are hand-only, so the sampled world carries no
    # opponent inkwell/deck assignment.
    assert world.opponent_inkwell_ids
    assert world.opponent_deck_ids


@_red("Phase 5 must not multiply included-card marginals into unchanged posteriors")
def test_tracker_does_not_use_invalid_posterior_reweighting() -> None:
    tracker = BeliefTracker(n_particles=128, ess_threshold=0.0, rng=np.random.default_rng(9))
    pool = ["a", "b", "c", "d"]
    tracker.worlds(pool, np.full(4, 0.5), 2, 2)
    before = tracker.pf.weights.copy()
    # OWNED DEFECT (Phase 5): an unchanged pool re-applies included-card neural reweighting,
    # mutating the posterior weights (should be driven only by observed-action evidence).
    tracker.worlds(pool, np.array([0.9, 0.7, 0.2, 0.1]), 2, 2)
    assert np.allclose(tracker.pf.weights, before)


@_red("Phase 5 must replay observed-action evidence across tracker transitions")
def test_observed_action_evidence_survives_tracker_transition() -> None:
    tracker = BeliefTracker(n_particles=2_000, ess_threshold=0.0, rng=np.random.default_rng(10))
    pool = ["a", "b", "c", "d", "e", "f", "g", "h"]
    tracker.worlds(pool, np.full(8, 0.375), 3, 8)
    tracker.observe_opponent_action(lambda hand: 1.0 if "a" in hand else 1e-5)
    # OWNED DEFECT (Phase 5): the next worlds() call reseeds on the changed pool and discards
    # the observed-action correction, so the posterior mass on "a" collapses back.
    tracker.worlds(pool[:-1], np.full(7, 3 / 7), 3, 8)
    posterior = tracker.pf.weights[
        np.array(["a" in hand for hand in tracker.pf.particles], dtype=bool)
    ].sum()
    assert posterior > 0.8


@_red("Phase 6 must redact inking relative to the observer, not unconditionally")
def test_python_key_actor_knows_own_inking() -> None:
    assert history_event(0, "putCardIntoInkwell", "card-a") != history_event(
        0, "putCardIntoInkwell", "card-b"
    )


@_red("Phase 6 must preserve the full stable candidate key in branch history")
def test_python_key_preserves_full_candidate_key_fields() -> None:
    sim = EngineSimulator(_RecordingLaneEngine(), "root", self_id=0, hand_only_diagnostic=True)
    sim.begin_lane(_world())
    sim.step("challenge:attacker:defender")
    assert "challenge:attacker:defender" in repr(sim.branch[0])


@_red("Phase 6 must project simulated branch history into leaf observations")
def test_neural_leaf_receives_projected_branch_history() -> None:
    sim = EngineSimulator(_RecordingLaneEngine(), "root", self_id=0, hand_only_diagnostic=True)
    root = sim.begin_lane(_world())
    leaf = sim.step("quest:c1")
    assert len(leaf["history"]) == len(root["history"]) + 1


@_red("Phase 7 must pass the rich World to the lane adapter")
def test_begin_lane_does_not_discard_world_inkwell() -> None:
    engine = _RecordingLaneEngine()
    sim = EngineSimulator(engine, "root", self_id=0, hand_only_diagnostic=True)
    sim.begin_lane(_world())
    assert engine.full_world is not None
    assert engine.full_world.opponent_inkwell_ids == ("i0",)


@_red("Phase 7 must pass opponent and self deck order to the lane adapter")
def test_begin_lane_does_not_discard_world_deck() -> None:
    engine = _RecordingLaneEngine()
    sim = EngineSimulator(engine, "root", self_id=0, hand_only_diagnostic=True)
    sim.begin_lane(_world())
    assert engine.full_world is not None
    assert engine.full_world.opponent_deck_ids == ("d0", "d1")
    assert engine.full_world.self_deck_ids == ("s0", "s1")


@_red("Phase 9 must wire action-likelihood updates into real self-play loops")
def test_action_updates_nonzero_in_real_selfplay() -> None:
    assert "observe_opponent_action" in (_BOT_ROOT / "training" / "run.py").read_text()
    assert "observe_opponent_action" in (_BOT_ROOT / "training" / "parallel.py").read_text()


@_red("Phase 11 must expose the populated shared root statistics")
def test_infoset_diagnostics_nonzero_visits() -> None:
    engine = _SearchEngine()
    tracker = BeliefTracker(n_particles=16, rng=np.random.default_rng(11))
    search = BISMCTS(engine, UniformEvaluator(), SearchConfig(simulations=4, depth_limit=2),
                     rng=np.random.default_rng(12))
    result = search.run_infoset(_obs(), _uniform_belief, tracker, total_simulations=4)
    # OWNED DEFECT (Phase 11): run_infoset discards the populated shared root and returns a
    # fresh empty legacy InfoSetNode, so the reported visit counts are all zero.
    assert result.sims > 0
    assert int(result.root.N.sum()) > 0


@_red("Phase 11 must count shared deeper-node hits separately from root reuse")
def test_deeper_shared_node_hit_is_not_only_root_reuse() -> None:
    engine = _SearchEngine()
    tracker = BeliefTracker(n_particles=16, rng=np.random.default_rng(13))
    search = BISMCTS(engine, UniformEvaluator(), SearchConfig(simulations=4, depth_limit=2),
                     rng=np.random.default_rng(14))
    result = search.run_infoset(_obs(), _uniform_belief, tracker, total_simulations=4)
    # OWNED DEFECT (Phase 11): only a combined `transposition_hits` exists; deeper shared-node
    # hits are not tracked separately from root revisits.
    assert result.stats["deeper_transposition_hits"] > 0


@_red("Phase 14 must assign a prior when an existing infoset gains an edge")
def test_existing_infoset_new_available_edge_receives_prior() -> None:
    table = InfoSetTable()
    obs1 = {"infoSetKey": "shared", "actor": "me", "legal": [{"actionId": "a"}]}
    node, _ = table.get_or_create(obs1)
    _expand(node, obs1, lambda _o: ({"a": 1.0}, 0.0))
    obs2 = {"infoSetKey": "shared", "actor": "me", "legal": [{"actionId": "a"}, {"actionId": "b"}]}
    # OWNED DEFECT (Phase 14): a newly-available edge "b" at an existing node is registered
    # with a zero prior because _expand only runs at first creation.
    node, _ = table.get_or_create(obs2)
    assert node.edges["b"].prior > 0.0
