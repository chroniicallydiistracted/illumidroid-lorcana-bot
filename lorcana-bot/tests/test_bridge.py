"""Phase 1A — cross-boundary determinism + action/snapshot semantics."""

import json

import pytest

from engine.bridge import LorcanaEngine, LorcanaEnv


def _fingerprint(obs: dict) -> str:
    s = obs["players"]["self"]
    o = obs["players"]["opp"]
    return json.dumps({
        "turn": obs["turn"], "phase": obs["phase"], "status": obs["status"],
        "actor": obs["actor"], "self": s, "opp": o,
        "n_legal": len(obs["legal"]),
        "keys": [a["stableKey"] for a in obs["legal"]],
    }, sort_keys=True)


@pytest.fixture
def engine():
    e = LorcanaEngine()
    yield e
    e.close()


def test_reset_has_legal_actions(engine):
    obs = engine.reset("t-reset")
    assert obs["status"] == "playing"
    assert len(obs["legal"]) >= 1
    assert obs["actor"] is not None


def test_same_seed_identical_scripted_trajectory():
    """Same seed => bit-identical trajectory across two independent processes."""
    def trace(seed, n=40):
        e = LorcanaEngine()
        try:
            obs = e.reset(seed)
            fps = [_fingerprint(obs)]
            for _ in range(n):
                if obs["done"]:
                    break
                obs = e.step_auto("best")["obs"]
                fps.append(_fingerprint(obs))
            return fps
        finally:
            e.close()

    a = trace("determinism-seed")
    b = trace("determinism-seed")
    assert a == b, "same seed produced divergent trajectories"


def test_different_seed_diverges():
    """Different seeds => different shuffles => divergent action sequences.

    Compared at the executed-action level (keys carry card *instance* ids), since
    the coarse per-step fingerprint can coincide with homogeneous fixture decks.
    Hidden-RNG divergence itself is proven in Phase 0.
    """
    def action_trace(seed, n=60):
        e = LorcanaEngine()
        try:
            obs = e.reset(seed)
            keys = []
            for _ in range(n):
                if obs["done"]:
                    break
                r = e.step_auto("best")
                keys.append(r.get("executed"))
                obs = r["obs"]
            return keys
        finally:
            e.close()

    assert action_trace("seed-A") != action_trace("seed-B")


def test_chosen_action_executes_exactly(engine):
    obs = engine.reset("t-choose")
    # advance to a richer decision
    for _ in range(4):
        if obs["done"]:
            break
        obs = engine.step_auto("best")["obs"]
    if obs["done"]:
        pytest.skip("game ended during warmup")
    key = obs["legal"][0]["stableKey"]
    res = engine.step(key)
    assert res["success"]
    # passTurn fallbacks aside, the requested key should be what executed
    if res["executed"] != "passTurn":
        assert res["matched"], f"requested {key} but executed {res['executed']}"


def test_snapshot_restore_roundtrip(engine):
    obs = engine.reset("t-snap")
    for _ in range(5):
        if obs["done"]:
            break
        obs = engine.step_auto("best")["obs"]
    before = _fingerprint(obs)
    snap = engine.snapshot()
    # mutate
    if not obs["done"]:
        engine.step(obs["legal"][0]["stableKey"])
    restored = engine.restore(snap)
    assert _fingerprint(restored) == before
    engine.drop_snapshot(snap)


def test_env_plays_full_random_game():
    import random
    env = LorcanaEnv(max_steps=400)
    try:
        obs, mask = env.reset("t-env")
        rng = random.Random(1)
        done = False
        steps = 0
        while not done:
            a = rng.randrange(len(env.legal))
            obs, mask, reward, done, info = env.step(a)
            steps += 1
        assert obs["done"] or info["truncated"]
        if obs["done"]:
            assert reward in (-1.0, 0.0, 1.0)
    finally:
        env.close()
