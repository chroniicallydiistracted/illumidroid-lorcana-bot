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


def test_runpaths_exact_executes_requested_key(engine):
    """A legal stable key descended by search executes exactly (no invalidPath)."""
    obs = engine.reset("t-exact")
    snap = engine.snapshot()
    key = obs["legal"][0]["stableKey"]
    [leaf] = engine.run_paths(snap, [[key]])
    assert not leaf.get("invalidPath"), "a legal key must execute exactly, not fall back"
    engine.drop_snapshot(snap)


def test_runpaths_rejects_missing_key_no_fallback(engine):
    """A bogus stable key must NOT silently execute a fallback candidate — the
    leaf is flagged invalidPath so search can discard it (doc #1 / #7)."""
    obs = engine.reset("t-bogus")
    snap = engine.snapshot()
    [leaf] = engine.run_paths(snap, [["challenge:NOTAREALCARD:ALSOBOGUS"]])
    assert leaf.get("invalidPath") is True, "missing key should flag invalidPath, not fall back"
    assert leaf.get("failedAtDepth") == 0
    engine.drop_snapshot(snap)


def test_automation_grammar_gap_is_only_known_residual(engine):
    """PROOF (regression): the automation enumeration the bot uses covers the FULL
    engine grammar EXCEPT a known, measured residual — nested triggered-ability
    (bag/effect) resolution ("exceeds the v1 automation support matrix"), ~0.1% of
    decisions (see illumidroid_grammar_gap_proof.md). The invariant we guard: any
    `unsupported-shape` the engine reports must be in the resolveBag/resolveEffect
    family. If a NEW gap appears in a real action family (playCard/quest/challenge/
    ability/ink/move) — the only case that would justify the raw getAvailableMoves
    grammar — this test fails."""
    KNOWN_GAP = {"resolveBag", "resolveEffect"}
    decisions = 0
    bad: list[dict] = []
    for g in range(4):
        obs = engine.reset(f"grammar-gap-{g}")
        steps = 0
        while not obs.get("done") and steps < 120:
            pr = engine.grammar_probe()
            if pr.get("actor"):
                decisions += 1
                for d in pr.get("unsupported", []):
                    if d.get("family") not in KNOWN_GAP:
                        bad.append(d)
            obs = engine.step_auto("best")["obs"]
            steps += 1
    assert decisions > 50, "expected to sample real decisions"
    assert not bad, f"NEW grammar gap outside the known bag/effect residual: {bad[:5]}"


def test_determinize_is_leak_free(engine):
    """#1: a determinized world must RANDOMIZE the opponent's hidden inkwell (real
    inked identities are hidden), not leave the real ones in place. Across seeds
    the sampled inkwell varies, while the full hidden multiset is conserved."""
    import pytest
    obs = engine.reset("det-leak")
    for _ in range(50):
        if obs.get("done"):
            break
        h = obs.get("hidden", {})
        if len(h.get("inkwell", [])) >= 1 and len(h.get("deck", [])) >= 3:
            break
        obs = engine.step_auto("best")["obs"]
    hid = obs.get("hidden", {})
    if len(hid.get("inkwell", [])) < 1 or len(hid.get("deck", [])) < 3:
        pytest.skip("did not reach a state with opp inkwell + deck")
    self_id = obs["self"]
    real_ink = frozenset(c["id"] for c in hid["inkwell"])
    hand_ids = [c["id"] for c in hid["hand"]]
    allpool = set(hand_ids) | {c["id"] for c in hid["deck"]} | set(real_ink)
    snap = engine.snapshot()
    samples = []
    for s in ("a", "b", "c", "d"):
        engine.restore(snap)
        engine.determinize(self_id, hand_ids, seed=s)
        dh = engine.observe()["hidden"]
        ink = frozenset(c["id"] for c in dh["inkwell"])
        assert len(ink) == len(real_ink)                       # count-consistent
        full = set(c["id"] for c in dh["hand"]) | {c["id"] for c in dh["deck"]} | set(ink)
        assert full == allpool                                 # conservation (no leak in/out)
        samples.append(ink)
    engine.drop_snapshot(snap)
    # leak-free: the inkwell is RESAMPLED, so it is not always the real inked set
    assert any(s != real_ink for s in samples), "opp inkwell never randomized (leak)"


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
