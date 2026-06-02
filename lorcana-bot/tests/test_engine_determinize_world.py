"""Tier-A Phase 3 — full-`World` determinization RPC (real engine, bot-owned bridge/server).

Proves `determinize_world`:
  * HONORS the exact opponent hand/inkwell/deck partition + optional self-deck order;
  * is observer-aware — it pins ONLY ACTOR-VISIBLE facts (cards the actor sees via the fog
    projection + reveal `visibleTo` / scry / static top-deck), positionally for decks, across all
    four repartitioned zones; a PRIVATE engine reference to a hidden card does NOT pin its identity
    (count-only metrics are ignored; identity-bearing private refs quarantine the state) — so the
    belief is never conditioned on hidden truth;
  * never leaks an OPPONENT hidden-zone identity into the actor-visible `obs["cards"]`
    (Finding #3) while keeping the privileged `obs["hidden"]` witness for the sampler;
  * FAILS CLOSED (raises, live state untouched) on malformed / duplicate / non-conserving /
    protected-fact-violating specs and invalid seats / seeds — no silent repair;
  * is deterministic from `World.seed` (never ambient randomness), pinning protected positions.

These exercise the REAL Bun engine through the bridge — not a mock — per the Agent Requirement
Contract. Skips are narrow and specific (a precise engine precondition is absent at the reached
state), never "flaky"/"not implemented"."""

import pathlib

import pytest

from engine.bridge import LorcanaEngine, BridgeError
from search.determinize import World

_SERVER_TS = pathlib.Path(__file__).resolve().parents[1] / "engine" / "node_server" / "server.ts"


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def midgame():
    """A real mid-game state where the opponent has >=2 hidden inkwell + >=2 deck cards and the
    actor has a self deck of >=2 (so a free inkwell<->deck swap and an observable self-deck order
    exist)."""
    e = LorcanaEngine()
    obs = e.reset("p3-determinize-world")
    for _ in range(60):
        if obs["done"]:
            break
        opp, me = obs["players"]["opp"], obs["players"]["self"]
        if (opp["inkwell"] >= 2 and opp["deckCount"] >= 2 and me["deckCount"] >= 2
                and len(obs["hidden"]["inkwell"]) >= 2 and len(obs["hidden"]["deck"]) >= 2):
            break
        obs = e.step_auto("best")["obs"]
    yield e, obs
    e.close()


@pytest.fixture
def self_deck_pin_state():
    """A real state where the CURRENT actor has a PROTECTED self-deck position — a top-of-deck
    card whose identity (and position) the actor legally knows. Used to prove the self deck is no
    longer ignored: the seeded shuffle keeps the known card in place, and a world that displaces
    it is rejected."""
    e = None
    found = None
    for seed in ("s7", "p3-reveal", "s1", "s3", "s5", "deck-scry"):
        e = LorcanaEngine()
        obs = e.reset(seed)
        for _ in range(60):
            if obs["done"]:
                break
            pf = e.protected_facts()
            sid = pf["self"]
            if sid:
                spins = pf["pins"].get(f"deck:{sid}", [])
                if spins and obs["players"]["self"]["deckCount"] >= 3:
                    found = (obs, spins)
                    break
            obs = e.step_auto("best")["obs"]
        if found:
            break
        e.close()
        e = None
    try:
        if not found:
            pytest.skip("no protected self-deck position reached across the searched seeds")
        yield e, found[0], found[1]
    finally:
        if e is not None:
            e.close()


@pytest.fixture
def pinned_state():
    """A real state where a PROTECTED FACT lands inside the current actor's repartitioned pool —
    a card the determinization must not move (reveal window / live reference). Reliably produced
    by manual inking (which calls the engine's `zones.reveal`, revealing the inked card to all)."""
    e = LorcanaEngine()
    obs = e.reset("p3-reveal")
    found = None
    for _ in range(80):
        if obs["done"]:
            break
        pinned = set(e.protected_facts()["pinnedIds"])
        hand, ink, deck = _opp_pool(obs)
        pool = set(hand + ink + deck)
        if (pinned & pool) and len(ink) >= 1 and len(deck) >= 1:
            found = (obs, pinned & pool)
            break
        obs = e.step_auto("best")["obs"]
    yield e, found
    e.close()


@pytest.fixture
def opp_private_reveal_state():
    """A real state where a card in the current actor's OPPONENT repartitioned pool is in an
    active reveal window that is NOT visible to the actor (an opponent-private reveal). The actor
    must NOT treat it as known (Finding #2) — it is excluded from the protected-facts ledger and
    the belief stays free to vary it."""
    e = None
    found = None
    for seed in ("s0", "s1", "s2", "s3", "s4", "p3-reveal"):
        e = LorcanaEngine()
        obs = e.reset(seed)
        for _ in range(60):
            if obs["done"]:
                break
            revealed = {r["id"] for r in e.check_consistency()["revealedIds"]}
            if revealed:
                pinned = set(e.protected_facts()["pinnedIds"])
                hand, ink, deck = _opp_pool(obs)
                pool = set(hand + ink + deck)
                private = (revealed & pool) - pinned     # revealed, in opp pool, but not actor-known
                if private and len(deck) >= 1 and len(ink) >= 1:
                    found = (obs, private)
                    break
            obs = e.step_auto("best")["obs"]
        if found:
            break
        e.close()
        e = None
    try:
        if not found:
            pytest.skip("no opponent-private reveal in the actor pool across the searched seeds")
        yield e, found[0], found[1]
    finally:
        if e is not None:
            e.close()


@pytest.fixture
def count_only_ref_state():
    """A real state where an OPPONENT-HIDDEN card is referenced by a COUNT-ONLY engine metric
    (e.g. a top-deck-to-inkwell effect storing its private id in `turnMetadata.inkedThisTurn` /
    `cardsPutIntoInkwellThisTurn`, consumed only as a count). The actor cannot see the card, so the
    ledger must NOT pin it — the belief stays free to reassign it (the prior merged-pin code pinned
    it, conditioning the posterior on hidden truth)."""
    e = None
    found = None
    for seed in ("s2", "s0", "s4", "s6", "s9", "topdeck"):
        e = LorcanaEngine()
        obs = e.reset(seed)
        for _ in range(50):
            if obs["done"]:
                break
            pf = e.protected_facts()
            opp_hidden = {c["id"] for z in ("hand", "inkwell", "deck") for c in obs["hidden"][z]}
            co = [i for i in pf.get("countOnlyRefIds", []) if i in opp_hidden]
            if co:
                found = (obs, co)
                break
            obs = e.step_auto("best")["obs"]
        if found:
            break
        e.close()
        e = None
    try:
        if not found:
            pytest.skip("no opponent-hidden count-only engine reference reached across the searched seeds")
        yield e, found[0], found[1]
    finally:
        if e is not None:
            e.close()


# --- helpers -----------------------------------------------------------------
def _opp_pool(obs):
    h = obs["hidden"]
    return ([c["id"] for c in h["hand"]],
            [c["id"] for c in h["inkwell"]],
            [c["id"] for c in h["deck"]])


def _ready(obs):
    if obs["done"]:
        pytest.skip("game ended before a usable hidden-zone midgame state")
    hand, ink, deck = _opp_pool(obs)
    if not (len(ink) >= 1 and len(deck) >= 1):
        pytest.skip("opponent has no hidden inkwell+deck cards to repartition at this state")
    return hand, ink, deck


def _world(hand, ink, deck, *, self_deck=None, seed="w"):
    return World(opponent_hand_ids=tuple(hand), opponent_inkwell_ids=tuple(ink),
                 opponent_deck_ids=tuple(deck),
                 self_deck_ids=(None if self_deck is None else tuple(self_deck)), seed=seed)


def _pinned(eng):
    """The FULL observer-aware protected-facts ledger for the current actor (reveal / scry /
    static top-deck + every live runtime reference), as the determinizer enforces it."""
    return set(eng.protected_facts()["pinnedIds"])


def _free(ids, pinned):
    return [c for c in ids if c not in pinned]


def _swap_in_place(zone, a, b):
    """Replace id `a` with id `b` in `zone`, KEEPING every other position (so any pinned card
    keeps its index). Caller guarantees `a in zone` and `b not in zone`."""
    return [b if c == a else c for c in zone]


# --- honors requested partitions (ledger-respecting in-place swaps) ----------
def test_determinize_world_honors_requested_opponent_inkwell(midgame):
    """Two different requested inkwell assignments produce two different ACTUAL inkwell
    assignments, each matching its request EXACTLY."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    pinned = _pinned(eng)
    fi, fd = _free(ink, pinned), _free(deck, pinned)
    if not fi or not fd:
        pytest.skip("no free ink+deck pair to swap at this state")
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, seed="A"))
        ink_a = set(eng.last_world_realized["opponentInkwellIds"])
        eng.restore(root)
        ink2 = _swap_in_place(ink, fi[0], fd[0])
        deck2 = _swap_in_place(deck, fd[0], fi[0])
        eng.determinize_world(self_id, _world(hand, ink2, deck2, seed="B"))
        ink_b = set(eng.last_world_realized["opponentInkwellIds"])
        assert ink_a == set(ink) and ink_b == set(ink2)          # each honored exactly
        assert ink_a != ink_b                                     # different requests -> different actuals
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_honors_requested_opponent_deck(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    pinned = _pinned(eng)
    fi, fd = _free(ink, pinned), _free(deck, pinned)
    if not fi or not fd:
        pytest.skip("no free ink+deck pair to swap at this state")
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, seed="A"))
        deck_a = set(eng.last_world_realized["opponentDeckIds"])
        eng.restore(root)
        ink2 = _swap_in_place(ink, fi[0], fd[0])
        deck2 = _swap_in_place(deck, fd[0], fi[0])
        eng.determinize_world(self_id, _world(hand, ink2, deck2, seed="B"))
        deck_b = set(eng.last_world_realized["opponentDeckIds"])
        assert deck_a == set(deck) and deck_b == set(deck2)
        assert deck_a != deck_b
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_honors_requested_self_deck(midgame):
    """A SUPPLIED self-deck order is honored EXACTLY (order, not just membership)."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="s0"))
        s = list(eng.last_world_realized["selfDeckIds"])
        if len(s) < 2:
            pytest.skip("self deck < 2; ordering not observable")
        eng.restore(root)
        # reverse only the FREE self-deck positions so any pinned self-deck card keeps its slot
        pins = {p["index"] for p in eng.protected_facts()["pins"].get(f"deck:{self_id}", [])}
        free_idx = [i for i in range(len(s)) if i not in pins]
        target = list(s)
        for k, i in enumerate(free_idx):
            target[i] = s[free_idx[len(free_idx) - 1 - k]]
        if target == s:
            pytest.skip("self deck order not permutable (all positions pinned)")
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=target, seed="s1"))
        assert eng.last_world_realized["selfDeckIds"] == target    # exact order honored
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


# --- Findings #1/#2: observer-aware protected-facts ledger -------------------
def test_protected_facts_surface_is_real_and_nonempty(pinned_state):
    """The ledger surface is real: at a reached state a protected fact lands in the actor's
    repartitioned pool (reveal window / live reference), exposed per-zone with its index."""
    eng, found = pinned_state
    if not found:
        pytest.skip("no protected fact landed in the actor pool within the search horizon")
    obs, pinpool = found
    pf = eng.protected_facts()
    assert pf["self"] == obs["self"]
    flat = {p["id"] for zone in pf["pins"].values() for p in zone}
    assert pinpool <= set(pf["pinnedIds"]) and pinpool <= flat


def test_determinize_world_rejects_moving_a_protected_card(pinned_state):
    """Finding #1/#2 (rejection): a world that MOVES a protected (actor-known / live-referenced)
    card to a different zone is rejected fail-closed; a world that keeps every protected card in
    place is accepted."""
    eng, found = pinned_state
    if not found:
        pytest.skip("no protected fact landed in the actor pool within the search horizon")
    obs, pinpool = found
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    target = next(iter(pinpool))
    root = eng.snapshot()
    try:
        if target in ink:
            free_deck = _free(deck, set())
            w = _world(hand, _swap_in_place(ink, target, deck[0]),
                       _swap_in_place(deck, deck[0], target), seed="rv")
        elif target in deck:
            w = _world(hand, _swap_in_place(ink, ink[0], target),
                       _swap_in_place(deck, target, ink[0]), seed="rv")
        else:  # in hand
            w = _world(_swap_in_place(hand, target, deck[0]), ink,
                       _swap_in_place(deck, deck[0], target), seed="rv")
        with pytest.raises(BridgeError, match="actor-visible fact"):
            eng.determinize_world(self_id, w)
        # keeping every protected card in place is NOT rejected for this reason
        eng.determinize_world(self_id, _world(hand, ink, deck, seed="keep"))
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_failed_protected_violation_preserves_state(pinned_state):
    """Finding #6 (staged rollback): a world rejected for a protected-fact violation leaves the
    live engine state byte-identical (observation, legal moves, consistency)."""
    eng, found = pinned_state
    if not found:
        pytest.skip("no protected fact landed in the actor pool within the search horizon")
    obs, pinpool = found
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    target = next(iter(pinpool))
    root = eng.snapshot()
    try:
        before, c_before = eng.observe(), eng.check_consistency()
        if target in ink:
            w = _world(hand, _swap_in_place(ink, target, deck[0]),
                       _swap_in_place(deck, deck[0], target), seed="rv")
        elif target in deck:
            w = _world(hand, _swap_in_place(ink, ink[0], target),
                       _swap_in_place(deck, target, ink[0]), seed="rv")
        else:
            w = _world(_swap_in_place(hand, target, deck[0]), ink,
                       _swap_in_place(deck, deck[0], target), seed="rv")
        with pytest.raises(BridgeError, match="actor-visible fact"):
            eng.determinize_world(self_id, w)
        after = eng.observe()
        assert [a["stableKey"] for a in after["legal"]] == [a["stableKey"] for a in before["legal"]]
        assert after["cards"] == before["cards"]
        assert eng.check_consistency() == c_before
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_preserves_protected_deck_position(pinned_state):
    """Finding #1 (positional): a deck card whose position is known (top/bottom reveal, scry,
    static top-deck) must keep its EXACT index; a same-membership reorder that displaces it is
    rejected. Skips precisely if no protected DECK card is present at the reached state."""
    eng, found = pinned_state
    if not found:
        pytest.skip("no protected fact landed in the actor pool within the search horizon")
    obs, _ = found
    self_id, opp_id = obs["self"], obs["opp"]
    pf = eng.protected_facts()
    deck_key = f"deck:{opp_id}"
    deck_pins = pf["pins"].get(deck_key, [])
    hand, ink, deck = _ready(obs)
    if not deck_pins or len(deck) < 2:
        pytest.skip("no protected opponent-deck position at this state")
    pin = deck_pins[0]
    # find a free deck position to swap the pinned card with (a pure reorder, same membership)
    free_positions = [i for i in range(len(deck)) if i != pin["index"]
                      and deck[i] not in set(pf["pinnedIds"])]
    if not free_positions:
        pytest.skip("no free deck position to reorder against the pin")
    j = free_positions[0]
    reordered = list(deck)
    reordered[pin["index"]], reordered[j] = reordered[j], reordered[pin["index"]]
    root = eng.snapshot()
    try:
        with pytest.raises(BridgeError, match="actor-visible fact"):
            eng.determinize_world(self_id, _world(hand, ink, reordered, seed="reorder"))
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_seeded_self_deck_shuffle_preserves_protected_position(self_deck_pin_state):
    """Finding #1 (self deck no longer ignored): when the world omits the self-deck order the
    server shuffles it DETERMINISTICALLY, but PINS every protected (actor-known) self-deck
    position in place — the known top-deck card stays at its index while the free remainder is
    permuted."""
    eng, obs, spins = self_deck_pin_state
    hand, ink, deck = _opp_pool(obs)          # opponent pool passed as identity; focus is self deck
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="pinA"))
        realized = list(eng.last_world_realized["selfDeckIds"])
        for p in spins:
            assert realized[p["index"]] == p["id"], "seeded shuffle moved a protected self-deck card"
        # the FREE remainder is actually shuffled (different seed -> different free order), proving
        # the pin is a targeted constraint, not a no-op identity shuffle.
        pinned_idx = {p["index"] for p in spins}
        free_now = [c for i, c in enumerate(realized) if i not in pinned_idx]
        if len(free_now) >= 2:
            eng.restore(root)
            eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="pinB"))
            r2 = list(eng.last_world_realized["selfDeckIds"])
            for p in spins:
                assert r2[p["index"]] == p["id"]            # still pinned under a different seed
            assert r2 != realized                            # free remainder genuinely permuted
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_displacing_protected_self_deck_position(self_deck_pin_state):
    """Finding #1 (rejection): a SUPPLIED self-deck order that moves a protected self-deck card off
    its known index is rejected fail-closed."""
    eng, obs, spins = self_deck_pin_state
    hand, ink, deck = _opp_pool(obs)          # opponent pool passed as identity; focus is self deck
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="base"))
        s = list(eng.last_world_realized["selfDeckIds"])
        eng.restore(root)
        pin = spins[0]
        other = next(i for i in range(len(s)) if i != pin["index"])
        displaced = list(s)
        displaced[pin["index"]], displaced[other] = displaced[other], displaced[pin["index"]]
        with pytest.raises(BridgeError, match="actor-visible fact"):
            eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=displaced, seed="bad"))
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


# --- Finding #3: opponent hidden identity never leaks into obs["cards"] ------
def test_obs_never_exposes_opponent_hidden_identity(midgame):
    """No opponent hidden-zone instance id (inkwell/limbo are tagged with the RAW id by the
    fog projection) appears in the actor-visible obs["cards"]; opponent hidden rows are redacted
    to stable slot placeholders with null identity."""
    eng, obs = midgame
    _ready(obs)
    card_ids = {c["id"] for c in obs["cards"]}
    opp_hidden = {c["id"] for z in ("hand", "inkwell", "deck") for c in obs["hidden"][z]}
    assert not (opp_hidden & card_ids), "opponent hidden id leaked into obs['cards']"
    for c in obs["cards"]:
        if c["owner"] == 1 and c["hidden"]:
            assert c["id"].startswith("oppslot:")          # placeholder, not a real instance id
            assert c["definitionId"] is None and c["name"] is None


def test_hidden_swap_is_observationally_invariant_to_actor(midgame):
    """An ordinary opponent inkwell<->deck identity swap changes the privileged sampler witness
    (obs["hidden"]) but leaves the ACTOR-visible observation (obs["cards"]) byte-identical — the
    opponent's hidden identities do not reach the actor (Finding #3)."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    pinned = _pinned(eng)
    fi, fd = _free(ink, pinned), _free(deck, pinned)
    if not fi or not fd:
        pytest.skip("no free ink+deck pair to swap at this state")
    before = eng.observe()
    root = eng.snapshot()
    try:
        ink2 = _swap_in_place(ink, fi[0], fd[0])
        deck2 = _swap_in_place(deck, fd[0], fi[0])
        eng.determinize_world(self_id, _world(hand, ink2, deck2, seed="inv"))
        after = eng.observe()
        assert after["cards"] == before["cards"], "actor-visible cards changed under a hidden swap (leak)"
        assert after["hidden"] != before["hidden"], "the privileged sampler witness did not change"
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_obs_card_ids_are_unique(midgame):
    """Finding #4: every redacted opponent hidden card gets a DISTINCT slot placeholder (native
    projection assigns slot 0 to all hidden limbo cards, which under a `zoneIndex`-based scheme
    would collide and overwrite token positions). No two obs["cards"] rows share an id."""
    eng, obs = midgame
    _ready(obs)
    ids = [c["id"] for c in obs["cards"]]
    assert len(ids) == len(set(ids)), "duplicate obs['cards'] id (placeholder collision risk)"


def test_opponent_hidden_redaction_is_zone_specific(midgame):
    """Finding #5: redaction forwards only fields PROVEN public for that hidden zone. Opponent
    hidden HAND/DECK rows expose no per-card physical state (neutral exerted/ready/damage); only
    the inkwell may forward exertion (public spent ink)."""
    eng, obs = midgame
    _ready(obs)
    saw_nonink = False
    for c in obs["cards"]:
        if c["owner"] == 1 and c["hidden"] and c["zone"] in ("hand", "deck", "limbo"):
            saw_nonink = True
            assert c["exerted"] is False and c["ready"] is True and c["damage"] == 0 \
                and c["drying"] is False, f"non-inkwell hidden row leaked physical state: {c}"
    if not saw_nonink:
        pytest.skip("no opponent hidden hand/deck/limbo rows at this state")


def test_opponent_private_reveal_is_not_actor_knowledge(opp_private_reveal_state):
    """Finding #2: a reveal window NOT visible to the actor (opponent-private) must NOT pin the
    card — the actor does not legally know it, so the belief stays free to reassign it. The card
    is absent from the ledger and a world that MOVES it is accepted (the prior blind reveal scan,
    which pinned every active reveal regardless of `visibleTo`, would have rejected the move)."""
    eng, obs, private = opp_private_reveal_state
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    target = next(iter(private))
    assert target not in set(eng.protected_facts()["pinnedIds"])      # not treated as actor knowledge
    root = eng.snapshot()
    try:
        if target in ink:
            w = _world(hand, _swap_in_place(ink, target, deck[0]),
                       _swap_in_place(deck, deck[0], target), seed="mv")
        elif target in deck:
            w = _world(hand, _swap_in_place(ink, ink[0], target),
                       _swap_in_place(deck, target, ink[0]), seed="mv")
        else:
            w = _world(_swap_in_place(hand, target, deck[0]), ink,
                       _swap_in_place(deck, deck[0], target), seed="mv")
        eng.determinize_world(self_id, w)                            # accepted: belief may vary it
        moved = (list(eng.last_world_realized["opponentInkwellIds"])
                 + list(eng.last_world_realized["opponentDeckIds"])
                 + list(eng.last_world_realized["opponentHandIds"]))
        assert target in moved                                       # still conserved, just reassigned
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


# --- regression guards: the production information-set logic is correct -------
def _ledger_src():
    src = _SERVER_TS.read_text()
    start = src.index("private collectProtectedFacts(")
    end = src.index("protectedFacts(): {", start)
    return src[start:end]


def test_protected_facts_projection_fails_closed_not_open():
    """Finding #1: the actor-visible projection is REQUIRED — `collectProtectedFacts` must NOT
    swallow a projection failure and proceed without actor pins. The body throws on a missing
    projection and contains no bare catch-swallow."""
    body = _ledger_src()
    assert "catch {" not in body and "catch (" not in body, "projection failure is swallowed (fail-open)"
    assert "projection unavailable" in body and "throw new Error" in body, "no fail-closed projection guard"


def test_protected_facts_reveal_scan_is_visibleto_filtered():
    """Finding #2: any reveal-window read for actor knowledge must apply the native
    `isCardVisibleViaReveal` `visibleTo` rule (`"all"` or a window naming the actor), so an
    opponent-PRIVATE reveal is never treated as actor knowledge."""
    body = _ledger_src()
    assert "zones.reveals" in body, "reveal-window actor knowledge not consulted at all"
    assert 'vt === "all"' in body and "vt.includes(selfId)" in body, "reveal scan ignores visibleTo"


def test_protected_facts_scans_card_ids_stored_as_map_keys():
    """Finding #3: the live-reference scan must inspect MAP KEYS (engine stores card ids as keys,
    e.g. continuousEffects.byTarget, turnMetadata.cardsUnderThisTurn), not only values."""
    body = _ledger_src()
    assert "idUniverse.has(k)" in body and "classify(k" in body, "scan does not check map keys"


# --- engine refs SEPARATE from actor-visible pins (count-only ignored) --------
def test_count_only_private_reference_does_not_pin_hidden_identity(count_only_ref_state):
    """The cited leak: a private engine reference to a HIDDEN card in a COUNT-ONLY metric must NOT
    pin its exact identity (that conditions the actor's posterior on hidden truth). The card is
    absent from `pins`, present in `countOnlyRefIds`, and a world that REASSIGNS it is accepted —
    the prior code unioned engine refs into the pin map and would have rejected the move."""
    eng, obs, co = count_only_ref_state
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    pf = eng.protected_facts()
    target = co[0]
    assert target not in set(pf["pinnedIds"]), "count-only-referenced hidden card was pinned (leak)"
    assert target in set(pf["countOnlyRefIds"])
    root = eng.snapshot()
    try:
        if target in ink:
            w = _world(hand, _swap_in_place(ink, target, deck[0]),
                       _swap_in_place(deck, deck[0], target), seed="co")
        elif target in deck:
            w = _world(hand, _swap_in_place(ink, ink[0], target),
                       _swap_in_place(deck, target, ink[0]), seed="co")
        else:
            w = _world(_swap_in_place(hand, target, deck[0]), ink,
                       _swap_in_place(deck, deck[0], target), seed="co")
        eng.determinize_world(self_id, w)                       # accepted: belief free to reassign it
        moved = (list(eng.last_world_realized["opponentInkwellIds"])
                 + list(eng.last_world_realized["opponentDeckIds"])
                 + list(eng.last_world_realized["opponentHandIds"]))
        assert target in moved
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_protected_facts_sets_are_disjoint(midgame):
    """The three ledger classes are represented SEPARATELY and never overlap: an id is at most one
    of actor-visible (`pinnedIds`), identity-bearing-private (`quarantineIds`), or count-only
    (`countOnlyRefIds`)."""
    eng, obs = midgame
    _ready(obs)
    pf = eng.protected_facts()
    pins = set(pf["pinnedIds"]); quar = set(pf["quarantineIds"]); co = set(pf["countOnlyRefIds"])
    assert pins.isdisjoint(quar) and pins.isdisjoint(co) and quar.isdisjoint(co)


def test_scry_revealed_self_deck_card_is_actor_visible_not_quarantined(self_deck_pin_state):
    """The actor's OWN scry (a reveal `visibleTo` the actor over mid-deck cards the board
    projection does not surface) is ACTOR knowledge: such cards are pinned (exact id) and the
    determinization proceeds — NOT quarantined. Guards against the projection-only ledger wrongly
    quarantining the actor's own scry decisions."""
    eng, obs, spins = self_deck_pin_state
    hand, ink, deck = _opp_pool(obs)
    self_id = obs["self"]
    pf = eng.protected_facts()
    assert not pf["quarantineIds"], f"actor's own knowledge wrongly quarantined: {pf['quarantineIds']}"
    assert spins[0]["id"] in set(pf["pinnedIds"])               # the known self-deck card is pinned
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="scry"))
        realized = list(eng.last_world_realized["selfDeckIds"])
        assert realized[spins[0]["index"]] == spins[0]["id"]   # pinned position preserved
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


# --- state integrity: three structures agree, owner+controller preserved -----
def test_determinize_world_state_stays_consistent_and_preserves_ink(midgame):
    """After a cross-zone repartition the engine's cardIndex agrees with zoneCards (no stale
    location), no card is duplicated, owner AND controller are intact (badOwner==0), reverse-index
    orphans and public zone summaries are unchanged, and the opponent's public available-ink
    (ready count) is preserved — proven before AND while playing 12 ply forward."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id, opp_id = obs["self"], obs["opp"]
    ink_key = f"inkwell:{opp_id}"
    pinned = _pinned(eng)
    fh, fi, fd = _free(hand, pinned), _free(ink, pinned), _free(deck, pinned)
    if len(fd) < 2 or len(fh) < 1 or len(fi) < 1:
        pytest.skip("not enough free cards to construct a cross-zone repartition")
    root = eng.snapshot()
    try:
        c0 = eng.check_consistency()
        assert c0["indexMismatches"] == 0 and c0["multiZone"] == 0      # real state is clean
        ready_before = c0["inkReady"].get(ink_key, 0)
        summaries_before, orphan_before = c0["summaries"], c0["orphanIndex"]
        # in-place moves preserving every pinned position: fd0->hand slot of fh0, fd1->ink slot
        # of fi0, displaced fh0 & fi0 into the freed deck slots.
        new_hand = _swap_in_place(hand, fh[0], fd[0])
        new_ink = _swap_in_place(ink, fi[0], fd[1])
        new_deck = [fh[0] if c == fd[0] else (fi[0] if c == fd[1] else c) for c in deck]
        eng.determinize_world(self_id, _world(new_hand, new_ink, new_deck, seed="cons"))
        c1 = eng.check_consistency()
        assert c1["indexMismatches"] == 0, "cardIndex stale after repartition (finding 1)"
        assert c1["multiZone"] == 0, "a card instance is in two zones after repartition (finding 1)"
        assert c1["badOwner"] == 0, "owner/controller corrupted (findings 4/7)"
        assert c1["orphanIndex"] == orphan_before, "reverse-index orphans changed (finding 7)"
        assert c1["summaries"] == summaries_before, "public zone-summary drift (finding 5)"
        assert c1["inkReady"].get(ink_key, 0) == ready_before           # public available-ink preserved
        o = eng.observe()
        for _ in range(12):
            if o["done"]:
                break
            o = eng.step_auto("best")["obs"]
            cc = eng.check_consistency()
            assert cc["indexMismatches"] == 0 and cc["multiZone"] == 0 and cc["badOwner"] == 0
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


# --- fail-closed rejections (no silent repair) -------------------------------
def test_determinize_world_rejects_duplicate_hidden_ids(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    if not hand:
        pytest.skip("opponent hand empty")
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        dup = _world(hand, ink, [hand[0]] + deck[1:], seed="x")   # hand[0] duplicated into deck
        with pytest.raises(BridgeError, match="duplicate"):
            eng.determinize_world(self_id, dup)
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_missing_hidden_pool_ids(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        ghost = _world(hand, ink, ["GHOST_CARD"] + deck[1:], seed="x")   # invented id, count preserved
        with pytest.raises(BridgeError, match="hidden pool"):
            eng.determinize_world(self_id, ghost)
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def _realized_self_deck(eng, obs, hand, ink, deck, root):
    eng.determinize_world(obs["self"], _world(hand, ink, deck, self_deck=None, seed="s0"))
    s = list(eng.last_world_realized["selfDeckIds"])
    eng.restore(root)
    return s


def test_determinize_world_rejects_malformed_self_deck_ids(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        s = _realized_self_deck(eng, obs, hand, ink, deck, root)
        if not s:
            pytest.skip("self deck empty")
        for bad_id in (7, ""):                                    # non-string and empty-string
            bad = _world(hand, ink, deck, self_deck=[bad_id] + s[1:], seed="s")
            with pytest.raises(BridgeError, match="non-string/empty"):
                eng.determinize_world(self_id, bad)
            eng.restore(root)
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_duplicate_self_deck_ids(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        s = _realized_self_deck(eng, obs, hand, ink, deck, root)
        if len(s) < 2:
            pytest.skip("self deck < 2; cannot form a count-preserving duplicate")
        dup = _world(hand, ink, deck, self_deck=s[:-1] + [s[0]], seed="s")   # s[0] duplicated, count kept
        with pytest.raises(BridgeError, match="duplicate id in selfDeckIds"):
            eng.determinize_world(self_id, dup)
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_self_deck_conservation_failure(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    n = obs["players"]["self"]["deckCount"]
    if n < 1:
        pytest.skip("self deck empty")
    root = eng.snapshot()
    try:
        ghosts = [f"GHOST{i}" for i in range(n)]                  # right count, unique, all strings, none real
        bad = _world(hand, ink, deck, self_deck=ghosts, seed="s")
        with pytest.raises(BridgeError, match="conservation"):
            eng.determinize_world(self_id, bad)
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_invalid_seat(midgame):
    """A non-canonical seat is rejected before any cloning/mutation."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    root = eng.snapshot()
    try:
        with pytest.raises(BridgeError, match="invalid seat"):
            eng.determinize_world("bogus-seat", _world(hand, ink, deck, seed="x"))
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_wrong_canonical_seat(midgame):
    """F6: a VALID canonical seat that is NOT the current actor is rejected — a caller may not
    mutate the non-acting perspective and receive the current actor's observation."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    opp_id = obs["opp"]                      # a real canonical seat, but not the current actor
    root = eng.snapshot()
    try:
        with pytest.raises(BridgeError, match="not the current actor"):
            eng.determinize_world(opp_id, _world(hand, ink, deck, seed="x"))
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_rejects_empty_or_whitespace_seed(midgame):
    """A non-empty (trimmed) seed is required for EVERY full-world RPC — both when the self-deck
    order is omitted (server would shuffle) and when it is supplied."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        for bad_seed in ("", "   ", "\t\n"):
            with pytest.raises(BridgeError, match="non-empty string"):
                eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed=bad_seed))
            eng.restore(root)
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="s0"))
        s = list(eng.last_world_realized["selfDeckIds"])
        eng.restore(root)
        with pytest.raises(BridgeError, match="non-empty string"):
            eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=s, seed="   "))
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_failed_request_preserves_state(midgame):
    """A rejected determinize_world leaves the live engine state UNCHANGED (validation runs
    entirely before any mutation/loadState)."""
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        before = eng.observe()
        before_legal = [a["stableKey"] for a in before["legal"]]
        before_players = (before["players"]["self"], before["players"]["opp"])
        c_before = eng.check_consistency()
        with pytest.raises(BridgeError, match="hidden pool"):
            eng.determinize_world(self_id, _world(hand, ink, ["GHOST"] + deck[1:], seed="x"))
        after = eng.observe()
        assert [a["stableKey"] for a in after["legal"]] == before_legal
        assert (after["players"]["self"], after["players"]["opp"]) == before_players
        assert eng.check_consistency() == c_before
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


# --- reproducibility / no ambient randomness ---------------------------------
def test_determinize_world_is_reproducible_with_same_world_seed(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="REPRO"))
        r1 = list(eng.last_world_realized["selfDeckIds"])
        if len(r1) < 2:
            pytest.skip("self deck < 2; shuffle determinism not observable")
        eng.restore(root)
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="REPRO"))
        r2 = list(eng.last_world_realized["selfDeckIds"])
        assert r1 == r2                                          # same seed -> same realized self deck
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)


def test_determinize_world_never_uses_ambient_randomness(midgame):
    eng, obs = midgame
    hand, ink, deck = _ready(obs)
    self_id = obs["self"]
    root = eng.snapshot()
    try:
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="AAA"))
        ra = list(eng.last_world_realized["selfDeckIds"])
        eng.restore(root)
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="BBB"))
        rb = list(eng.last_world_realized["selfDeckIds"])
        eng.restore(root)
        eng.determinize_world(self_id, _world(hand, ink, deck, self_deck=None, seed="AAA"))
        ra2 = list(eng.last_world_realized["selfDeckIds"])
        assert ra == ra2                                        # deterministic from seed
        if len([c for c in ra if c not in _pinned(eng)]) >= 2:
            assert ra != rb                                     # different seeds -> different free order
    finally:
        eng.restore(root)
        eng.drop_snapshot(root)
    # static guard: the determinizeWorld method body contains no ambient-randomness fallback
    src = _SERVER_TS.read_text()
    start = src.index("determinizeWorld(selfId")
    end = src.index("checkConsistency(): {", start)
    assert "Math.random" not in src[start:end]


# --- bridge defense-in-depth: exact (order-sensitive) realized agreement -----
def test_bridge_determinize_world_rejects_reversed_realized_order():
    """The bridge compares realized vs requested EXACTLY (order-sensitive) — a reversed realized
    opponent deck OR self deck is a disagreement, not an acceptable match."""
    w = World(opponent_hand_ids=("a", "b"), opponent_inkwell_ids=("c",),
              opponent_deck_ids=("d", "e"), self_deck_ids=("s1", "s2"), seed="x")

    def _engine_with_rpc(rpc):
        eng = LorcanaEngine.__new__(LorcanaEngine)     # bypass __init__ (no subprocess)
        eng.last_world_realized = None
        eng._rpc = rpc
        return eng

    def reversed_opp_deck(req):
        sp = req["world"]
        return {"ok": True, "obs": {}, "realized": {**{k: sp[k] for k in
                ("opponentHandIds", "opponentInkwellIds", "selfDeckIds")},
                "opponentDeckIds": list(reversed(sp["opponentDeckIds"]))}}

    def reversed_self_deck(req):
        sp = req["world"]
        return {"ok": True, "obs": {}, "realized": {**{k: sp[k] for k in
                ("opponentHandIds", "opponentInkwellIds", "opponentDeckIds")},
                "selfDeckIds": list(reversed(sp["selfDeckIds"]))}}

    with pytest.raises(BridgeError, match="different opponentDeckIds"):
        _engine_with_rpc(reversed_opp_deck).determinize_world("player_one", w)
    with pytest.raises(BridgeError, match="different self deck"):
        _engine_with_rpc(reversed_self_deck).determinize_world("player_one", w)
