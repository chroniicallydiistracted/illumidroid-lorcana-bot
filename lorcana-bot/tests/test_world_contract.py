"""Tier-A Phase 1 — the canonical full hidden-zone `World` contract.

Proves `World.validate_against_obs` enforces an exact partition of the opponent's
hidden pool (counts, no duplicates, pool equality), the optional self-deck semantics
(None = unsupplied vs () = supplied-empty), and the non-empty clean-label seed rule.
Engine-free: obs uses the real bridge schema (players dict + hidden id lists)."""

import numpy as np
import pytest

from search.determinize import World, WorldContractError, sample_worlds


def _obs(hand=3, ink=2, deck=4, self_deck=5):
    """Real-schema obs: opponent hidden pool of `hand+ink+deck` cards + public counts."""
    hand_ids = [f"h{i}" for i in range(hand)]
    ink_ids = [f"i{i}" for i in range(ink)]
    deck_ids = [f"d{i}" for i in range(deck)]
    return {
        "players": {"self": {"deckCount": self_deck},
                    "opp": {"handCount": hand, "inkwell": ink, "deckCount": deck}},
        "hidden": {"hand": [{"id": x, "def": "X"} for x in hand_ids],
                   "inkwell": [{"id": x, "def": "X"} for x in ink_ids],
                   "deck": [{"id": x, "def": "X"} for x in deck_ids]},
    }


def _valid_world(self_deck=("s0", "s1", "s2", "s3", "s4"), seed="game:seat=0:dec=1:sim=0:p=0"):
    return World(
        opponent_hand_ids=("h0", "h1", "h2"),
        opponent_inkwell_ids=("i0", "i1"),
        opponent_deck_ids=("d0", "d1", "d2", "d3"),
        self_deck_ids=self_deck,
        seed=seed,
    )


def test_world_partition_has_no_duplicates():
    _valid_world().validate_against_obs(_obs())                 # clean partition passes
    dup = World(opponent_hand_ids=("h0", "h1", "h0"),           # h0 duplicated into hand
                opponent_inkwell_ids=("i0", "i1"),
                opponent_deck_ids=("d0", "d1", "d2", "d3"),
                self_deck_ids=("s0", "s1", "s2", "s3", "s4"), seed="s")
    with pytest.raises(WorldContractError, match="duplicate"):
        dup.validate_against_obs(_obs())


def test_world_partition_matches_hidden_pool():
    # a deck id not present in the known hidden pool (z9) -> partition != pool
    bad = World(opponent_hand_ids=("h0", "h1", "h2"),
                opponent_inkwell_ids=("i0", "i1"),
                opponent_deck_ids=("d0", "d1", "d2", "z9"),
                self_deck_ids=("s0", "s1", "s2", "s3", "s4"), seed="s")
    with pytest.raises(WorldContractError, match="hidden pool"):
        bad.validate_against_obs(_obs())


def test_world_rejects_missing_inkwell_assignment():
    # opp has 2 hidden inkwell cards but the world assigns none -> count mismatch
    missing_ink = World(opponent_hand_ids=("h0", "h1", "h2"),
                        opponent_inkwell_ids=(),
                        opponent_deck_ids=("d0", "d1", "d2", "d3", "i0", "i1"),
                        self_deck_ids=("s0", "s1", "s2", "s3", "s4"), seed="s")
    with pytest.raises(WorldContractError, match="inkwell"):
        missing_ink.validate_against_obs(_obs())


def test_world_rejects_hand_count_mismatch():
    short_hand = World(opponent_hand_ids=("h0", "h1"),          # 2 != handCount 3
                       opponent_inkwell_ids=("i0", "i1"),
                       opponent_deck_ids=("d0", "d1", "d2", "d3", "h2"),
                       self_deck_ids=("s0", "s1", "s2", "s3", "s4"), seed="s")
    with pytest.raises(WorldContractError, match="hand"):
        short_hand.validate_against_obs(_obs())


def test_world_preserves_self_deck_when_known():
    _valid_world().validate_against_obs(_obs(self_deck=5))      # 5 == self.deckCount
    wrong = _valid_world(self_deck=("s0", "s1"))                # 2 != self.deckCount 5
    with pytest.raises(WorldContractError, match="self_deck"):
        wrong.validate_against_obs(_obs(self_deck=5))


def test_world_rejects_empty_seed_for_clean_label_search():
    no_seed = _valid_world(seed="")
    with pytest.raises(WorldContractError, match="seed"):
        no_seed.validate_against_obs(_obs())                   # require_seed defaults True
    no_seed.validate_against_obs(_obs(), require_seed=False)    # explicit diagnostic opt-out


def test_world_distinguishes_unsupplied_self_deck_from_empty_self_deck():
    unsupplied = World(opponent_hand_ids=("h0", "h1", "h2"), opponent_inkwell_ids=("i0", "i1"),
                       opponent_deck_ids=("d0", "d1", "d2", "d3"), self_deck_ids=None, seed="s")
    empty = World(opponent_hand_ids=("h0", "h1", "h2"), opponent_inkwell_ids=("i0", "i1"),
                  opponent_deck_ids=("d0", "d1", "d2", "d3"), self_deck_ids=(), seed="s")
    assert unsupplied != empty                                  # None and () are distinct
    assert unsupplied.self_deck_ids is None and empty.self_deck_ids == ()
    # unsupplied: self-deck check is SKIPPED (passes even though self.deckCount=5)
    unsupplied.validate_against_obs(_obs(self_deck=5))
    # supplied-empty: self-deck check ENFORCED -> 0 != 5 fails
    with pytest.raises(WorldContractError, match="self_deck"):
        empty.validate_against_obs(_obs(self_deck=5))
    # supplied-empty is valid only when the public self deck is actually empty
    empty.validate_against_obs(_obs(self_deck=0))
    # unsupplied + require_self_deck -> rejected
    with pytest.raises(WorldContractError, match="self_deck"):
        unsupplied.validate_against_obs(_obs(self_deck=0), require_self_deck=True)


# --- review findings 1/2/3 (NO-GO) regression guards --------------------------
def test_world_rejects_missing_hidden_block_fail_closed():
    """Finding 1: a missing hidden block can no longer fail OPEN — without the pool we
    cannot prove the partition, so any world (even count-matching) is rejected."""
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 3, "inkwell": 0, "deckCount": 0}}}  # NO hidden
    w = World(opponent_hand_ids=("h0", "h1", "h2"), self_deck_ids=None, seed="s")
    with pytest.raises(WorldContractError, match="fail-closed"):
        w.validate_against_obs(obs)


def test_world_rejects_invented_card_against_coherent_pool():
    """Finding 1: with a COHERENT witness (zone cardinalities match public counts), a world
    that swaps in invented ids of the right COUNT is still rejected by the partition check."""
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 3, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": ["h0", "h1", "h2"], "inkwell": [], "deck": []}}
    ghost = World(opponent_hand_ids=("g0", "g1", "g2"), self_deck_ids=None, seed="s")
    with pytest.raises(WorldContractError, match="hidden pool"):
        ghost.validate_against_obs(obs)


def test_world_accepts_empty_world_when_hidden_supplied_empty():
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [], "inkwell": [], "deck": []}}
    World(self_deck_ids=None, seed="s").validate_against_obs(obs)   # empty pool admits empty world


def test_hand_only_determinization_is_gated_for_clean_label():
    """Finding 2: the hand-only determinization path fails closed at the SEARCH boundary
    (not only at training writers) unless explicitly opted into the diagnostic flag."""
    from search.engine_sim import EngineSimulator
    from training.tier_a_guard import TierARemediationIncompleteError

    class _FakeEngine:
        def __init__(self):
            self.calls = 0

        def restore(self, _snap):
            return {}

        def determinize(self, _self_id, _hand_ids, seed=None):
            self.calls += 1
            return {"players": {}, "legal": [], "cards": []}

    eng = _FakeEngine()
    gated = EngineSimulator(eng, "root", self_id=0)            # default -> fail-closed
    with pytest.raises(TierARemediationIncompleteError):
        gated.begin_lane(World(seed="s"))
    assert eng.calls == 0                                      # determinization never ran
    diag = EngineSimulator(eng, "root", self_id=0, hand_only_diagnostic=True)
    obs = diag.begin_lane(World(seed="s"))
    assert eng.calls == 1 and "infoSetKey" in obs


def test_default_seed_changes_when_belief_probs_change():
    """Finding 3: the fallback base seed hashes the belief probs, so reversing the belief
    changes the world seeds (the seed is content-derived, not just index-derived)."""
    pool = ["a", "b", "c"]
    fwd = sample_worlds(pool, np.array([0.8, 0.4, 0.2]), 2, 3, rng=np.random.default_rng(4))
    rev = sample_worlds(pool, np.array([0.2, 0.4, 0.8]), 2, 3, rng=np.random.default_rng(4))
    assert all(w.seed for w in fwd) and all(w.seed for w in rev)
    assert [w.seed for w in fwd] != [w.seed for w in rev]


# --- re-review findings: STRICT presence (incomplete dict-shaped obs must fail closed) ---
def _empty_world():
    return World(self_deck_ids=None, seed="s")


def test_world_rejects_empty_hidden_dict():
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}}, "hidden": {}}
    with pytest.raises(WorldContractError, match="hidden zone 'hand'"):
        _empty_world().validate_against_obs(obs)


def test_world_rejects_hidden_missing_a_zone_key():
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [], "inkwell": []}}            # missing 'deck'
    with pytest.raises(WorldContractError, match="hidden zone 'deck'"):
        _empty_world().validate_against_obs(obs)


def test_world_rejects_hidden_none_valued_zones():
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": None, "inkwell": None, "deck": None}}
    with pytest.raises(WorldContractError, match="not list-like"):
        _empty_world().validate_against_obs(obs)


def test_world_rejects_missing_public_counters():
    obs = {"players": {"self": {"deckCount": 0}, "opp": {}},   # opp counters absent
           "hidden": {"hand": [], "inkwell": [], "deck": []}}
    with pytest.raises(WorldContractError, match="handCount"):
        _empty_world().validate_against_obs(obs)


def test_world_rejects_supplied_self_deck_with_missing_self_deckcount():
    obs = {"players": {"self": {}, "opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [], "inkwell": [], "deck": []}}
    supplied_empty = World(self_deck_ids=(), seed="s")        # self_deck supplied -> self.deckCount required
    with pytest.raises(WorldContractError, match="self is missing public count 'deckCount'"):
        supplied_empty.validate_against_obs(obs)


# --- re-review findings: opponent_hidden_pool strict (F1) + non-integer counters (F4) ---
def test_opponent_hidden_pool_strict_fail_closed():
    """F1: the documented sampler-exposure accessor must NOT fail open — incomplete
    evidence raises instead of returning a silently-empty pool."""
    from search.determinize import World as W
    with pytest.raises(WorldContractError):
        W.opponent_hidden_pool({})                                   # no players
    with pytest.raises(WorldContractError, match="hidden"):
        W.opponent_hidden_pool({"players": {"opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}},
                                "hidden": {}})                       # empty hidden block
    with pytest.raises(WorldContractError, match="not list-like"):
        W.opponent_hidden_pool({"players": {"opp": {"handCount": 0, "inkwell": 0, "deckCount": 0}},
                                "hidden": {"hand": None, "inkwell": None, "deck": None}})
    # a complete obs returns the full pool ids + counts
    pool = W.opponent_hidden_pool(_obs())
    assert pool["hand_count"] == 3 and pool["inkwell_count"] == 2 and pool["deck_count"] == 4
    assert pool["hand_ids"] == ["h0", "h1", "h2"] and pool["inkwell_ids"] == ["i0", "i1"]


@pytest.mark.parametrize("bad", [True, 1.9, "1", ["x"]])
def test_world_rejects_non_integer_public_count(bad):
    """F4: a malformed public counter (bool / float / str / list) is rejected, not coerced."""
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": bad, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [], "inkwell": [], "deck": []}}
    with pytest.raises(WorldContractError, match="handCount"):
        World(self_deck_ids=None, seed="s").validate_against_obs(obs)


def test_world_accepts_numpy_integer_count():
    """F4 robustness: NumPy integers remain valid (real obs counts are ints; np ints OK)."""
    obs = {"players": {"self": {"deckCount": np.int64(0)},
                       "opp": {"handCount": np.int64(0), "inkwell": np.int64(0),
                               "deckCount": np.int64(0)}},
           "hidden": {"hand": [], "inkwell": [], "deck": []}}
    World(self_deck_ids=None, seed="s").validate_against_obs(obs)


# --- re-review findings: witness integrity (W1) + string seed contract (W2) ---
def test_witness_rejects_duplicate_source_ids():
    """W1: the same id appearing across hidden zones (collapsed by a set previously) is now
    rejected at the witness, for both validate_against_obs and opponent_hidden_pool."""
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 1, "inkwell": 0, "deckCount": 1}},
           "hidden": {"hand": ["a"], "inkwell": [], "deck": ["a"]}}   # 'a' duplicated
    with pytest.raises(WorldContractError, match="duplicate id across hidden"):
        World.opponent_hidden_pool(obs)
    with pytest.raises(WorldContractError, match="duplicate id across hidden"):
        World(opponent_hand_ids=("a",), opponent_deck_ids=("a",),
              self_deck_ids=None, seed="s").validate_against_obs(obs)


def test_witness_rejects_contradictory_cardinality():
    """W1: a hidden zone whose size disagrees with its public count is rejected (no
    contradictory witness)."""
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 1, "inkwell": 0, "deckCount": 1}},
           "hidden": {"hand": ["a", "b"], "inkwell": [], "deck": ["c"]}}  # hand 2 != handCount 1
    with pytest.raises(WorldContractError, match="witness count"):
        World.opponent_hidden_pool(obs)


def test_witness_rejects_non_string_id():
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 1, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [7], "inkwell": [], "deck": []}}          # non-string id
    with pytest.raises(WorldContractError, match="non-string/empty id"):
        World.opponent_hidden_pool(obs)


def test_witness_rejects_empty_string_id():
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 1, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [""], "inkwell": [], "deck": []}}
    with pytest.raises(WorldContractError, match="non-string/empty id"):
        World.opponent_hidden_pool(obs)


def test_witness_dict_entry_missing_id_raises_contract_error_not_keyerror():
    """W1: a dict card entry without 'id' raises WorldContractError, never a raw KeyError."""
    obs = {"players": {"self": {"deckCount": 0},
                       "opp": {"handCount": 1, "inkwell": 0, "deckCount": 0}},
           "hidden": {"hand": [{"def": "X"}], "inkwell": [], "deck": []}}  # no 'id'
    with pytest.raises(WorldContractError, match="missing 'id'"):
        World.opponent_hidden_pool(obs)


@pytest.mark.parametrize("bad_seed", [True, 7, 1.5, None, b"x"])
def test_world_rejects_non_string_seed(bad_seed):
    """W2: World.seed is declared str — a non-string seed is rejected even though some are
    truthy (True / 7)."""
    with pytest.raises(WorldContractError, match="seed must be a string"):
        World(self_deck_ids=None, seed=bad_seed).validate_against_obs(_obs())


def test_world_rejects_whitespace_only_seed():
    with pytest.raises(WorldContractError, match="empty/whitespace"):
        _valid_world(seed="   ").validate_against_obs(_obs())
