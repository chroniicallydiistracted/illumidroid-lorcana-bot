"""Tier-A #2 — full ISMCTS: leak-free info-set key, shared node-edge statistics,
availability-aware selection, and the strategy-fusion proof (spec §13.1/§13.3/§13.4).
Engine-free: the search is driven by a synthetic imperfect-information fixture."""

import numpy as np

from search.infoset import info_set_key, history_event
from search.infoset_tree import (
    SharedInfoSetNode, EdgeStats, InfoSetTable, run_infoset, value_for,
)
from search.ismcts import SearchConfig


# ---------------------------------------------------------------------------
# §13.4 leak tests — the key uses only actor-visible info
# ---------------------------------------------------------------------------
def _obs_with_hidden(hand_defs, deck_defs, visible_def="vis-1"):
    cards = [
        {"id": "v1", "owner": 0, "zone": "play", "definitionId": visible_def, "cost": 3,
         "strength": 2, "willpower": 3, "lore": 1, "damage": 0, "exerted": False,
         "drying": False, "hidden": False},
        # opponent hidden cards: filtered board redacts identity (definitionId None, hidden)
        *[{"id": f"oh{i}", "owner": 1, "zone": "hand", "definitionId": None,
           "hidden": True} for i in range(len(hand_defs))],
    ]
    return {
        "actor": "player_one", "selfIdx": 0, "turn": 4, "phase": "main", "step": None,
        "status": "playing", "forced": False,
        "players": {"self": {"lore": 2, "handCount": 3, "deckCount": 50, "inkwell": 4,
                             "discard": 1, "play": 1},
                    "opp": {"lore": 1, "handCount": len(hand_defs), "deckCount": len(deck_defs),
                            "inkwell": 3, "discard": 0, "play": 0}},
        "cards": cards,
        "legal": [{"actionId": "a1"}, {"actionId": "a2"}],
        "hidden": {"hand": [{"id": f"oh{i}", "def": d} for i, d in enumerate(hand_defs)],
                   "deck": [{"id": f"od{i}", "def": d} for i, d in enumerate(deck_defs)],
                   "inkwell": []},
    }


def test_2_key_ignores_opponent_hidden_identities_and_deck_order():
    base = _obs_with_hidden(["X", "Y", "Z"], ["P", "Q", "R"])
    k0 = info_set_key(base)
    # scramble opponent hidden hand identities + deck order: key MUST NOT change
    scram = _obs_with_hidden(["Z", "X", "Y"], ["R", "Q", "P"])
    assert info_set_key(scram) == k0
    # mutate obs["hidden"] in place to anything: still ignored
    base2 = _obs_with_hidden(["X", "Y", "Z"], ["P", "Q", "R"])
    base2["hidden"]["hand"][0]["def"] = "TOTALLY-DIFFERENT"
    assert info_set_key(base2) == k0


def test_2_key_changes_on_visible_state_and_history():
    k0 = info_set_key(_obs_with_hidden(["X"], ["P"]))
    # a different VISIBLE card identity is actor-knowable -> different info set
    assert info_set_key(_obs_with_hidden(["X"], ["P"], visible_def="vis-2")) != k0
    # different observer-visible history -> different info set (perfect recall §6.4)
    h = (history_event(0, "playCard", "playCard:c5"),)
    assert info_set_key(_obs_with_hidden(["X"], ["P"]), h) != k0


def test_2_history_redacts_face_down_inking():
    """Face-down inking hides the card identity, so the branch event redacts it
    (two different inked cards produce the SAME observer-projected event)."""
    assert history_event(1, "putCardIntoInkwell", "card-A") == \
           history_event(1, "putCardIntoInkwell", "card-B")
    # but a public play keeps its identity
    assert history_event(1, "playCard", "card-A") != history_event(1, "playCard", "card-B")


# ---------------------------------------------------------------------------
# §13.1 pure node tests — shared statistics, availability-aware selection
# ---------------------------------------------------------------------------
def test_2_table_merges_same_key_and_counts_transpositions():
    t = InfoSetTable()
    o = {"infoSetKey": "K", "actor": "me", "legal": [{"actionId": "a"}, {"actionId": "b"}]}
    n1, c1 = t.get_or_create(o)
    n2, c2 = t.get_or_create({"infoSetKey": "K", "actor": "me", "legal": [{"actionId": "a"}]})
    assert c1 and not c2 and n1 is n2          # same key -> same shared node
    assert t.transposition_hits == 1


def test_2_availability_increments_every_available_sibling():
    """Availability tracks how often each action was AVAILABLE, separately from
    visits — so a rarely-available action is not penalized as if always available."""
    node = SharedInfoSetNode(key="K", actor="me", exemplar_obs={})
    node.ensure_edges([{"actionId": "a"}, {"actionId": "b"}, {"actionId": "c"}])
    # simulate two backups where {a,b} available (selected a), then {a,c} (selected c)
    for avail, sel in [(["a", "b"], "a"), (["a", "c"], "c")]:
        for aid in avail:
            node.edges[aid].availability += 1.0
        node.edges[sel].visits += 1.0
    assert node.edges["a"].availability == 2.0     # available in both
    assert node.edges["b"].availability == 1.0
    assert node.edges["c"].availability == 1.0
    assert node.edges["a"].visits == 1.0 and node.edges["c"].visits == 1.0


def test_2_select_never_returns_unavailable_action():
    node = SharedInfoSetNode(key="K", actor="me", exemplar_obs={})
    node.ensure_edges([{"actionId": "a"}, {"actionId": "b"}, {"actionId": "c"}])
    for e in node.edges.values():
        e.prior_sum, e.prior_samples = 1.0, 1
    cfg = SearchConfig(c_puct=1.5, pw_c=10.0, pw_alpha=1.0)
    for _ in range(50):
        assert node.select(["b", "c"], cfg) in ("b", "c")   # never "a"


def test_2_value_for_flips_by_actor():
    leaf_win_me = {"done": True, "winner": "me"}
    assert value_for("me", leaf_win_me, 0.0) == 1.0
    assert value_for("opp", leaf_win_me, 0.0) == -1.0
    leaf_nonterm = {"done": False, "actor": "me"}
    assert value_for("me", leaf_nonterm, 0.7) == 0.7
    assert value_for("opp", leaf_nonterm, 0.7) == -0.7


# ---------------------------------------------------------------------------
# §13.3 the strategy-fusion proof — a synthetic imperfect-information game
# ---------------------------------------------------------------------------
class _SynthSim:
    """Hidden world A or B, indistinguishable at the root.
      X: wins in A, loses in B.   Y: wins in B, loses in A.   wait -> shared node N
      -> fold (loses in both). The true value of the root is 0 (any fixed action
      wins half the worlds). PIMC/determinized-UCT would let each world's tree pick
      its own winner and OVER-value the root to ~+1 (strategy fusion)."""

    def __init__(self):
        self.world = None
        self.state = "root"
        self.unavailable_violation = False
        self.reached_N_from = set()

    def begin_lane(self, world):
        self.world = world
        self.state = "root"
        return self._obs()

    def _obs(self):
        if self.state == "root":
            return {"infoSetKey": "root", "actor": "me", "done": False,
                    "legal": [{"actionId": "X"}, {"actionId": "Y"}, {"actionId": "wait"}]}
        return {"infoSetKey": "N", "actor": "me", "done": False,
                "legal": [{"actionId": "fold"}]}

    def step(self, action_id):
        avail = {"root": {"X", "Y", "wait"}, "N": {"fold"}}[self.state]
        if action_id not in avail:
            self.unavailable_violation = True
        if self.state == "root":
            if action_id == "X":
                return self._term("me" if self.world == "A" else "opp")
            if action_id == "Y":
                return self._term("me" if self.world == "B" else "opp")
            self.state = "N"
            self.reached_N_from.add(self.world)
            return self._obs()
        return self._term("opp")            # fold always loses

    @staticmethod
    def _term(winner):
        return {"infoSetKey": f"term:{winner}", "actor": "me", "done": True,
                "winner": winner, "legal": []}

    def end(self):
        self.state = "root"


def _uniform_eval(obs):
    n = len(obs.get("legal", []))
    return ({str(a["actionId"]): 1.0 / n for a in obs["legal"]} if n else {}), 0.0


def test_2_full_ismcts_removes_strategy_fusion():
    sim = _SynthSim()
    rng = np.random.default_rng(0)
    cfg = SearchConfig(simulations=4000, depth_limit=4, c_puct=1.5,
                       dirichlet_eps=0.0, temperature=1.0, assert_root_invariant=True)
    root_obs = sim.begin_lane("A")
    sampler = lambda r: "A" if r.random() < 0.5 else "B"
    res = run_infoset(root_obs, sim, sampler, _uniform_eval, cfg, rng)

    # (1) ONE shared root node; every world resolved the root to the same key
    assert "root" in res.table.nodes
    assert res.sims == cfg.simulations and not sim.unavailable_violation

    # (2) the deeper node N is SHARED — reached from BOTH worlds, one node, revisited
    assert "N" in res.table.nodes
    assert res.table.nodes["N"].successful_visits() > 0
    assert sim.reached_N_from == {"A", "B"}
    assert res.table.transposition_hits > 0

    # (3) NO strategy fusion: the gambles average out to ≈0, root is NOT over-valued
    root = res.root
    qX, qY = root.edges["X"].q, root.edges["Y"].q
    assert abs(qX) < 0.3 and abs(qY) < 0.3        # each wins half the worlds
    assert root.edges["wait"].q < -0.5            # wait -> fold loses in both

    # (4) the shared-tree root value stays near 0, far below the PIMC over-valuation
    pimc_pooled = np.mean([max(+1, -1, -1), max(-1, +1, -1)])   # per-world best = +1 each
    assert pimc_pooled == 1.0
    assert res.value < 0.4                        # shared tree ≈0, not ~+1
    assert pimc_pooled - res.value > 0.5          # strategy fusion would over-value

    # (5) root policy is aligned to the real legal order by actionId (len + indices)
    assert res.pi.shape == (3,) and abs(res.pi.sum() - 1.0) < 1e-5
    assert int(res.pi.argmax()) in (0, 1)         # prefers a gamble over guaranteed-loss wait
