"""Phase 3 — league/PSRO: PFSP, Elo, KL trust region, match routing.

Most tests are engine-free (a fake engine drives match routing). One real-engine
smoke (NetPlayer vs ScriptedPlayer) is at the bottom.
"""

import copy

import numpy as np
import pytest
import torch

from training.league import League, ScriptedPlayer, NetPlayer, Player, frozen_net_copy, SEAT_ONE, SEAT_TWO
from training.match import play_match
from training.learner import loss_terms, _masked_logp, _build_batch, Sample
from engine.serialization import encode_obs, collate
from network.model import LorcanaNet


# ---------------------------------------------------------------- Elo / PFSP
def _league():
    lg = League(rng=np.random.default_rng(0))
    lg.register(ScriptedPlayer("main", "best"), "main")  # id only; kind irrelevant here
    lg.register(ScriptedPlayer("weak", "best"), "anchor")
    lg.register(ScriptedPlayer("tough", "best"), "anchor")
    return lg


def test_elo_moves_with_results():
    lg = _league()
    base = lg.elo["main"]
    for _ in range(10):
        lg.record_result("main", "weak", "main")   # main keeps beating weak
    assert lg.elo["main"] > base
    assert lg.elo["weak"] < 1000.0
    assert lg.win_prob("main", "weak") > 0.8


def test_win_prob_symmetry_and_prior():
    lg = _league()
    assert lg.win_prob("main", "tough") == pytest.approx(0.5)  # no games -> prior
    lg.record_result("main", "tough", "tough")
    lg.record_result("main", "tough", "tough")
    assert lg.win_prob("main", "tough") < 0.5
    assert lg.win_prob("tough", "main") > 0.5                   # symmetric


def test_pfsp_weights_focus_on_hard_opponents():
    lg = _league()
    for _ in range(8):
        lg.record_result("main", "weak", "main")    # we crush weak
        lg.record_result("main", "tough", "tough")  # we lose to tough
    cands = ["weak", "tough"]
    w = lg.pfsp_weights("main", cands, mode="hard")
    assert w.sum() == pytest.approx(1.0)
    assert w[cands.index("tough")] > w[cands.index("weak")]     # PFSP favors the hard one


def test_frozen_net_copy_is_independent_and_frozen():
    net = LorcanaNet(d_model=16, n_layers=1)
    clone = frozen_net_copy(net)
    assert all(not p.requires_grad for p in clone.parameters())
    # mutate original; clone unchanged
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    a = next(net.parameters()).detach().clone()
    b = next(clone.parameters()).detach().clone()
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------- KL term
def _fake_obs(n_cards=4, n_actions=3):
    cards = [{"id": f"c{i}", "owner": i % 2, "zone": "play", "cardType": "character",
              "cost": i, "strength": 2, "willpower": 3, "lore": 1, "damage": 0,
              "exerted": False, "drying": False, "ready": True, "keywords": [],
              "definitionId": f"def-{i}", "hidden": False} for i in range(n_cards)]
    legal = [{"idx": j, "stableKey": f"k{j}", "family": "playCard",
              "src": f"c{j % n_cards}", "tgt": None} for j in range(n_actions)]
    return {"turn": 1, "phase": "main", "status": "playing", "done": False, "winner": None,
            "actor": "player_one", "self": "player_one", "opp": "player_two", "forced": False,
            "players": {"self": {"lore": 0, "handCount": 3, "deckCount": 50, "inkwell": 1,
                                 "discard": 0, "play": 1},
                        "opp": {"lore": 0, "handCount": 4, "deckCount": 52, "inkwell": 1,
                                "discard": 0, "play": 1}},
            "cards": cards, "legal": legal}


def test_kl_zero_to_self_positive_after_drift():
    net = LorcanaNet(d_model=24, n_layers=1)
    samples = [Sample(enc=encode_obs(_fake_obs()), pi=np.array([1, 0, 0], np.float32), z=0.0)
               for _ in range(4)]
    tb, pi_t, z = _build_batch(samples, torch.device("cpu"))
    ref = copy.deepcopy(net)
    with torch.no_grad():
        ref_logp = _masked_logp(ref.forward(tb)["policy_logits"], tb["action_mask"])
    _, s_same = loss_terms(net, tb, pi_t, z, ref_logp=ref_logp, c_kl=1.0)
    assert s_same["kl"] == pytest.approx(0.0, abs=1e-5)         # net == reference
    # drift the net, KL should become positive
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5 * torch.randn_like(p))
    _, s_drift = loss_terms(net, tb, pi_t, z, ref_logp=ref_logp, c_kl=1.0)
    assert s_drift["kl"] > 1e-4


# ---------------------------------------------------------------- match routing (fake engine)
class _FakeEngine:
    """Scripted 4-decision game: P1, P2, P1, P2 then done with player_one winner."""
    def __init__(self):
        self.t = 0

    def _obs(self, actor, done=False):
        return {"done": done, "winner": "player_one" if done else None, "actor": actor,
                "self": actor, "opp": "player_two" if actor == "player_one" else "player_one",
                "legal": [] if done else [{"stableKey": f"k{self.t}", "family": "quest",
                                           "src": None, "tgt": None}],
                "turn": self.t, "players": {"self": {}, "opp": {}}, "cards": [],
                "hidden": {"hand": [], "deck": [], "inkwell": []}}

    def reset(self, seed="m", d1=None, d2=None):
        self.t = 0
        return self._obs("player_one")

    def step_auto(self, strategy="best"):
        self.t += 1
        if self.t >= 4:
            return {"obs": self._obs("player_one", done=True)}
        return {"obs": self._obs("player_one" if self.t % 2 == 0 else "player_two")}


class _RecordingPlayer(Player):
    """Records a trivial sample each decision (engine-free)."""
    def __init__(self, pid):
        super().__init__(pid, record=True)

    def act(self, engine, obs):
        actor = obs["actor"]
        new_obs = engine.step_auto()["obs"]
        raw = (encode_obs(obs), np.array([1.0], np.float32), None, actor)
        return new_obs, raw


def test_play_match_routes_and_assigns_outcome():
    eng = _FakeEngine()
    a = _RecordingPlayer("A")  # player_one seat
    b = _RecordingPlayer("B")  # player_two seat
    res = play_match(eng, {SEAT_ONE: a, SEAT_TWO: b}, seed="t", max_steps=10)
    assert res.winner_seat == "player_one"
    assert res.winner_id == "A"
    # player_one won -> A's samples z=+1, B's z=-1
    assert all(s.z == 1.0 for s in res.samples["A"])
    assert all(s.z == -1.0 for s in res.samples["B"])
    assert len(res.samples["A"]) >= 1 and len(res.samples["B"]) >= 1


# ---------------------------------------------------------------- engine smoke
@pytest.mark.slow
def test_netplayer_vs_scripted_real_engine():
    from engine.bridge import LorcanaEngine
    from search.ismcts import SearchConfig
    eng = LorcanaEngine(timeout=240)
    try:
        net = LorcanaNet(d_model=32, n_layers=2)
        main = NetPlayer("main", net, SearchConfig(simulations=4, depth_limit=2),
                         record=True, use_belief=True, n_worlds=2)
        opp = ScriptedPlayer("anchor", "best")
        res = play_match(eng, {SEAT_ONE: main, SEAT_TWO: opp}, seed="smoke",
                         max_steps=200)
        assert res.winner_seat in ("player_one", "player_two", None)
        assert len(res.samples["main"]) >= 1     # main recorded decisions
        assert all(s.z in (-1.0, 0.0, 1.0) for s in res.samples["main"])
    finally:
        eng.close()
