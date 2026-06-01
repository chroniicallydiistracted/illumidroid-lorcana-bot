"""Phase 2 — belief head: shapes, count-consistency, leak-freeness, learning."""

import numpy as np
import torch

from engine.serialization import encode_obs, encode_belief, collate, collate_belief
from network.model import LorcanaNet
from training.learner import Sample, ReplayBuffer, Learner


def _obs(n_cards=4, n_actions=3, opp_hand=3, opp_deck=5, opp_ink=0):
    cards = [{
        "id": f"c{i}", "owner": i % 2, "zone": "play", "cardType": "character",
        "cost": i, "strength": 2, "willpower": 3, "lore": 1, "damage": 0,
        "exerted": False, "drying": False, "ready": True, "keywords": [],
        "definitionId": f"def-{i}", "hidden": False,
    } for i in range(n_cards)]
    legal = [{"idx": j, "stableKey": f"k{j}", "family": "playCard",
              "src": f"c{j % n_cards}", "tgt": None} for j in range(n_actions)]
    hidden = {
        "hand": [{"id": f"h{i}", "def": f"opp-def-{i}"} for i in range(opp_hand)],
        "deck": [{"id": f"d{i}", "def": f"opp-def-{100 + i}"} for i in range(opp_deck)],
        "inkwell": [{"id": f"w{i}", "def": f"opp-def-{200 + i}"} for i in range(opp_ink)],
    }
    return {
        "turn": 3, "phase": "main", "status": "playing", "done": False, "winner": None,
        "actor": "player_one", "self": "player_one", "opp": "player_two", "forced": False,
        "players": {"self": {"lore": 2, "handCount": 4, "deckCount": 50, "inkwell": 3,
                             "discard": 1, "play": 2},
                    "opp": {"lore": 1, "handCount": opp_hand, "deckCount": opp_deck,
                            "inkwell": 2, "discard": 0, "play": 1}},
        "cards": cards, "legal": legal, "hidden": hidden,
    }


def test_encode_belief_shapes_and_labels():
    bel = encode_belief(_obs(opp_hand=3, opp_deck=5))
    assert int(bel["n_pool"]) == 8
    assert int(bel["belief_handcount"]) == 3
    assert bel["belief_label"].sum() == 3          # first 3 (hand) are positive
    assert bel["belief_label"][:3].sum() == 3 and bel["belief_label"][3:].sum() == 0


def test_inkwell_belief_target_and_learning():
    # #10: pool includes inkwell; a second target (in_inkwell) is emitted + learned
    bel = encode_belief(_obs(opp_hand=3, opp_deck=5, opp_ink=2))
    assert int(bel["n_pool"]) == 10                 # 3 hand + 5 deck + 2 inkwell
    assert int(bel["belief_inkcount"]) == 2
    assert bel["belief_label_ink"][-2:].sum() == 2  # last 2 (inkwell) flagged
    assert bel["belief_label"].sum() == 3 and (bel["belief_label"] * bel["belief_label_ink"]).sum() == 0
    # the head learns inkwell membership on a fixed obs
    net = LorcanaNet(d_model=32, n_layers=2)
    learner = Learner(net, lr=3e-3)
    buf = ReplayBuffer()
    o = _obs(opp_hand=3, opp_deck=5, opp_ink=2)
    for _ in range(32):
        buf.add(Sample(enc=encode_obs(o), pi=np.array([1, 0, 0], np.float32), z=0.0,
                       belief=encode_belief(o)))
    s0 = learner.train_epoch(buf, batch_size=16)
    for _ in range(30):
        s1 = learner.train_epoch(buf, batch_size=16)
    assert "belief_ink" in s1 and s1["belief_ink"] < s0["belief_ink"]
    assert s1["belief_ink_sep"] > 0.3              # separates inked from not


def test_belief_probs_normalize_to_handcount():
    net = LorcanaNet(d_model=32, n_layers=1)
    batch = {**collate([encode_obs(_obs())]), **collate_belief([encode_belief(_obs())])}
    bp = net.belief_probs(batch, normalize=True)
    assert abs(float(bp.sum()) - 3.0) < 1e-3       # sums to known hand size


def test_belief_is_leak_free():
    """Scrambling belief candidate ids must NOT change policy/value (no leak),
    but must change the belief output."""
    net = LorcanaNet(d_model=32, n_layers=2)
    base = {**collate([encode_obs(_obs())]), **collate_belief([encode_belief(_obs())])}
    scrambled = {k: v.copy() for k, v in base.items()}
    from engine.serialization import VOCAB_SIZE
    scrambled["belief_ids"] = np.random.randint(2, VOCAB_SIZE, size=base["belief_ids"].shape).astype(np.int64)
    o1 = net.forward({k: torch.as_tensor(v) for k, v in base.items()})
    o2 = net.forward({k: torch.as_tensor(v) for k, v in scrambled.items()})
    assert torch.allclose(o1["policy_logits"], o2["policy_logits"], atol=1e-6)
    assert torch.allclose(o1["value_logits"], o2["value_logits"], atol=1e-6)
    assert not torch.allclose(o1["belief_logits"], o2["belief_logits"], atol=1e-4)


def _hist_event(family, actor, defId, turn):
    return {"family": family, "actor": actor, "defId": defId,
            "lore": [2, 1], "ink": [3, 2], "hand": [4, 5], "turn": turn}


def test_history_influences_belief_and_is_leak_free():
    """§2.3: the PUBLIC history must reach the belief head (and policy/value) via
    the trunk — changing which cards the opponent has PLAYED changes the belief —
    while the existing hidden-card leak-freeness is unaffected (history carries
    only public ids)."""
    from engine.serialization import HIST_FEAT_DIM

    def batch_for(history):
        o = _obs(opp_hand=3, opp_deck=5, opp_ink=2)
        o["history"] = history
        o["selfIdx"] = 0
        enc = encode_obs(o)
        assert enc["hist_feats"].shape == (len(history), HIST_FEAT_DIM)
        return {**collate([enc]), **collate_belief([encode_belief(o)])}

    h1 = [_hist_event("playCard", 1, "opp-def-100", 4),
          _hist_event("putCardIntoInkwell", 1, None, 4)]
    h2 = [_hist_event("playCard", 1, "opp-def-200", 4),   # opponent played a DIFFERENT card
          _hist_event("putCardIntoInkwell", 1, None, 4)]

    net = LorcanaNet(d_model=32, n_layers=2)
    net.eval()
    o1 = net.forward({k: torch.as_tensor(v) for k, v in batch_for(h1).items()})
    o2 = net.forward({k: torch.as_tensor(v) for k, v in batch_for(h2).items()})
    # different public history -> different belief (history reaches the belief head)
    assert not torch.allclose(o1["belief_logits"], o2["belief_logits"], atol=1e-4)
    # and it also moves policy/value (public info fused into the trunk)
    assert not torch.allclose(o1["value_logits"], o2["value_logits"], atol=1e-4)
    # identical history -> identical output (determinism)
    a = net.forward({k: torch.as_tensor(v) for k, v in batch_for(h1).items()})
    assert torch.allclose(o1["belief_logits"], a["belief_logits"], atol=1e-6)


def test_aux_head_learns_consequences():
    """The auxiliary head regresses trajectory-derived race/clock targets and
    learns them (masked MSE drops; prediction converges to the fixed target)."""
    net = LorcanaNet(d_model=32, n_layers=2)
    learner = Learner(net, lr=3e-3)
    buf = ReplayBuffer()
    o = _obs(opp_hand=3, opp_deck=5, opp_ink=2)
    target = np.array([0.8, 0.2, 0.3], np.float32)   # self lore high, opp low, mid clock
    for _ in range(32):
        buf.add(Sample(enc=encode_obs(o), pi=np.array([1, 0, 0], np.float32), z=0.0,
                       belief=encode_belief(o), aux=target.copy()))
    s0 = learner.train_epoch(buf, batch_size=16)
    for _ in range(40):
        s1 = learner.train_epoch(buf, batch_size=16)
    assert "aux" in s1 and s1["aux"] < s0["aux"]
    batch = {**collate([encode_obs(o)]), **collate_belief([encode_belief(o)])}
    pred = torch.sigmoid(net.forward({k: torch.as_tensor(v) for k, v in batch.items()})["aux_logits"])[0]
    assert float(((pred - torch.as_tensor(target)) ** 2).mean()) < 0.02


def test_belief_learns_fixed_pattern():
    """On a fixed obs (so identities carry signal), the head should learn which
    candidates are in hand: loss drops and in-hand/out separation grows."""
    net = LorcanaNet(d_model=32, n_layers=2)
    learner = Learner(net, lr=3e-3)
    buf = ReplayBuffer()
    o = _obs(opp_hand=3, opp_deck=5)
    for _ in range(32):
        buf.add(Sample(enc=encode_obs(o), pi=np.array([1, 0, 0], np.float32), z=0.0,
                       belief=encode_belief(o)))
    s0 = learner.train_epoch(buf, batch_size=16)
    for _ in range(30):
        s1 = learner.train_epoch(buf, batch_size=16)
    assert s1["belief"] < s0["belief"]
    assert s1["belief_sep"] > 0.3                  # clearly separates in-hand cards
    assert s1["belief_count_mae"] < 1.0
