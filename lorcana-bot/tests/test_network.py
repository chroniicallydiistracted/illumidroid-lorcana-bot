"""Phase 1B — forward-pass shapes, legal masking, distributional value."""

import numpy as np
import torch

from engine.serialization import (
    encode_obs, collate, NUM_CATEGORIES, CARD_FEATURE_DIM, GLOBAL_FEATURE_DIM,
)
from network.model import LorcanaNet


def _fake_obs(n_cards=5, n_actions=4):
    cards = [{
        "id": f"c{i}", "owner": i % 2, "zone": "play", "cardType": "character",
        "cost": i, "strength": 2, "willpower": 3, "lore": 1, "damage": 0,
        "exerted": False, "drying": False, "ready": True,
        "keywords": ["Evasive"] if i == 0 else [], "definitionId": f"def-{i}",
        "hidden": False,
    } for i in range(n_cards)]
    legal = [{
        "idx": j, "stableKey": f"k{j}", "family": "playCard",
        "src": f"c{j % n_cards}", "tgt": None,
    } for j in range(n_actions)]
    return {
        "turn": 3, "phase": "main", "status": "playing", "done": False,
        "winner": None, "actor": "player_one", "forced": False,
        "players": {"self": {"lore": 2, "handCount": 4, "deckCount": 50,
                             "inkwell": 3, "discard": 1, "play": 2},
                    "opp": {"lore": 1, "handCount": 5, "deckCount": 52,
                            "inkwell": 2, "discard": 0, "play": 1}},
        "cards": cards, "legal": legal,
    }


def test_encode_shapes():
    enc = encode_obs(_fake_obs(5, 4))
    assert enc["card_feats"].shape == (5, CARD_FEATURE_DIM)
    assert enc["card_ids"].shape == (5,)
    assert enc["globals"].shape == (GLOBAL_FEATURE_DIM,)
    assert enc["action_cat"].shape == (4,)


def test_collate_variable_sizes():
    b = collate([encode_obs(_fake_obs(3, 2)), encode_obs(_fake_obs(7, 5))])
    assert b["card_feats"].shape == (2, 7, CARD_FEATURE_DIM)
    assert b["action_mask"].shape == (2, 5)
    assert b["action_mask"][0].sum() == 2  # first obs had 2 actions
    assert b["action_mask"][1].sum() == 5


def test_forward_and_masking():
    net = LorcanaNet(d_model=32, n_layers=2)
    batch = collate([encode_obs(_fake_obs(3, 2)), encode_obs(_fake_obs(7, 5))])
    out = net.forward({k: torch.as_tensor(v) for k, v in batch.items()})
    assert out["policy_logits"].shape == (2, 5)
    assert out["value_logits"].shape == (2, 51)
    assert out["value"].shape == (2,)
    assert torch.all(out["value"] >= -1.0) and torch.all(out["value"] <= 1.0)

    priors, values = net.infer(batch)
    mask = batch["action_mask"]
    assert np.allclose(priors[~mask], 0.0), "illegal actions received probability"
    # legal rows sum to ~1
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-4)


def test_two_hot_roundtrip():
    net = LorcanaNet(d_model=16, n_layers=1)
    z = torch.tensor([1.0, -1.0, 0.0, 0.5])
    dist = net.value.two_hot_target(z)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(4), atol=1e-5)
    recovered = (dist * net.value.atoms).sum(dim=-1)
    assert torch.allclose(recovered, z, atol=1e-5)
