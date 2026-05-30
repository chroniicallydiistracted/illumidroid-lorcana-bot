"""Phase 1A serialization — engine observation -> network tensors.

Turns a bridge `obs` dict into the permutation-invariant token representation
the set-transformer trunk consumes (architecture doc §2.1):

  * card tokens   : per-card numeric features + a definition-vocab id
  * global scalars: turn, per-player lore/hand/deck/ink/discard/play, flags
  * actions       : each legal action factored into (category, source card,
                    target card) -> the factored policy head (§2.5)

Card-source/target positions are 1-based with 0 reserved for a NULL token that
the model prepends; this lets actions with no source/target (e.g. passTurn)
point at a learned null embedding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- fixed vocabularies (network + search must agree on these) ---------------
CATEGORIES = [
    "chooseWhoGoesFirst",
    "alterHand",
    "resolveBag",
    "resolveEffect",
    "putCardIntoInkwell",
    "playCard",
    "activateAbility",
    "quest",
    "challenge",
    "moveCharacterToLocation",
    "passTurn",
]
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
NUM_CATEGORIES = len(CATEGORIES)

ZONES = ["deck", "hand", "play", "inkwell", "discard", "limbo"]
ZONE_INDEX = {z: i for i, z in enumerate(ZONES)}
NUM_ZONES = len(ZONES) + 1  # + unknown bucket

CARD_TYPES = ["character", "action", "item", "location"]
CARD_TYPE_INDEX = {t: i for i, t in enumerate(CARD_TYPES)}
NUM_CARD_TYPES = len(CARD_TYPES) + 1  # + unknown

KEYWORDS = [
    "Shift", "Evasive", "Rush", "Bodyguard", "Ward", "Resist",
    "Reckless", "Challenger", "Support", "Singer", "Vanish",
    "Sing Together", "Boost", "Alert",
]
KEYWORD_INDEX = {k.lower(): i for i, k in enumerate(KEYWORDS)}
NUM_KEYWORDS = len(KEYWORDS)

# definition-identity vocabulary (hashing trick). 0 = padding, 1 = hidden/unknown.
VOCAB_SIZE = 4096
PAD_ID = 0
HIDDEN_ID = 1

# per-card feature layout
CARD_FEATURE_DIM = (
    1            # owner (0 self / 1 opp)
    + NUM_ZONES
    + NUM_CARD_TYPES
    + 5          # cost, strength, willpower, lore, damage (scaled)
    + 4          # exerted, drying, ready, hidden
    + NUM_KEYWORDS
)

GLOBAL_FEATURE_DIM = (
    1            # turn (scaled)
    + 6          # self: lore, hand, deck, ink, discard, play
    + 6          # opp:  lore, hand, deck, ink, discard, play
    + 1          # forced flag
    + 1          # n legal (scaled)
)


def vocab_id(definition_id: Any, hidden: bool) -> int:
    if hidden or definition_id is None:
        return HIDDEN_ID
    h = hash(str(definition_id)) % (VOCAB_SIZE - 2)
    return h + 2  # avoid PAD_ID / HIDDEN_ID


def _scale(x: float, denom: float) -> float:
    return float(x) / denom


def encode_card(c: dict) -> np.ndarray:
    f = np.zeros(CARD_FEATURE_DIM, dtype=np.float32)
    i = 0
    f[i] = 1.0 if c.get("owner", 0) == 1 else 0.0
    i += 1
    z = ZONE_INDEX.get(c.get("zone", ""), NUM_ZONES - 1)
    f[i + z] = 1.0
    i += NUM_ZONES
    t = CARD_TYPE_INDEX.get(c.get("cardType", ""), NUM_CARD_TYPES - 1)
    f[i + t] = 1.0
    i += NUM_CARD_TYPES
    f[i + 0] = _scale(c.get("cost", 0), 10.0)
    f[i + 1] = _scale(c.get("strength", 0), 10.0)
    f[i + 2] = _scale(c.get("willpower", 0), 10.0)
    f[i + 3] = _scale(c.get("lore", 0), 5.0)
    f[i + 4] = _scale(c.get("damage", 0), 10.0)
    i += 5
    f[i + 0] = 1.0 if c.get("exerted") else 0.0
    f[i + 1] = 1.0 if c.get("drying") else 0.0
    f[i + 2] = 1.0 if c.get("ready") else 0.0
    f[i + 3] = 1.0 if c.get("hidden") else 0.0
    i += 4
    for kw in c.get("keywords", []) or []:
        ki = KEYWORD_INDEX.get(str(kw).lower())
        if ki is not None:
            f[i + ki] = 1.0
    return f


def encode_globals(obs: dict) -> np.ndarray:
    g = np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32)
    s = obs.get("players", {}).get("self", {})
    o = obs.get("players", {}).get("opp", {})
    g[0] = _scale(obs.get("turn", 0), 20.0)
    base = 1
    for j, key in enumerate(["lore", "handCount", "deckCount", "inkwell", "discard", "play"]):
        denom = 20.0 if key == "lore" else 60.0 if key == "deckCount" else 12.0
        g[base + j] = _scale(s.get(key, 0), denom)
        g[base + 6 + j] = _scale(o.get(key, 0), denom)
    base += 12
    g[base] = 1.0 if obs.get("forced") else 0.0
    g[base + 1] = _scale(len(obs.get("legal", [])), 30.0)
    return g


def encode_obs(obs: dict) -> dict[str, np.ndarray]:
    """Single observation -> dict of numpy arrays (unbatched)."""
    cards = obs.get("cards", [])
    n = len(cards)
    card_feats = np.zeros((n, CARD_FEATURE_DIM), dtype=np.float32)
    card_ids = np.zeros(n, dtype=np.int64)
    pos_of: dict[str, int] = {}
    for idx, c in enumerate(cards):
        card_feats[idx] = encode_card(c)
        card_ids[idx] = vocab_id(c.get("definitionId"), bool(c.get("hidden")))
        pos_of[str(c.get("id"))] = idx + 1  # +1: index 0 reserved for NULL token

    legal = obs.get("legal", [])
    a = len(legal)
    cat_idx = np.zeros(a, dtype=np.int64)
    src_pos = np.zeros(a, dtype=np.int64)
    tgt_pos = np.zeros(a, dtype=np.int64)
    for j, act in enumerate(legal):
        cat_idx[j] = CATEGORY_INDEX.get(act.get("family", ""), 0)
        src_pos[j] = pos_of.get(str(act.get("src")), 0)
        tgt_pos[j] = pos_of.get(str(act.get("tgt")), 0)

    return {
        "card_feats": card_feats,           # [N, Fc]
        "card_ids": card_ids,               # [N]
        "globals": encode_globals(obs),     # [G]
        "action_cat": cat_idx,              # [A]
        "action_src": src_pos,              # [A] (0 = null)
        "action_tgt": tgt_pos,              # [A] (0 = null)
        "n_cards": np.int64(n),
        "n_actions": np.int64(a),
    }


def collate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pad a list of encoded observations into a batch with masks."""
    B = len(batch)
    max_n = max(int(b["n_cards"]) for b in batch) if B else 0
    max_a = max(int(b["n_actions"]) for b in batch) if B else 0
    max_n = max(max_n, 1)
    max_a = max(max_a, 1)

    card_feats = np.zeros((B, max_n, CARD_FEATURE_DIM), dtype=np.float32)
    card_ids = np.zeros((B, max_n), dtype=np.int64)
    card_mask = np.zeros((B, max_n), dtype=bool)
    globals_ = np.zeros((B, GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_cat = np.zeros((B, max_a), dtype=np.int64)
    action_src = np.zeros((B, max_a), dtype=np.int64)
    action_tgt = np.zeros((B, max_a), dtype=np.int64)
    action_mask = np.zeros((B, max_a), dtype=bool)

    for i, b in enumerate(batch):
        n = int(b["n_cards"])
        a = int(b["n_actions"])
        if n:
            card_feats[i, :n] = b["card_feats"]
            card_ids[i, :n] = b["card_ids"]
            card_mask[i, :n] = True
        globals_[i] = b["globals"]
        if a:
            action_cat[i, :a] = b["action_cat"]
            action_src[i, :a] = b["action_src"]
            action_tgt[i, :a] = b["action_tgt"]
            action_mask[i, :a] = True

    return {
        "card_feats": card_feats,
        "card_ids": card_ids,
        "card_mask": card_mask,
        "globals": globals_,
        "action_cat": action_cat,
        "action_src": action_src,
        "action_tgt": action_tgt,
        "action_mask": action_mask,
    }
