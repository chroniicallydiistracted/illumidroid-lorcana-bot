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

import hashlib
import json
import os
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

# definition-identity vocabulary. 0 = padding, 1 = hidden/unknown.
#
# We use a COMPACT, CONTIGUOUS map over the exact card universe that appears in
# the deck pool (`engine/card_vocab.json`, built by `engine.build_vocab`): the 25
# real decks contain only ~155 distinct cards, so a 32768-row hash table wasted
# ~99.5% of its rows AND starved each card's embedding of gradient updates. Known
# cards get a dedicated, collision-free row (2 .. 1+K); a small OOV band absorbs
# anything outside the pool via a process-stable hash (blake2b, NOT builtin hash()
# which is PYTHONHASHSEED-salted). The file is committed so every spawned worker
# and the learner agree on the same ids (determinism).
PAD_ID = 0
HIDDEN_ID = 1
_OOV_BUCKETS = 256

_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_vocab.json")
try:
    with open(_VOCAB_PATH) as _f:
        _CARD_MAP: dict[str, int] = json.load(_f)["map"]
except (OSError, ValueError, KeyError):
    _CARD_MAP = {}                       # fall back to pure-hash if not yet built

_KNOWN = len(_CARD_MAP)
_OOV_BASE = 2 + _KNOWN
VOCAB_SIZE = _OOV_BASE + _OOV_BUCKETS    # embedding row count (trunk reads this)

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

# --- rich action-candidate descriptor (ACE) ----------------------------------
# Each legal candidate is encoded as (a) a scalar/categorical feature vector and
# (b) a set of role-tagged pointers into the card-token set, so the candidate
# transformer can distinguish actions that share family/src/first-target but
# differ in singer / shift target / 2nd target / named card / banish-discard-
# exert cost / destination / optional / choice-index.
COST_TYPES = ["none", "standard", "free", "shift", "sing"]
COST_TYPE_INDEX = {t: i for i, t in enumerate(COST_TYPES)}
DEST_ZONES = ["deck", "hand", "play", "inkwell", "discard"]
DEST_ZONE_INDEX = {z: i for i, z in enumerate(DEST_ZONES)}

# token roles (0 = pad). The named-card token is handled separately (cand_named_id).
NUM_ROLES = 8
ROLE_SRC, ROLE_TGT, ROLE_SINGER, ROLE_BANISH, ROLE_DISCARD, ROLE_EXERT, ROLE_SHIFT = 1, 2, 3, 4, 5, 6, 7
_CAND_CAP = {"targets": 4, "singers": 3, "banish": 2, "discard": 2, "exert": 2}
CAND_TOK_MAX = 16          # 1 src + 4 tgt + 3 singer + 1 shift + 2 banish + 2 discard + 2 exert
CAND_FEAT_DIM = (
    len(COST_TYPES)        # cost-type one-hot
    + 1                    # ink cost (scaled)
    + 1                    # choiceIndex (scaled)
    + 1                    # resolveOptional
    + 1                    # abilityIndex (scaled)
    + 1                    # has named card
    + len(DEST_ZONES)      # destination zones (multi-hot)
    + 3                    # arity counts: n_targets, n_singers, n_costcards (scaled)
)

# --- public-history encoder (§2.3) -------------------------------------------
# A rolling window of recent PUBLIC events. Each event is a token: what family of
# move, by self or opponent, the public counts after it, recency, and the played-
# card identity (the key belief signal — a card that left the opponent's hand;
# inked cards stay hidden so defId is 0). Everything here is public => leak-free.
HIST_MAX = 48              # must match the bridge's HIST_MAX
HIST_FEAT_DIM = (
    NUM_CATEGORIES         # family one-hot
    + 1                    # is-self (actor == viewer)
    + 6                    # lore/ink/hand for self & opp (scaled)
    + 1                    # turn (scaled)
    + 1                    # recency (events-ago, scaled)
    + 1                    # has played-card identity
)


def vocab_id(definition_id: Any, hidden: bool) -> int:
    if hidden or definition_id is None:
        return HIDDEN_ID
    key = str(definition_id)
    idx = _CARD_MAP.get(key)
    if idx is not None:
        return 2 + idx                   # dedicated, collision-free row
    # outside the known pool -> stable hash into the small OOV band.
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return _OOV_BASE + int.from_bytes(digest, "big") % _OOV_BUCKETS


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


def _encode_candidate(act: dict, pos_of: dict[str, int]):
    """One legal candidate -> (feat [CAND_FEAT_DIM], tok_pos/tok_role/tok_mask
    [CAND_TOK_MAX], named_id). Pointers index the card-token set (0 = NULL)."""
    f = np.zeros(CAND_FEAT_DIM, dtype=np.float32)
    i = 0
    ct = COST_TYPE_INDEX.get(act.get("costType", "none"), 0)
    f[i + ct] = 1.0
    i += len(COST_TYPES)
    f[i] = _scale(max(int(act.get("inkCost", -1)), -1) + 1, 10.0); i += 1
    f[i] = _scale(max(int(act.get("choice", -1)), -1) + 1, 8.0);   i += 1
    f[i] = 1.0 if act.get("resolveOptional") else 0.0;             i += 1
    f[i] = _scale(max(int(act.get("abilityIndex", -1)), -1) + 1, 8.0); i += 1
    f[i] = 1.0 if act.get("namedCard") else 0.0;                   i += 1
    for z in act.get("destinationZones", []) or []:
        zi = DEST_ZONE_INDEX.get(z)
        if zi is not None:
            f[i + zi] = 1.0
    i += len(DEST_ZONES)
    tgts = act.get("targets", []) or []
    sing = act.get("singers", []) or []
    banish, discard, exert = (act.get("banish") or [], act.get("discard") or [], act.get("exert") or [])
    f[i] = _scale(len(tgts), 4.0)
    f[i + 1] = _scale(len(sing), 3.0)
    f[i + 2] = _scale(len(banish) + len(discard) + len(exert), 4.0)

    toks: list[tuple[str, int]] = []
    if act.get("src"):
        toks.append((act["src"], ROLE_SRC))
    toks += [(x, ROLE_TGT) for x in tgts[: _CAND_CAP["targets"]]]
    toks += [(x, ROLE_SINGER) for x in sing[: _CAND_CAP["singers"]]]
    if act.get("shiftTarget"):
        toks.append((act["shiftTarget"], ROLE_SHIFT))
    toks += [(x, ROLE_BANISH) for x in banish[: _CAND_CAP["banish"]]]
    toks += [(x, ROLE_DISCARD) for x in discard[: _CAND_CAP["discard"]]]
    toks += [(x, ROLE_EXERT) for x in exert[: _CAND_CAP["exert"]]]

    tok_pos = np.zeros(CAND_TOK_MAX, dtype=np.int64)
    tok_role = np.zeros(CAND_TOK_MAX, dtype=np.int64)
    tok_mask = np.zeros(CAND_TOK_MAX, dtype=bool)
    for k, (cid, role) in enumerate(toks[:CAND_TOK_MAX]):
        tok_pos[k] = pos_of.get(str(cid), 0)
        tok_role[k] = role
        tok_mask[k] = True
    nm = act.get("namedCard")
    named_id = vocab_id(nm, hidden=False) if nm else 0
    return f, tok_pos, tok_role, tok_mask, named_id


def encode_history(obs: dict):
    """Public event history -> (feats [H, HIST_FEAT_DIM], ids [H]). All public,
    so it is safe to feed the trunk (policy/value/belief). Self/opp are resolved
    relative to the current viewer via `selfIdx`. Index 0 = oldest in window."""
    hist = obs.get("history", []) or []
    H = len(hist)
    feats = np.zeros((H, HIST_FEAT_DIM), dtype=np.float32)
    ids = np.zeros(H, dtype=np.int64)
    so = int(obs.get("selfIdx", 0))
    op = 1 - so
    for i, e in enumerate(hist):
        f = feats[i]
        j = 0
        f[CATEGORY_INDEX.get(e.get("family", ""), 0)] = 1.0
        j += NUM_CATEGORIES
        f[j] = 1.0 if int(e.get("actor", 0)) == so else 0.0
        j += 1
        lore = e.get("lore", [0, 0]); ink = e.get("ink", [0, 0]); hand = e.get("hand", [0, 0])
        f[j] = _scale(lore[so], 20.0); f[j + 1] = _scale(lore[op], 20.0); j += 2
        f[j] = _scale(ink[so], 12.0);  f[j + 1] = _scale(ink[op], 12.0);  j += 2
        f[j] = _scale(hand[so], 12.0); f[j + 1] = _scale(hand[op], 12.0); j += 2
        f[j] = _scale(e.get("turn", 0), 20.0); j += 1
        f[j] = _scale(H - 1 - i, HIST_MAX); j += 1     # events-ago (newest = 0)
        defid = e.get("defId")
        f[j] = 1.0 if defid else 0.0
        ids[i] = vocab_id(defid, hidden=False) if defid else 0
    return feats, ids


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
    tgt2_pos = np.zeros(a, dtype=np.int64)
    choice = np.zeros(a, dtype=np.float32)
    cand_feats = np.zeros((a, CAND_FEAT_DIM), dtype=np.float32)
    cand_tok_pos = np.zeros((a, CAND_TOK_MAX), dtype=np.int64)
    cand_tok_role = np.zeros((a, CAND_TOK_MAX), dtype=np.int64)
    cand_tok_mask = np.zeros((a, CAND_TOK_MAX), dtype=bool)
    cand_named_id = np.zeros(a, dtype=np.int64)
    for j, act in enumerate(legal):
        cat_idx[j] = CATEGORY_INDEX.get(act.get("family", ""), 0)
        src_pos[j] = pos_of.get(str(act.get("src")), 0)
        tgt_pos[j] = pos_of.get(str(act.get("tgt")), 0)
        tgt2_pos[j] = pos_of.get(str(act.get("tgt2")), 0)   # 2nd target (0 = null)
        choice[j] = _scale(max(int(act.get("choice", -1)), -1) + 1, 8.0)  # modal choiceIndex
        f, tp, tr, tm, nid = _encode_candidate(act, pos_of)
        cand_feats[j] = f
        cand_tok_pos[j] = tp
        cand_tok_role[j] = tr
        cand_tok_mask[j] = tm
        cand_named_id[j] = nid

    hist_feats, hist_ids = encode_history(obs)

    return {
        "card_feats": card_feats,           # [N, Fc]
        "card_ids": card_ids,               # [N]
        "globals": encode_globals(obs),     # [G]
        "hist_feats": hist_feats,           # [H, HIST_FEAT_DIM] (public events)
        "hist_ids": hist_ids,               # [H] (played-card vocab id, 0 = none/hidden)
        "n_hist": np.int64(len(hist_ids)),
        "action_cat": cat_idx,              # [A]
        "action_src": src_pos,              # [A] (0 = null)  (legacy)
        "action_tgt": tgt_pos,              # [A] (0 = null)  (legacy)
        "action_tgt2": tgt2_pos,            # [A] (0 = null)  (legacy)
        "action_choice": choice,            # [A] (legacy)
        "cand_feats": cand_feats,           # [A, CAND_FEAT_DIM]
        "cand_tok_pos": cand_tok_pos,       # [A, CAND_TOK_MAX] (pointers, 0 = null)
        "cand_tok_role": cand_tok_role,     # [A, CAND_TOK_MAX] (role ids, 0 = pad)
        "cand_tok_mask": cand_tok_mask,     # [A, CAND_TOK_MAX] (bool)
        "cand_named_id": cand_named_id,     # [A] (named-card vocab id, 0 = none)
        "n_cards": np.int64(n),
        "n_actions": np.int64(a),
    }


def encode_belief(obs: dict) -> dict[str, np.ndarray]:
    """Belief target/conditioning from the opponent's true hidden zones.

    Candidate pool = the opponent's hidden cards (hand + deck), i.e. the known
    composition in self-play (deck-conditioning, architecture doc §6). For each
    candidate we emit its identity-vocab id and a binary label `in_hand`. The
    belief head scores each candidate against the (leak-free) trunk; these arrays
    NEVER enter the trunk, so policy/value cannot see opponent identities.
    """
    hidden = obs.get("hidden") or {}
    hand = hidden.get("hand", []) or []
    deck = hidden.get("deck", []) or []
    ink = hidden.get("inkwell", []) or []
    # pool = ALL hidden opponent cards; two targets per card: in_hand, in_inkwell
    # (in_deck = neither). Inkwell identity is hidden+strategically important.
    pool = ([(c, 1.0, 0.0) for c in hand]
            + [(c, 0.0, 0.0) for c in deck]
            + [(c, 0.0, 1.0) for c in ink])
    p = len(pool)
    bel_ids = np.zeros(p, dtype=np.int64)
    bel_label = np.zeros(p, dtype=np.float32)       # in opp hand
    bel_label_ink = np.zeros(p, dtype=np.float32)   # in opp inkwell
    for i, (c, lab, lab_ink) in enumerate(pool):
        bel_ids[i] = vocab_id(c.get("def"), hidden=False)  # identity is known (conditioning)
        bel_label[i] = lab
        bel_label_ink[i] = lab_ink
    return {
        "belief_ids": bel_ids,                       # [P]
        "belief_label": bel_label,                   # [P] (1 = in opp hand)
        "belief_label_ink": bel_label_ink,           # [P] (1 = in opp inkwell)
        "belief_handcount": np.int64(int(bel_label.sum())),
        "belief_inkcount": np.int64(int(bel_label_ink.sum())),
        "n_pool": np.int64(p),
    }


def collate_belief(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pad a list of encoded belief targets into a batch with a candidate mask."""
    B = len(batch)
    max_p = max((int(b["n_pool"]) for b in batch), default=1)
    max_p = max(max_p, 1)
    bel_ids = np.zeros((B, max_p), dtype=np.int64)
    bel_label = np.zeros((B, max_p), dtype=np.float32)
    bel_label_ink = np.zeros((B, max_p), dtype=np.float32)
    bel_mask = np.zeros((B, max_p), dtype=bool)
    handcount = np.zeros(B, dtype=np.float32)
    inkcount = np.zeros(B, dtype=np.float32)
    for i, b in enumerate(batch):
        p = int(b["n_pool"])
        if p:
            bel_ids[i, :p] = b["belief_ids"]
            bel_label[i, :p] = b["belief_label"]
            bel_label_ink[i, :p] = b.get("belief_label_ink", np.zeros(p, np.float32))
            bel_mask[i, :p] = True
        handcount[i] = float(b["belief_handcount"])
        inkcount[i] = float(b.get("belief_inkcount", 0))
    return {
        "belief_ids": bel_ids,
        "belief_label": bel_label,
        "belief_label_ink": bel_label_ink,
        "belief_mask": bel_mask,
        "belief_handcount": handcount,
        "belief_inkcount": inkcount,
    }


def collate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pad a list of encoded observations into a batch with masks."""
    B = len(batch)
    max_n = max(int(b["n_cards"]) for b in batch) if B else 0
    max_a = max(int(b["n_actions"]) for b in batch) if B else 0
    max_h = max((int(b.get("n_hist", 0)) for b in batch), default=0) if B else 0
    max_n = max(max_n, 1)
    max_a = max(max_a, 1)
    max_h = max(max_h, 1)

    card_feats = np.zeros((B, max_n, CARD_FEATURE_DIM), dtype=np.float32)
    card_ids = np.zeros((B, max_n), dtype=np.int64)
    card_mask = np.zeros((B, max_n), dtype=bool)
    globals_ = np.zeros((B, GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_cat = np.zeros((B, max_a), dtype=np.int64)
    action_src = np.zeros((B, max_a), dtype=np.int64)
    action_tgt = np.zeros((B, max_a), dtype=np.int64)
    action_tgt2 = np.zeros((B, max_a), dtype=np.int64)
    action_choice = np.zeros((B, max_a), dtype=np.float32)
    action_mask = np.zeros((B, max_a), dtype=bool)
    cand_feats = np.zeros((B, max_a, CAND_FEAT_DIM), dtype=np.float32)
    cand_tok_pos = np.zeros((B, max_a, CAND_TOK_MAX), dtype=np.int64)
    cand_tok_role = np.zeros((B, max_a, CAND_TOK_MAX), dtype=np.int64)
    cand_tok_mask = np.zeros((B, max_a, CAND_TOK_MAX), dtype=bool)
    cand_named_id = np.zeros((B, max_a), dtype=np.int64)
    hist_feats = np.zeros((B, max_h, HIST_FEAT_DIM), dtype=np.float32)
    hist_ids = np.zeros((B, max_h), dtype=np.int64)
    hist_mask = np.zeros((B, max_h), dtype=bool)

    for i, b in enumerate(batch):
        n = int(b["n_cards"])
        a = int(b["n_actions"])
        h = int(b.get("n_hist", 0))
        if h:
            hist_feats[i, :h] = b["hist_feats"]
            hist_ids[i, :h] = b["hist_ids"]
            hist_mask[i, :h] = True
        if n:
            card_feats[i, :n] = b["card_feats"]
            card_ids[i, :n] = b["card_ids"]
            card_mask[i, :n] = True
        globals_[i] = b["globals"]
        if a:
            action_cat[i, :a] = b["action_cat"]
            action_src[i, :a] = b["action_src"]
            action_tgt[i, :a] = b["action_tgt"]
            action_tgt2[i, :a] = b.get("action_tgt2", np.zeros(a, np.int64))
            action_choice[i, :a] = b.get("action_choice", np.zeros(a, np.float32))
            action_mask[i, :a] = True
            if "cand_feats" in b:
                cand_feats[i, :a] = b["cand_feats"]
                cand_tok_pos[i, :a] = b["cand_tok_pos"]
                cand_tok_role[i, :a] = b["cand_tok_role"]
                cand_tok_mask[i, :a] = b["cand_tok_mask"]
                cand_named_id[i, :a] = b["cand_named_id"]

    return {
        "card_feats": card_feats,
        "card_ids": card_ids,
        "card_mask": card_mask,
        "globals": globals_,
        "action_cat": action_cat,
        "action_src": action_src,
        "action_tgt": action_tgt,
        "action_tgt2": action_tgt2,
        "action_choice": action_choice,
        "action_mask": action_mask,
        "cand_feats": cand_feats,
        "cand_tok_pos": cand_tok_pos,
        "cand_tok_role": cand_tok_role,
        "cand_tok_mask": cand_tok_mask,
        "cand_named_id": cand_named_id,
        "hist_feats": hist_feats,
        "hist_ids": hist_ids,
        "hist_mask": hist_mask,
    }
