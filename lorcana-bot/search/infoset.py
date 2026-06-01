"""Tier-A #2 — leak-free information-set identity.

An information-set key may include ONLY information available to the player whose
decision is being selected at that node (architecture doc §2; spec §6). It must
NEVER include `obs["hidden"]` (the authoritative opponent zones), opponent hidden
instance identities, or either deck's draw order — doing so would split
indistinguishable hidden worlds into separate nodes and recreate strategy fusion.

Two paths that reach the same visible board can still be different information sets
when the actor remembers a different past, so the key folds in a leak-free,
observer-projected branch history (face-down inking is redacted; public plays /
quests / challenges keep their identity).
"""

from __future__ import annotations

import hashlib
import json

# actions whose moved card identity is HIDDEN to observers (face-down inking):
# redact the card so the branch history never leaks the inked identity.
_REDACT_FAMILIES = {"putCardIntoInkwell"}


def _public(p: dict) -> list:
    return [p.get("lore", 0), p.get("handCount", 0), p.get("deckCount", 0),
            p.get("inkwell", 0), p.get("discard", 0), p.get("play", 0)]


def canonical_cards(obs: dict) -> tuple[list, dict]:
    """Split the filtered board into (sorted visible card rows, hidden-zone counts).

    Visible cards contribute their gameplay-relevant fields (the actor legitimately
    knows them). Hidden cards (opponent hand/deck/inkwell) contribute ONLY an
    aggregate `(owner, zone)` count — never their instance id or identity.
    """
    rows: list = []
    hidden: dict = {}
    for c in obs.get("cards", []):
        if c.get("hidden"):
            k = f"{c.get('owner')}:{c.get('zone')}"
            hidden[k] = hidden.get(k, 0) + 1
        else:
            rows.append([
                c.get("owner"), c.get("zone"), c.get("definitionId"),
                c.get("cost", 0), c.get("strength", 0), c.get("willpower", 0),
                c.get("lore", 0), c.get("damage", 0),
                int(bool(c.get("exerted"))), int(bool(c.get("drying"))),
            ])
    rows.sort(key=lambda r: tuple("" if x is None else x for x in r))
    return rows, dict(sorted(hidden.items()))


def history_event(actor_seat, family: str, card_id) -> list:
    """One observer-projected branch event (leak-free): redact the card for
    face-down inking, keep public action identities."""
    return [actor_seat, family, None if family in _REDACT_FAMILIES else card_id]


def info_set_key(obs: dict, branch_events: tuple | list = ()) -> str:
    """Deterministic, leak-free information-set key for the CURRENT actor's decision.

    Built only from actor-visible fields + an observer-projected branch history.
    `obs["hidden"]` is never read, so scrambling opponent hidden identities / deck
    order cannot change the key (proven in tests)."""
    players = obs.get("players", {})
    rows, hidden = canonical_cards(obs)
    actions = sorted(str(a.get("actionId", a.get("stableKey"))) for a in obs.get("legal", []))
    payload = {
        "actor": obs.get("actor"),
        "seat": obs.get("selfIdx"),
        "turn": obs.get("turn"),
        "phase": obs.get("phase"),
        "step": obs.get("step"),
        "status": obs.get("status"),
        "forced": bool(obs.get("forced")),
        "self": _public(players.get("self", {})),
        "opp": _public(players.get("opp", {})),
        "cards": rows,
        "hidden": hidden,                 # aggregate counts only — NO identities
        "actions": actions,
        "hist": [list(e) for e in branch_events],
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
