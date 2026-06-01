"""Build a compact, deterministic card-identity vocabulary.

The old `vocab_id` hashed every card into a 32768-row table (`blake2b % 32768`),
of which only the few hundred cards that actually appear in the 25 real decks
were ever trained — 60% of the whole net was a dead lookup table, and each card's
row updated only on the rare steps it appeared, so belief/policy embeddings barely
moved.

This harvests the EXACT card universe that can appear in self-play (every card in
all 25 decks) by resetting each deck as the opponent once and reading its full
hidden composition (hand + deck + inkwell). The result is a fixed, contiguous
`def -> index` map committed as `card_vocab.json`, so every spawned worker and the
learner agree on it (determinism) while the embedding shrinks to ~the real catalog
size. Re-run this whenever the deck pool changes.

    python -m engine.build_vocab        # writes engine/card_vocab.json
"""

from __future__ import annotations

import json
import os

from engine.bridge import LorcanaEngine

VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_vocab.json")


def harvest() -> list[str]:
    defs: set[str] = set()
    with LorcanaEngine(timeout=300) as eng:
        deck_ids = [d["id"] for d in eng.list_decks()]
        n = len(deck_ids)
        for i, did in enumerate(deck_ids):
            # put deck `did` in the OPPONENT seat so its full composition shows up
            # in the fog-filtered `hidden` block; rotate the self seat so every
            # deck is harvested exactly once.
            obs = eng.reset(f"vocab-{i}", deck_ids[(i + 1) % n], did)
            hidden = obs.get("hidden") or {}
            for zone in ("hand", "deck", "inkwell"):
                for c in hidden.get(zone, []) or []:
                    d = c.get("def")
                    if d:
                        defs.add(str(d))
            for c in obs.get("cards", []) or []:        # own visible cards too
                d = c.get("definitionId")
                if d and not c.get("hidden"):
                    defs.add(str(d))
    return sorted(defs)


def main() -> None:
    cards = harvest()
    mapping = {d: i for i, d in enumerate(cards)}
    with open(VOCAB_PATH, "w") as f:
        json.dump({"cards": cards, "map": mapping}, f, indent=0)
    print(f"harvested {len(cards)} distinct card definitions -> {VOCAB_PATH}")


if __name__ == "__main__":
    main()
