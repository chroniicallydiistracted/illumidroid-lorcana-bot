"""Grammar-gap proof: measure where illumidroid's action enumeration differs from
the engine's full grammar, across REAL decision states.

For each decision we compare three enumerations of the SAME state:
  * capped automation   = enumerateAutomatedActions({searchCaps: 24/48/16/32})  (what the bot uses)
  * uncapped automation = same with effectively-infinite caps                   (Botcana-style full)
  * raw grammar         = engine enumerateMovesForPlayer(actor)                  (true legal move ids)

Answers, with numbers:
  1. Do illumidroid's caps truncate the action space?  (uncapped > capped, how often / how much)
  2. Are there ZERO-candidate automation states?       (the infinite-loop cause)
  3. In those, does the raw grammar expose a legal ACTION the automation missed?

Run:  python -m tools.grammar_gap_probe [n_games]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from engine.bridge import LorcanaEngine

# raw move ids that are not "real" actions (always-available / synthetic)
_NON_ACTION = {"passTurn", "concede", "pass"}


def main() -> None:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    decisions = 0
    cap_truncated = 0           # uncapped candidate count > capped
    trunc_deltas: list[int] = []
    capped_overflow_decisions = 0   # decisions where the bot's caps dropped combinations
    zero_capped = 0
    zero_uncapped = 0
    zero_uncapped_with_unsupported = 0  # automation empty AND a legal move it can't represent
    unsupported_decisions = 0           # decisions with >=1 unsupported-shape (true grammar gap)
    unsupported_reasons: Counter = Counter()
    auto_family_freq: Counter = Counter()
    gap_examples: list[dict] = []

    with LorcanaEngine(timeout=300) as eng:
        deck_ids = [d["id"] for d in eng.list_decks()]
        for g in range(n_games):
            # rotate deck pairs (incl. the matchup that previously stalled)
            p1 = deck_ids[g % len(deck_ids)]
            p2 = deck_ids[(g + 1) % len(deck_ids)]
            obs = eng.reset(f"probe-{g}", p1, p2)
            steps = 0
            while not obs.get("done") and steps < 250:
                pr = eng.grammar_probe()
                if pr.get("actor"):
                    decisions += 1
                    cc, uc = pr["cappedCount"], pr["uncappedCount"]
                    if uc > cc:
                        cap_truncated += 1
                        trunc_deltas.append(uc - cc)
                    if pr.get("cappedOverflow", 0) > 0:
                        capped_overflow_decisions += 1
                    if cc == 0:
                        zero_capped += 1
                    for f in pr.get("uncappedFamilies", []):
                        auto_family_freq[f] += 1
                    unsup = pr.get("unsupported", []) or []   # true grammar gap
                    if unsup:
                        unsupported_decisions += 1
                        for d in unsup:
                            unsupported_reasons[f"{d['family']}: {d['reason']}"] += 1
                        if len(gap_examples) < 10:
                            gap_examples.append({"turn": pr["turn"], "phase": pr["phase"],
                                                 "uncappedCount": uc, "unsupported": unsup})
                    if uc == 0:
                        zero_uncapped += 1
                        if unsup:
                            zero_uncapped_with_unsupported += 1
                res = eng.step_auto("best")
                obs = res["obs"]
                steps += 1

    avg_delta = (sum(trunc_deltas) / len(trunc_deltas)) if trunc_deltas else 0.0
    out = {
        "n_games": n_games,
        "decisions_sampled": decisions,
        "Q1_cap_truncation": {
            "decisions_where_uncapped>capped": cap_truncated,
            "pct": round(100 * cap_truncated / max(decisions, 1), 1),
            "decisions_with_capped_overflow_skip": capped_overflow_decisions,
            "avg_extra_candidates_when_truncated": round(avg_delta, 1),
            "max_extra_candidates": max(trunc_deltas) if trunc_deltas else 0,
        },
        "Q2_grammar_gap_unsupported_shape": {
            "decisions_with_unsupported_move": unsupported_decisions,
            "pct": round(100 * unsupported_decisions / max(decisions, 1), 1),
            "reasons": dict(unsupported_reasons.most_common()),
            "examples": gap_examples,
        },
        "Q3_zero_candidate_states": {
            "uncapped_zero": zero_uncapped,
            "pct": round(100 * zero_uncapped / max(decisions, 1), 1),
            "zero_AND_grammar_had_unsupported_move": zero_uncapped_with_unsupported,
            "note": "zero candidates with NO unsupported move == genuine pass states (normal); "
                    "zero WITH an unsupported move == automation missed a forced legal move (stuck cause)",
        },
        "automation_families_seen": dict(auto_family_freq.most_common()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
