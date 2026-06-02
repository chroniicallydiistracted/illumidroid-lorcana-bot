#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess

STEP = os.environ.get("AGENT_BLUEPRINT_STEP_NUMBER")
BASE = os.environ.get("AGENT_BASE_REF", "HEAD")

RULES = [
    (28, ("play-card.ts", "moves/core/play-card", "PlayCard"), "full playCard"),
    (26, ("resolve-bag.ts", "bag/"), "Bag resolution"),
    (27, ("replacement-effects.ts", "replacements/"), "replacement/prevention effects"),
    (31, ("lorcana-cards/src/cards/", "packages/lorcana/lorcana-cards/src/cards/"), "full card catalog"),
    (43, ("legal_action", "legal-actions", "legal_actions"), "legal-action generation"),
]


def changed_files() -> list[str]:
    try:
        out = subprocess.check_output(["git", "diff", "--name-only", BASE], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            out = subprocess.check_output(["git", "diff", "--name-only", "--cached"], text=True)
        except Exception:
            out = ""
    return [x.strip() for x in out.splitlines() if x.strip()]


def main() -> int:
    if not STEP:
        print("[dependency] AGENT_BLUEPRINT_STEP_NUMBER not set; skipping hard dependency gates.")
        return 0

    try:
        step_num = int(STEP)
    except ValueError:
        print(f"[dependency] invalid AGENT_BLUEPRINT_STEP_NUMBER={STEP!r}")
        return 1

    files = changed_files()
    fail = False

    for min_step, patterns, label in RULES:
        if step_num >= min_step:
            continue
        for f in files:
            if any(p in f for p in patterns):
                print(f"[dependency] BLOCKER: {label} touched before step {min_step}: {f}")
                fail = True

    if fail:
        return 1

    print("[dependency] passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
