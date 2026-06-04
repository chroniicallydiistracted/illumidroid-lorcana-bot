#!/usr/bin/env bash
set -euo pipefail

BASE_REF="${AGENT_BASE_REF:-HEAD}"
STEP="${AGENT_BLUEPRINT_STEP_NUMBER:-}"

changed_files() {
  git diff --name-only "$BASE_REF" 2>/dev/null || git diff --name-only --cached || true
}

FILES="$(changed_files)"
# The vendored frozen oracle (oracle/source/**) is a byte-exact reference copy,
# not port-authored code. It legitimately contains card files, play-card.ts,
# .svelte components, packet animations, etc. Exempt it from the dependency/UI
# path blocks; its integrity is policed by the oracle freeze tests/verifier.
FILES="$(printf '%s\n' "$FILES" | grep -v '^oracle/source/' || true)"
if [[ -z "$FILES" ]]; then
  echo "[paths] no changed files detected (after excluding frozen oracle/source/)."
  exit 0
fi

FAIL=0

while IFS= read -r file; do
  case "$file" in
    packages/lorcana/lorcana-simulator/*|lorcana-simulator/packages/lorcana/lorcana-simulator/*)
      echo "[paths] REVIEW/BLOCKER: UI simulator path changed: $file"
      ;;
    *runtime-game/lorcanaPacketAnimations.ts|*.svelte)
      echo "[paths] REVIEW/BLOCKER: visual/UI path changed: $file"
      ;;
  esac
done <<< "$FILES"

if [[ -n "$STEP" ]]; then
  while IFS= read -r file; do
    if [[ "$STEP" -lt 31 && "$file" == *"lorcana-cards/src/cards/"* ]]; then
      echo "[paths] BLOCKER: card file touched before card catalog step: $file"
      FAIL=1
    fi
    if [[ "$STEP" -lt 28 && "$file" == *"play-card.ts"* ]]; then
      echo "[paths] BLOCKER: playCard touched before full playCard step: $file"
      FAIL=1
    fi
    if [[ "$STEP" -lt 26 && "$file" == *"resolve-bag.ts"* ]]; then
      echo "[paths] BLOCKER: Bag resolution touched before Bag step: $file"
      FAIL=1
    fi
    if [[ "$STEP" -lt 27 && "$file" == *"replacement-effects.ts"* ]]; then
      echo "[paths] BLOCKER: replacement effects touched before replacement step: $file"
      FAIL=1
    fi
    if [[ "$STEP" -lt 43 && "$file" == *"legal_action"* ]]; then
      echo "[paths] BLOCKER: legal-action generator touched before legal-action step: $file"
      FAIL=1
    fi
  done <<< "$FILES"
else
  echo "[paths] AGENT_BLUEPRINT_STEP_NUMBER not set; dependency path blocks are warnings only."
fi

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

echo "[paths] passed."
