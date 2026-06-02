#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_GOVERNANCE_AUDIT:-}" == "true" ]]; then
  exit 0
fi

INPUT="$(cat)"
LEVEL="${GOVERNANCE_LEVEL:-standard}"
BLOCK="${BLOCK_ON_THREAT:-false}"
LOG_DIR="${GOVERNANCE_LOG_DIR:-logs/agent/governance}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/audit.log"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

PROMPT="$INPUT"
if command -v jq >/dev/null 2>&1; then
  parsed="$(printf '%s' "$INPUT" | jq -r '.userMessage // .prompt // .message // empty' 2>/dev/null || true)"
  if [[ -n "$parsed" && "$parsed" != "null" ]]; then
    PROMPT="$parsed"
  fi
fi

THREATS=()
check() {
  local pattern="$1"
  local severity="$2"
  local message="$3"
  if printf '%s' "$PROMPT" | grep -qiE "$pattern"; then
    THREATS+=("$severity|$message|$pattern")
  fi
}

check 'ignore (all )?(previous|prior|system|developer|agent) instructions' high 'instruction override attempt'
check 'do not (tell|mention|report|log)' medium 'concealment request'
check 'skip (tests|conformance|audit|registry)' high 'quality gate bypass'
check 'modify (the )?TypeScript oracle' high 'oracle modification request'
check 'force push|git reset --hard|rm -rf' high 'dangerous repo operation'
check 'use Math\.random|rand::StdRng|thread_rng' high 'determinism violation request'
check 'start with playCard|start with Bag|port card files first' high 'dependency-order violation request'

if [[ "${#THREATS[@]}" -gt 0 ]]; then
  {
    echo "[$TIMESTAMP] level=$LEVEL threats=${#THREATS[@]}"
    printf '%s\n' "${THREATS[@]}"
    echo "prompt=$(printf '%s' "$PROMPT" | tr '\n' ' ' | cut -c1-500)"
  } >> "$LOG_FILE"
  echo "[governance] threats detected:"
  printf '  - %s\n' "${THREATS[@]}"
  if [[ "$BLOCK" == "true" ]]; then
    exit 1
  fi
fi

echo "[governance] passed."
