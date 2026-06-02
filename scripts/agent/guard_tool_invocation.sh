#!/usr/bin/env bash
set -euo pipefail

# Adapted guard pattern for local agent safety.
# Reads JSON or plain text from stdin and blocks dangerous operations.

if [[ "${SKIP_TOOL_GUARD:-}" == "true" ]]; then
  exit 0
fi

INPUT="$(cat)"
MODE="${GUARD_MODE:-block}"
LOG_DIR="${TOOL_GUARD_LOG_DIR:-logs/agent/tool-guardian}"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/guard.log"

TOOL_NAME=""
TOOL_INPUT="$INPUT"

if command -v jq >/dev/null 2>&1; then
  TOOL_NAME="$(printf '%s' "$INPUT" | jq -r '.toolName // .tool_name // empty' 2>/dev/null || true)"
  parsed="$(printf '%s' "$INPUT" | jq -r '.toolInput // .tool_input // .command // .args // .input // empty' 2>/dev/null || true)"
  if [[ -n "$parsed" && "$parsed" != "null" ]]; then
    TOOL_INPUT="$parsed"
  fi
fi

ALLOWLIST="${TOOL_GUARD_ALLOWLIST:-}"
if [[ -n "$ALLOWLIST" ]]; then
  IFS=',' read -ra ALLOW_PATTERNS <<< "$ALLOWLIST"
  for pat in "${ALLOW_PATTERNS[@]}"; do
    if [[ "$TOOL_INPUT" == *"$pat"* ]]; then
      echo "[$TIMESTAMP] allowlisted tool invocation: $pat" >> "$LOG_FILE"
      exit 0
    fi
  done
fi

THREATS=()
check() {
  local pattern="$1"
  local severity="$2"
  local message="$3"
  if printf '%s' "$TOOL_INPUT" | grep -qiE "$pattern"; then
    THREATS+=("$severity|$message|$pattern")
  fi
}

check 'rm[[:space:]]+-rf[[:space:]]+(/|\*|\$HOME|~|\.)' critical 'destructive recursive delete'
check 'git[[:space:]]+push[[:space:]].*--force' high 'force push'
check 'git[[:space:]]+reset[[:space:]]+--hard' high 'hard reset'
check 'git[[:space:]]+clean[[:space:]]+-fd' high 'destructive git clean'
check 'curl[^|;]*\|[[:space:]]*(sh|bash)' critical 'curl pipe to shell'
check 'wget[^|;]*\|[[:space:]]*(sh|bash)' critical 'wget pipe to shell'
check 'chmod[[:space:]]+777' medium 'world-writable chmod'
check 'sudo[[:space:]]+' medium 'sudo command'
check '(drop|truncate)[[:space:]]+(database|table)' critical 'destructive database command'
check 'lorcana-simulator/packages/lorcana/lorcana-engine/.+>[[:space:]]*/dev/null' high 'possible hidden oracle modification'

if [[ "${#THREATS[@]}" -gt 0 ]]; then
  {
    echo "[$TIMESTAMP] tool=$TOOL_NAME mode=$MODE threats=${#THREATS[@]}"
    printf '%s\n' "${THREATS[@]}"
    echo "input=$(printf '%s' "$TOOL_INPUT" | tr '\n' ' ' | cut -c1-500)"
  } >> "$LOG_FILE"

  echo "[tool-guardian] threats detected:"
  printf '  - %s\n' "${THREATS[@]}"

  if [[ "$MODE" == "block" ]]; then
    exit 1
  fi
fi

echo "[tool-guardian] passed."
