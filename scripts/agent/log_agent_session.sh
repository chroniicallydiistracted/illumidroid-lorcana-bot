#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_AGENT_LOGGING:-}" == "true" ]]; then
  exit 0
fi

EVENT="${1:-event}"
INPUT="$(cat || true)"
LOG_DIR="${AGENT_SESSION_LOG_DIR:-logs/agent/session}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/session.jsonl"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
CWD="$(pwd)"

if command -v jq >/dev/null 2>&1; then
  jq -Rn \
    --arg timestamp "$TIMESTAMP" \
    --arg event "$EVENT" \
    --arg cwd "$CWD" \
    --arg input "$INPUT" \
    '{timestamp:$timestamp,event:$event,cwd:$cwd,input_preview:($input|.[0:1000])}' >> "$LOG_FILE"
else
  printf '{"timestamp":"%s","event":"%s","cwd":"%s"}\n' "$TIMESTAMP" "$EVENT" "$CWD" >> "$LOG_FILE"
fi

echo "[session-log] recorded $EVENT"
