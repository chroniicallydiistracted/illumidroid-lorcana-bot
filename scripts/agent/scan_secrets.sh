#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_SECRETS_SCAN:-}" == "true" ]]; then
  exit 0
fi

MODE="${SCAN_MODE:-warn}"
SCOPE="${SCAN_SCOPE:-diff}"
BASE_REF="${AGENT_BASE_REF:-HEAD}"
LOG_DIR="${SECRETS_LOG_DIR:-logs/agent/secrets}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/secrets.log"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

PATTERNS=(
  'AWS_ACCESS_KEY|critical|AKIA[0-9A-Z]{16}'
  'AWS_SECRET_KEY|critical|aws_secret_access_key[[:space:]]*[:=][[:space:]]*["'"'']?[A-Za-z0-9/+=]{40}'
  'GITHUB_PAT|critical|ghp_[0-9A-Za-z]{36}'
  'GITHUB_FINE_GRAINED_PAT|critical|github_pat_[0-9A-Za-z_]{20,}'
  'SLACK_TOKEN|high|xox[baprs]-[0-9A-Za-z-]+'
  'PRIVATE_KEY|critical|-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'
  'JWT|medium|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'
  'GENERIC_API_KEY|medium|(api[_-]?key|secret|token)[[:space:]]*[:=][[:space:]]*["'"'']?[A-Za-z0-9_\-]{24,}'
)

changed_files() {
  if [[ "$SCOPE" == "staged" ]]; then
    git diff --cached --name-only
  else
    git diff --name-only "$BASE_REF" 2>/dev/null || git diff --cached --name-only || true
  fi
}

FILES="$(changed_files)"
if [[ -z "$FILES" ]]; then
  echo "[secrets] no changed files detected."
  exit 0
fi

FAIL=0
while IFS= read -r file; do
  [[ -f "$file" ]] || continue
  case "$file" in
    *.png|*.jpg|*.jpeg|*.gif|*.zip|*.pdf|*.lock) continue ;;
  esac
  for entry in "${PATTERNS[@]}"; do
    IFS='|' read -r name severity regex <<< "$entry"
    if grep -InE "$regex" "$file" >/tmp/secret_hits.$$ 2>/dev/null; then
      echo "[$TIMESTAMP] $severity $name in $file" >> "$LOG_FILE"
      cat /tmp/secret_hits.$$ >> "$LOG_FILE"
      echo "[secrets] $severity finding $name in $file"
      if [[ "$MODE" == "block" || "$severity" == "critical" ]]; then
        FAIL=1
      fi
    fi
  done
done <<< "$FILES"
rm -f /tmp/secret_hits.$$ || true

if [[ "$FAIL" -ne 0 ]]; then
  echo "[secrets] failed."
  exit 1
fi

echo "[secrets] passed."
