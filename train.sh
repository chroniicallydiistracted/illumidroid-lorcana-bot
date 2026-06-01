#!/usr/bin/env bash
# All-in-one launcher. Two trainers:
#   ./train.sh [flags]          -> training/run.py     (use --no-belief during Tier-A remediation)
#   ./train.sh league [flags]   -> training/league_train.py (use --no-belief during Tier-A remediation)
#
# Examples:
#   ./train.sh --no-belief                  # belief-guided label writing is fail-closed for now
#   ./train.sh --no-belief --actors 8 --rounds 50 --eval-every 5
#   ./train.sh league --no-belief --iterations 20 --games 6 --exploit
#   ./train.sh --help        |  ./train.sh league --help
set -euo pipefail
cd "$(dirname "$0")"
VENV="./lorcana-bot-venv/bin/python"
[ -x "$VENV" ] || { echo "venv missing at $VENV"; exit 1; }
if [ "${1:-}" = "league" ]; then
  shift
  exec "$VENV" lorcana-bot/training/league_train.py "$@"
fi
exec "$VENV" lorcana-bot/training/run.py "$@"
