#!/usr/bin/env bash
# All-in-one launcher. Two trainers:
#   ./train.sh [flags]          -> training/run.py     (parallel self-play, the default)
#   ./train.sh league [flags]   -> training/league_train.py (Phase-3 PFSP/league + Elo + exploitability)
#
# Examples:
#   ./train.sh                              # ~13M net, 6 actors, belief self-play, runs forever
#   ./train.sh --actors 8 --rounds 50 --eval-every 5
#   ./train.sh league --iterations 20 --games 6 --exploit
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
