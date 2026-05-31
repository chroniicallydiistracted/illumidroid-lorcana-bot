#!/usr/bin/env bash
# All-in-one launcher: scaled net on GPU + batched inference + belief self-play
# + live monitor. Ctrl-C saves and exits. Pass through any run.py flags.
#
#   ./train.sh                       # defaults (~13M net, runs forever)
#   ./train.sh --rounds 50           # bounded run
#   ./train.sh --sims 24 --batch 48  # more search / bigger leaf batches
#   ./train.sh --help
set -euo pipefail
cd "$(dirname "$0")"
VENV="./lorcana-bot-venv/bin/python"
[ -x "$VENV" ] || { echo "venv missing at $VENV"; exit 1; }
exec "$VENV" lorcana-bot/training/run.py "$@"
