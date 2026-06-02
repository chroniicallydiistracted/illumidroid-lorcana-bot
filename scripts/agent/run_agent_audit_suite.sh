#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

echo "[suite] git diff --check"
git diff --check

echo "[suite] forbidden paths"
scripts/agent/check_forbidden_paths.sh

echo "[suite] dependency step"
scripts/agent/check_dependency_step.py

echo "[suite] symbol registry"
scripts/agent/check_symbol_registry_required.py

echo "[suite] determinism bans"
scripts/agent/check_determinism_bans.sh

echo "[suite] secrets scan"
scripts/agent/scan_secrets.sh

echo "[suite] passed."
