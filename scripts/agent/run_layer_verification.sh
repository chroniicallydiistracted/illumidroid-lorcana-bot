#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

echo "[verify] git status"
git status --short

echo "[verify] git diff --check"
git diff --check

echo "[verify] agent audit suite"
scripts/agent/run_agent_audit_suite.sh

if [[ -f "lorcana-rs/Cargo.toml" ]]; then
  echo "[verify] Rust workspace detected."
  (
    cd lorcana-rs
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    if command -v cargo-nextest >/dev/null 2>&1; then
      cargo nextest run
    else
      cargo test
    fi
  )
else
  echo "[verify] lorcana-rs/Cargo.toml not present; skipping Rust commands."
fi

if [[ "${RUN_ORACLE_TESTS:-0}" == "1" ]]; then
  echo "[verify] RUN_ORACLE_TESTS=1; attempting TypeScript oracle tests."
  if [[ -d "lorcana-simulator/packages/lorcana/lorcana-engine" ]]; then
    (cd lorcana-simulator/packages/lorcana/lorcana-engine && bun test src)
  elif [[ -d "packages/lorcana/lorcana-engine" ]]; then
    (cd packages/lorcana/lorcana-engine && bun test src)
  else
    echo "[verify] no known oracle engine package path found."
  fi
fi

echo "[verify] complete."
