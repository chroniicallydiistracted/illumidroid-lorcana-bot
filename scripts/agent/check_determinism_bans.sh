#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

TARGETS=()
if [[ -d "lorcana-rs/crates/lorcana-core" ]]; then
  TARGETS+=("lorcana-rs/crates/lorcana-core")
elif [[ -d "lorcana-rs" ]]; then
  TARGETS+=("lorcana-rs")
else
  echo "[determinism] lorcana-rs not present yet; skipping Rust scan."
  exit 0
fi

FAIL=0

hard_patterns=(
  'rand::StdRng'
  'thread_rng'
  'Math\.random'
  'tokio::'
  'use tokio'
  'static mut'
  'lazy_static!'
)

order_patterns=(
  'HashMap<'
  'HashSet<'
  'std::collections::HashMap'
  'std::collections::HashSet'
)

for target in "${TARGETS[@]}"; do
  echo "[determinism] scanning $target"

  for pattern in "${hard_patterns[@]}"; do
    if grep -RInE "$pattern" "$target" \
      --exclude-dir=target \
      --exclude-dir=.git \
      --exclude='*.lock' \
      | grep -v 'agent-allow-determinism-exception' ; then
      echo "[determinism] BLOCKER: banned deterministic-core pattern: $pattern"
      FAIL=1
    fi
  done

  for pattern in "${order_patterns[@]}"; do
    if grep -RInE "$pattern" "$target" \
      --exclude-dir=target \
      --exclude-dir=.git \
      --exclude='*.lock' \
      | grep -v 'agent-allow-unordered-review' ; then
      echo "[determinism] REVIEW REQUIRED: unordered collection may define observable behavior: $pattern"
      FAIL=1
    fi
  done
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[determinism] failed."
  exit 1
fi

echo "[determinism] passed."
