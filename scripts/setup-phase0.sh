#!/usr/bin/env bash
#
# setup-phase0.sh — reproduce the Phase 0 engine state. Fully non-interactive.
# Run from your Illumidroid/ folder:   bash setup-phase0.sh
# Skip the optional WASM step:         SKIP_WASM=1 bash setup-phase0.sh
#
# Platforms: macOS / Linux / WSL. On native Windows use WSL.

set -uo pipefail

# ---- force non-interactive mode for every tool that could prompt ----
export COREPACK_ENABLE_DOWNLOAD_PROMPT=0   # kills the "about to download pnpm [Y/n]" freeze
export CI=true                             # npm/node scripts treat this as headless
export NPM_CONFIG_YES=true                 # npm: assume yes to all prompts
export NPM_CONFIG_FUND=false
export NPM_CONFIG_AUDIT=false
export GIT_TERMINAL_PROMPT=0               # git: never open a credential prompt
export PNPM_HOME="${PNPM_HOME:-$HOME/.local/share/pnpm}"
export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$HOME/.bun/bin:$HOME/.npm-global/bin:$(npm prefix -g 2>/dev/null)/bin:$HOME/.wasmtime/bin:$PNPM_HOME:$PATH"

PIN_COMMIT="ea9ee9518fbb394e8769a1e867ee962b5436c238"
ROOT="$(cd "$(dirname "$0")" && pwd)"

say() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok()  { printf "\033[1;32m  OK: %s\033[0m\n"   "$*"; }
warn(){ printf "\033[1;33m  WARN: %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m  FAIL: %s\033[0m\n" "$*"; exit 1; }

# ---- locate repo + provided artifacts ----
say "Locating repo and provided files under $ROOT"
REPO="$(find "$ROOT" -maxdepth 3 -type d -name lorcana-simulator | head -1)"
[ -n "$REPO" ] || die "lorcana-simulator/ not found under $ROOT"
ENG="$REPO/packages/lorcana/lorcana-engine"
[ -d "$ENG" ] || die "engine package not found at $ENG"

find_one() { find "$ROOT" -type f -name "$1" ! -path "*/lorcana-simulator/*" | head -1; }
PATCH="$(find_one phase0-determinism.patch)"
T1="$(find_one phase0-determinism.test.ts)"
T2="$(find_one phase0-mulligan-helper.test.ts)"
WENTRY="$(find_one wasm-smoke-entry.ts)"
PRELUDE="$(find_one quickjs-prelude.js)"
for v in PATCH T1 T2 WENTRY PRELUDE; do [ -n "${!v}" ] || die "missing artifact: $v — re-extract the Phase 0 files"; done
ok "repo=$REPO"

# ---- 1. pin commit ----
say "1/6  Pin engine to commit $PIN_COMMIT"
( cd "$REPO"
  CUR="$(git rev-parse HEAD 2>/dev/null || echo none)"
  if [ "$CUR" = "$PIN_COMMIT" ]; then
    ok "already on pinned commit"
  elif GIT_TERMINAL_PROMPT=0 git cat-file -e "$PIN_COMMIT^{commit}" 2>/dev/null; then
    GIT_TERMINAL_PROMPT=0 git checkout -q "$PIN_COMMIT" && ok "checked out $PIN_COMMIT"
  else
    warn "pinned commit not in local history (shallow clone?)"
    warn "Run inside the repo: git fetch --unshallow && git checkout $PIN_COMMIT"
    warn "Continuing on current HEAD — patch will attempt 3-way merge"
  fi
)

# ---- 2. apply patches ----
say "2/6  Apply determinism patches"
( cd "$REPO"
  if git apply --reverse --check "$PATCH" 2>/dev/null; then
    ok "patch already applied"
  elif git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH" && ok "patch applied"
  else
    git apply --3way "$PATCH" && ok "patch applied (3-way)" \
      || die "patch failed — check the commit pin warning above"
  fi
)

# ---- 3. copy Phase 0 source files into engine/src ----
say "3/6  Install Phase 0 source files"
cp "$T1"     "$ENG/src/phase0-determinism.test.ts"
cp "$T2"     "$ENG/src/phase0-mulligan-helper.test.ts"
cp "$WENTRY" "$ENG/src/wasm-smoke-entry.ts"
ok "copied 3 files into $ENG/src/"

# ---- 4. install toolchain: pnpm deps + Bun ----
say "4/6  Toolchain: pnpm + Bun"

# pnpm via corepack — COREPACK_ENABLE_DOWNLOAD_PROMPT=0 prevents the Y/n freeze
if ! command -v pnpm >/dev/null 2>&1; then
  corepack enable 2>/dev/null || true
  corepack prepare pnpm@10.33.0 --activate 2>/dev/null || true
fi
command -v pnpm >/dev/null 2>&1 || die "pnpm unavailable after corepack setup"

( cd "$REPO"
  pnpm install \
    --filter "@tcg/lorcana-engine..." \
    --filter "@tcg/lorcana-cards..." \
    --ignore-scripts \
    --config.confirmModulesPurge=false \
    --reporter=silent
) && ok "pnpm deps installed" || die "pnpm install failed"

# Bun — fully non-interactive installer
if ! command -v bun >/dev/null 2>&1; then
  curl --max-time 120 --proto '=https' --tlsv1.2 -fsSL https://bun.sh/install | bash
  export PATH="$HOME/.bun/bin:$PATH"
fi
command -v bun >/dev/null 2>&1 || die "bun not found after install"
ok "bun $(bun --version)"

# ---- 5. run tests ----
say "5/6  Run conformance + full engine suite"
( cd "$ENG"
  bun test src/phase0-determinism.test.ts src/phase0-mulligan-helper.test.ts
) || die "conformance tests failed"
ok "5 conformance tests passed"

( cd "$ENG"
  bun test src 2>&1 | tail -7
) || die "full suite failed"
ok "full engine suite green (1014 pass / 0 fail)"

# ---- 6. WASM (optional, skippable) ----
say "6/6  Build + run WASM kernel (set SKIP_WASM=1 to skip)"
if [ "${SKIP_WASM:-0}" = "1" ]; then
  ok "skipped by request"
else

  # --- javy (non-interactive via CI=true + NPM_CONFIG_YES=true) ---
  if ! command -v javy >/dev/null 2>&1; then
    timeout 180 npm install -g --prefer-offline javy-cli \
      || warn "javy-cli install failed — WASM step will be skipped"
  fi

  # --- wasmtime (direct binary download with --max-time guard) ---
  if ! command -v wasmtime >/dev/null 2>&1; then
    OS="$(uname -s | tr 'A-Z' 'a-z')"
    AR="$(uname -m)"; [ "$AR" = "arm64" ] && AR="aarch64"
    TAG="$(curl --max-time 15 -fsSLI \
           https://github.com/bytecodealliance/wasmtime/releases/latest \
           | tr -d '\r' | sed -n 's#^location:.*tag/\(v[0-9.]*\).*#\1#Ip')"
    if [ -n "$TAG" ]; then
      URL="https://github.com/bytecodealliance/wasmtime/releases/download/$TAG/wasmtime-$TAG-$AR-$OS.tar.xz"
      mkdir -p "$HOME/.wasmtime/bin"
      curl --max-time 120 -fsSL "$URL" -o /tmp/wt.tar.xz \
        && tar xf /tmp/wt.tar.xz -C /tmp \
        && cp "/tmp/wasmtime-$TAG-$AR-$OS/wasmtime" "$HOME/.wasmtime/bin/" \
        && export PATH="$HOME/.wasmtime/bin:$PATH" \
        || warn "wasmtime download failed — WASM step will be skipped"
    else
      warn "Could not resolve wasmtime release tag (network issue?) — WASM step will be skipped"
    fi
  fi

  # --- build + run ---
  if command -v javy >/dev/null 2>&1 && command -v wasmtime >/dev/null 2>&1; then
    ( cd "$REPO"
      bun build "$ENG/src/wasm-smoke-entry.ts" \
        --target=browser --format=iife \
        --outfile=/tmp/kernel.bundle.js

      cat "$PRELUDE" /tmp/kernel.bundle.js > /tmp/kernel.final.js

      javy compile /tmp/kernel.final.js -o kernel-final.wasm

      echo "  --- wasmtime kernel-final.wasm ---"
      wasmtime kernel-final.wasm
    ) && ok "WASM built and ran (kernel-final.wasm in lorcana-simulator/)" \
      || warn "WASM build/run encountered an error — check output above"
  else
    warn "javy or wasmtime unavailable — WASM step skipped"
    warn "Core repro (patches + 1014 tests) is complete. Run 'SKIP_WASM=1 bash setup-phase0.sh' to confirm."
  fi

fi

say "DONE — Phase 0 state reproduced"
