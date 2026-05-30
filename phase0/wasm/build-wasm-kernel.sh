#!/usr/bin/env bash
# Reproducible Phase 0 WASM build pipeline (verified working: Bun 1.3, Javy 3.0.1, wasmtime 45).
# Compiles real engine determinism code to QuickJS->WASM and runs it.
set -euo pipefail
ENGINE=packages/lorcana/lorcana-engine
ENTRY=$ENGINE/src/wasm-smoke-entry.ts          # or your trimmed kernel-entry.ts

# 1) Bundle to a single self-contained file. --target=browser avoids the node:module
#    require-preamble that QuickJS lacks; --format=iife yields no import/export.
bun build "$ENTRY" --target=browser --format=iife --outfile=/tmp/kernel.bundle.js

# 2) Prepend the QuickJS prelude (defines process/__require/structuredClone/timers
#    BEFORE the bundle's UMD wrappers run). See quickjs-prelude.js.
cat quickjs-prelude.js /tmp/kernel.bundle.js > /tmp/kernel.final.js

# 3) Compile to WASM (Javy v3 uses `compile`; v4+ uses `build`).
javy compile /tmp/kernel.final.js -o kernel.wasm

# 4) Run in any WASM runtime (here wasmtime). In production, embed via wasmtime/wasmer
#    bindings inside each search actor and call exported functions instead of main().
wasmtime kernel.wasm
