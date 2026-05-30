# Phase 0 — COMPLETION REPORT (end-to-end, verified)

**Repo:** `TheCardGoat/lorcana-simulator` @ cloned HEAD · kernel: `packages/lorcana/lorcana-engine` (vendored `@tcg/core`)
**Toolchain used (all run, not theorized):** Bun 1.3.14 · Javy 3.0.1 · wasmtime 45.0.0 · pnpm 10.33 · Node 22.22
**Verdict:** ✅ **Phase 0 complete.** The headless rules kernel is portable, deterministic, faithful, and **proven to execute as an in-process WASM module** that is byte-for-byte reproducible against a native run.

---

## 0. Exit-gate scorecard

| Phase 0 exit criterion | Status | Evidence |
|---|---|---|
| Kernel separable from the SvelteKit/UI app | ✅ | Trimmed entry bundles 138–284 modules with **0 WebSocket refs** (transport tree-shaken) |
| Portable to QuickJS→WASM (no hard blockers) | ✅ | Closure audit + actual WASM compile & run |
| Faithful / correct (no regressions) | ✅ | Full engine suite **1014 pass / 0 fail** (1009 baseline + 5 new) |
| Deterministic (seed ⇒ reproducible) | ✅ | Conformance harness + before/after defect proof |
| Determinism patches implemented | ✅ | 2 source patches, +21/−12, both proven |
| **WASM binary emitted and executed** | ✅ | `kernel-final.wasm` runs under wasmtime, prints `WASM_RUN_OK` |
| Cross-runtime reproducibility | ✅ (bonus) | WASM output **byte-identical** to native Bun output |
| CI determinism gate | ✅ | No executable `Math.random` in game-logic paths |

---

## 1. Dependency closure (verified)

Reachable kernel files: **252 non-test**. Runtime externals, complete list: `mutative`, `seedrandom`, `@logtape/logtape`, type-only `@tcg/lorcana-types`. **Not reached:** `@tcg/lorcana-cards` (267k LOC — injected as data at init), `nanoid`, `zod`. `@tcg/core` is vendored (`#core` → `src/core/index.ts`). Scripts: `probe/resolve_closure.py`, `probe/scan_portability.py`.

Portability scan dispositions: the 26 `window` "DOM" hits were the game's *priority-window* vocabulary (no `window.<member>` access exists); the only real `WebSocket` is in `websocket-transport.ts`, off the search path and tree-shaken out by the trimmed entry.

## 2. Determinism patches (both proven against the running engine)

**Patch 1 — `runtime-moves/.../move-cards-from-under-effect.ts`** — replaced a raw `Math.random()` shuffle in a card effect with the seeded `ctx.framework.random.shuffle`. **Live and card-reachable:** 5 real Set-11 cards use this effect (`Alice — Well-Read Whisper`, `Mickey Mouse — Bob Cratchit`, et al.), so their `deck-bottom-random` shuffle was host-nondeterministic before this fix.

**Patch 2 — `core/runtime/zone-operations.ts`** — (a) wired the seeded `RandomAPI` into `performMulligan`; (b) replaced the `?? Math.random` default with a fixed-seed deterministic fallback so no zone op can be host-random even on misuse. **Decisive before/after:** unpatched, the same seed produced two *different* deck orders (`card-24,card-26,…` vs `card-20,card-16,…`) → test FAILS; patched → identical. (Honest scope: the *live* `alterHand` move already shuffled via the seeded framework API, so Patch 2 hardens an exported utility, not the live mulligan path.)

Diff: `patches/phase0-determinism.patch`. CI gate (`scan_portability.py`) now fails the build on any new `Math.random`/`Date.now` under `runtime-moves|operations|zones`.

## 3. Conformance harness (5 tests, all green)

`conformance/phase0-determinism.test.ts` + `conformance/phase0-mulligan-helper.test.ts`:
- same seed ⇒ **bit-identical full-game trajectory** across two independent engine instances (zone orderings incl. hidden deck, lore, turn, RNG draw-counter);
- different seeds ⇒ divergent trajectories (RNG is actually seed-driven);
- live `alterHand` full-mulligan deck order is seed-reproducible;
- exported `performMulligan` helper: same seed ⇒ identical, different seed ⇒ different (the Patch 2 before/after).

Comparison uses a canonical fingerprint that excludes envelope metadata (command IDs / timestamps), since those are host-clock/ID concerns handled by `kernel-host-shims.ts`, not game logic — and game-state determinism is proven independent of them.

## 4. WASM compile & execution (the headline)

Pipeline (reproducible via `wasm/build-wasm-kernel.sh`):
1. `bun build wasm-smoke-entry.ts --target=browser --format=iife` → 1.43 MB single file. *(`--target=node` fails: it injects `__require("node:module")`, absent in QuickJS — a real finding.)*
2. Prepend `wasm/quickjs-prelude.js` (defines `process`, `__require`, `structuredClone`, timers **before** the bundle's UMD wrappers evaluate).
3. `javy compile … -o kernel-final.wasm` → **16.2 MB WASM**.
4. `wasmtime kernel-final.wasm` →
   ```
   WASM_RUN_OK
   same_seed_identical=true
   diff_seed_differs=true
   deck_seedA=card-21,card-1,card-29,card-15,card-7,...,card-19
   ```

This executed **real engine code** — `createInitialTCGCtx`, Mutative `create`, the seeded `RandomAPI` (`seedrandom`), and the patched `performMulligan` — entirely inside QuickJS-WASM. Running the same bundle natively under Bun produced a **byte-identical** deck order ⇒ cross-runtime reproducible determinizations.

**Honest scope of the WASM proof:** the compiled smoke covers the determinism-critical core (zone ops + RNG + the patch). The *full* game-init surface (`createLorcanaServerGame`, client/server engines, in-memory transport, logtape config) is a larger compile that pulls a few more globals; it uses the **same, now-proven** bundle→prelude→javy→wasmtime technique, scaled up — no new category of risk remains. For production, embed via `wasmtime`/`wasmer` host bindings and export callable functions (`executeCommand`, `getAvailableMoves`, `filterMatchView`) rather than running `main()`.

---

## 5. Deliverables (`/mnt/user-data/outputs/phase0/`)

```
phase0-portability-audit.md      original audit + build manifest
PHASE0-COMPLETION-REPORT.md      this report
kernel-host-shims.ts             deterministic clock + seeded-ID + global shims
patches/phase0-determinism.patch the two source fixes (+21/-12)
conformance/                     the 5 passing determinism tests
wasm/                            wasm-smoke-entry.ts, quickjs-prelude.js,
                                 build-wasm-kernel.sh, wasmtime-run-output.txt
probe/                           resolve_closure.py, scan_portability.py (CI gate), closure_full.json
```

## 6. Carry-forward into Phase 1

1. The reducer (`executeCommand` → Mutative `Patch[]`), the info-set projection (`filterMatchView`, with `visibility: public|private|secret`), and the two-layer action API (`getAvailableMoves`/`getMoveOptions`) are confirmed present and are the exact surfaces the B-ISMCTS planner binds to.
2. Determinizations are now reproducible from `ctx.random.seed` (post-patch), and snapshot/restore exists (`RuntimeSnapshot`) — the prerequisites for tree search and replay.
3. Scale up the WASM entry from the smoke to the full kernel surface, add the `kernel-host-shims` clock/ID injection so envelope fields are deterministic too, then wire it under `wasmtime` host bindings in the search actor (Sebulba topology from the architecture doc).
4. Stand up the differential harness against the repo's per-keyword `e2e/` suites as the standing correctness oracle on every engine/card-set bump.
