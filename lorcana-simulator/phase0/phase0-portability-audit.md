# Phase 0 — Engine Portability Audit & Kernel Extraction Plan

**Target:** `TheCardGoat/lorcana-simulator` → `packages/lorcana/lorcana-engine` (vendored `@tcg/core` runtime).
**Question:** Can the headless rules kernel compile to QuickJS→WASM and run in-process inside the search actors, or does it reach DOM/Node-only/non-deterministic APIs that break that plan?
**Method:** static import-graph closure resolution from the kernel entry points, then a severity-tagged API scan over *only the reachable closure* (not the whole repo — that would over-report). Scripts in `./probe/` are reusable CI gates.

## Verdict: **GO** ✅

The kernel is portable to QuickJS→WASM with **no hard blockers on the search path.** Required work is a bounded *host-shim layer*, a *trimmed entry point*, and a small *determinism patch set* (one confirmed effect-level bug + one latent fallback). The codebase was already written host-agnostically (guarded `typeof window`, `globalThis as {process?...}`, a Bun branch), which is why this comes back clean.

---

## 1. Dependency closure (what actually must compile)

Reachable internal files from the kernel surface: **252 non-test `.ts` files** (the full public `index.ts` reaches 256 — the `#core` barrel pulls nearly everything, so transitive files must *compile* even if the search never *calls* them).

Reachable **runtime** externals — the entire list:

| Specifier | Reached by | Portable? |
|---|---|---|
| `@tcg/lorcana-types` | 122 sites | ✅ types — erased at compile |
| `mutative` | 13 sites | ✅ pure JS (Immer-lineage) |
| `@logtape/logtape` | 11 sites | ✅ pure JS; route to a null sink |
| `seedrandom` | 1 site | ✅ pure JS |

**Key wins confirmed by the graph:**
- `@tcg/lorcana-cards` (267k LOC, 5,660 files) is **not** in the closure — cards are injected as a `cardCatalog` at `init`, so the card behemoth never needs compiling into the WASM kernel. (Only mention in engine src is a doc-comment in `serialization.ts`.)
- `nanoid` and `zod` are declared in `package.json` but **not reached** by the engine's own import graph — no Web-Crypto (`nanoid`) dependency on the search path.
- `@tcg/core` is vendored at `lorcana-engine/src/core/` (`#core` → `./src/core/index.ts`) — no cross-package version risk.

---

## 2. Portability scan results (reachable closure only)

| Category | Raw hits | Real disposition |
|---|---|---|
| **BLOCKER** | 35 | **0 true blockers on the search path** (see below) |
| **DETERMINISM** | 68 | 2 must-fix code defects; the rest are envelope/ID/timestamp → inject deterministic clock + seeded IDs |
| **SHIMMABLE** | 12 | trivial host shims |
| **LOGGING** | 18 | bundle logtape + null sink |

### 2.1 "BLOCKER" — dispositioned to zero on the search path

- **`window` ×26 (4 files) — FALSE POSITIVE.** These are the game-domain *"priority window"* concept (`openWindow(ctx,{window:"after-challenge"})`, time-control `"window" | "reserve"`), not the DOM global. Verified: there is **no `window.<member>` access anywhere** in the closure. The only `typeof window` is a *guarded* browser feature-detect in `config/logtape/configure.ts:14` (`const isBrowser = typeof window !== "undefined"`) — returns `false` under QuickJS, never throws.
- **`WebSocket` ×9 (1 file) — REAL but off-path.** Confined to `core/runtime/websocket-transport.ts`, reachable only because the `#core` barrel re-exports it. **Disposition: exclude via a trimmed WASM entry point** (§3.2). Not compiled, not a blocker.

### 2.2 DETERMINISM — the findings that actually matter

Game randomness **is** properly seeded: `match-runtime.random-apis.ts` derives every draw/shuffle from `ctx.random.seed` + a monotonic `draws` counter via `seedrandom(`${seed}:${draws}`)`. Shuffles and draws are therefore reproducible given the seed — the precondition the determinization design depends on. **But the scan found two leaks of raw nondeterminism into game-affecting paths:**

1. **CONFIRMED BUG — `runtime-moves/resolution/action-effects/move-cards-from-under-effect.ts:24`**
   ```ts
   const swapIndex = Math.floor(Math.random() * (index + 1));   // ← raw Math.random in a CARD EFFECT
   ```
   Any card using this effect is non-reproducible across determinizations/replays. **Must redirect to the seeded RandomAPI.**

2. **LATENT FALLBACK — `core/runtime/zone-operations.ts:97`**
   ```ts
   const random = options?.random ?? Math.random;   // ← defaults to Math.random if caller forgets to wire seeded rng
   ```
   Determinism here depends on every `createZoneOperations` call passing `options.random` = the seeded API. **Must eliminate the fallback (throw instead) so a missing wire fails loud, not silently nondeterministic.**

Everything else in DETERMINISM (`Date.now` ×59, `Math.random` in IDs ×~6, `new Date()` ×1, `performance.now` ×1) is **envelope metadata** — command IDs, match/game IDs, auth tokens, client/server timestamps, a perf cache. These do not corrupt state transitions, but they (a) break bit-exact replay/snapshot hashing and (b) would poison the transposition fingerprint if included. Disposition: inject a **deterministic logical clock** and a **seeded ID generator**, and ensure `computeAutomatedActionStateFingerprint` excludes envelope fields (it operates on board state, so verify).

### 2.3 SHIMMABLE / LOGGING

`setTimeout`/`setInterval` ×6, `process.env` ×3 (already guarded via `globalThis`), `Buffer` ×2, `structuredClone` ×1, `performance.now` ×1, `console` ×7, `@logtape/logtape` top-level imports ×11. All covered by the host-shim module (`kernel-host-shims.ts`) + a logtape null sink. None block instantiation.

---

## 3. Build manifest (the actionable output)

### 3.1 Host-shim layer
Provide `kernel-host-shims.ts` (included) as the **first import** of the WASM entry. It injects: a deterministic clock (`now()` from an injected logical tick), a seeded ID source (replaces `Date.now()-Math.random()` ID strings), microtask-based `setTimeout(fn,0)`, a `Buffer`/`TextEncoder` polyfill, a `structuredClone` polyfill, a no-op `console`, and a logtape null sink. The host (search worker) advances the logical clock deterministically per command.

### 3.2 Trimmed WASM entry point
Do **not** compile the `#core` barrel. Author `kernel-entry.ts` that imports only:
`createLorcanaServerGame` / `executeCommand` (reducer) · `getAvailableMoves` + `getMoveOptions` (mask) · `filterMatchView` (info set) · projection (`EngineProjectionSnapshot`) · `RuntimeSnapshot` save/restore · seeded RandomAPI.
This statically excludes `websocket-transport.ts`, `multiplayer-engine.ts`, persistence adapters, `auth.ts`, and the client/server network engines — removing the only real `WebSocket` hit and most envelope-ID code.

### 3.3 Determinism patch set (2 items, both bounded)
- Patch `move-cards-from-under-effect.ts:24` to draw from `ctx.framework.random` (the seeded API) instead of `Math.random`.
- Patch `zone-operations.ts:97`: replace the `?? Math.random` fallback with a hard assert/throw so determinism wiring can't silently regress.
- Add a CI guard: grep-gate (`scan_portability.py`) that fails the build if any **new** `Math.random`/`Date.now`/`new Date()` appears under `runtime-moves/`, `operations/`, `zones/`, `triggered-abilities/`.

### 3.4 Toolchain
QuickJS→WASM via **Javy** (faithful, low-effort) for v1. Profile `executeCommand` + `getAvailableMoves`; if the combat/availability hot paths dominate, hand-port just those to Rust/AssemblyScript later. `wasmtime`/`wasmer` host in the Python (or Rust) search worker.

---

## 4. Conformance harness (Phase 0 exit gate)

Differential test the WASM kernel against the source TS engine; **a divergence silently teaches the agent wrong lines.**

1. **Seeded random-play generator** drives identical command streams into both engines.
2. After every command assert equality of: (a) the canonicalized `MatchState` (post-shim, envelope fields excluded), and (b) the full `getAvailableMoves`/`getMoveOptions` output.
3. **Reuse the repo's own `e2e/` suites** as a correctness oracle — it ships per-keyword tests: `shift`, `evasive`, `ward`, `bodyguard`, `support`, `challenger-resist`, `reckless-rush`, `vanish`, `boost-mechanics`, `look-at-the-top`, `name-a-card`, plus `regressions/`. Run them through the WASM kernel.
4. Pin the engine commit + `@tcg/lorcana-cards` catalog version together; re-run on every set release.

**Phase 0 is "done" when:** the trimmed kernel compiles to WASM, the host-shim layer makes it deterministic, and the differential harness + the imported `e2e` suites pass at 100% over ≥10⁶ random legal commands.

---

## 5. Files in this deliverable
- `probe/resolve_closure.py` — import-graph closure resolver (run `full` or `kernel`).
- `probe/scan_portability.py` — severity-tagged portability scanner (also the CI determinism gate).
- `probe/closure_full.json` — the resolved 256-file reachable set.
- `kernel-host-shims.ts` — the host-shim module from §3.1.
