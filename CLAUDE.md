# Lorcana AI Bot — Project Memory for Claude Code

## What this project is

A champion-level Machine Learning AI player for the Disney Lorcana TCG.
The engine is the open-source **Lorcanito** TypeScript simulator
(`lorcana-simulator/`). The ML stack will be Python (PyTorch/JAX).

Full architecture design: `lorcana-simulator/lorcana-bot-architecture.md`

---

## Current state: Phase 1 COMPLETE (Phase 0 also complete)

Phase 1 (plain neural-ISMCTS, end-to-end self-play loop) is done. The Python ML
stack lives in `lorcana-bot/` and drives the real Lorcanito kernel:
- **1A bridge** — Bun subprocess hosts the kernel; `LorcanaEnv` gym wrapper;
  engine-authoritative legal-action mask; snapshot/restore for tree search.
- **1B network** — `LorcanaNet`: set-transformer trunk + factored pointer policy
  head + distributional (C51) value head. (Belief head deferred to Phase 2.)
- **1C search** — `BISMCTS`: PUCT, progressive widening, depth-limited neural
  leaves, zero-sum perspective backup. Determinization N=1 (realized world);
  belief-weighted multi-determinization seam left for Phase 2.
- **1D training** — behaviour cloning from the scripted oracle (BC loss
  2.55→0.49) + AlphaZero-style self-play actor/learner loop.

14/14 tests pass (`pytest lorcana-bot/tests`). Full write-up:
`lorcana-simulator/phase1/PHASE1-COMPLETION-REPORT.md`.

Phase 0 exit criteria — all met:

| Criterion | Evidence |
|---|---|
| Kernel portable to QuickJS→WASM | Closure audit — 0 hard blockers on search path |
| Faithful / correct | `bun test src` — **1014 pass / 0 fail** |
| Deterministic | Conformance harness + before/after defect proof |
| Patches applied | 2 source fixes, +21/−12, both proven |
| WASM binary runs | `kernel-final.wasm` runs under wasmtime, prints `WASM_RUN_OK` |
| Cross-runtime reproducible | WASM output byte-identical to native Bun output |
| CI determinism gate | No executable `Math.random` in game-logic paths |

Full Phase 0 completion report:
`lorcana-simulator/phase0/PHASE0-COMPLETION-REPORT.md`

---

## What was changed from upstream

Only two files were patched from the original repo (commit `ea9ee95`):

### Patch 1 — `move-cards-from-under-effect.ts`
**File:** `packages/lorcana/lorcana-engine/src/runtime-moves/resolution/action-effects/move-cards-from-under-effect.ts`
**Fix:** Replaced `Math.random()` shuffle with `ctx.framework.random.shuffle()`
(the seeded RandomAPI). 5 real Set-11 cards use this effect; their
`deck-bottom-random` shuffle was host-nondeterministic before this fix.

### Patch 2 — `zone-operations.ts`
**File:** `packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts`
**Fix:** (a) Wired seeded `RandomAPI` into `performMulligan` helper.
(b) Replaced `?? Math.random` fallback with a fixed-seed deterministic
stream so no zone op is ever host-nondeterministic.

Patch diff: `lorcana-simulator/phase0/patches/phase0-determinism.patch`

---

## Phase 0 artifacts (all in `lorcana-simulator/`)

```
phase0/
  PHASE0-COMPLETION-REPORT.md     full write-up
  phase0-portability-audit.md     closure audit + build manifest
  patches/
    phase0-determinism.patch      the two source fixes
  conformance/
    phase0-determinism.test.ts    → copied to engine/src/
    phase0-mulligan-helper.test.ts → copied to engine/src/
  wasm/
    wasm-smoke-entry.ts           → copied to engine/src/
    quickjs-prelude.js            QuickJS global shims
    build-wasm-kernel.sh          reproducible WASM build pipeline
    wasmtime-run-output.txt       proof of WASM execution
  probe/
    resolve_closure.py            import-graph resolver (CI gate)
    scan_portability.py           portability scanner (CI gate)
    closure_full.json             resolved 256-file reachable set
kernel-host-shims.ts              deterministic clock + seeded-ID shims
kernel-final.wasm                 the compiled WASM artifact
```

Additional Phase 0 source files live directly in the engine `src/`:
```
packages/lorcana/lorcana-engine/src/
  phase0-determinism.test.ts      5 conformance tests (all green)
  phase0-mulligan-helper.test.ts  performMulligan determinism test
  wasm-smoke-entry.ts             WASM smoke (real engine code)
```

---

## Engine contract (what the search drives)

**Runtime:** `@tcg/core` MatchRuntime (boardgame.io-style pure reducer)
**Package location:** `packages/lorcana/lorcana-engine/` (vendored core at `src/core/`)

The four operations the search needs — confirmed present and working:

```typescript
// 1. Initialize  (real signature — decks + seed live inside `init`, NOT positional)
createLorcanaServerGame(playersInfo: { player: Player }[], init: LorcanaEngineInit) → LorcanaServer

// 2. Transition (the reducer) — returns Mutative patches, not a full clone
executeCommand(state, command) → { state', patches: Patch[] }

// 3. Legal-action mask — two-layer API
getAvailableMoves(state, playerID)           // Layer 1: categories + selectable cards
getMoveOptions(state, playerID, moveId, cardId) // Layer 2: targets/costs for a selection

// 4. Information set (fog of war — engine-enforced)
filterMatchView(state, { role, playerID }) → FilteredMatchView

// 5. Snapshot/restore (for MCTS node cloning)
saveSnapshot() / getSnapshotAtStateID(stateID) → RuntimeSnapshot
```

**State shape:** `MatchState = { G: LorcanaG, ctx: TCGCtx }`
- `ctx.random.seed` + `ctx.random.draws` → seeded RNG (determinism guaranteed post-patch)
- `ctx.zones.private.zoneCards` → full zone contents incl. hidden deck order
- Zone visibility: `"public" | "private" | "secret"` (+ `faceDown`/`ownerScoped`)

**Key visibility mapping for the belief net:**
- `public` → board, discard, lore, ink count
- `private` (ownerScoped) → opponent hand (the hidden variable to model)
- `secret` + ordered → deck ordering (top-deck distribution)
- `faceDown` + ownerScoped → inkwell identities (secondary belief target)

**Action grammar — factored (mirrors the 2-layer engine API):**
```
π_cat  : over 12 AvailableMoveId categories
         (playCard, singCard, shiftCard, putCardIntoInkwell, quest, challenge,
          moveCharacterToLocation, activateAbility, passTurn, questWithAll,
          chooseWhoGoesFirst, concede)
π_src  : over Layer-1 selectableCardIds
π_tgt  : over Layer-2 getMoveOptions (targets, costs, singers, ability index)
```

**Automation layer (use for bootstrap + league anchors):**
- `bestDeckAwareOracleLoreRaceAutomatedActionStrategy` — strongest scripted AI
- `BEST_AI_DECK_DOSSIERS` — initial deck distribution for training
- `computeAutomatedActionStateFingerprint` — transposition-table hash
- `createAutomatedActionBoardSnapshot` — board feature snapshot
- `engine.asServer().takeAutomatedActionForCurrentActor({ strategy })` — drives a full game

---

## Architecture decisions (do not re-litigate these)

**Core algorithm: B-ISMCTS** (Belief-guided neural Information-Set MCTS)
- NOT pure AlphaZero (can't handle hidden info without cheating)
- NOT pure CFR (action space too large and sequential within a turn)
- ISMCTS is the spine; AlphaZero's dual-headed net provides priors + leaf values

**PUCT selection:**
```
a* = argmax_a [ Q(I,a) + c_puct · P(a|I) · √(Σ_b N(I,b)) / (1 + N(I,a)) ]
```

**Network topology — three heads on one trunk:**
```
I → Trunk (set transformer) ─┬─→ Policy head  p(a|I)   factored: π_cat × π_src × π_tgt
                              ├─→ Value head   v(I)     categorical/two-hot, trained on z
                              └─→ Belief head  b(world|I)  P(opponent hand + inkwell)
```

**State encoder:** permutation-invariant transformer over card tokens.
Feed `EngineBoardProjection`/`EngineProjectionSnapshot` (from `engine/projection.ts`),
NOT raw `G`. Never let the encoder see `filterMatchView`'s hidden zones directly.

**Determinization:** importance-weighted (not uniform).
Sample `w_i ~ b(world|I)`, weight by `ρ_i = b(w_i|I)/q(w_i)`.

**Value head training:** on game outcome `z` (win/loss), NOT lore count.
Distributional (C51/two-hot) to handle top-deck variance.
Bootstrap with MCTS root value as auxiliary target.

**Bridge topology (Sebulba/Podracer):**
- K actor processes, each owns: engine instance (WASM in-process) + MCTS search loop
- 1 central GPU inference server with dynamic batching
- Actors batch leaf states → inference server; never cross the boundary per node

**Why NOT per-node IPC:** at 10^5–10^7 transitions/decision, any per-step
JS↔Python crossing caps you at hobby strength. The kernel must run in-process.

---

## Development commands

```bash
# From lorcana-simulator/packages/lorcana/lorcana-engine/
bun test src                                    # full suite (expect 1014 pass / 0 fail)
bun test src/phase0-determinism.test.ts \
         src/phase0-mulligan-helper.test.ts     # Phase 0 conformance only

# From lorcana-simulator/ (repo root)
pnpm install \
  --filter "@tcg/lorcana-engine..." \
  --filter "@tcg/lorcana-cards..." \
  --ignore-scripts \
  --config.confirmModulesPurge=false

# WASM build (from lorcana-simulator/ with quickjs-prelude.js present)
bun build packages/lorcana/lorcana-engine/src/wasm-smoke-entry.ts \
  --target=browser --format=iife --outfile=/tmp/kernel.bundle.js
cat phase0/wasm/quickjs-prelude.js /tmp/kernel.bundle.js > /tmp/kernel.final.js
javy compile /tmp/kernel.final.js -o kernel-final.wasm
wasmtime kernel-final.wasm
# Expect: WASM_RUN_OK / same_seed_identical=true / diff_seed_differs=true
```

### Phase 1 ML stack (from repo root; venv at `lorcana-bot-venv/`)

```bash
# tests (network is fast; bridge/search spawn the Bun engine)
lorcana-bot-venv/bin/python -m pytest lorcana-bot/tests        # 14 pass
# behaviour-clone the scripted oracle, then self-play from that prior
cd lorcana-bot
../lorcana-bot-venv/bin/python -m training.bootstrap --games 20 --epochs 5 --out checkpoints/bc.pt
../lorcana-bot-venv/bin/python -m training.selfplay  --init checkpoints/bc.pt --iterations 3 --games 4 --sims 32
```
The Bun engine server resolves the workspace via relative imports into
`lorcana-simulator/packages/...`; `bridge.py` runs it with cwd = `lorcana-simulator/`.

---

## Phase 1 — Plain neural-ISMCTS (DONE)

Goal (met): a working end-to-end self-play loop that can play full games and
generate training data. No league, no belief net — just the basic loop.
Implemented in `lorcana-bot/`; see the completion report and the README there.
**Next is Phase 2** (belief net + importance-weighted determinization + Bayesian
filtering — the strength inflection point; see architecture doc §2.4/§3.2).

### Phase 1 sub-steps (all complete — each gated the next):

```
1A  Bridge       Full kernel callable from Python:
                 executeCommand / getAvailableMoves / filterMatchView
                 exposed as a proper Python interface with:
                 - legal-action mask returned as a numpy bitarray
                 - state serialized for the network encoder
                 - LorcanaEnv gym-style wrapper: step(action) → (obs, mask, done)

1B  Network      PyTorch dual-headed net:
                 - Set transformer encoder over card tokens
                 - Factored policy head (π_cat, π_src, π_tgt) — legal-masked
                 - Value head (categorical/two-hot, 51 atoms)
                 - Shared trunk, three heads

1C  Search       B-ISMCTS in Python:
                 - PUCT selection over information-set nodes
                 - Uniform determinization (no belief net yet — Phase 2)
                 - Depth-limited (d=2 full turns), leaf value from value head
                 - Progressive widening on branching factor

1D  Bootstrap    Behaviour cloning from automation strategies:
                 - Generate games: engine.takeAutomatedActionForCurrentActor()
                 - Supervised training: policy head → automation's choices
                 - Value head → game outcome z
                 - Gives a competent prior before any self-play
```

### Start here: **1A — the bridge**

The WASM smoke proves the engine runs in QuickJS-WASM. Phase 1A makes it
callable in a tight Python loop. Two approaches (evaluate in 1A):

**Option A — WASM exported functions (Javy dynamic model or wasmer-py)**
Pros: true in-process, no subprocess overhead, matches the architecture.
Cons: Javy v3 is WASI-command (stdin/stdout); need v4+ or wasmer's JS plugin.

**Option B — Node subprocess with in-memory transport (fallback)**
Pros: works immediately with existing engine, no porting needed.
Cons: ~1 IPC per batch (vectorized), not per-node; acceptable for Phase 1.
Implementation: Node server exposing engine via `in-memory-transport.ts`,
Python client sends batches of (stateRef, command) via Unix socket.

**Recommended for Phase 1A:** implement Option B first (unblocks the ML work
immediately), prototype Option A in parallel, promote when ready.

### Python project structure to create:

```
lorcana-bot/                     (new, sits alongside lorcana-simulator/)
  engine/
    bridge.py                    LorcanaEnv + Node subprocess wrapper
    serialization.py             state → tensor, action mask → numpy
    node_server/
      server.ts                  Node process hosting the headless kernel
      protocol.ts                binary message framing
  network/
    trunk.py                     set transformer encoder
    heads.py                     policy (factored) + value (distributional)
    model.py                     combined LorcanaNet
  search/
    ismcts.py                    B-ISMCTS with PUCT
    node.py                      InfoSetNode with visit/value pooling
  training/
    bootstrap.py                 behaviour-cloning from automation data
    selfplay.py                  actor loop (Sebulba pattern)
    learner.py                   gradient updates
  tests/
    test_bridge.py               cross-boundary determinism smoke
    test_network.py              forward pass shapes
    test_search.py               1-step ISMCTS sanity check
  requirements.txt               torch, numpy, websockets, flatbuffers, ...
  pyproject.toml
```

---

## Session pattern for Claude Code

At the start of each session, read this file and the relevant phase report.
Before ending a session with significant decisions, update this file.
Use `phase0/`, `phase1/`, etc. folders for phase-specific deliverables.
All new Python code goes under `lorcana-bot/` (create it alongside the repo).
