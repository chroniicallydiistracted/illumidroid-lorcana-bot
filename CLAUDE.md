# Lorcana AI Bot — Project Memory for Claude Code

## What this project is

A champion-level Machine Learning AI player for the Disney Lorcana TCG.
The engine is the open-source **Lorcanito** TypeScript simulator
(`lorcana-simulator/`). The ML stack will be Python (PyTorch/JAX).

Full architecture design: `lorcana-simulator/lorcana-bot-architecture.md`

---

## Current state: Phase 3 COMPLETE (Phases 0–2 also complete)

Phase 3 (League/PSRO self-play + exploitability gating — "becomes hard to beat")
is done, in `lorcana-bot/training/`:
- **League** (`league.py`) — population of players: trainable `main`, frozen
  `past_k` mains, scripted `anchor`s (`best`/`deckAware`), exploiters. `Player`
  interface advances one decision; `NetPlayer` acts via B-ISMCTS, `ScriptedPlayer`
  via the shipped automation.
- **PFSP** opponent sampling (`mode="hard"` weights ∝ `1−P(win)`) + **Elo** +
  pairwise results matrix.
- **Match routing** (`match.py`) — `play_match` runs two (possibly different)
  policies through one real-deck game, routing each decision to the seat to move,
  collecting each learning seat's samples with z from its POV.
- **Anti-collapse** — KL trust-region term (`learner.py` `c_kl`/`set_reference`)
  + entropy bonus + Phase-2 deck-diversity sampling.
- **Exploitability + gauntlet** (`exploitability.py`) — gauntlet win-rate vs
  anchors/checkpoints (gating signal) + a main-exploiter exploitability proxy.
- **Orchestrator** (`league_train.py`) — generate → train → freeze → gate.

Run: `python -m training.league_train --init checkpoints/bc_realdecks.pt
--iterations N --games G --sims S [--exploit]`. Tests: `test_league.py` (PFSP,
Elo, KL, match routing + engine smoke). Report:
`lorcana-simulator/phase3/PHASE3-COMPLETION-REPORT.md`.
**Next is Phase 4** (ReBeL/PBS subgame solving + distributional/auxiliary value
heads; §5.4/§3.2). The §2.3 public-history encoder to sharpen the belief is the
other high-value enrichment.

---

Phase 2 (belief net + importance-weighted determinization + Bayesian filtering —
the strength inflection point) is done, in `lorcana-bot/`:
- **Belief net** — a third, leak-free head (`network/heads.py:BeliefHead`) that
  predicts `P(card ∈ opp hand)` over the opponent's known pool, count-consistent
  to the public hand size. The trunk is built only from the filtered view, so
  policy/value cannot see opponent identities (proven by a leak-free test).
  Trained on the true hidden world (free in self-play; bridge `hidden` block).
- **Importance-weighted determinization** — `bridge.determinize()` repartitions
  the opponent's hidden cards via state surgery; `search/determinize.py` samples
  N worlds from the belief with weights `ρ_i = b/q`; `BISMCTS.run_belief` pools
  per-world Phase-1 searches at the shared root (legal set is world-invariant,
  since determinization only touches hidden cards).
- **Bayesian filtering** — `search/belief_filter.py`: SIR particle filter
  (neural belief = proposal, observed action = correction, ESS-triggered resample).

Tests: 27/27 pass. Report: `lorcana-simulator/phase2/PHASE2-COMPLETION-REPORT.md`,
plan: `lorcana-simulator/phase2/PHASE2-PLAN.md`.
**Next is Phase 3** (League/PSRO self-play + exploitability gating).

**Real decks (done):** 25 real tournament-winning decklists live in
`lorcana-bot/decks/*.json` (fixture shape `{name, cards}`; provenance in
`lorcana-bot/decks/docs/`). The bridge resolves them against the full card
catalog (`all001..all012Cards` + `resolveLorcanaDeckListTextFromPool`) and uses
them by default; `reset(seed, deck_p1, deck_p2)` selects a pair (deterministic
from the seed if unspecified), `list_decks` enumerates them, and
bootstrap/self-play sample distinct pairs for metagame diversity (§6). All 25
resolve to full decks with 0 unresolved cards. Pass `deck_p1="placeholder"` to
force the old synthetic fallback. This makes the belief/determinization
strategically meaningful (real shared card identities across games).

---

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

## Throughput / hardware (training infra)

Dev box: RTX 3070 Ti (8 GB, reachable from WSL2 via `/dev/dxg`), i5-12600K
(16 logical cores), 32 GB. **torch is the CUDA build** (`2.6.0+cu124`,
`torch.cuda.is_available()==True`) — reinstall with
`pip install torch --index-url https://download.pytorch.org/whl/cu124` if the venv
is rebuilt.

**The bottleneck is engine COMPUTE, not IPC and not inference** (measured: native
in-Bun engine ~12–24 decisions/s; a net forward is ~1 ms). The GPU accelerates
the **learner**, not the actors.

**Option 1 (DONE) — in-process search.** The MCTS no longer drives the engine one
step per IPC round-trip. Two changes:
- **O(1) snapshot/restore.** Engine state is immutable (Mutative structural
  sharing), so `snapshot` just holds the `getAuthoritativeState()` reference and
  `restore` does `loadState(shallowProtect(ref))` (shallow-copy top+G+ctx so
  `invalidateStaticEffects` can't corrupt the canonical snapshot). The old
  `structuredClone` (~2.7 ms/op) is gone.
- **`run_paths` op + path-based descent** (`server.ts` + `search/ismcts.py`): the
  Python MCTS tree-walks the in-memory tree (no engine) to find the descent path,
  then ONE `run_paths` IPC executes the whole path in-process and returns the
  leaf obs. Sims that re-hit terminal/depth nodes touch the engine zero times.
  Replaces O(sims×depth) per-step round-trips with O(new-leaves) batched ones.

Measured after Option 1: **~24–25 sims/s plain, ~39 sims/s belief** (≈0.75–0.81
decisions/s/actor) — ~5–8× the old per-step search; the engine is now the
in-process limiter. With the parallel actor pool (`training/distributed.py`,
~3× at ~8 actors) that's the practical training rate.

```bash
python -m training.distributed --init lorcana-bot/checkpoints/bc_fair.pt \
  --rounds R --actors 8 --games-per-actor 1 --sims 8 --n-worlds 4   # [--no-belief]
```

**Leaf-batching + batched GPU inference (DONE).** `SearchConfig.batch_size>1`
enables virtual-loss wave batching (`InfoSetNode.vloss`, `BISMCTS.run_batched`):
each wave descends B sims with virtual loss so they fan out, executes all paths
in ONE `run_paths` IPC, and evaluates all leaves in ONE batched net forward
(`NetEvaluator.batch_eval`) on the GPU. Collisions (same new leaf) are deduped.

**Net size:** default scaled to **d512×8 ≈ 13M params** (the useful ceiling for
this box — VRAM allows ~37M fp32 but data/latency cap usefulness lower). GPU
engages at batch ≥256 + net ≥5M; the small 0.7M net leaves the GPU idle.

**All-in-one runner:** `./train.sh [flags]` → `training/run.py`. **Parallel by
default** (`--actors 6`): K spawn-isolated CPU workers (`training/parallel.py`)
each own a Bun engine + run batched belief self-play, **streaming** progress +
samples to a shared queue; the main process runs the GPU learner (KL+entropy) and
updates a **`LiveMonitor`** (`training/monitor.py`) heartbeat every 3 s (phase,
sims/s, dec/s, games/h, buffer, loss, winrate, GPU mem, ETA, per-worker result)
so a round is never a silent/blocked wait. `--actors 1` = single-process. Fresh
workers reload the latest weights each round (spawn keeps them CUDA-free; only
the main process touches the GPU). Ctrl-C saves and exits.
```bash
./train.sh                                   # ~13M net, 6 actors, runs forever
./train.sh --actors 8 --rounds 50 --sims 24 --batch 48 --eval-every 5
./train.sh --actors 1                        # single-process fallback
```
Gotchas baked in (were bugs): worker rng needs an int seed; all worker setup is
inside try/except so a crash reports instead of hanging; the drain loop has a
liveness guard so a dead worker can't stall the round.

**Rust engine port** is the ceiling-raiser but a 92k-LOC / 205-suite trap (see
`lorcana-bot/PORT-AUDIT.md`) — deferred indefinitely.

Clean prior checkpoint: `bc_fair.pt` (fair data). Pre-fairness-fix checkpoints
were deleted as oracle-tainted.

## Information-policy fairness (training-data hygiene — important)

The engine's automation planner **defaults an unset `informationPolicy` to
`"oracle"`** (`planner.ts:3088`), and oracle = full-deck visibility into the
opponent's hidden cards. Only `bestDeckAwareLoreRaceAutomatedActionStrategy` is
explicitly fair; the oracle and all composer strategies cheat. **Oracle play
must never generate training targets** (BC would clone unseeable decisions; value
z would be off-distribution). Observation features are safe either way (obs is
always the fog-filtered `getBoard(view)`), but targets are not.

The bridge is **fair-by-default**: `best`/`fairBest`/`fairDefault`/`fairControl`/
`fairAggro`/`deckAware` are all fair (a `fair()` wrapper forces it); `oracle`/
`oracleDeckAware` are EVAL-ONLY. `step_auto` returns `policy` ("fair"|"oracle");
`bootstrap.generate_game` raises on oracle; league anchors are fair. Checkpoints
made before this fix are oracle-tainted at the target level — retrain on fair
data.

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
# tests (network/belief/determinize are fast; bridge/search spawn the Bun engine)
lorcana-bot-venv/bin/python -m pytest lorcana-bot/tests        # 27 pass (P1+P2)
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
(Phase 2 — belief net + importance-weighted determinization + Bayesian filtering
— is now also complete; see the state summary at the top of this file.)

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
