# Lorcana AI Bot — Project Memory for Claude Code

## What this project is

A champion-level Machine Learning AI player for the Disney Lorcana TCG.
The engine is the open-source **Lorcanito** TypeScript simulator
(`lorcana-simulator/`). The ML stack will be Python (PyTorch/JAX).

Full architecture design: `lorcana-simulator/lorcana-bot-architecture.md`

---

## Current Safety Checkpoint: Tier-A Remediation Active

The historical Phase 0–3 milestone reports remain useful architecture context,
but they do not prove belief-guided clean-label training correctness. The
follow-up Tier-A audit found active-path gaps. The authoritative implementation
sequence is `tier-a-belief-search-remediation-plan.md`.

Audited checkpoint:

```text
Tier-A Phase 0   complete — belief-guided clean-label sample writing fails closed
Tier-A Phase 13  baseline installed — 5 passing probes (2 Phase-1 + 3 Phase-2) + 15 strict expected-red probes
Tier-A Phase 1   complete — audited GO for the canonical full hidden-zone World boundary
Tier-A Phase 2   complete — audited GO for exact structured sampling + importance semantics
Tier-A Phase 3   IN REMEDIATION (NO-GO) — full-`World` RPC built + bounded findings fixed;
                 protected-facts ledger partial (reveals guarded) + obs sanitization (F3) unresolved
Next             close Tier-A Phase 3 blockers before Phase 7
```

Latest verification: `134 passed / 15 xfailed` (`149` collected), with clean
`compileall` and `git diff --check`. Use `--no-belief` for diagnostics only.
Do not resume belief-guided clean-label self-play until the Tier-A final release
gate passes.

---

## Historical milestone state: Phase 3 COMPLETE (Phases 0–2 also complete)

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

Diagnostic run during Tier-A remediation: `python -m training.league_train
--no-belief --init checkpoints/bc_realdecks.pt --iterations N --games G --sims S
[--exploit]`. Tests: `test_league.py` (PFSP,
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
- **Historical importance-weighted determinization path** — `bridge.determinize()` repartitions
  the opponent's hidden cards via state surgery; `search/determinize.py` samples
  N worlds from the belief with weights `ρ_i = b/q`; the legacy root-pooled PIMC
  implementation is now diagnostic-only as `BISMCTS.run_pimc_diagnostic`.
- **Bayesian filtering prototype** — `search/belief_filter.py`: SIR particle
  filter seam. Tier-A Phase 5 owns the full-world persistent replacement and
  Phase 10 owns observed-action integration in real self-play.

Historical tests: 27/27 passed. Report: `lorcana-simulator/phase2/PHASE2-COMPLETION-REPORT.md`,
plan: `lorcana-simulator/phase2/PHASE2-PLAN.md`.

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
in-process limiter. The parallel actor pool (`training/parallel.py`, driven by
`run.py --actors N`) adds ~3× at ~8 actors — that's the practical training rate.
(The old `training/distributed.py` was removed; `run.py`/`parallel.py` supersede it.)

**Leaf-batching + batched GPU inference (DONE).** `SearchConfig.batch_size>1`
enables virtual-loss wave batching (`InfoSetNode.vloss`, `BISMCTS.run_batched`):
each wave descends B sims with virtual loss so they fan out, executes all paths
in ONE `run_paths` IPC, and evaluates all leaves in ONE batched net forward
(`NetEvaluator.batch_eval`) on the GPU. Collisions (same new leaf) are deduped.

**Net size:** default scaled to **d512×8 ≈ 13M params** (the useful ceiling for
this box — VRAM allows ~37M fp32 but data/latency cap usefulness lower). GPU
engages at batch ≥256 + net ≥5M; the small 0.7M net leaves the GPU idle.

**All-in-one runner:** `./train.sh [flags]` → `training/run.py`. During Tier-A
remediation, pass `--no-belief`; belief-guided clean-label sample writing fails
closed. **Parallel by default** (`--actors 6`): K spawn-isolated CPU workers
(`training/parallel.py`) each own a Bun engine + run batched diagnostic
self-play, **streaming** progress +
samples to a shared queue; the main process runs the GPU learner (KL+entropy) and
updates a **`LiveMonitor`** (`training/monitor.py`) heartbeat every 3 s (phase,
sims/s, dec/s, games/h, buffer, loss, winrate, GPU mem, ETA, per-worker result)
so a round is never a silent/blocked wait. `--actors 1` = single-process. Fresh
workers reload the latest weights each round (spawn keeps them CUDA-free; only
the main process touches the GPU). Ctrl-C saves and exits.
```bash
./train.sh --no-belief                                   # ~13M net, 6 actors
./train.sh --no-belief --actors 8 --rounds 50 --sims 24 --batch 48 --eval-every 5
./train.sh --no-belief --actors 1                        # single-process fallback
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

### Python ML stack (from repo root; venv at `lorcana-bot-venv/`)

```bash
# tests (network/belief/determinize are fast; bridge/search spawn the Bun engine)
lorcana-bot-venv/bin/python -m pytest lorcana-bot/tests        # 111 pass / 18 expected xfail
# behaviour-clone the scripted oracle, then run non-belief diagnostics only
cd lorcana-bot
../lorcana-bot-venv/bin/python -m training.bootstrap --games 20 --epochs 5 --out checkpoints/bc.pt
../lorcana-bot-venv/bin/python -m training.selfplay --no-belief --init checkpoints/bc.pt --iterations 3 --games 4 --sims 32
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

---

# Agent Requirement Contract

You must not merely make code run. You must prove that the implementation is correct, aligned with the stated remediation goals, guarded against drift, and resistant to false-positive test completion.

This contract applies to all remediation work and any follow-up development that affects any repository/codebase wide features or functionality.

---

# 1. Non-negotiable objective

You objective is:

```text
Implement the requested development task exactly, prove it with meaningful tests, prevent hidden drift, and explicitly report any remaining uncertainty or unchecked area.
```

You must not optimize for:

```text
passing existing tests only
minimal patches that preserve incorrect legacy behavior
backward compatibility with flawed logic
silencing failures
reducing scope without saying so
claiming completion without proof
```

Correctness has priority over preserving old APIs, old tests, old abstractions, or old assumptions.

---

# 2. Required source-of-truth hierarchy

You must resolve conflicts using this priority order:

```text
1. Lorcana TCG rules correctness
2. TheCardGoats / Lorcanito simulator behavior
3. Tier A remediation requirements
4. Current repository architecture
5. Existing tests
6. Legacy helper names, legacy APIs, legacy comments
```

If an old test conflicts with correct behavior, the test must be rewritten.

If an old helper conflicts with correct behavior, the helper must be replaced, removed, or quarantined as diagnostic-only.

If preserving an API would preserve incorrect logic, the API must be changed.

---

# 3. Hard forbidden actions

You must not do any of the following.

## 3.1 Forbidden code behavior

```text
Do not use hidden opponent card identities in actor-visible search keys.
Do not leak opponent inkwell identities into hand/deck belief sampling.
Do not discard sampled hidden-zone world fields.
Do not silently repair invalid hidden-zone worlds.
Do not use unseeded randomness in reproducible search paths.
Do not fall back to Math.random in full ISMCTS determinization.
Do not continue clean-label training through PIMC/root-pooled determinized search.
Do not leave legacy belief search available as an active training path.
Do not collapse distinct legal actions that differ by visible target, choice, cost, destination, or mulligan fields.
Do not redact information from the acting player that the acting player legally knows.
Do not expose opponent private information to an observer.
Do not mutate server public history during simulated search transitions.
Do not feed stale root history to neural leaf evaluation when simulated branch history exists.
Do not return empty diagnostic roots when the real shared search tree has populated statistics.
Do not normalize or rebase a value while still documenting it as raw.
Do not multiply only included-card probabilities and call it a valid posterior update.
```

## 3.2 Forbidden testing behavior

```text
Do not delete failing tests to claim success.
Do not weaken assertions to make failures pass.
Do not replace exact tests with loose smoke tests.
Do not skip tests unless the skip condition is real, narrow, and justified in code.
Do not mark deterministic correctness tests as flaky.
Do not use mocks where real-engine behavior is required by the remediation.
Do not rely only on transposition_hits > 0 as proof of deeper shared-tree correctness.
Do not test only isolated helper math when the failure is in the runtime path.
Do not accept tests that would pass on the pre-remediation broken code.
```

## 3.3 Forbidden reporting behavior

```text
Do not claim 100% completion unless all required verification commands pass.
Do not claim a file was audited unless it was actually inspected.
Do not claim alignment with Lorcanito/GameEngine behavior without checking the relevant source path.
Do not omit failed commands from the final report.
Do not hide skipped tests, xfails, warnings, or partial failures.
Do not say a task is complete when any acceptance criterion remains unproven.
```

---

# 4. Required pre-work before editing code

Before changing code, You must identify and report:

```text
1. The exact remediation phase being implemented.
2. The finding(s) being addressed.
3. The current flawed behavior.
4. The files and functions involved.
5. The downstream files likely affected.
6. The tests that currently miss the failure.
7. The new tests that will prove the fix.
8. Whether any legacy API/test must be removed, renamed, or rewritten.
```

You must inspect every directly relevant file before editing it.

If a function imports or calls another helper that controls the same behavior, that helper must also be inspected.

If a TypeScript bridge/server wrapper calls into TheCardGoats GameEngine, You must distinguish:

```text
bot-owned wrapper code: may be changed
native GameEngine code: must not be changed
```

---

# 5. Required implementation discipline

## 5.1 Dependency order

You must implement remediation in dependency order.

Do not implement downstream behavior before the upstream contract exists.

Example:

```text
Do not wire EngineSimulator.begin_lane() to pass full worlds before bridge/server determinize_world() exists.
Do not force StructuredBeliefEvaluator into training before BeliefTracker can store full World particles.
Do not claim full ISMCTS reproducibility before all worlds carry deterministic seeds.
```

## 5.2 One behavior, one authoritative path

You must avoid duplicate active implementations.

If replacing legacy behavior, the new behavior must become the only clean-label path.

Legacy behavior may remain only when all are true:

```text
It is explicitly named diagnostic or legacy.
It is not reachable from clean-label training.
Tests prove clean-label training does not call it.
Its docstring warns that it is not valid for Tier A training labels.
```

## 5.3 Fail closed

Invalid state must fail closed.

Required examples:

```text
Invalid World partition -> raise error, do not silently repair.
Missing full hidden-zone assignment -> raise error in clean-label path.
Hand-only belief evaluator passed to full ISMCTS -> raise error.
Unseeded full-ISMCTS world -> raise error.
Attempt to use run_belief for clean-label training -> raise error.
Engine returned world differs from requested World -> raise error.
```

## 5.4 No silent fallback

You must remove or guard silent fallbacks that hide correctness failures.

Forbidden patterns:

```python
try:
    critical_correctness_operation()
except Exception:
    pass
```

```python
if invalid_world:
    world = World()
```

```python
if exact_action_failed:
    execute_some_other_action()
```

```ts
const rng = seed ? seededRng(seed) : Math.random;
```

```python
if tests_fail:
    relax_assertion()
```

---

# 6. Required test standards

Every meaningful fix must include tests that prove:

```text
The old bug would fail.
The new behavior passes.
The runtime path uses the new behavior.
The implementation fails closed on invalid inputs.
The implementation does not leak hidden information.
The implementation remains reproducible when seeded.
```

## 6.1 Test categories required when applicable

For each remediation phase, You must include at least one test from each applicable category:

```text
Unit test:
  Pure helper/math/projection behavior.

Contract test:
  API shape, required fields, fail-closed behavior.

Runtime integration test:
  Real path used by training/search/engine bridge.

Regression test:
  Reproduces the audited failure.

Negative test:
  Invalid or unsafe behavior raises instead of silently continuing.

Reproducibility test:
  Same seed produces same result when deterministic behavior is required.

Anti-leak test:
  Hidden opponent information does not affect actor-visible output.

Anti-drift test:
  Legacy path is unreachable from clean-label training.
```

## 6.2 Tests must not be performative

A test is invalid if it only proves that code executed.

Weak:

```python
assert res.pi.sum() > 0
assert transposition_hits > 0
assert len(worlds) > 0
```

Strong:

```python
assert requested_inkwell_ids == actual_inkwell_ids
assert root_history_len < leaf_history_len
assert returned_root_visit_sum > 0
assert two_visible_target_differences_produce_different_keys
assert two_hidden_opponent_ink_identity_differences_produce_same_observer_key
assert clean_label_training_did_not_call_run_belief
```

## 6.3 No unjustified skips

A skip is allowed only when the condition is unavoidable and specific.

Allowed:

```python
if obs["done"]:
    pytest.skip("game ended before required midgame state")
```

Forbidden:

```python
pytest.skip("flaky")
pytest.skip("not implemented")
pytest.skip("engine issue")
pytest.mark.xfail
```

Any `xfail`, broad skip, or test weakening must be treated as incomplete remediation unless explicitly approved.

---

# 7. Required audit after implementation

After making changes, You must perform a self-audit before claiming completion.

The audit must include:

```text
1. Search for forbidden legacy calls.
2. Search for hand-only belief usage.
3. Search for unseeded randomness in search/determinization paths.
4. Search for Math.random in full-ISMCTS paths.
5. Search for broad exception swallowing.
6. Search for new or existing TODO/FIXME near touched logic.
7. Search for skipped/xfail tests.
8. Search for tests that assert only shape/execution without semantic correctness.
9. Confirm no native GameEngine source was modified.
10. Confirm all required new tests were added.
```

Suggested commands:

```bash
grep -R "BeliefEvaluator" -n lorcana-bot/search lorcana-bot/training lorcana-bot/tests
grep -R "run_belief" -n lorcana-bot/search lorcana-bot/training lorcana-bot/tests
grep -R "Math.random" -n lorcana-bot/engine/node_server lorcana-bot/search lorcana-bot/training
grep -R "pytest.skip\|xfail\|pass  #\|except Exception" -n lorcana-bot/tests lorcana-bot/search lorcana-bot/training lorcana-bot/engine
grep -R "TODO\|FIXME\|temporary\|legacy" -n lorcana-bot/search lorcana-bot/training lorcana-bot/engine lorcana-bot/tests
git diff --name-only
git diff --check
```

If these commands find concerning results, You must resolve them or explicitly list them as unresolved blockers.

---

# 8. Required verification commands

You must run the baseline verification:

```bash
../lorcana-bot-venv/bin/python -m pytest -q
git diff --check
python -m compileall -q engine network search training tests
```

You must also run the Tier A-specific verification suite:

```bash
../lorcana-bot-venv/bin/python -m pytest -q \
  tests/test_world_contract.py \
  tests/test_engine_determinize_world.py \
  tests/test_belief_tracker_full_world.py \
  tests/test_structured_belief_wiring.py \
  tests/test_clean_label_search_mode.py \
  tests/test_engine_sim_full_world.py \
  tests/test_infoset_perfect_recall.py \
  tests/test_lane_local_history.py \
  tests/test_observed_action_correction_runtime.py \
  tests/test_infoset_diagnostics.py \
  tests/test_reproducibility_tier_a.py \
  tests/test_tier_a_runtime_regressions.py
```

If any file does not yet exist, You must create it when it is required by the remediation phase.

A phase is not complete if its required test file does not exist.

---

# 9. Required final response format from the agent

At the end of each implementation phase, You must report in this exact structure.

```markdown
# Phase completion report

## Phase implemented

- Phase:
- Findings addressed:
- Commit/branch:

## Files changed

- `path/to/file.py`
  - What changed:
  - Why it changed:
  - Risk:

## Legacy logic removed or quarantined

- Legacy path:
- Status:
- Proof it is not used by clean-label training:

## Tests added or updated

- `tests/test_name.py::test_name`
  - Proves:
  - Would fail before this fix:

## Verification commands run

```bash
<command>
```

Result:

```text
<exact result summary>
```

## Forbidden-pattern audit

- Hand-only belief path checked:
- PIMC clean-label path checked:
- Unseeded randomness checked:
- Math.random checked:
- broad exception swallowing checked:
- skipped/xfail tests checked:
- native GameEngine modifications checked:

## Remaining blockers

- None

or

- Blocker:
  - Why it remains:
  - Required next action:
```

You must not replace this with a generic summary.

---

# 10. Required outstanding-work check

Before ending any response, You must explicitly answer:

```text
Did I inspect every file/function that this phase depends on?
Did I update every active caller?
Did I remove or guard every obsolete legacy path?
Did I add tests that would fail on the broken implementation?
Did I run the required tests?
Did I check for hidden fallbacks, skipped tests, and broad exception swallowing?
Did I confirm no native GameEngine code was modified?
Is there any remaining unverified behavior?
```

If any answer is not an unqualified yes, the phase is incomplete.

---

# 11. Strict definition of done

A remediation item is done only when all are true:

```text
The flawed logic is removed or made unreachable.
The replacement logic is implemented in the active runtime path.
The replacement logic is covered by meaningful tests.
The tests would fail on the previous flawed logic.
The full verification command passes.
The agent audited for forbidden patterns.
The agent reports no unresolved blockers.
```

A remediation item is not done when:

```text
Only helper code was added.
Only a test mock passes.
Only existing tests pass.
The old path remains reachable.
The final behavior depends on caller discipline.
The code silently falls back to unsafe behavior.
The agent did not inspect downstream callers.
The agent did not run verification.
```

---

# 12. Tier A-specific forbidden drift list

For Tier A belief-search correctness, these are strict blockers.

```text
BeliefEvaluator used in clean-label training.
run_belief used in clean-label training.
World.opponent_inkwell_ids ignored by engine simulation.
World.opponent_deck_ids ignored by engine simulation.
World.self_deck_ids ignored when supplied.
sample_worlds() returns empty default Worlds for non-empty hidden pools.
rho stored as rebased value while documented as raw b/q.
Structured uniform proposal forces rho == 1.
BeliefTracker stores only hand sets.
BeliefTracker.sample_world() returns hand-only Worlds.
BeliefTracker applies included-only posterior reweighting.
observe_opponent_action() exists but is not called during real self-play.
action_updates remains zero after real opponent actions.
info_set_key includes obs["hidden"] identities.
history_event truncates visible action identity.
Actor’s own ink/mulligan identity is redacted from actor.
Opponent hidden ink/mulligan identity is visible to observer.
stepExact mutates server public history.
Neural leaf obs receives stale root history only.
run_infoset returns empty legacy diagnostic root.
DecisionTracer reports zero visits for populated full ISMCTS.
Tracker worlds lack seeds.
TypeScript full-ISMCTS determinization can fall back to Math.random.
Root-invariance tests omit opponent inkwell.
Runtime tests do not exercise the real bridge/server path.
```

Any one of these means development is incomplete.

---

# 13. Pull request merge contract

A remediation/implementation must not be finalized unless the description includes:

```text
1. Phase(s) implemented.
2. Findings closed.
3. Files changed.
4. Tests added.
5. Verification commands and results.
6. Forbidden-pattern audit results.
7. Confirmation that no native TheCardGoats GameEngine source was modified.
8. Remaining blockers, if any.
```

A remediation/implementation must be rejected if:

```text
It weakens tests.
It skips failing tests without justification.
It changes native GameEngine code.
It leaves legacy clean-label paths reachable.
It does not include tests for the remediated behavior.
It claims completion without command output.
It introduces broad silent fallback behavior.
```
