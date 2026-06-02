# Headless Lorcana Engine Port Symbol Registry

## 1. How To Use This Registry

This document is the compact symbol memory for the headless Rust port. It is
designed for AI developer agents after context compaction.

Update this file in the same change whenever a port task creates, renames,
removes, deprecates, or changes the meaning of a crate, module, constant,
variable, enum, struct/class, trait, type alias, function, method, command
variant, serialized field, generated artifact, test fixture, oracle fixture, or
source-of-truth path.

Use exact symbol names. Keep entries short, factual, and searchable.

Status values:

```text
planned
implemented
test-only
deprecated
quarantined
removed
```

## 2. Source-Of-Truth Documents And Oracle Paths

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `docs/port/headless_lorcana_engine_porting_blueprint.md` | document | `docs/port` | Active dependency-driven porting plan | n/a | Must be read before port work | implemented | 2026-06-02 |
| `docs/port/headless_lorcana_engine_porting_symbol_registry.md` | document | `docs/port` | Compact symbol/name registry for future agents | n/a | Must be updated with symbol changes | implemented | 2026-06-02 |
| `docs/port/port-status.md` | document | `docs/port` | Operational status tracker for each blueprint step | n/a | Tracks progress only; not a behavior oracle | implemented | 2026-06-02 |
| `AGENTS.md` | document | repo root | Root agent instructions | n/a | Controls active objective and Rust stack | implemented | 2026-06-02 |
| `CLAUDE.md` | document | repo root | Root Claude-specific instructions | n/a | Must match root objective in `AGENTS.md` | implemented | 2026-06-02 |
| `lorcana-simulator/packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.random-apis.ts` | TypeScript source | simulator engine | Oracle RNG and shuffle behavior | TypeScript GameEngine | Rust RNG must match default `seedrandom` called with the seed/draws string | implemented | 2026-06-02 |
| `lorcana-simulator/packages/lorcana/lorcana-engine/package.json` | TypeScript package metadata | simulator engine | Source for `seedrandom` version | TypeScript GameEngine | Shows `seedrandom` dependency version | implemented | 2026-06-02 |
| `lorcana-simulator/packages/lorcana/lorcana-engine/src/core/runtime/types.ts` | TypeScript source | simulator engine | Oracle match state and command envelope types | TypeScript GameEngine | Rust state schema mirrors this source | implemented | 2026-06-02 |
| `lorcana-simulator/packages/lorcana/lorcana-engine/src/types/runtime-state.ts` | TypeScript source | simulator engine | Oracle Lorcana `G`, metadata, Bag, replacement, pending effect state | TypeScript GameEngine | Rust state schema mirrors this source | implemented | 2026-06-02 |
| `lorcana-simulator/packages/lorcana/lorcana-engine/src/zones/runtime-zone-config.ts` | TypeScript source | simulator engine | Oracle zone IDs, visibility, ordering, ownership | TypeScript GameEngine | Rust zones mirror this source | implemented | 2026-06-02 |
| `lorcana-simulator/packages/lorcana/lorcana-engine/src/runtime-moves/index.ts` | TypeScript source | simulator engine | Oracle move registry | TypeScript GameEngine | Rust production registry excludes debug/manual moves | implemented | 2026-06-02 |

## 3. Rust Workspace Crates And Module Map

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `lorcana-rs` | workspace | repo root planned path | Rust port workspace root | n/a | Contains all Rust crates | planned | 2026-06-02 |
| `lorcana-schema` | crate | `lorcana-rs/crates/lorcana-schema` | Rust enums/structs for schemas and DSL nodes | `lorcana-types/src/**`, engine state types | No engine mutation logic | planned | 2026-06-02 |
| `lorcana-card-ir` | crate | `lorcana-rs/crates/lorcana-card-ir` | Generated normalized card catalog IR loader | `lorcana-cards/src/cards/**`, helpers | Avoid hand-porting all cards first | planned | 2026-06-02 |
| `lorcana-core` | crate | `lorcana-rs/crates/lorcana-core` | Pure deterministic headless engine | `lorcana-engine/src/**` | No PyO3, no CLI-only logic | planned | 2026-06-02 |
| `lorcana-conformance` | crate | `lorcana-rs/crates/lorcana-conformance` | TypeScript-vs-Rust lockstep harness | Frozen TypeScript oracle | Compares after every command | planned | 2026-06-02 |
| `lorcana-py` | crate | `lorcana-rs/crates/lorcana-py` | PyO3 Python extension for ML stack | `lorcana-core` APIs | Thin wrapper only | planned | 2026-06-02 |
| `lorcana-cli` | crate | `lorcana-rs/crates/lorcana-cli` | CLI for oracle freeze, replay, snapshot, diff, benchmarks | Rust port + TypeScript oracle | Tooling only | planned | 2026-06-02 |

## 4. Toolchain And Dependency Constants

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `RUST_TOOLCHAIN_VERSION` | constant | `rust-toolchain.toml` | Pinned Rust version | n/a | Initial value `1.96.0` | planned | 2026-06-02 |
| `RUST_EDITION` | constant | workspace `Cargo.toml` | Rust language edition | n/a | Initial value `2024` | planned | 2026-06-02 |
| `PYTHON_BINDING_CRATE` | dependency role | `lorcana-py/Cargo.toml` | Python extension bridge | n/a | Use `pyo3` | planned | 2026-06-02 |
| `PYTHON_PACKAGE_TOOL` | tool role | `lorcana-py/pyproject.toml` | Python build/install tooling | n/a | Use `maturin` | planned | 2026-06-02 |
| `DENSE_OBSERVATION_EXPORT_CRATE` | optional dependency role | `lorcana-py/Cargo.toml` | NumPy tensor export | n/a | Use `numpy` only when dense tensors need it | planned | 2026-06-02 |
| `SERIALIZATION_CRATES` | dependency role | workspace `Cargo.toml` | Schema and JSON handling | TypeScript JSON-like DSL | Use `serde`, `serde_json` | planned | 2026-06-02 |
| `ORDERED_MAP_CRATE` | dependency role | workspace `Cargo.toml` | Observable insertion-order maps/sets | TypeScript object/array order | Use `indexmap` where insertion order matters | planned | 2026-06-02 |
| `PROPERTY_TEST_CRATE` | dependency role | dev-dependencies | Property tests | n/a | Use `proptest` | planned | 2026-06-02 |
| `FUZZING_TOOLS` | tool role | fuzz workspace | Fuzzing structured inputs | n/a | Use `cargo-fuzz` + `arbitrary` | planned | 2026-06-02 |
| `SNAPSHOT_TEST_CRATE` | dependency role | dev-dependencies | Snapshot tests | n/a | Use `insta` | planned | 2026-06-02 |
| `BENCHMARK_CRATE` | dependency role | dev-dependencies | Microbenchmarks | n/a | Use `criterion` after parity | planned | 2026-06-02 |

## 5. TypeScript Oracle Constants And Source Symbols

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `seedrandom@3.0.5` | dependency version | `lorcana-engine/package.json` | Oracle PRNG package | TypeScript engine | Rust engine RNG must match default `seedrandom` behavior | implemented | 2026-06-02 |
| `ctx.random.seed` | state field | `core/runtime/types.ts` | Per-match random seed | TypeScript engine | Preserve exactly | implemented | 2026-06-02 |
| `ctx.random.draws` | state field | `core/runtime/types.ts` | Random draw counter | TypeScript engine | Increment before calling `seedrandom` with the seed/draws string | implemented | 2026-06-02 |
| `createRandomAPIForDraft` | function | `match-runtime.random-apis.ts` | Creates `random()` and `shuffle()` for draft state | TypeScript engine | Rust RNG module mirrors this behavior | implemented | 2026-06-02 |
| `random` | function | `match-runtime.random-apis.ts` | Produces next deterministic float | TypeScript engine | Uses string seed `${seed}:${draws}` | implemented | 2026-06-02 |
| `shuffle` | function | `match-runtime.random-apis.ts` | Fisher-Yates shuffle | TypeScript engine | Rust shuffle must match order exactly | implemented | 2026-06-02 |
| `MIN_DECK_SIZE` | constant | `lorcana-types/src/cards/deck-validation.ts` | Minimum legal deck size | TypeScript types | Value `60` | implemented | 2026-06-02 |
| `MAX_INK_TYPES` | constant | `lorcana-types/src/cards/deck-validation.ts` | Maximum ink colors in a deck | TypeScript types | Value `2` | implemented | 2026-06-02 |
| `MAX_COPIES_PER_CARD` | constant | `lorcana-types/src/cards/deck-validation.ts` | Maximum copies of one card | TypeScript types | Value `4` | implemented | 2026-06-02 |
| `MatchState` | type | `core/runtime/types.ts` | Authoritative state envelope `{ G, ctx }` | TypeScript engine | Rust state root mirrors this shape | implemented | 2026-06-02 |
| `LorcanaG` | type | `types/runtime-state.ts` | Lorcana-specific rules state | TypeScript engine | Rust `G` mirrors this before normalization | implemented | 2026-06-02 |
| `LorcanaCardMeta` | type | `types/runtime-state.ts` | Mutable per-card metadata | TypeScript engine | Preserve default field shape for parity first | implemented | 2026-06-02 |

## 6. Rust Port Constants

No Rust implementation constants exist yet. Add entries here as soon as the Rust
workspace is created.

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `ENGINE_RNG_IS_SEEDRANDOM_COMPATIBLE` | invariant | `lorcana-core` planned | Documents that engine RNG is oracle-compatible, not `rand`-based | `match-runtime.random-apis.ts` | Must be proven by golden vectors | planned | 2026-06-02 |

## 7. Runtime State Structs / Classes / Types

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `MatchState` | planned Rust type | `lorcana-schema` or `lorcana-core` | Root authoritative state | `core/runtime/types.ts` | Must serialize to oracle-equivalent snapshot | planned | 2026-06-02 |
| `TCGCtx` | planned Rust type | `lorcana-schema` or `lorcana-core` | Runtime/system context | `core/runtime/types.ts` | Preserve `_stateID`, phase, priority, zones, random fields | planned | 2026-06-02 |
| `LorcanaG` | planned Rust type | `lorcana-schema` or `lorcana-core` | Lorcana game state | `types/runtime-state.ts` | Preserve fields before performance normalization | planned | 2026-06-02 |
| `LorcanaCardMeta` | planned Rust type | `lorcana-schema` or `lorcana-core` | Mutable card metadata | `types/runtime-state.ts` | Preserve explicit `undefined`/missing-field semantics via serialization policy | planned | 2026-06-02 |
| `HeadlessEngine` | planned Rust type | `lorcana-core` | Engine owner for static resources, state, caches, replay log | Blueprint section 11.2 | Must not use process-global rule cache | planned | 2026-06-02 |
| `EngineLane` | planned Rust type | `lorcana-core` later optimization | Optimized apply/undo lane for MCTS | Reducer parity tests | Add only after reducer parity exists | planned | 2026-06-02 |

## 8. Command / Action Variants

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `ChooseWhoGoesFirst` | action variant | planned Rust action enum | Setup first-player choice | `moves/setup/choose-who-goes-first.ts` | Production action | planned | 2026-06-02 |
| `AlterHand` | action variant | planned Rust action enum | Mulligan / alter hand | `moves/setup/alter-hand.ts` | Production action | planned | 2026-06-02 |
| `PutCardIntoInkwell` | action variant | planned Rust action enum | Ink action | `runtime-moves/moves/resources/*` | Production action | planned | 2026-06-02 |
| `PlayCard` | action variant | planned Rust action enum | Standard play, Shift, Sing, Sing Together, alternatives | `moves/core/play-card.ts` | High dependency step; do not port early | planned | 2026-06-02 |
| `Quest` | action variant | planned Rust action enum | Quest with one character | `moves/core/quest.ts` | Production action | planned | 2026-06-02 |
| `QuestWithAll` | action variant | planned Rust action enum | Quest with all eligible characters | `runtime-moves/index.ts` | Production only if oracle supports current behavior | planned | 2026-06-02 |
| `Challenge` | action variant | planned Rust action enum | Challenge action | `moves/core/challenge.ts`, `rules/challenge-rules.ts` | Production action | planned | 2026-06-02 |
| `MoveCharacterToLocation` | action variant | planned Rust action enum | Location movement | `runtime-moves/moves/location/*` | Production action | planned | 2026-06-02 |
| `ActivateAbility` | action variant | planned Rust action enum | Activated ability use | `runtime-moves/moves/abilities/*` | Production action | planned | 2026-06-02 |
| `ResolveBag` | action variant | planned Rust action enum | Resolve triggered Bag item | `resolution/resolve-bag.ts` | High dependency step | planned | 2026-06-02 |
| `ResolveEffect` | action variant | planned Rust action enum | Resolve pending effect input | `resolution/resolve-effect.ts` | High dependency step | planned | 2026-06-02 |
| `PassTurn` | action variant | planned Rust action enum | End current turn | `moves/turn/pass-turn.ts` | Production action | planned | 2026-06-02 |
| `Concede` | action variant | planned Rust action enum | Concede game | `moves/turn/*` | Production if enabled by oracle | planned | 2026-06-02 |
| `ForfeitGame` | action variant | planned Rust action enum | Forfeit game | `moves/turn/*` | Production if enabled by oracle | planned | 2026-06-02 |
| `manual*` debug moves | action family | test-only registry | Manual simulator/debug state edits | `debug/manual-moves.ts` | Must not appear in production legal actions | test-only | 2026-06-02 |

## 9. Engine API Functions And Methods

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `new_game` | planned function | `lorcana-core` | Create a new game from config | `runtime-game/definition.ts` | Must match setup/deck shuffle parity | planned | 2026-06-02 |
| `get_state` | planned function | `lorcana-core` | Return authoritative state snapshot | `MatchState` | Snapshot must be canonicalized for parity | planned | 2026-06-02 |
| `get_observation` | planned function | `lorcana-core` | Return player-visible observation | `filterMatchView` / projection paths | Must not leak hidden info | planned | 2026-06-02 |
| `get_legal_actions` | planned function | `lorcana-core` | Enumerate command-compatible legal actions | move validators + targeting + flow | Every returned action must validate | planned | 2026-06-02 |
| `validate_action` | planned function | `lorcana-core` | Validate action without mutation | `match-runtime.validation.ts` | Must preserve preflight/final distinction if oracle has it | planned | 2026-06-02 |
| `apply_action` | planned function | `lorcana-core` | Apply one action and return transition result | `match-runtime.commands.ts` | Reducer source of truth | planned | 2026-06-02 |
| `clone_game` | planned function | `lorcana-core` | Clone game state for search | TypeScript snapshot/restore behavior | Start with `Clone`; optimize later | planned | 2026-06-02 |
| `serialize_game` | planned function | `lorcana-core` | Serialize state | `serialization.ts`, `core/runtime/serialization.ts` | Must match canonical snapshot policy | planned | 2026-06-02 |
| `deserialize_game` | planned function | `lorcana-core` | Restore serialized state | `serialization.ts`, replay | Must fail closed on invalid state | planned | 2026-06-02 |
| `export_replay` | planned function | `lorcana-core` | Export replay | `core/runtime/replay.ts` | Deterministic structure, not human UI logs | planned | 2026-06-02 |
| `import_replay` | planned function | `lorcana-core` | Import replay | `core/runtime/replay.ts` | Must replay through reducer | planned | 2026-06-02 |

## 10. Serialization, Replay, And Snapshot Artifacts

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `oracle/source-hash.txt` | artifact | planned `oracle/` | Frozen TypeScript source hash | TypeScript repo | Required before Step 1 | planned | 2026-06-02 |
| `oracle/card-catalog-hash.txt` | artifact | planned `oracle/` | Frozen card catalog hash | `lorcana-cards/src/**` | Required before card IR parity claims | planned | 2026-06-02 |
| `oracle/ruleset-hash.txt` | artifact | planned `oracle/` | Frozen ruleset hash | engine/types/cards source | Required before parity claims | planned | 2026-06-02 |
| `oracle/replay-corpus/` | artifact directory | planned `oracle/` | Replay fixtures for conformance | TypeScript oracle | Compare after every command | planned | 2026-06-02 |
| `oracle/snapshot-schema/` | artifact directory | planned `oracle/` | Snapshot schema fixtures | TypeScript oracle | Drives Rust serialization parity | planned | 2026-06-02 |

## 11. Test Fixtures And Conformance Corpora

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `rng_golden_vectors` | fixture | planned `oracle/` or `lorcana-conformance` | Verify Rust RNG output against TypeScript | `match-runtime.random-apis.ts` | Required before setup shuffle parity | planned | 2026-06-02 |
| `shuffle_golden_vectors` | fixture | planned `oracle/` or `lorcana-conformance` | Verify Fisher-Yates output against TypeScript | `match-runtime.random-apis.ts` | Must include deck-order examples | planned | 2026-06-02 |
| `initial_state_snapshots` | fixture | planned `oracle/snapshot-schema/` | Verify setup state shape | `runtime-game/definition.ts` | Include seeded real-deck examples | planned | 2026-06-02 |
| `random_legal_action_streams` | corpus | planned `oracle/replay-corpus/` | Stress differential legal command streams | TypeScript oracle legal actions | Must compare after every command | planned | 2026-06-02 |

## 12. Generated Artifacts

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `normalized_card_ir.json` | planned generated artifact | `lorcana-rs` generated path TBD | Language-agnostic card catalog IR | `lorcana-cards/src/cards/**` | Generated before Rust card loading | planned | 2026-06-02 |
| `schema_discriminator_manifest.json` | planned generated artifact | `lorcana-rs` generated path TBD | List of effect/condition/target/trigger discriminators | `lorcana-types/src/**`, engine registries | Used to prevent missing DSL variants | planned | 2026-06-02 |

## 13. Deprecated Or Quarantined Symbols

| Symbol / Name | Kind | Location | Purpose | Oracle source | Parity notes | Status | Updated |
|---|---|---|---|---|---|---|---|
| `tokio` | dependency | n/a | Async runtime | n/a | Not allowed in `lorcana-core`; unnecessary for deterministic CPU-bound transitions | quarantined | 2026-06-02 |
| `rand::StdRng` | dependency symbol | n/a | Standard Rust RNG | n/a | Not allowed for engine transitions; may be used only in tests/tools if explicitly scoped | quarantined | 2026-06-02 |
| `HashMap` iteration order | behavior | n/a | Unordered map iteration | n/a | Must not define rules-visible order | quarantined | 2026-06-02 |
| `cffi` primary bridge | integration option | n/a | Python FFI alternative | n/a | Not selected; PyO3/maturin is primary | quarantined | 2026-06-02 |

## 14. Update Log

| Date | Change | Notes |
|---|---|---|
| 2026-06-02 | Created initial symbol registry | Captures Rust stack, planned crates, oracle paths, core API names, action variants, fixtures, and quarantined symbols. |
