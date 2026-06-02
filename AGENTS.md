# Lorcana Headless GameEngine Port - Root Agent Instructions

## Active Objective

The active repository objective is a dependency-driven headless port of the
current Lorcanito / TheCardGoats Lorcana GameEngine into a faster language for
ML self-play and search.

The port has one non-negotiable requirement:

```text
Preserve Lorcana rules and current GameEngine behavior 1:1, with conformance
tests proving parity against the frozen TypeScript oracle.
```

Speed matters only after correctness has been proven. Do not trade away exact
GameEngine behavior for performance, ergonomics, simpler APIs, or cleaner
internal design.

The active porting source of truth is:

```text
headless_lorcana_engine_porting_blueprint.md
```

Read that file before starting any port-related work.

## Current Pause

The previous ML-bot roadmap, Tier-A belief-search remediation, league training,
self-play training, ReBeL/PBS work, and neural-network improvements are on hold
until the headless GameEngine port has passed its conformance gates.

Do not resume or extend bot training work unless the user explicitly asks for
that separate task. Existing bot/training documents remain historical context
only.

Allowed ML-adjacent work during the port:

- Defining the headless engine API needed by ML.
- Building legal-action, observation, clone, serialize, and replay interfaces.
- Building conformance and differential replay harnesses.
- Benchmarking after parity is already proven for the layer being measured.

## Source-Of-Truth Order

For the active port, resolve conflicts in this order:

1. Current Lorcanito / TheCardGoats TypeScript engine behavior as the frozen
   oracle for compatibility.
2. Official Lorcana TCG rules, when deciding whether the oracle itself needs a
   separate upstream/runtime correction.
3. `headless_lorcana_engine_porting_blueprint.md`.
4. Current repository architecture.
5. Existing tests.
6. Legacy helper names, comments, APIs, or convenience abstractions.

If the TypeScript oracle appears to conflict with Lorcana rules, do not silently
"fix" the port. Preserve oracle behavior in the port, document the suspected
rules discrepancy, and only change behavior when the TypeScript oracle and its
tests are intentionally updated.

## Porting Language Status

The final port language is not locked by old Rust-port notes.

Before implementing Step 1 of the port, perform and document a language decision
using the blueprint's constraints:

- Exact parity and deterministic replay support.
- Fast clone/apply/legal-action loops for ML self-play.
- Safe state ownership and cache invalidation.
- First-class Python integration for the learner/search stack.
- Ability to express a large rules engine without hiding fallbacks.
- Tooling for differential tests, fuzz/property tests, profiling, and CI.
- Practical maintainability for hundreds of thousands of lines of card/rule
  behavior.

Rust remains a serious candidate, but do not assume it by default. Compare it
against at least C++, Go, Zig, and a TypeScript/WASM or native-TS optimization
path before committing.

## Hard Porting Rules

Never start by porting card files, `playCard`, Bag resolution, or high-level card
effects. Follow the dependency graph in the blueprint.

Never use the Svelte simulator UI as a runtime source of truth for the headless
engine. Required packages are:

```text
packages/lorcana/lorcana-types
packages/lorcana/lorcana-engine
packages/lorcana/lorcana-cards
```

Mostly ignore for the production headless core:

```text
packages/lorcana/lorcana-simulator
packages/lorcana/lorcana-interaction
runtime-game/lorcanaPacketAnimations.ts
Svelte routes/components/screens/menus/visual tests
debug manual moves, except in test-only registries
```

Do not modify native TypeScript oracle logic during the port unless the user
explicitly asks for an oracle-side rules fix. Adding oracle tests, snapshot
exporters, replay fixtures, or differential harnesses is allowed.

Do not redesign behavior while porting. First reproduce current behavior; then
propose separately-tested improvements after parity exists.

Do not introduce:

- Unseeded randomness.
- `Math.random` or host RNG fallbacks in deterministic paths.
- Silent repair of invalid states.
- Broad exception swallowing around correctness logic.
- Process-global rule caches in the new headless engine.
- Debug/manual moves in production legal-action generation.
- UI observation leakage of hidden information.
- Legal-action generators that return actions not accepted by command
  validation.

## Dependency Spine

The required implementation order is:

```text
types / card schemas
-> static card resources
-> authoritative match state
-> zones / card instances / metadata
-> deterministic RNG
-> runtime command reducer
-> flow / priority / turn status
-> card runtime queries
-> derived state
-> static effects
-> conditions
-> targeting
-> costs
-> primitive state mutations
-> core moves
-> action effects
-> pending effect resolution
-> triggered abilities
-> Bag resolution
-> replacement/prevention effects
-> full play-card execution
-> activated abilities
-> full card catalog
-> legal-action generation for ML
-> replay/differential validation
```

Each step must depend only on completed upstream contracts. If a downstream
system is needed for a test, build the smallest explicit test harness instead of
porting downstream behavior early.

## Required Pre-Work Before Editing Port Code

Before editing implementation files for any port step, identify and report:

1. The exact blueprint step being implemented.
2. The TypeScript source files/functions being mirrored.
3. The current oracle behavior and invariants.
4. The new port files/functions involved.
5. Upstream dependencies already completed.
6. Downstream systems likely affected.
7. The conformance tests that will prove parity.
8. The negative/fail-closed tests that will catch unsafe behavior.
9. Whether any legacy helper/API must be removed, renamed, gated, or kept only
   as diagnostic/test-only.

Inspect every directly relevant source file before editing. If a function calls
another helper that controls the same behavior, inspect that helper too.

## Conformance Requirements

Every meaningful port step must include tests that prove:

- The port matches the TypeScript oracle for that layer.
- The runtime path uses the new port behavior, not a mock-only helper.
- Invalid input fails closed.
- Seeded behavior is reproducible.
- State serialization/replay shape does not drift.
- Hidden information is not exposed through player observations.
- Production legal actions exclude debug/manual moves.

For differential validation, compare after every command, not only final winner:

```text
state.G
state.ctx
zones
card metadata
deck order
discard order
Bag
pending effects
replacement state
triggered state
priority
logs
events
legal actions
winner/outcome
```

Tests that only prove code executed are not enough.

## Oracle Freeze

The first real port step is freezing the TypeScript oracle.

Required artifacts:

```text
oracle/source-hash.txt
oracle/card-catalog-hash.txt
oracle/ruleset-hash.txt
oracle/replay-corpus/
oracle/snapshot-schema/
```

Do not trust performance numbers, replay results, or parity claims unless they
state which oracle hash they were compared against.

## Verification Discipline

Run the narrow test suite for the layer being changed plus any affected
differential/conformance tests. Also run formatting or diff checks available for
the touched language.

For current TypeScript oracle checks, use the existing simulator commands from
the relevant package. From the engine package:

```bash
bun test src
```

From the simulator root, install dependencies with the existing filtered pnpm
command when needed:

```bash
pnpm install \
  --filter "@tcg/lorcana-engine..." \
  --filter "@tcg/lorcana-cards..." \
  --ignore-scripts \
  --config.confirmModulesPurge=false
```

Always run:

```bash
git diff --check
```

If a command cannot be run, report it explicitly with the reason.

## Required Final Reporting

For any port implementation phase, report:

- Blueprint step implemented.
- TypeScript oracle files inspected.
- Port files changed.
- Legacy/debug behavior removed, gated, or left test-only.
- Tests added or updated, including what would have failed before.
- Verification commands run and exact result summary.
- Conformance gaps or remaining uncertainty.
- Confirmation whether native TypeScript oracle logic was modified.

Do not claim completion if any required conformance check is missing.

## Nested Instructions

When working inside nested packages, also read their local `AGENTS.md` or
`CLAUDE.md`. Root instructions control the active objective; nested instructions
add package-specific conventions only.
