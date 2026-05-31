# Lorcana-engine port-complexity audit (TS → Rust)

Scan date 2026-05-31. Question: how hard is a faithful native port, where does the
risk live, and what's easy vs. doable vs. high-complexity. All figures measured
from the source.

## Headline finding: cards are DATA, the engine is a DSL interpreter

Card definitions are **100% declarative data** — only **1** non-test card file
contains a closure (`=>`); the other ~2,750 are pure objects. An ability is a
nested discriminated union, e.g.:

```ts
{ type: "triggered", trigger: {...},
  effect: { type: "optional", chooser: "CONTROLLER",
            effect: { type: "move-damage", amount: 1,
                      from: { selector: "chosen", filter: [{type:"damaged"}], ... },
                      to:   { selector: "chosen", owner: "opponent", ... } } } }
```

**Implication that dominates everything:** you do **not** port 2,754 cards. You
port the **interpreter** that reads the effect DSL, and the card data ships as
JSON (language-agnostic, zero logic risk). The DSL has **204 distinct node
types** (effects + selectors + conditions + triggers) — that is the real surface.

## Size of the actual port surface (measured, non-test)

| Module | LOC | Needed? |
|---|---:|---|
| runtime-moves (moves + effect resolution) | 33,475 | **yes — core** |
| core (runtime: reducer, zones, RNG, validation) | 14,303 | **yes** |
| targeting (selectors/filters) | 5,271 | **yes** |
| rules (static effects, combat, keywords) | 4,955 | **yes** |
| runtime-game (phases, move registry) | 3,257 | **yes** |
| types (schema) | 3,134 | yes (→ Rust enums/structs) |
| triggered-abilities (the bag) | 2,375 (1 file) | **yes — hardest** |
| operations / flow / zones / projection | ~1,300 | yes |
| automation (scripted AI) | 11,668 | **NO — MCTS replaces it** |
| testing / i18n / support-probe | ~4,600 | no (but testing = the oracle) |

**Port surface ≈ 65k LOC of interpreter/runtime logic** (cards excluded — they're
data). The ~12k LOC of `automation` is *not* needed; our search replaces it.

## Language differences (TS → Rust)

**Helps the port (good fit):**
- The effect DSL is discriminated unions → **Rust enums (sum types)** are an
  almost 1:1 match, and the compiler forces you to handle every variant
  (exhaustive `match`) — this turns "missed a case" from a silent bug into a
  compile error. Big correctness win.
- Code is **well-typed**: only **18 `: any`** in the whole engine. The 3,395
  `as` casts are mostly TS working around its own type system; they largely
  vanish with real Rust types.
- Stats are **small integers** (cost/strength/willpower/lore/damage) → i32; no
  floating-point semantics to worry about.
- String-keyed zones (`"hand:player_one"`) → `HashMap`, trivial.

**Fights the port (real friction):**
- **Immutable state + structural sharing.** The engine is a Mutative reducer:
  every move returns a new immutable state + inverse patches (cheap branching).
  Rust has no Mutative; you must choose a strategy — persistent data structures
  (`im`/`rpds`), or an arena + undo-log, or copy-on-write. Architectural, then
  mechanical, but it touches everything.
- **RNG determinism.** `seedrandom` (2 files) must be replicated **bit-for-bit**
  (its Alea PRNG) or conformance fails on every shuffle/draw. Bounded but exact.
- **Closures as control flow.** 3,970 arrow functions — most are local helpers
  (fine), but the resolution pipeline passes effect-handler callbacks/sinks;
  those become trait objects / enums in Rust. Manageable, not mechanical.
- **Lazy/derived state** (static-effect version invalidation, cached
  enumerations) — must be reproduced or redesigned.

## Complexity tiers

### Easy (mechanical, low risk)
- **Card data → JSON** loaded by Rust. No logic. (2,754 cards.)
- **Schema → Rust enums/structs** (the 204 DSL node types as enum variants).
  Tedious typing, compiler-verified.
- Zones, turn/phase/step counters, lore tracking, integer stats, simple queries.

### Doable (real work, bounded, per-piece testable)
- **The ~74 leaf effect handlers** (deal-damage, draw, banish, modify-stat,
  gain-lore, exert/ready, discard, return-to-hand, scry, put-into-inkwell…).
  Each is a scoped state transform with an existing per-effect test suite to
  check against. This is the bulk of runtime-moves — tedious but tractable.
- **Targeting / selectors** (5.3k LOC): `chosen`/filters/`up-to`/count — a query
  language over the board; finite set of selector + filter + condition types.
- **The 12 move categories** (play/quest/challenge/ink/sing/shift/activate/
  move-to-location/pass…): legality + application.
- **RNG replication**, the reducer skeleton, and the chosen immutable/undo model.

### High complexity (the silent-divergence hotspots)
- **The triggered-ability "bag"** (2,375 LOC, one file): event detection,
  ordering of simultaneous triggers, `when/whenever/at-start` windows, chooser
  resolution, sequence numbers. Subtle ordering — wrong order doesn't crash, it
  mis-trains.
- **Effect *composition*:** `optional`, `or`, `up-to`, `sequence`, `conditional`,
  `if-you-do`, `once-per-turn`, modal choices — effects that wrap/sequence other
  effects with mid-resolution player decisions. (535 `optional`, 397 `sequence`,
  111 `conditional`, 50 `if-you-do`…)
- **Continuous / static effects** (467 `static`, 363 `modify-stat`, 281
  `gain-keyword`): derived stat/keyword recomputation with **layering** (apply
  order) and duration — the classic source of silent rules bugs.
- **Keyword interactions** (Ward/Bodyguard/Evasive/Resist/Challenger/Rush/Shift/
  Singer/Reckless/Support/Vanish) × combat × targeting — combinatorial.
- **The conformance harness itself** (meta): the *only* way to prove parity is to
  run both engines on the **205 test suites + millions of random legal action
  streams** and assert identical `MatchState` and legal-move sets at every step,
  re-run per set release. This is a project in its own right and is mandatory —
  without it a port is untrustworthy.

## Logic-fallout risks (why "it compiles" ≠ "it's correct")
- **Silent divergence**: one of 204 node types × keyword interactions resolving
  subtly wrong trains the bot on *almost-Lorcana* with no error surfaced.
- **Ordering/layering**: trigger order and static-effect layering are
  semantically load-bearing and under-specified outside the code itself.
- **RNG drift**: any deviation in the PRNG or draw order desyncs every game.
- **Per-set churn**: Lorcana ships sets/errata; the port must re-sync forever.

## Bottom line
- **Not "port 2,754 cards"** — port a ~65k-LOC **DSL interpreter + runtime**,
  load cards as JSON. That's the reframe that makes it conceivable at all.
- **The data model ports cleanly** (unions→enums, well-typed, integer math); the
  **dynamic runtime** (immutable/patches, RNG, triggered-bag, static layering)
  is the hard ~10–15k LOC where divergence hides.
- **Effort (solo):** ~6–18 months to faithful parity, gated less by typing the
  handlers than by the bag + static layering + the mandatory conformance harness.
- **Payoff if completed:** ~20–200× on the ~11 ms/move resolution → genuinely
  GPU-bound, championship-depth search. Real, large — but the risk is completion
  and trust, not feasibility.

**Recommendation:** if attempted, build it **interpreter-first, data-as-JSON,
conformance-harness from commit one**, and port effect handlers in priority order
(most-frequent node types first: triggered/optional/static/sequence/modify-stat/
draw cover the majority of cards). Treat the triggered-bag and static layering as
the make-or-break modules. Otherwise, the finishable path remains: keep the TS
engine, maximize strength-per-sim (bigger net + leaf-batching), parallel actors.

---

# Deep dive: the high-complexity subsystems (where divergence hides)

## 1. The triggered-ability "bag" — `triggered-abilities/index.ts` (2,375 LOC, one file)
**State:** `{ pendingEvents[], registrations[] (floating + delayed), bag:{nextSeq, items[]}, usageLedger:{occurrences, resolutions} }`.
**Windows:** `challenge-declaration | after-challenge | start-of-turn | end-of-turn`, plus `when/whenever/at-start` timings.
**The crux — `finalizeResolutionBoundary()`:** for each pending event it scans printed-board + floating candidates, matches them, then runs a gauntlet of rules-specific filters before enqueuing:
- composite-trigger expansion (`leave-play` → `banish` + `banish-in-challenge`, deduped so it fires once)
- discard-batch and sing-batch dedup
- `n-times-per-turn` skip (via `usageLedger`)
- skip-if-no-valid-targets
- it cites the **official Comprehensive Rules (CRD 6.2.7)**: conditions are evaluated at *resolution* time, except `turn-metric` conditions which are pre-fizzled to avoid pointless prompts.

**Ordering:** `nextSeq` sequence numbers + `getNextBagResolver` (turn player resolves their triggers first, in an order they choose; non-turn player after) + priority handoff *during* bag resolution.
**Lifecycle:** floating (`startsAtTurn`/`expiresAtTurn`) and delayed (start/end-of-turn) registrations, pruned on expiry.
**Why it's the #1 divergence risk:** every dedup/skip/ordering rule is load-bearing and rules-citation-specific, and getting trigger *order* wrong **doesn't crash** — it silently changes outcomes. This is the single hardest module to port faithfully.

## 2. Effect composition — a *resumable coroutine*, not a function
`composed-effect-resolver.ts` (+ `conditional-effect`, `resolve-effect`, `resolve-bag`). Effect resolution is a **suspendable state machine**: an effect runs until it needs player input (choose target / `optional` yes-no / `or` modal / `up-to N`), then **suspends** — serializing its resolution state — and **continues** when input arrives, via `continuation` / `stagedSequence`. The code explicitly guards against re-suspend/replay loops.
- `sequence` shares an `eventSnapshot` across steps, consumes targets per step, and has `select-target` steps interleaved.
- `if-you-do` must detect whether the *prior* effect actually resolved; `conditional` is checked at resolution time (CRD 6.2.7).
**Why it's hard to port:** you must reproduce a **serializable, replay-safe, resumable** resolution state that interleaves with the bag and with mid-resolution player choices. This is the gnarliest *control flow* in the engine — far harder than the leaf handlers.

## 3. Continuous / static effects + conditions — ~4,400 LOC, implicit layering
`derived-state.ts` (1,397) + `static-effect-registry.ts` (1,482) + `condition-evaluator.ts` (1,558).
Derived stats (strength/willpower/lore/keywords) are recomputed by applying all active static effects (`modify-stat`, `gain-keyword`, `restriction`) filtered per player/kind, with conditions evaluated dynamically.
**Key risk — layering is *implicit*:** there is no explicit numbered layer system (à la MTG). The application/accumulation order is an **emergent property of registry-iteration order**. A port must reproduce that emergent order *exactly* or derived stats diverge — and a wrong strength/willpower silently flips combat math. Plus the full 1,558-LOC condition vocabulary (`strength-comparison`, `cost-comparison`, `filtered-count`, `turn-metric`, `has-classification`…) must match, and the `staticEffectsVersion` lazy-invalidation timing.

## 4. Keywords × combat × targeting — combinatorial, non-localized
Keywords are woven through many files (not localized): **Shift 20, Support 16, Rush 13, Evasive 12, Reckless 11, Bodyguard 9, Resist/Ward/Challenger/Singer 6–7, Vanish 4**. They touch legality, targeting, combat (`challenge-rules.ts`, `moves/core/challenge.ts`), continuous-effects, and derived-state simultaneously:
- Resist (−damage), Challenger (+strength attacking), Bodyguard (forced targeting), Evasive (only Evasive challenges Evasive), Ward (can't be *chosen*), Rush (challenge while ink-wet), Reckless (must challenge / can't quest), banish thresholds.
- Interactions are where bugs cluster — e.g. there's a regression test `bug-2-ward-chosen-target` specifically about **granted-via-static** Ward correctly filtering a "chosen" target. A port must get keyword × static-grant × targeting interactions right, combinatorially.

## 5. The conformance harness — mandatory, a sub-project of its own
The oracle is the in-repo **205 test suites** + per-keyword e2e (shift/evasive/ward/bodyguard/support/challenger-resist/reckless-rush/vanish/boost). A trustworthy port needs a **lockstep differential harness**:
- feed BOTH engines identical command streams — the 205 suites **and** millions of random *legal* action streams,
- canonicalize state (exclude envelope metadata — Phase 0 already built this fingerprint),
- assert identical `MatchState` **and** `getAvailableMoves` at **every** step,
- re-run on every engine commit **and** every Lorcana set release (rules are versioned).
Without this you cannot know which of the 204 node types × keyword interactions diverged. It is multi-week, non-optional, and the thing that actually makes a port *provable*.

## Net assessment of the hard tier
The leaf handlers are tedious-but-safe. The danger concentrates in **~10–15k LOC**: the **bag** (ordering/dedup), the **resumable composition machine** (suspend/continue), and **static-effect layering** (implicit order) — all three are *silent* on error and only catchable by the conformance harness. Budget the majority of port time and risk here, not on the 74 handlers.
