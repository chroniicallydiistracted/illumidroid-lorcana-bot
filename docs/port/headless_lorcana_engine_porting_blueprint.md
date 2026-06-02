# Dependency-driven headless porting blueprint for `lorcana-simulator`

## 0. Executive verdict

A production-quality headless Lorcana engine is feasible, but the port must follow the engine’s dependency graph, not the file tree.

The correct dependency spine is:

```text
types / card schemas
→ static card resources
→ authoritative match state
→ zones / card instances / metadata
→ deterministic RNG
→ runtime command reducer
→ flow / priority / turn status
→ card runtime queries
→ derived state
→ static effects
→ conditions
→ targeting
→ costs
→ primitive state mutations
→ core moves
→ action effects
→ pending effect resolution
→ triggered abilities
→ Bag resolution
→ replacement/prevention effects
→ full play-card execution
→ activated abilities
→ full card catalog
→ legal-action generation for ML
→ replay/differential validation
```

The required engine logic is concentrated in:

```text
packages/lorcana/lorcana-types
packages/lorcana/lorcana-engine
packages/lorcana/lorcana-cards
```

The UI and presentation app is mostly ignorable:

```text
packages/lorcana/lorcana-simulator
packages/lorcana/lorcana-interaction
runtime-game/lorcanaPacketAnimations.ts
projectBoard / playerView / packet animations
Svelte routes/components
menus/screens/visual tests
```

Do not begin by porting card files. Do not begin with `playCard`. Do not begin with Bag resolution. Those sit near the top of the dependency graph and depend on nearly everything below them.

---

# 1. Repository architecture map

## 1.1 Package-level map

| Package | Role | Headless status |
|---|---|---|
| `packages/lorcana/lorcana-types` | Shared card, ability, condition, cost, effect, target, deck-validation types | **Required** |
| `packages/lorcana/lorcana-engine` | Runtime reducer, game state, rules, moves, targeting, effects, triggers, replay | **Required** |
| `packages/lorcana/lorcana-cards` | Card catalog and card-specific ability/effect definitions | **Required** |
| `packages/shared` | Generic adapter contracts, deck parsing, utilities | Optional / integration |
| `packages/lorcana/lorcana-interaction` | Renderer-agnostic UI interaction/view-model layer | Optional, not core |
| `packages/lorcana/lorcana-server-adapter` | Server adapter around the engine | Optional |
| `packages/lorcana/lorcana-simulator` | Svelte UI simulator app | Ignore for headless core |
| `packages/tools/replay-cli` | Replay tooling | Optional validation tool |

Measured code scale from the attached ZIP:

| Area | Files | Nonblank LOC | Headless relevance |
|---|---:|---:|---|
| `lorcana-engine/src` | 499 TS files | ~114k | Required |
| `lorcana-types/src` | 46 TS files | ~13.5k | Required |
| `lorcana-cards/src` | 8,172 TS files | ~434k | Required |
| `lorcana-interaction/src` | 11 TS files | ~3.3k | Optional |
| `lorcana-server-adapter/src` | 6 TS files | ~654 | Optional |
| `lorcana-simulator/src` | 1,314 TS/Svelte files | ~155k | Ignore except scenario intent/tests |

---

## 1.2 `lorcana-engine/src` architecture

| Folder / file | Purpose | Headless status |
|---|---|---|
| `core/runtime/*` | Generic match runtime: command execution, validation, zones, state IDs, RNG, events, serialization, replay | **Required** |
| `core/engine/*` | Client/server engine wrapper contracts | Optional facade |
| `types/runtime-state.ts` | Lorcana-specific `G` state, card metadata, Bag, triggers, replacement state, pending effects | **Required** |
| `runtime-game/definition.ts` | Lorcana runtime config: setup, board setup, moves, flow, zones, projection hooks | Partially required |
| `runtime-game/lorcanaPacketAnimations.ts` | Packet animations | UI-only |
| `flow/runtime-flow-config.ts` | Segment/phase/step flow, valid move windows, game-end checks | **Required** |
| `zones/runtime-zone-config.ts` | Deck/hand/play/discard/inkwell/limbo config | **Required** |
| `runtime-moves/index.ts` | Move registry | **Required**, excluding debug moves |
| `runtime-moves/moves/setup/*` | Choose first player, mulligan | **Required** |
| `runtime-moves/moves/core/*` | Play card, quest, challenge | **Required** |
| `runtime-moves/moves/resources/*` | Put card into inkwell | **Required** |
| `runtime-moves/moves/abilities/*` | Activate abilities | **Required** |
| `runtime-moves/moves/location/*` | Move character to location | **Required** |
| `runtime-moves/moves/turn/*` | Pass turn, concede, forfeit | **Required** |
| `runtime-moves/resolution/*` | Resolve action effects and Bag entries | **Required** |
| `runtime-moves/effects/*` | Triggered, replacement, delayed, temporary, continuous effect helpers | **Required** |
| `runtime-moves/rules/*` | Cost rules, challenge rules, move rules, static ability helpers | **Required** |
| `runtime-moves/state/*` | Turn metrics, shift stack, lethal damage sweep, game-state checks, derived-card cache | **Required** |
| `rules/*` | Derived state, conditions, static effect registry | **Required** |
| `targeting/*` | Target descriptors, target resolution, availability, target analysis, slotted targets | **Required** |
| `triggered-abilities/index.ts` | Trigger window capture, candidate matching, Bag enqueueing | **Required** |
| `operations/*` | Low-level game mutations | **Required** if used by moves/effects |
| `projection/card-derived.ts` | Runtime-derived card values | **Required** for rules; not UI-only |
| `automation/*` | Built-in bot/planner/strategy logic | Optional reference |
| `testing/*` | Unit harness, test engine helpers | Required for validation, not runtime |
| `debug/manual-moves.ts` | Manual debug state edits | Test-only / ignore in production |
| `i18n/*` | Human-facing logs/text | Optional |

---

## 1.3 `lorcana-types/src` architecture

| Area | Source | Headless status |
|---|---|---|
| Card schemas | `cards/*`, `cards/deck-validation.ts` | Required |
| Ability types | `abilities/ability-types.ts` | Required |
| Cost types | `abilities/cost-types.ts` | Required |
| Condition types | `abilities/condition-types.ts` | Required |
| Effect types | `abilities/effect-types/*` | Required |
| Target types | `abilities/targets/*` | Required |
| Keyword/classification/card-type definitions | multiple type modules | Required |

Important source references:

```text
packages/lorcana/lorcana-types/src/abilities/ability-types.ts
packages/lorcana/lorcana-types/src/abilities/cost-types.ts
packages/lorcana/lorcana-types/src/abilities/condition-types.ts
packages/lorcana/lorcana-types/src/abilities/effect-types/*
packages/lorcana/lorcana-types/src/cards/deck-validation.ts
```

Deck-construction rules exist in:

```text
packages/lorcana/lorcana-types/src/cards/deck-validation.ts
```

Relevant constants:

```text
MIN_DECK_SIZE = 60
MAX_INK_TYPES = 2
MAX_COPIES_PER_CARD = 4
```

---

## 1.4 `lorcana-cards/src` architecture

| Area | Purpose | Headless status |
|---|---|---|
| `cards/001` through `cards/012` | Card definitions by set | Required |
| `cards/index.ts` / set exports | Card catalog export surface | Required |
| `helpers/*` | Card-construction helpers | Required |
| `data/*` | Generated/static card data | Required if consumed |
| card tests | Card behavior examples | Required validation corpus |

Card definitions are executable TypeScript data/code, not plain static JSON.

Examples:

```text
packages/lorcana/lorcana-cards/src/cards/003/locations/204-the-bayou-mysterious-swamp.ts:4-65
packages/lorcana/lorcana-cards/src/cards/004/items/200-fortisphere.ts:4-68
packages/lorcana/lorcana-cards/src/cards/010/characters/124-lady-tremaine-sinister-socialite.ts:37-85
packages/lorcana/lorcana-cards/src/cards/002/characters/172-beast-selfless-protector.ts:33-45
```

These cards use triggered effects, activated abilities, temporary effects, replacement effects, and play-from-zone behavior. The card catalog must be ported after the effect system exists.

---

# 2. True engine logic vs non-headless logic

## 2.1 Required for headless engine

| System | Required source |
|---|---|
| Match state and command reducer | `core/runtime/types.ts`, `match-runtime.commands.ts`, `match-runtime.validation.ts` |
| Lorcana state model | `types/runtime-state.ts` |
| Game setup | `runtime-game/definition.ts:28-74` |
| Flow and phases | `flow/runtime-flow-config.ts` |
| Zones | `zones/runtime-zone-config.ts`, `core/runtime/zone-operations.ts` |
| Card instance registry | `core/runtime/static-resources.ts`, `core/runtime/card-instance-bootstrap.ts` |
| Card runtime query API | `core/runtime/card-runtime.ts` |
| RNG/shuffle | `core/runtime/match-runtime.random-apis.ts`, `zone-operations.ts` |
| Moves | `runtime-moves/index.ts`, `runtime-moves/moves/**` |
| Costs | `runtime-moves/rules/play-card-rules.ts` |
| Challenge rules | `runtime-moves/rules/challenge-rules.ts` |
| Derived state | `rules/derived-state.ts`, `projection/card-derived.ts`, `runtime-moves/state/runtime-card-derived.ts` |
| Static effects | `rules/static-effect-registry.ts` |
| Conditions | `rules/condition-evaluator.ts` |
| Targeting | `targeting/runtime/*`, `targeting/slotted-targets.ts`, `targeting/variants/*` |
| Action effects | `runtime-moves/resolution/action-effects/*` |
| Pending effects | `runtime-moves/resolution/resolve-effect.ts` |
| Bag resolution | `runtime-moves/resolution/resolve-bag.ts` |
| Triggered abilities | `triggered-abilities/index.ts` |
| Replacement/prevention effects | `runtime-moves/effects/replacement-effects.ts` |
| Continuous/temporary/delayed effects | `runtime-moves/effects/*` |
| Turn metadata and state checks | `runtime-moves/state/*` |
| Serialization/replay | `serialization.ts`, `core/runtime/serialization.ts`, `core/runtime/replay.ts` |
| Card definitions | `lorcana-cards/src/cards/**` |

---

## 2.2 Optional but useful

| System | Source | Use |
|---|---|---|
| Test harness | `lorcana-engine/src/testing/*` | Port validation |
| Replay CLI | `packages/tools/replay-cli` | Differential replay validation |
| Built-in automation bot | `lorcana-engine/src/automation/*` | Reference bot behavior only |
| Server adapter | `lorcana-server-adapter/src/*` | Existing server integration |
| Interaction model | `lorcana-interaction/src/*` | Optional action-surface reference |
| Shared deck parser | `packages/shared` | Input tooling |

---

## 2.3 Ignore for production headless engine

| System | Source |
|---|---|
| Svelte UI app | `packages/lorcana/lorcana-simulator/src/routes`, components, stores, visual UI |
| Visual rendering | Svelte files, CSS, assets |
| Packet animations | `runtime-game/lorcanaPacketAnimations.ts` |
| Screen/menu flow | simulator app |
| UI-specific player view projection | `runtime-game/definition.ts:80-109`, unless reused for hidden-information observations |
| Debug manual moves | `debug/manual-moves.ts`, except test-only |
| Human i18n/log formatting | `i18n/*`, unless logs must be human-readable |

---

# 3. Current engine dependency graph

## 3.1 Foundational layer

```text
lorcana-types
core/runtime/types.ts
types/runtime-state.ts
static-resources.ts
card-instance-bootstrap.ts
zones/runtime-zone-config.ts
```

This layer defines what a game is.

Do not port moves before this exists.

---

## 3.2 Runtime layer

```text
match-runtime.commands.ts
match-runtime.validation.ts
match-runtime.framework-api.ts
match-runtime.utils.ts
match-runtime.zone-apis.ts
zone-operations.ts
match-runtime.random-apis.ts
serialization.ts
replay.ts
```

This layer defines how commands enter, mutate, validate, emit events, and produce new states.

---

## 3.3 Lorcana flow layer

```text
runtime-game/definition.ts
flow/runtime-flow-config.ts
runtime-moves/index.ts
zones/runtime-zone-config.ts
```

This layer defines setup, phase progression, valid move windows, current player, priority, game-end checks, and which moves exist.

---

## 3.4 Query/derived rules layer

```text
card-runtime.ts
runtime-card-derived.ts
projection/card-derived.ts
rules/derived-state.ts
rules/static-effect-registry.ts
rules/condition-evaluator.ts
targeting/runtime/*
targeting/slotted-targets.ts
targeting/variants/*
```

This layer defines what cards currently are, what targets exist, what restrictions apply, and what the current legal context is.

---

## 3.5 Move legality layer

```text
runtime-moves/rules/play-card-rules.ts
runtime-moves/rules/challenge-rules.ts
runtime-moves/moves/setup/*
runtime-moves/moves/resources/*
runtime-moves/moves/core/quest.ts
runtime-moves/moves/core/challenge.ts
runtime-moves/moves/core/play-card.ts
runtime-moves/moves/location/*
runtime-moves/moves/abilities/*
runtime-moves/moves/turn/*
```

This layer depends on state, zones, card queries, derived state, targeting, conditions, static effects, and costs.

---

## 3.6 Effect execution layer

```text
runtime-moves/resolution/action-effects/*
runtime-moves/resolution/resolve-effect.ts
runtime-moves/resolution/resolve-bag.ts
runtime-moves/effects/*
triggered-abilities/index.ts
```

This layer depends on almost everything below it.

---

## 3.7 Card catalog layer

```text
lorcana-cards/src/cards/**
lorcana-cards/src/helpers/**
```

This layer should be integrated after the generic effect/move/target/condition systems exist.

---

# 4. Critical current engine systems

## 4.1 State model

Core match state:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:1-8
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:27-30
```

The engine state shape is:

```ts
MatchState<G> = {
  G,
  ctx
}
```

`ctx` contains runtime/system state:

```text
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:42-55
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:61-77
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:83-93
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:99-122
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:285-289
```

It tracks:

- protocol version
- match ID
- game ID
- ruleset hash
- `_stateID`
- player IDs
- zones
- game segment
- phase
- step
- turn
- winner/game-ended status
- priority holder
- pending choice
- random seed/state/draw count
- time control

`G` contains Lorcana-specific rules state:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:561-614
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:626-690
```

It tracks:

- lore totals
- lore-to-win
- turn metadata
- triggered ability state
- pending effects
- pending turn transition
- continuous effects
- temporary player restrictions
- play-from-under permissions
- replacement effects
- current challenge state
- static effect version

Card metadata:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:497-552
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:695-718
```

Card metadata tracks:

- ready/exerted
- damage
- drying
- public face state
- location attachment
- cards under
- Shift stack parent
- cost type used
- temporary keywords
- temporary abilities
- temporary restrictions
- activated ability uses
- replacement abilities
- after-play destination

Porting note: `createDefaultCardMeta` intentionally uses `undefined` for many fields. A headless port must decide whether to preserve the exact JSON shape or normalize it. For differential testing against TypeScript, preserve it first.

---

## 4.2 Zones

Zone configuration:

```text
packages/lorcana/lorcana-engine/src/zones/runtime-zone-config.ts:13-102
```

Zones:

| Zone | Visibility | Ordering | Scope | Face state |
|---|---|---|---|---|
| `deck` | secret | ordered | owner-scoped | face down |
| `hand` | private | unordered | owner-scoped | private |
| `play` | public | unordered | owner-scoped | public |
| `discard` | public | ordered | owner-scoped | public |
| `inkwell` | public | unordered | owner-scoped | face down |
| `limbo` | public | ordered | owner-scoped | public |

Mutation/query implementation:

```text
packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts:15-76
packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts:175-222
packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts:229-278
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.zone-apis.ts:18-99
```

Zone operations are not pure list edits. They also:

- update `cardIndex`
- emit zone events
- invalidate static effects
- update public zone summaries
- preserve deck ordering semantics
- distinguish top and bottom deck access

Deck order conventions are critical:

```text
packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts:298-348
packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts:576-603
```

---

## 4.3 RNG and deterministic shuffle

RNG source:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.random-apis.ts:10-27
```

The runtime uses:

```ts
seedrandom(`${seed}:${draws}`)()
```

Then increments `draws`.

Shuffle uses Fisher-Yates with that RNG.

A headless ML engine must either:

1. reproduce this exactly, or  
2. define a new deterministic RNG and accept that replay parity against the TypeScript engine requires supplying fixed deck orders instead of only a seed.

For port parity, reproduce it exactly.

---

## 4.4 Game setup

Setup and board initialization:

```text
packages/lorcana/lorcana-engine/src/runtime-game/definition.ts:28-74
```

Behavior:

- requires exactly 2 players
- creates initial `G`
- creates player deck zones
- creates deterministic card instance records
- fills private zone card lists
- shuffles each deck
- builds card index
- initializes public zone summaries

The port should separate:

```text
static card catalog
card instance registry
mutable match state
```

Do not store full card definitions inside mutable game state.

---

## 4.5 Flow, phases, and priority

Flow config:

```text
packages/lorcana/lorcana-engine/src/flow/runtime-flow-config.ts:8-269
```

Phase/step definitions:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:40-45
```

Phases:

```text
beginning
main
end
```

Beginning steps:

```text
ready
set
draw
```

Important auto-advance guard:

```text
packages/lorcana/lorcana-engine/src/flow/runtime-flow-config.ts:8-15
```

The beginning phase does not auto-advance when any of these exist:

- pending turn transition
- Bag items
- priority pending choice
- pending effects

Main valid moves:

```text
packages/lorcana/lorcana-engine/src/flow/runtime-flow-config.ts:197-223
```

Includes:

- `playCard`
- `quest`
- `questWithAll`
- `challenge`
- `moveCharacterToLocation`
- `activateAbility`
- `putCardIntoInkwell`
- `passTurn`
- `resolveBag`
- `resolveEffect`
- `concede`
- `forfeitGame`
- debug manual moves

Debug moves must be excluded or feature-gated in the headless production engine.

---

## 4.6 Command reducer

Command execution:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.commands.ts:68-176
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.commands.ts:252-331
```

Validation:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.validation.ts:31-176
```

Command envelope:

```text
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:306-312
```

Validation order:

1. input exists
2. state ID is not stale
3. game has not ended
4. move exists
5. server-only restrictions
6. flow permits move
7. actor is valid
8. priority permits action
9. move-specific validation passes

Execution side effects:

- mutates draft state
- emits move events
- emits domain events
- logs move entries
- increments `_stateID`
- expires reveals
- checks game end
- emits patches if patch capture is enabled

Headless port requirement: make command execution a deterministic pure transition from:

```text
state + command + static resources → new state + events + logs + validation result
```

Even if internally mutable, externally it should behave as a reducer.

---

## 4.7 Runtime framework API

Framework write API:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.framework-api.ts:23-184
```

It exposes mutation surfaces for:

- card metadata
- zones
- time
- random
- events
- undo
- status
- priority
- logs

Hidden dependency: `setMeta` and `patchMeta` invalidate static effects when relevant metadata changes.

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.framework-api.ts:23-86
```

Porting rule: cache invalidation must be explicit. Do not bury it in unrelated setters.

---

## 4.8 Static resources and card instances

Static resources:

```text
packages/lorcana/lorcana-engine/src/core/runtime/static-resources.ts:14-37
packages/lorcana/lorcana-engine/src/core/runtime/static-resources.ts:135-188
```

Card instance bootstrap:

```text
packages/lorcana/lorcana-engine/src/core/runtime/card-instance-bootstrap.ts:4-78
```

The engine separates:

- card definition catalog
- card instance registry
- mutable card metadata
- zone membership

This is the correct headless design. Preserve it.

---

## 4.9 Card runtime query API

Card query runtime:

```text
packages/lorcana/lorcana-engine/src/core/runtime/card-runtime.ts:60-75
packages/lorcana/lorcana-engine/src/core/runtime/card-runtime.ts:106-190
packages/lorcana/lorcana-engine/src/core/runtime/card-runtime.ts:249-329
```

Responsibilities:

- resolve card instance by ID
- resolve definition by public card ID
- get metadata
- query zone
- derive runtime card values
- query target DSL
- cache derived runtime card views by state ID

Many rule systems call this. It must exist before move legality.

---

## 4.10 Derived state

Derived runtime card projection:

```text
packages/lorcana/lorcana-engine/src/projection/card-derived.ts:92-336
packages/lorcana/lorcana-engine/src/runtime-moves/state/runtime-card-derived.ts:30-93
```

Derived values include:

- strength
- willpower
- lore
- move cost
- play cost
- Shift play cost
- classifications
- keywords
- temporary abilities
- temporary restrictions
- granted abilities
- keyword sources
- stat modifier sources

Derived rules:

```text
packages/lorcana/lorcana-engine/src/rules/derived-state.ts
```

Porting rule: do not compute card values ad hoc inside moves. Centralize derived state.

---

## 4.11 Static effects

Static effect registry:

```text
packages/lorcana/lorcana-engine/src/rules/static-effect-registry.ts:1-15
packages/lorcana/lorcana-engine/src/rules/static-effect-registry.ts:46-87
packages/lorcana/lorcana-engine/src/rules/static-effect-registry.ts:202-360
packages/lorcana/lorcana-engine/src/rules/static-effect-registry.ts:512-745
```

Materialized effect kinds include:

```text
modify-stat
stat-floor
damage-source-stat-override
gain-keyword
lose-keyword
grant-classification
grant-ability
grant-abilities-while-here
restriction
cost-reduction
cost-increase
property-modification
```

The registry uses:

- suppression pre-pass
- stat layer pass
- derived-condition reevaluation
- keyword/restriction/grant pass
- by-target indexes
- by-player indexes
- global indexes
- by-source indexes

Hidden coupling exists around the default static registry provider:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.utils.ts:85-99
```

Porting rule: the Rust/headless engine should make the registry instance-owned, not process-global.

---

## 4.12 Conditions

Condition evaluator:

```text
packages/lorcana/lorcana-engine/src/rules/condition-evaluator.ts:28-86
packages/lorcana/lorcana-engine/src/rules/condition-evaluator.ts:121-153
packages/lorcana/lorcana-engine/src/rules/condition-evaluator.ts:443-599
```

Implemented condition families include:

```text
and
or
not
if
if-you-do
turn
during-turn
your-turn
first-turn-non-otp
in-challenge
being-challenged
exerted
is-exerted
has-any-damage
has-no-damage
self-has-damage
damage-comparison
stat-threshold
strength
willpower
lore
resource-count
inkwell-count
has-card-under
has-another-character
has-item-count
has-location-count
has-location-in-play
has-named-character
has-named-item
has-character-count
has-character-with-keyword
has-character-with-strength
has-character-with-classification
opponent-has-damaged-character
turn-metric
used-shift
play-context
target-query
target-aggregate-comparison
revealed-is-card-type
revealed-matches-named
revealed-matches-chosen-name
trigger-subject-had-card-under
discarded-card-has-classification
returned-card-has-classification
returned-card-is-named
returned-card-is-princess
```

Porting rule: conditions must be ported before static effects and card effects are considered complete.

---

## 4.13 Targeting

Target resolver:

```text
packages/lorcana/lorcana-engine/src/targeting/runtime/target-resolver.ts:234-1401
```

Target analysis:

```text
packages/lorcana/lorcana-engine/src/targeting/runtime/target-analysis.ts:105-126
packages/lorcana/lorcana-engine/src/targeting/runtime/target-analysis.ts:1822-1908
```

Target availability:

```text
packages/lorcana/lorcana-engine/src/targeting/runtime/target-availability.ts
```

Slotted targets:

```text
packages/lorcana/lorcana-engine/src/targeting/slotted-targets.ts:15-87
```

Targeting supports:

- owner/controller filters
- zone filters
- type filters
- classification filters
- name filters
- strength/cost/damage filters
- exerted/ready/drying filters
- Ward
- source/self references
- event references
- trigger-subject references
- different-target requirements
- selected targets
- slotted targets
- deferred target choices
- min/max target validation

Porting rule: targeting is a foundational query system, not a UI system. Legal-action generation for ML depends on it.

---

## 4.14 Costs and play-card rules

Play-card cost/rule helpers:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/rules/play-card-rules.ts:419-831
```

Important functions:

```text
getAvailableInk
spendInk
isSongCard
getSingerThreshold
getSingerThresholdForInstance
getSingTogetherThreshold
getShiftRules
resolveShiftTargetCandidates
validateExertCost
validateBasicCost
payBasicCost
```

`playCard` itself is high-level:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/moves/core/play-card.ts:960-2350
```

It depends on:

- zones
- card runtime
- derived state
- static effects
- costs
- targeting
- conditions
- restrictions
- Shift
- Sing
- Sing Together
- alternative costs
- pending effects
- action effect resolution
- trigger emission
- replacement effects
- zone movement

Do not port it before those systems exist.

---

## 4.15 Quest

Quest validation and execution:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/moves/core/quest.ts:78-207
packages/lorcana/lorcana-engine/src/runtime-moves/moves/core/quest.ts:244-312
packages/lorcana/lorcana-engine/src/runtime-moves/moves/core/quest.ts:339-514
```

Quest depends on:

- card in play
- character type
- ready/exerted state
- drying state
- Reckless restrictions
- quest restrictions
- lore gain restrictions
- static effects
- turn metrics
- trigger emission

---

## 4.16 Challenge

Challenge rules:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/rules/challenge-rules.ts:645-912
packages/lorcana/lorcana-engine/src/runtime-moves/rules/challenge-rules.ts:915-1037
```

Important mechanics:

- valid attacker
- valid defender
- exerted defender requirement
- challenge-ready exceptions
- Bodyguard
- Evasive
- drying restrictions
- challenge limits
- damage modifiers
- Resist
- no-damage effects
- challenge damage replacement/prevention
- lethal damage sweep
- banish triggers

Fragile source:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/rules/challenge-rules.ts:915-924
```

There are TODO-style dynamic combat modifier placeholders. Preserve current behavior first; redesign only after differential tests prove the replacement.

---

## 4.17 Action effect resolver

Resolver type list:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/action-effects/composed-effect-resolver.ts:1268-1326
```

Executable action effect families include:

```text
gain-keyword
gain-keywords
modify-stat
sequence
play-card
conditional
draw
optional
gain-lore
restriction
banish
deal-damage
return-to-hand
remove-damage
discard
mill
put-into-inkwell
put-under
enable-play-from-under
pay-cost
put-on-bottom
put-on-top
ready
select-target
scry
for-each
for-each-opponent
return-from-discard
return-random-from-inkwell
exert
choice
or
lose-lore
shuffle-into-deck
reveal
reveal-top-card
reveal-until-match
name-a-card
reveal-hand
reveal-inkwell
search-deck
put-damage
grant-ability
cost-reduction
additional-inkwell
put-in-hand
move-to-location
move-damage
count
move-cards-from-under
draw-until-hand-size
create-triggered-ability
create-replacement-effect
support
property-modification
lose-keyword
reveal-and-route
```

Resolution entry:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/action-effects/composed-effect-resolver.ts:3508-3564
```

Important fragile behavior:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/action-effects/composed-effect-resolver.ts:1388-1460
```

The sequence resolver shares an `eventSnapshot` reference across shallow-copied effects. Preserve or explicitly redesign with tests.

---

## 4.18 Pending effects

Pending effect state:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:430-489
```

Resolver:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/resolve-effect.ts:654-1352
```

Pending effects include:

```text
discard-choice
choice-selection
name-card-selection
optional-selection
scry-selection
target-selection
```

Porting rule: pending effects are not UI prompts. They are part of the authoritative state machine. The ML bot must see them as required decision actions.

---

## 4.19 Triggered abilities and Bag

Triggered state:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:221-371
```

Bag state:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:284-309
```

Triggered ability system:

```text
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts:1503-1628
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts:1661-1688
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts:1880-1923
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts:1925-2017
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts:2083-2195
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts:2259-2375
```

Bag resolution:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/resolve-bag.ts:317-484
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/resolve-bag.ts:771-794
packages/lorcana/lorcana-engine/src/runtime-moves/resolution/resolve-bag.ts:1066-1248
```

Bag resolution depends on:

- trigger candidate collection
- occurrence ledgers
- controller/chooser logic
- pending choices
- action effects
- turn transition continuation
- challenge continuation
- priority holder
- auto-resolve behavior

This is one of the last systems to port.

---

## 4.20 Replacement and prevention effects

Replacement state:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:373-411
```

Replacement implementation:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/effects/replacement-effects.ts:23-87
packages/lorcana/lorcana-engine/src/runtime-moves/effects/replacement-effects.ts:125-207
packages/lorcana/lorcana-engine/src/runtime-moves/effects/replacement-effects.ts:661-813
```

Implemented replacement event kinds:

```text
modify-stat
deal-damage
put-damage
remove-damage
challenge-damage
gain-lore
zone-change
```

Important behavior:

- repeated replacement passes
- `MAX_REPLACEMENT_PASSES = 100`
- printed replacement abilities
- registered replacement effects
- usage ledger
- event mutation
- sibling duplicate handling

This system must be ported after event/effect state exists.

---

# 5. Dependency analysis and hidden coupling

## 5.1 Core dependency table

| System | Depends on | Required before |
|---|---|---|
| `lorcana-types` | none | everything |
| static resources | card definitions/types | setup, card runtime |
| card instances | static resources | zones, setup |
| `MatchState` / `ctx` / `G` | runtime types | all stateful systems |
| zones | state, card instances | setup, moves, targeting, effects |
| RNG | ctx random | setup, shuffle, random effects |
| command reducer | state, move registry, validation | every action |
| flow | state, move registry | legal action windows |
| card runtime | zones, static resources, meta | derived state, targeting, moves |
| derived state | card runtime, static registry | legality, costs, effects |
| static registry | derived state, conditions, card abilities | legality, card projection |
| conditions | state, card runtime, targeting | static/effect/card rules |
| targeting | card runtime, zones, conditions | play/effects/legal actions |
| costs | zones, derived state, static effects | play/activate |
| primitive moves | zones, meta, events | higher moves |
| challenge/quest | costs, restrictions, derived state | triggers/effects/gameplay |
| action effects | targeting, conditions, mutations | playCard, activate, Bag |
| pending effects | action effects, state | player decisions |
| triggered abilities | events, card abilities, targeting, conditions | Bag |
| Bag | triggered abilities, action effects, pending | full game flow |
| replacement effects | events, effect state, card abilities | damage/lore/zone changes |
| playCard | almost all prior systems | full card gameplay |
| card catalog | effect/cost/target/condition schema | full game |
| legal-action generation | validation + targeting + state | ML bot |
| replay | command reducer + serialization | validation/training |

---

## 5.2 Circular or hidden coupling

### Static effect registry singleton

Source:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.utils.ts:85-99
```

The current runtime uses a default static-effect registry provider with process-wide behavior.

Headless redesign:

```text
EngineInstance {
  static_effect_cache
  derived_card_cache
  rng
  state
}
```

Do not make static-effect state global.

---

### Derived-state/static-registry cycle

Source:

```text
packages/lorcana/lorcana-engine/src/rules/derived-state.ts:181-188
packages/lorcana/lorcana-engine/src/rules/static-effect-registry.ts
```

The code avoids a circular runtime import by duplicating lightweight registry accessors.

Headless redesign:

```text
DerivedStateService
StaticEffectService
ConditionService
TargetService
```

Use explicit service interfaces instead of cross-import shortcuts.

---

### Mutation triggers static invalidation

Sources:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.framework-api.ts:23-86
packages/lorcana/lorcana-engine/src/core/runtime/zone-operations.ts:229-249
```

Changing card meta or moving zones invalidates static effects.

Headless redesign:

```text
StateMutationResult {
  changed_zones
  changed_card_meta
  emitted_events
  invalidated_rule_caches
}
```

Cache invalidation must be testable.

---

### Event emission drives triggered abilities

Sources:

```text
packages/lorcana/lorcana-engine/src/core/runtime/types.ts:407-547
packages/lorcana/lorcana-engine/src/triggered-abilities/index.ts
```

Zone movement and damage are not just state changes. They emit events that later produce triggers.

Headless rule:

```text
No mutation without event policy.
```

Every mutation must declare whether it emits a game event.

---

### Debug moves mixed into valid move windows

Source:

```text
packages/lorcana/lorcana-engine/src/flow/runtime-flow-config.ts:79-94
packages/lorcana/lorcana-engine/src/flow/runtime-flow-config.ts:197-223
packages/lorcana/lorcana-engine/src/runtime-moves/index.ts
```

Debug/manual moves are registered alongside normal moves.

Headless redesign:

```text
ProductionMoveRegistry
TestMoveRegistry
DebugMoveRegistry
```

ML training must use only production moves.

---

### `sing` and `singTogether` move entries are unimplemented

Source:

```text
packages/lorcana/lorcana-engine/src/runtime-moves/index.ts:38-53
packages/lorcana/lorcana-engine/src/runtime-moves/index.ts:68-70
```

Standalone `sing` and `singTogether` moves are TODO/unimplemented. Song play modes are handled through `playCard` cost paths.

Headless port:

```text
Do not model Sing as a separate action unless intentionally redesigning.
Model it as playCard with cost mode = sing / singTogether.
```

---

### Dynamic JS object semantics

Sources:

```text
packages/lorcana/lorcana-engine/src/types/runtime-state.ts:695-718
packages/lorcana/lorcana-engine/src/core/runtime/serialization.ts
packages/lorcana/lorcana-engine/src/core/runtime/replay.ts
```

The TypeScript engine relies on object shape, missing fields, `undefined`, JSON stringify cloning, and insertion order.

Headless redesign can normalize later. During parity validation, preserve exact serialization.

---

# 6. Dependency-ordered porting ladder

## Step 0 — Freeze the TypeScript oracle

| Field | Detail |
|---|---|
| Ported concept | Behavioral oracle definition |
| Source | Entire attached ZIP |
| Depends on | none |
| Required by | every validation layer |
| Status | Required |
| Validation before next step | Record commit/source hash, package versions, card catalog hash, ruleset hash, deterministic test corpus |

Deliverables:

```text
oracle/source-hash.txt
oracle/card-catalog-hash.txt
oracle/ruleset-hash.txt
oracle/replay-corpus/
oracle/snapshot-schema/
```

---

## Step 1 — Port primitive type/schema layer

| Field | Detail |
|---|---|
| Ported concept | Card, ability, condition, effect, cost, target, keyword, deck schema types |
| Source | `lorcana-types/src/**` |
| Depends on | Step 0 |
| Required by | all engine systems |
| Status | Required |
| Tests | Schema round-trip tests; card definition load tests; deck-validation tests |

Key files:

```text
lorcana-types/src/abilities/ability-types.ts
lorcana-types/src/abilities/cost-types.ts
lorcana-types/src/abilities/condition-types.ts
lorcana-types/src/abilities/effect-types/*
lorcana-types/src/cards/deck-validation.ts
```

Validation:

```text
same accepted card definitions
same rejected malformed definitions
same deck validation errors
same keyword/effect/condition discriminator names
```

---

## Step 2 — Port static card resources and card instance registry

| Field | Detail |
|---|---|
| Ported concept | Static catalog, instance registry, deterministic card instance IDs |
| Source | `core/runtime/static-resources.ts`, `card-instance-bootstrap.ts` |
| Depends on | Step 1 |
| Required by | setup, zones, card runtime |
| Status | Required |
| Tests | Deterministic instance-ID tests; duplicate card/owner rejection tests |

Key files:

```text
core/runtime/static-resources.ts:14-37
core/runtime/static-resources.ts:135-188
core/runtime/card-instance-bootstrap.ts:4-78
```

---

## Step 3 — Port authoritative state model

| Field | Detail |
|---|---|
| Ported concept | `MatchState`, `TCGCtx`, `LorcanaG`, `LorcanaCardMeta` |
| Source | `core/runtime/types.ts`, `types/runtime-state.ts` |
| Depends on | Steps 1-2 |
| Required by | all runtime systems |
| Status | Required |
| Tests | Initial state snapshot parity; JSON shape parity; card meta default parity |

Key files:

```text
core/runtime/types.ts:42-122
core/runtime/types.ts:285-312
types/runtime-state.ts:40-45
types/runtime-state.ts:113-146
types/runtime-state.ts:221-489
types/runtime-state.ts:497-614
types/runtime-state.ts:626-718
```

---

## Step 4 — Port zone configuration and zone storage

| Field | Detail |
|---|---|
| Ported concept | Zone IDs, visibility, ordering, ownership, card index |
| Source | `zones/runtime-zone-config.ts`, `match-runtime.zone-apis.ts` |
| Depends on | Steps 2-3 |
| Required by | setup, card runtime, moves, targeting |
| Status | Required |
| Tests | Zone creation, owner-scoped zone queries, top/bottom card ordering, card index parity |

Key files:

```text
zones/runtime-zone-config.ts:13-102
core/runtime/match-runtime.zone-apis.ts:18-99
```

---

## Step 5 — Port zone mutation operations

| Field | Detail |
|---|---|
| Ported concept | Move, draw, mill, shuffle, reveal, search, look-at, zone summaries |
| Source | `core/runtime/zone-operations.ts` |
| Depends on | Step 4 |
| Required by | setup, moves, effects, triggers |
| Status | Required |
| Tests | Draw order, mill order, shuffle order, move events, public summaries, reveal expiration |

Key files:

```text
core/runtime/zone-operations.ts:15-76
core/runtime/zone-operations.ts:113-126
core/runtime/zone-operations.ts:175-278
core/runtime/zone-operations.ts:298-348
core/runtime/zone-operations.ts:363-442
core/runtime/zone-operations.ts:468-603
```

---

## Step 6 — Port deterministic RNG

| Field | Detail |
|---|---|
| Ported concept | Runtime random API and shuffle source |
| Source | `match-runtime.random-apis.ts` |
| Depends on | Step 3 |
| Required by | setup shuffle, random effects, deterministic replay |
| Status | Required |
| Tests | `seed:draws` golden values; shuffle golden decks; draw-counter parity |

Key file:

```text
core/runtime/match-runtime.random-apis.ts:10-27
```

---

## Step 7 — Port card runtime query API

| Field | Detail |
|---|---|
| Ported concept | Runtime card lookup, definition lookup, meta lookup, target DSL query shell |
| Source | `card-runtime.ts` |
| Depends on | Steps 2-6 |
| Required by | derived state, targeting, moves |
| Status | Required |
| Tests | Runtime card view parity; zone-filtered card query parity; missing-card errors |

Key files:

```text
core/runtime/card-runtime.ts:60-75
core/runtime/card-runtime.ts:106-190
core/runtime/card-runtime.ts:249-329
```

---

## Step 8 — Port framework read/write context

| Field | Detail |
|---|---|
| Ported concept | Execution context, validation context, lifecycle context, mutation API |
| Source | `match-runtime.utils.ts`, `match-runtime.framework-api.ts` |
| Depends on | Steps 3-7 |
| Required by | command execution, move execution, effects |
| Status | Required |
| Tests | Meta mutation, status mutation, priority mutation, log/event sinks, cache invalidation hooks |

Key files:

```text
core/runtime/match-runtime.utils.ts:137-255
core/runtime/match-runtime.framework-api.ts:23-184
```

Porting change:

```text
Replace global static registry provider with engine-owned cache/service.
```

---

## Step 9 — Port command envelope and validation skeleton

| Field | Detail |
|---|---|
| Ported concept | Command input, stale-state check, actor/priority/move validation skeleton |
| Source | `core/runtime/types.ts`, `match-runtime.validation.ts` |
| Depends on | Steps 3-8 |
| Required by | all move execution |
| Status | Required |
| Tests | stale command, invalid actor, invalid move, server-only move, wrong priority, game-ended rejection |

Key files:

```text
core/runtime/types.ts:306-312
core/runtime/match-runtime.validation.ts:31-176
```

---

## Step 10 — Port runtime reducer

| Field | Detail |
|---|---|
| Ported concept | `executeCommand`, state transition, `_stateID`, events/logs, game-end check hook |
| Source | `match-runtime.commands.ts` |
| Depends on | Step 9 |
| Required by | all gameplay |
| Status | Required |
| Tests | command success/failure parity; state ID increment; event order; game-end event emission |

Key files:

```text
core/runtime/match-runtime.commands.ts:68-176
core/runtime/match-runtime.commands.ts:252-331
```

---

## Step 11 — Port game setup and initial board setup

| Field | Detail |
|---|---|
| Ported concept | Initial `G`, player deck zones, shuffled decks, static resources |
| Source | `runtime-game/definition.ts`, `createInitialLorcanaG` |
| Depends on | Steps 1-10 |
| Required by | all games |
| Status | Required |
| Tests | exact initial state snapshot; deck order parity with seed; two-player validation |

Key files:

```text
runtime-game/definition.ts:28-74
types/runtime-state.ts:626-690
```

Ignore from this file:

```text
runtime-game/definition.ts:80-109
runtime-game/lorcanaPacketAnimations.ts
```

unless building UI/observation projections.

---

## Step 12 — Port flow, phases, priority windows, and game-end checks

| Field | Detail |
|---|---|
| Ported concept | Setup segment, mulligan segment, beginning/main/end phases, valid move windows |
| Source | `flow/runtime-flow-config.ts` |
| Depends on | Steps 9-11 |
| Required by | legal-action generation and command validation |
| Status | Required |
| Tests | phase auto-advance, pending-effect blockers, valid move list by phase, lore win |

Key files:

```text
flow/runtime-flow-config.ts:8-15
flow/runtime-flow-config.ts:24-64
flow/runtime-flow-config.ts:68-78
flow/runtime-flow-config.ts:98-223
flow/runtime-flow-config.ts:249-269
```

Production change:

```text
Remove or gate debug moves from valid move windows.
```

---

## Step 13 — Port serialization and replay snapshots

| Field | Detail |
|---|---|
| Ported concept | Canonical state serialization, authoritative snapshots, replay records |
| Source | `serialization.ts`, `core/runtime/serialization.ts`, `core/runtime/replay.ts` |
| Depends on | Steps 1-12 |
| Required by | deterministic validation and ML training auditability |
| Status | Required |
| Tests | JSON snapshot parity; replay export/import parity; timestamp normalization policy |

Key files:

```text
serialization.ts
core/runtime/serialization.ts
core/runtime/replay.ts
```

---

## Step 14 — Port turn metadata and primitive state mutations

| Field | Detail |
|---|---|
| Ported concept | Turn counters, damage, exert/ready, lore, banish, state-based checks |
| Source | `runtime-moves/state/*`, `operations/*` |
| Depends on | Steps 3-13 |
| Required by | quest, challenge, effects, conditions |
| Status | Required |
| Tests | turn metric parity; damage/lethal sweep; lore gain/loss; exert/ready/drying |

Key files:

```text
runtime-moves/state/turn-metrics.ts
runtime-moves/state/turn-action-ink.ts
runtime-moves/state/shift-stack.ts
runtime-moves/state/lethal-damage-sweep.ts
runtime-moves/state/game-state-check.ts
operations/*
```

---

## Step 15 — Port condition evaluator

| Field | Detail |
|---|---|
| Ported concept | All condition variants and comparison value resolution |
| Source | `rules/condition-evaluator.ts` |
| Depends on | Steps 7, 14 |
| Required by | static effects, targeting, card effects |
| Status | Required |
| Tests | One unit test per condition variant; card-trigger condition parity |

Key files:

```text
rules/condition-evaluator.ts:28-86
rules/condition-evaluator.ts:121-153
rules/condition-evaluator.ts:443-599
```

---

## Step 16 — Port static-effect registry

| Field | Detail |
|---|---|
| Ported concept | Static effect materialization, layering, suppression, indexes |
| Source | `rules/static-effect-registry.ts` |
| Depends on | Steps 7, 14, 15 |
| Required by | derived state, move legality, costs, targeting |
| Status | Required |
| Tests | keyword grants, stat layers, suppression, restrictions, cost modifiers, cache invalidation |

Key files:

```text
rules/static-effect-registry.ts:1-15
rules/static-effect-registry.ts:46-87
rules/static-effect-registry.ts:202-360
rules/static-effect-registry.ts:512-745
```

Design change:

```text
Make registry cache per-engine/per-state, not process-global.
```

---

## Step 17 — Port derived card projection

| Field | Detail |
|---|---|
| Ported concept | Runtime-derived card values |
| Source | `projection/card-derived.ts`, `runtime-card-derived.ts`, `rules/derived-state.ts` |
| Depends on | Steps 15-16 |
| Required by | costs, challenge, quest, targeting, effects |
| Status | Required |
| Tests | strength/willpower/lore/cost/keyword/classification projection parity |

Key files:

```text
projection/card-derived.ts:92-336
runtime-moves/state/runtime-card-derived.ts:30-93
rules/derived-state.ts
```

---

## Step 18 — Port targeting

| Field | Detail |
|---|---|
| Ported concept | Target descriptor normalization, candidate resolution, selection validation, slotted targets |
| Source | `targeting/runtime/*`, `targeting/slotted-targets.ts`, `targeting/variants/*` |
| Depends on | Steps 7, 15, 17 |
| Required by | play-card, effects, activated abilities, legal actions |
| Status | Required |
| Tests | target candidates, Ward, source/self references, trigger-subject references, multi-slot targets, min/max validation |

Key files:

```text
targeting/runtime/target-resolver.ts:234-1401
targeting/runtime/target-analysis.ts:105-126
targeting/runtime/target-analysis.ts:1822-1908
targeting/runtime/target-availability.ts
targeting/slotted-targets.ts:15-87
targeting/variants/*
```

---

## Step 19 — Port play-card cost helpers

| Field | Detail |
|---|---|
| Ported concept | Ink availability, ink payment, exert costs, Shift, Sing, Sing Together |
| Source | `runtime-moves/rules/play-card-rules.ts` |
| Depends on | Steps 14-18 |
| Required by | play-card, activated abilities |
| Status | Required |
| Tests | ink spending, dry/ready singer checks, Shift target candidates, Sing Together thresholds, cost reductions/increases |

Key file:

```text
runtime-moves/rules/play-card-rules.ts:419-831
```

---

## Step 20 — Port setup moves

| Field | Detail |
|---|---|
| Ported concept | Choose first player, mulligan/alter hand |
| Source | `runtime-moves/moves/setup/*` |
| Depends on | Steps 5-13 |
| Required by | game start |
| Status | Required |
| Tests | first-player selection, pending mulligan state, mulligan draw/shuffle parity |

Key files:

```text
runtime-moves/moves/setup/choose-who-goes-first.ts
runtime-moves/moves/setup/alter-hand.ts
```

---

## Step 21 — Port simple resource and core moves

| Field | Detail |
|---|---|
| Ported concept | Put card into inkwell, quest, move to location |
| Source | `runtime-moves/moves/resources/*`, `quest.ts`, `moves/location/*` |
| Depends on | Steps 14-19 |
| Required by | normal gameplay, triggers |
| Status | Required |
| Tests | inkwell limit, quest legality, Reckless prevention, lore gain, location movement |

Key files:

```text
runtime-moves/moves/core/quest.ts:78-514
runtime-moves/moves/resources/*
runtime-moves/moves/location/*
```

---

## Step 22 — Port challenge legality and challenge damage

| Field | Detail |
|---|---|
| Ported concept | Challenge attacker/defender validation and challenge damage resolution |
| Source | `challenge.ts`, `challenge-rules.ts` |
| Depends on | Steps 14-19 |
| Required by | combat, triggers, replacement effects |
| Status | Required |
| Tests | Bodyguard, Evasive, Resist, drying, ready-challenge exceptions, lethal banish, no-damage effects |

Key files:

```text
runtime-moves/moves/core/challenge.ts
runtime-moves/rules/challenge-rules.ts:645-1037
```

Port TODO behavior deliberately:

```text
challenge-rules.ts:915-924
```

Do not “fix” it during the compatibility port unless the oracle changes.

---

## Step 23 — Port action-effect resolver primitives

| Field | Detail |
|---|---|
| Ported concept | Generic effect dispatch and primitive effect families |
| Source | `runtime-moves/resolution/action-effects/*`, `composed-effect-resolver.ts` |
| Depends on | Steps 14-22 |
| Required by | play-card, activated abilities, Bag |
| Status | Required |
| Tests | One direct test per resolver type; effect snapshot parity |

Key files:

```text
runtime-moves/resolution/action-effects/composed-effect-resolver.ts:1268-1326
runtime-moves/resolution/action-effects/composed-effect-resolver.ts:1330-1460
runtime-moves/resolution/action-effects/composed-effect-resolver.ts:3508-3564
runtime-moves/resolution/action-effects/*
```

---

## Step 24 — Port pending effect resolution

| Field | Detail |
|---|---|
| Ported concept | Optional choices, target choices, discard choices, scry, name-card, continuation |
| Source | `types/runtime-state.ts`, `resolve-effect.ts` |
| Depends on | Step 23 |
| Required by | complex card effects, Bag, ML decision API |
| Status | Required |
| Tests | suspend/resume parity; invalid choice rejection; pending-state serialization |

Key files:

```text
types/runtime-state.ts:430-489
runtime-moves/resolution/resolve-effect.ts:654-1352
```

---

## Step 25 — Port triggered ability capture

| Field | Detail |
|---|---|
| Ported concept | Event buffering, trigger candidate collection, trigger windows, occurrence ledgers |
| Source | `triggered-abilities/index.ts`, `types/runtime-state.ts` |
| Depends on | Steps 14-24 |
| Required by | Bag |
| Status | Required |
| Tests | play/draw/discard/quest/challenge/banish/start/end-turn triggers |

Key files:

```text
types/runtime-state.ts:221-371
triggered-abilities/index.ts:1503-2375
```

---

## Step 26 — Port Bag resolution

| Field | Detail |
|---|---|
| Ported concept | Bag chooser/controller logic, Bag item resolution, suspended Bag effects |
| Source | `resolve-bag.ts` |
| Depends on | Steps 23-25 |
| Required by | full gameplay |
| Status | Required |
| Tests | simultaneous triggers, optional Bag effects, Bag during challenge, Bag during turn transition |

Key file:

```text
runtime-moves/resolution/resolve-bag.ts:317-1248
```

---

## Step 27 — Port replacement and prevention effects

| Field | Detail |
|---|---|
| Ported concept | Replacement registrations, event replacement passes, prevention/redirect |
| Source | `replacement-effects.ts`, `types/runtime-state.ts` |
| Depends on | Steps 23-26 |
| Required by | damage, lore, zone-change, challenge correctness |
| Status | Required |
| Tests | repeated replacement pass, damage prevention, redirect damage, zone-change replacement, usage ledger |

Key files:

```text
types/runtime-state.ts:373-411
runtime-moves/effects/replacement-effects.ts:23-813
```

---

## Step 28 — Port full `playCard`

| Field | Detail |
|---|---|
| Ported concept | Standard play, Shift, Sing, Sing Together, alternative costs, targets, destinations, action resolution |
| Source | `play-card.ts` |
| Depends on | Steps 14-27 |
| Required by | full card gameplay |
| Status | Required |
| Tests | every cost mode, every card type, every destination, trigger-on-play, action resolution, failed validation parity |

Key file:

```text
runtime-moves/moves/core/play-card.ts:960-2350
```

Do not separately port unimplemented `sing` / `singTogether` moves unless intentionally redesigning.

---

## Step 29 — Port activated abilities

| Field | Detail |
|---|---|
| Ported concept | Activated ability validation, cost payment, effect execution, usage tracking |
| Source | `runtime-moves/moves/abilities/*` |
| Depends on | Steps 19, 23-28 |
| Required by | items, locations, characters with activated abilities |
| Status | Required |
| Tests | exert/ink/banish/discard costs, once-per-turn usage, ability targeting, effect resolution |

---

## Step 30 — Port pass turn, turn transition, delayed effects, and win conditions

| Field | Detail |
|---|---|
| Ported concept | End turn, start next turn, ready/set/draw, delayed triggers, win/loss |
| Source | `moves/turn/*`, `flow/runtime-flow-config.ts`, `state/game-state-check.ts`, effects helpers |
| Depends on | Steps 12, 14, 25-27 |
| Required by | full games |
| Status | Required |
| Tests | pass blocked by Bag/pending effects, Reckless obligations, start/end triggers, empty deck loss, lore win |

Key files:

```text
runtime-moves/moves/turn/pass-turn.ts
flow/runtime-flow-config.ts
runtime-moves/state/game-state-check.ts
runtime-moves/effects/delayed-triggers.ts
runtime-moves/effects/win-condition-effects.ts
```

---

## Step 31 — Integrate full card catalog

| Field | Detail |
|---|---|
| Ported concept | Card definitions and helper DSL |
| Source | `lorcana-cards/src/cards/**`, `lorcana-cards/src/helpers/**` |
| Depends on | Steps 1-30 |
| Required by | real Lorcana gameplay |
| Status | Required |
| Tests | Load every card; execute every reachable ability/effect; compare card test corpus |

Porting approach:

```text
First: generate/import a normalized card-effect IR.
Second: interpret that IR in the headless engine.
Third: hand-port only cards/helpers that cannot be represented generically.
```

Avoid hand-translating thousands of card files first.

---

## Step 32 — Build ML legal-action generation

| Field | Detail |
|---|---|
| Ported concept | Deterministic legal action enumeration for bot search |
| Source | move validators, targeting, flow valid moves |
| Depends on | Steps 9-31 |
| Required by | ML bot |
| Status | Required for ML |
| Tests | legal action set differential tests against TypeScript from randomized reachable states |

The action generator must enumerate:

```text
chooseWhoGoesFirst
alterHand
putCardIntoInkwell
playCard with cost mode and target/choice args
quest
challenge
moveCharacterToLocation
activateAbility
resolveBag
resolveEffect
passTurn
concede / forfeit if enabled
```

Exclude debug moves.

---

## Step 33 — Port replay and differential validation harness

| Field | Detail |
|---|---|
| Ported concept | Full TypeScript-vs-port replay comparison |
| Source | `core/runtime/replay.ts`, `testing/*`, card tests |
| Depends on | Steps 1-32 |
| Required by | trustworthiness |
| Status | Required |
| Tests | compare after every command, not only final result |

Validation result must compare:

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

---

# 7. Headless engine target design

## 7.1 Minimum public API

```ts
type HeadlessLorcanaEngine = {
  newGame(config: NewGameConfig): GameHandle;
  getState(game: GameHandle): MatchStateSnapshot;
  getObservation(game: GameHandle, playerId: PlayerId): PlayerObservation;
  getLegalActions(game: GameHandle, playerId: PlayerId): LegalAction[];
  validateAction(game: GameHandle, action: GameAction): ValidationResult;
  applyAction(game: GameHandle, action: GameAction): TransitionResult;
  clone(game: GameHandle): GameHandle;
  serialize(game: GameHandle): SerializedGame;
  deserialize(snapshot: SerializedGame): GameHandle;
  exportReplay(game: GameHandle): Replay;
  importReplay(replay: Replay): GameHandle;
}
```

For ML, the critical functions are:

```text
getLegalActions
applyAction
clone
serialize
deserialize
getObservation
```

---

## 7.2 Required state representation

Keep static and mutable state separate.

```text
Static resources:
  card catalog
  card instance registry
  owner map
  initial deck lists

Mutable state:
  ctx
    status
    priority
    zones
    random
    time/no-time
    _stateID

  G
    lore
    turn metadata
    challenge state
    continuous effects
    temporary restrictions
    pending effects
    pending turn transition
    triggered ability state
    Bag
    replacement effects

  card metadata:
    ready/exerted
    damage
    drying
    face state
    location
    cards under
    Shift stack parent
    temp keywords/abilities/restrictions
    activated ability usage
```

For ML training, disable wall-clock time by default:

```text
time.mode = "none"
```

---

## 7.3 Action model

Use explicit action variants.

```text
ChooseWhoGoesFirst
AlterHand
PutCardIntoInkwell
PlayCard
Quest
QuestWithAll
Challenge
MoveCharacterToLocation
ActivateAbility
ResolveBag
ResolveEffect
PassTurn
Concede
ForfeitGame
```

Production headless action model must exclude:

```text
manualMoveCard
manualExertCard
manualReadyCard
manualDryCard
manualSetDamage
manualSetLore
manualShuffleDeck
manualPassTurn
```

Debug actions can exist in a test-only registry.

---

## 7.4 Legal-action generation for ML

Legal action generation must be command-compatible, not UI-compatible.

A legal action should include:

```text
move ID
actor/player ID
card instance ID if applicable
target selections
choice selections
cost mode
cost payment details
ability index/key
Bag item ID
pending effect input
deterministic ordering key
```

Required behavior:

```text
getLegalActions(state, player)
→ returns exactly the commands that validate successfully
```

Validation mode must match the TypeScript engine’s preflight/final distinction.

---

## 7.5 Determinism model

For deterministic ML simulation:

```text
initial decks
+ initial seed
+ same first-player choice
+ same mulligan choices
+ same action sequence
+ same pending decisions
= same state sequence
```

Required deterministic components:

- RNG compatibility
- deck order
- event order
- Bag order
- target candidate order
- legal-action order
- serialization order
- replacement-effect pass order
- static-effect materialization order

---

## 7.6 Replay/log model

Every transition should produce:

```text
previous_state_id
command
validation result
new_state_id
domain events
move log entries
winner/game-ended result
optional canonical snapshot hash
```

For ML, logs do not need human text. They need deterministic structure.

---

# 8. Rules and card-effect inventory

## 8.1 Primitive game concepts

| Concept | Source |
|---|---|
| Match state | `core/runtime/types.ts`, `types/runtime-state.ts` |
| Players | `core/runtime/types.ts` |
| Zones | `zones/runtime-zone-config.ts` |
| Card instances | `static-resources.ts`, `card-instance-bootstrap.ts` |
| Ready/exerted | `LorcanaCardMeta`, zone/meta operations |
| Drying | `LorcanaCardMeta`, beginning phase, quest/challenge/play rules |
| Damage | card meta, damage operations, challenge/effects |
| Lore | `G.lore`, quest/effects/game-state check |
| Inkwell | `zones`, `putCardIntoInkwell`, cost rules |
| Location movement | location move and card meta `atLocationId` |
| Cards under / Shift stack | `shift-stack.ts`, card meta |
| Turn metrics | `TurnMetadata`, `turn-metrics.ts` |

---

## 8.2 Phases and turn flow

| Rule system | Source |
|---|---|
| Choose first player | `moves/setup/choose-who-goes-first.ts` |
| Mulligan | `moves/setup/alter-hand.ts`, `flow/runtime-flow-config.ts:24-64` |
| Beginning phase | `flow/runtime-flow-config.ts:98-195` |
| Ready step behavior | `flow/runtime-flow-config.ts:98-195` |
| Main phase actions | `flow/runtime-flow-config.ts:197-223` |
| End phase | `flow/runtime-flow-config.ts:225-246` |
| Pass turn | `moves/turn/pass-turn.ts` |
| Lore win | `flow/runtime-flow-config.ts:249-269`, `game-state-check.ts` |
| Empty deck loss | turn/draw handling |

---

## 8.3 Keywords and mechanics

Source:

```text
lorcana-types/src/abilities/ability-types.ts
rules/static-effect-registry.ts
projection/card-derived.ts
challenge-rules.ts
play-card-rules.ts
quest.ts
```

Implemented keyword/mechanic families include:

```text
Evasive
Reckless
Rush
Support
Bodyguard
Ward
Challenger
Resist
Singer
Sing Together
Shift
Boost
```

Additional mechanic/effect patterns present in the resolver/type system include:

```text
Vanish / chosen-target recording
play from discard
play from under
cost reduction
cost increase
temporary keyword grant
temporary restriction
replacement effect creation
delayed/floating triggered ability
additional inkwell action
```

---

## 8.4 Action types

| Action | Source | Dependency level |
|---|---|---|
| choose first player | `moves/setup/choose-who-goes-first.ts` | setup |
| alter hand / mulligan | `moves/setup/alter-hand.ts` | setup |
| put card into inkwell | `moves/resources/*` | basic |
| quest | `moves/core/quest.ts` | basic rules |
| challenge | `moves/core/challenge.ts`, `challenge-rules.ts` | combat |
| move to location | `moves/location/*` | board movement |
| play card | `moves/core/play-card.ts` | high |
| activate ability | `moves/abilities/*` | high |
| resolve effect | `resolution/resolve-effect.ts` | high |
| resolve Bag | `resolution/resolve-bag.ts` | high |
| pass turn | `moves/turn/pass-turn.ts` | high |
| concede/forfeit | `moves/turn/*` | basic |
| manual debug moves | `debug/manual-moves.ts`, registry | test-only |

---

## 8.5 Trigger event inventory

Source:

```text
types/runtime-state.ts:221-248
triggered-abilities/index.ts
```

Buffered triggered events include:

```text
play
sing
discard
draw
quest
ready
support
move
challenge
challenged
challenged-and-banished
banish
banish-in-challenge
remove-damage
return-to-hand
ink
start-turn
end-turn
be-chosen
boost
put-card-under
damage
deal-damage
exert
gain-lore
lose-lore
leave-discard
```

---

## 8.6 Replacement event inventory

Source:

```text
types/runtime-state.ts:373-411
runtime-moves/effects/replacement-effects.ts
```

Replacement event kinds:

```text
modify-stat
deal-damage
put-damage
remove-damage
challenge-damage
gain-lore
zone-change
```

---

## 8.7 Action effect inventory

Source:

```text
runtime-moves/resolution/action-effects/composed-effect-resolver.ts:1268-1326
```

Executable effect resolver families:

```text
gain-keyword
gain-keywords
modify-stat
sequence
play-card
conditional
draw
optional
gain-lore
restriction
banish
deal-damage
return-to-hand
remove-damage
discard
mill
put-into-inkwell
put-under
enable-play-from-under
pay-cost
put-on-bottom
put-on-top
ready
select-target
scry
for-each
for-each-opponent
return-from-discard
return-random-from-inkwell
exert
choice
or
lose-lore
shuffle-into-deck
reveal
reveal-top-card
reveal-until-match
name-a-card
reveal-hand
reveal-inkwell
search-deck
put-damage
grant-ability
cost-reduction
additional-inkwell
put-in-hand
move-to-location
move-damage
count
move-cards-from-under
draw-until-hand-size
create-triggered-ability
create-replacement-effect
support
property-modification
lose-keyword
reveal-and-route
```

---

## 8.8 Incomplete, fragile, or UI/debug-dependent implementations

| Issue | Source | Port handling |
|---|---|---|
| Standalone `sing` and `singTogether` moves are unimplemented | `runtime-moves/index.ts:38-53`, `68-70` | Do not port as production moves unless redesigning; preserve `playCard` cost modes |
| Dynamic combat modifier TODOs | `challenge-rules.ts:915-924` | Preserve current behavior first; flag for later correctness review |
| Debug moves registered in valid windows | `flow/runtime-flow-config.ts:79-94`, `197-223` | Exclude/gate in headless production |
| Static-effect registry global provider | `match-runtime.utils.ts:85-99` | Redesign as engine-owned cache |
| Projection/animation hooks in runtime definition | `runtime-game/definition.ts:80-109` | Ignore animations; keep only runtime card derivation if needed |
| Dynamic `unknown` effect payloads | `composed-effect-resolver.ts` and effect types | Use compatibility IR first; strengthen types later |
| Explicit `undefined` metadata defaults | `types/runtime-state.ts:695-718` | Preserve for parity, normalize only after tests |
| Event snapshot shared reference in sequences | `composed-effect-resolver.ts:1388-1460` | Preserve behavior or redesign with dedicated tests |

---

# 9. Porting risk register

| Risk | Severity | Why it breaks | Required mitigation |
|---|---:|---|---|
| RNG drift | Critical | Seed/draw counter and shuffle order affect all games | Golden RNG and shuffle tests |
| Deck top/bottom ordering drift | Critical | Draw, mill, reveal, search, look-at depend on array direction | Zone operation parity tests |
| Static-effect cache drift | Critical | Legal actions and derived card values become wrong | Per-state cache invalidation tests |
| Global registry state | High | Parallel ML games can contaminate each other | Engine-owned registry/cache |
| Pending effect drift | Critical | Choices/resumptions change state sequence | Suspend/resume snapshot tests |
| Bag ordering drift | Critical | Trigger order changes game outcomes | Bag differential tests |
| Replacement pass drift | Critical | Damage/lore/zone-change outcomes change | Replacement event pass tests |
| Targeting drift | Critical | Legal actions and card effects diverge | Target candidate differential tests |
| Cost-mode drift | High | Shift/Sing/SingTogether/free costs diverge | Cost-mode tests |
| Debug move leakage | High | ML bot can learn illegal simulator-only actions | Production move registry |
| Serialization shape drift | High | Replays and differential tests fail | Canonical JSON snapshots |
| Card catalog translation drift | Critical | Card text behavior diverges | Generated IR plus per-card tests |
| UI observation leakage | High | Bot may see hidden info if using full state | Explicit observation API |
| Preflight/final validation mismatch | High | Legal-action generator lies | Validate every generated action |
| Time controls | Medium | Wall-clock state can affect commands | Disable time for ML by default |
| Automation dependency confusion | Medium | Built-in bot is not engine legality | Treat `automation/*` as optional reference |

---

# 10. Validation strategy by porting layer

## 10.1 Foundational tests

| Layer | Required tests |
|---|---|
| Types/schema | Load every card definition; validate every discriminator; reject malformed samples |
| Static resources | Duplicate card IDs, missing owners, deterministic instance IDs |
| State model | Initial state snapshot parity, meta default parity |
| Zones | Move/draw/mill/shuffle/reveal/look-at/search parity |
| RNG | Golden random values and shuffle results |

---

## 10.2 Runtime tests

| Layer | Required tests |
|---|---|
| Command validation | stale state, wrong actor, wrong priority, game ended, invalid move |
| Command execution | `_stateID`, event order, log order, validation errors |
| Flow | setup → mulligan → main game, beginning auto-advance blockers |
| Serialization | canonical snapshot parity and replay round-trip |

---

## 10.3 Rules tests

| Layer | Required tests |
|---|---|
| Conditions | One test per condition variant |
| Static effects | stat layers, keyword grants, restrictions, suppression, cost modifiers |
| Derived state | strength/willpower/lore/cost/classification/keyword projection |
| Targeting | every target variant, Ward, source/self/event/trigger references |
| Costs | ink, exert, Shift, Sing, SingTogether, alternative/free costs |
| Quest | drying, Reckless, lore restriction, trigger emission |
| Challenge | Bodyguard, Evasive, Resist, ready-challenge, no-damage, lethal banish |

---

## 10.4 Effect tests

| Layer | Required tests |
|---|---|
| Action effects | One direct test per `ACTION_EFFECT_RESOLVER_TYPES` entry |
| Pending effects | optional, choice, target, scry, discard, name-card suspend/resume |
| Triggered abilities | play/draw/discard/quest/challenge/banish/start/end trigger windows |
| Bag | simultaneous triggers, chooser order, optional decline, auto-resolve |
| Replacement effects | damage prevention, redirect, zone-change, repeated pass, usage ledger |
| Play card | every cost mode, every card type, every destination, action resolution |
| Activated abilities | cost payment, ability use tracking, target/effect execution |

---

## 10.5 ML-specific validation

Minimum required before trusting for ML training:

```text
1. All deterministic setup/replay tests pass.
2. All existing TypeScript engine/card behavior tests have equivalent passing tests.
3. Every action-effect resolver type has at least one direct parity test.
4. Every condition variant has at least one direct parity test.
5. Every target variant has at least one direct parity test.
6. Legal-action generation is differentially tested against TypeScript on randomized reachable states.
7. Full-game simulations compare state after every command, not only final winner.
8. Production legal-action output excludes debug/manual moves.
9. Observation API proves hidden information is not leaked to the bot.
10. All supported cards load without unsupported effect/condition/target discriminators.
```

---

# 11. Recommended final headless architecture

## 11.1 Engine modules

```text
engine/
  types/
  cards/
  state/
  zones/
  rng/
  runtime/
  flow/
  queries/
  derived/
  static_effects/
  conditions/
  targeting/
  costs/
  mutations/
  moves/
  effects/
  triggers/
  bag/
  replacements/
  serialization/
  replay/
  legal_actions/
  testing/
```

---

## 11.2 Runtime ownership model

```text
HeadlessEngine {
  static_resources
  card_catalog
  match_state
  rule_cache
  derived_cache
  legal_action_cache
  replay_log
}
```

No process-global rule cache.

No UI projection in core state transitions.

No debug moves in production registry.

---

## 11.3 Transition model

```text
apply(command):
  validate generic command
  validate flow window
  validate priority
  validate move-specific rules
  execute mutation transaction
  emit events
  flush relevant trigger windows
  update pending effects / Bag
  check win/loss
  increment state ID
  return transition result
```

---

# 12. Final ordered checklist

1. Freeze TypeScript oracle and card catalog hash.
2. Port `lorcana-types` schemas.
3. Port deck-validation rules.
4. Port static resource and card instance registry.
5. Port `MatchState`, `TCGCtx`, `LorcanaG`, and card metadata defaults.
6. Port zone configuration.
7. Port zone query API.
8. Port zone mutation API.
9. Port RNG and shuffle exactly.
10. Port card runtime query API.
11. Port framework read/write context.
12. Replace global static registry behavior with engine-owned cache hooks.
13. Port command envelope and validation skeleton.
14. Port command reducer.
15. Port game setup and board setup.
16. Port phase/flow config.
17. Exclude or gate debug/manual moves.
18. Port serialization and replay snapshot format.
19. Port primitive mutations and turn metrics.
20. Port condition evaluator.
21. Port static-effect registry.
22. Port derived-card projection.
23. Port targeting resolver, target analysis, target availability, and slotted targets.
24. Port cost helpers: ink, exert, Shift, Sing, Sing Together.
25. Port setup moves: choose first player and mulligan.
26. Port put-card-into-inkwell.
27. Port quest.
28. Port move-character-to-location.
29. Port challenge legality.
30. Port challenge damage and lethal sweep.
31. Port primitive action-effect resolver families.
32. Port pending effect suspension/resumption.
33. Port triggered ability capture.
34. Port Bag resolution.
35. Port replacement/prevention effects.
36. Port full `playCard`.
37. Port activated abilities.
38. Port pass-turn and turn transition.
39. Port delayed/floating effects.
40. Port win/loss conditions.
41. Integrate card catalog through generated/normalized IR.
42. Hand-port only card helpers that cannot be represented in the generic IR.
43. Build production legal-action generator.
44. Build player observation API for hidden-information ML.
45. Build deterministic replay runner.
46. Build TypeScript-vs-port differential harness.
47. Run full unit, card, replay, legal-action, and full-game simulation parity tests.
48. Only after parity, redesign internals for performance.

---

# 13. Selected Rust stack

This section records the selected implementation stack for the headless port.
It does not replace any dependency ordering or conformance requirement above.

Rust is selected because this port is a large, deterministic rules-interpreter
project where exhaustive enums, strict ownership, fast CPU execution, strong
test tooling, and Python bindings matter more than rapid prototyping.

The dominant risks are silent rules drift, RNG drift, ordering drift, hidden
state mutation, and cache invalidation. The stack below is chosen to make those
risks visible and testable.

## 13.1 Toolchain baseline

Pin the Rust toolchain in `rust-toolchain.toml`.

```toml
[toolchain]
channel = "1.96.0"
components = ["clippy", "rustfmt", "llvm-tools-preview"]
```

Required baseline:

```text
Rust version: 1.96.0
Rust edition: 2024
Package manager/build: Cargo workspace
Core runtime: pure Rust library crate
Python bridge: PyO3/maturin binding crate
Async runtime: none in core
Engine RNG: custom TypeScript-oracle-compatible seedrandom implementation
```

`rustc` and `cargo` must be installed through `rustup` before port
implementation begins. If a local environment has no Rust toolchain installed,
that is a setup blocker, not a reason to change the stack.

## 13.2 Workspace layout

Target workspace:

```text
lorcana-rs/
  Cargo.toml
  rust-toolchain.toml
  crates/
    lorcana-schema/
    lorcana-card-ir/
    lorcana-core/
    lorcana-conformance/
    lorcana-py/
    lorcana-cli/
```

Crate responsibilities:

| Crate | Responsibility | Must not contain |
|---|---|---|
| `lorcana-schema` | Rust enums/structs for card, ability, condition, effect, cost, target, trigger, command, state schema | Engine mutation logic |
| `lorcana-card-ir` | Generated/normalized card catalog IR loader and validation | Hand-translated full card logic as the first approach |
| `lorcana-core` | Pure deterministic engine: state, zones, RNG, reducer, flow, queries, derived state, static effects, conditions, targeting, costs, moves, effects, triggers, Bag, replacements, legal actions, observation, serialization, replay | PyO3 types, CLI behavior, TypeScript subprocess control |
| `lorcana-conformance` | TypeScript-oracle runner, lockstep replay comparison, snapshot canonicalization, legal-action diffing, random legal stream generation | Production engine logic |
| `lorcana-py` | Thin PyO3/maturin Python extension exposing coarse batch APIs to the ML stack | Rules behavior not delegated to `lorcana-core` |
| `lorcana-cli` | Oracle freeze, card IR generation, replay, snapshot, diff, conformance, and benchmark commands | Hidden runtime state mutation not available through core APIs |

Keep the port's rules behavior in `lorcana-core`. Other crates wrap, generate,
test, or expose it.

## 13.3 Core dependency choices

Use a small, explicit dependency set. Add new crates only when they improve
correctness, observability, or measured performance without obscuring behavior.

| Purpose | Crate/tool | Why this is selected |
|---|---|---|
| Serialization/schema | `serde`, `serde_json` | The TypeScript card DSL is discriminated-union JSON-like data; Serde can map it to exhaustive Rust enums/structs. |
| Order-preserving maps/sets | `indexmap` | Observable order matters for JSON parity, action ordering, trigger ordering, and diagnostics; standard `HashMap` iteration must not define behavior. |
| Library errors | `thiserror` | Stable, typed error enums for fail-closed engine behavior. |
| CLI/test harness errors | `anyhow` | Good for tools where preserving context matters more than exposing a typed API. Do not use it inside core rules where exact error classification matters. |
| Small ordered vectors | `smallvec` | Useful for small target/action/effect lists without heap churn; preserve order explicitly. |
| Bit flags | `bitflags` | Useful for internal keyword/classification flags only when parity does not require raw string-list representation. |
| Diagnostics | `tracing` | Structured conformance mismatch and replay diagnostics. |
| CLI | `clap` | Stable command surface for oracle freeze, replay, diff, and benchmarks. |
| Parallel independent work | `rayon` | Use only above the deterministic core for independent games, conformance streams, batch observation, or benchmark workloads. |
| Python extension | `pyo3` | Native Python module bridge for Rust code. Keep it out of `lorcana-core`. |
| Python package/build | `maturin` | Build and install PyO3 modules into the Python environment. |
| Dense tensor export | `numpy` | Optional; use only for observation tensors where NumPy transfer is needed. |
| Property tests | `proptest` | Randomized invariant and legal-action stream tests with shrinking. |
| Fuzzing | `cargo-fuzz`, `arbitrary` | Structured fuzzing for malformed commands, serialized states, action streams, and parser/generator robustness. |
| Snapshot tests | `insta` | Canonical state/replay/schema snapshots. |
| Fast test runner | `cargo nextest` | Faster CI/local test execution and better test reporting than raw `cargo test`. |
| Benchmarks | `criterion` | Stable microbenchmarks for clone/apply/legal-action/observation after parity. |
| Coverage | `cargo llvm-cov` | Rust coverage reports for core and conformance suites. |
| Dependency audit | `cargo deny` | License/security/dependency policy gate. |
| Lint/format | `clippy`, `rustfmt` | Required quality gates. |

## 13.4 Python and ML integration

Use `lorcana-py` as a thin binding crate.

Required rules:

```text
PyO3 types do not appear in lorcana-core.
Python bridge functions call lorcana-core APIs.
Bindings expose coarse batch APIs, not one Python call per primitive transition.
The binding layer may release the GIL only around Rust-only work.
```

Preferred Python-facing API shape:

```text
new_game_batch(configs) -> handles
clone_batch(handles) -> handles
legal_actions_batch(handles, player_ids) -> encoded action batches
apply_actions_batch(handles, actions) -> transition results
observations_batch(handles, player_ids) -> compact observations / optional NumPy tensors
serialize_batch(handles) -> bytes or JSON snapshots
load_batch(snapshots) -> handles
```

Do not bind every internal helper. Binding too many helpers creates a second API
surface that can drift from the conformance-tested engine path.

## 13.5 Determinism and ordering rules

The TypeScript engine source currently uses:

```text
packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.random-apis.ts
seedrandom(`${seed}:${draws}`)()
```

with dependency:

```text
seedrandom@3.0.5
```

Therefore:

```text
Do not use rand::StdRng as the engine RNG.
Do not use rand_chacha as the engine RNG.
Do not use OS randomness in engine transitions.
Do not use HashMap or HashSet iteration order for rules-visible behavior.
```

Required implementation:

```text
Implement a custom seedrandom-compatible RNG module in lorcana-core.
Export golden TypeScript RNG vectors during Step 0.
Test random() output and shuffle order against the TypeScript oracle.
Track ctx.random.seed and ctx.random.draws exactly.
```

Allowed uses of Rust RNG crates:

```text
rand / rand_chacha may generate test cases, fuzz seeds, or randomized legal
conformance streams. They must never replace the engine's oracle RNG.
```

Use `Vec`, `IndexMap`, or `BTreeMap` for observed iteration order:

```text
Vec       = explicit sequence/order from engine logic
IndexMap  = insertion order matters
BTreeMap  = stable key order matters
HashMap   = only for lookup where iteration order cannot affect behavior
```

## 13.6 State strategy

Start with a simple, correctness-first reducer:

```text
apply_command(state, command, static_resources) -> TransitionResult
```

with owned Rust structs and explicit `Clone`.

Do not begin with:

```text
Persistent data structures as the primary design
Unsafe arena aliasing
Undo-log simulation as the only source of truth
Parallel mutation inside one transition
```

After layer parity is proven, add a separate optimized simulation lane:

```text
EngineLane {
  state
  undo_log
  caches
}
```

The lane may optimize MCTS apply/undo, but it must be verified against the
conformance-tested reducer.

## 13.7 Testing and CI commands

Baseline Rust checks once the workspace exists:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --workspace
cargo test --doc --workspace
cargo llvm-cov nextest --workspace
cargo deny check
git diff --check
```

Conformance checks must compare the Rust port against the frozen TypeScript
oracle after every command, not only at final game outcome.

Performance checks are allowed only after the relevant parity checks pass:

```bash
cargo bench --workspace
```

## 13.8 Explicit non-goals for the Rust stack

Do not use:

```text
tokio or async runtimes in lorcana-core
C ABI / cffi as the primary Python bridge
rand::StdRng for engine transitions
HashMap iteration for rules behavior
global static-effect or derived-state caches
hand-translated full card catalog as the first card integration strategy
performance-only rewrites before parity
```

---

# 14. Symbol registry requirement

Maintain this document throughout the port:

```text
docs/port/headless_lorcana_engine_porting_symbol_registry.md
```

The registry exists for AI developer agents that resume after context
compaction. It must make the current names, constants, variables, functions,
classes/structs, enums, modules, crates, generated artifacts, source paths, and
test fixtures discoverable without rereading the entire repository.

## 14.1 When to update it

Update the registry in the same change whenever any port work:

```text
creates a crate/module/file
creates or renames a constant
creates or renames a variable with cross-module meaning
creates or renames an enum, struct, class, trait, or type alias
creates or renames a function/method used outside one file
creates or renames a command/action variant
creates or renames a serialized field
creates or renames a test fixture, oracle fixture, snapshot, or generated artifact
changes the meaning of an existing symbol
removes or deprecates a symbol
adds a TypeScript oracle source path that future agents must know
```

If a future agent would need the name to continue safely after losing context,
the registry must include it.

## 14.2 Required organization

The registry must stay structured with these sections:

```text
1. How to use this registry
2. Source-of-truth documents and oracle paths
3. Rust workspace crates and module map
4. Toolchain and dependency constants
5. TypeScript oracle constants and source symbols
6. Rust port constants
7. Runtime state structs/classes/types
8. Command/action variants
9. Engine API functions and methods
10. Serialization, replay, and snapshot artifacts
11. Test fixtures and conformance corpora
12. Generated artifacts
13. Deprecated or quarantined symbols
14. Update log
```

Use tables with:

```text
Symbol / Name
Kind
Location
Purpose
Oracle source
Parity notes
Introduced / updated date
```

## 14.3 Registry quality bar

The registry must be:

```text
searchable by exact symbol name
organized by subsystem
explicit about TypeScript oracle source paths
explicit about whether a symbol is planned, implemented, deprecated, or test-only
kept small enough to scan, but complete enough to continue development
```

Do not use the registry for prose design discussion. Put design discussion in
the blueprint; put stable names and symbol facts in the registry.
