# Tier A Belief-Search Correctness Remediation Plan

## Purpose

Create a dependency-ordered remediation plan and implementation sequence to bring **Tier A — belief-search correctness** to true 100% end-to-end completion for `illumidroid-lorcana-bot`.

The current Tier A implementation is not complete. Although isolated sampler/search tests pass, active training paths still produce contaminated or incomplete search labels. This must be treated as a correctness remediation, not a compatibility-preserving patch. Legacy logic that conflicts with belief-search correctness must be replaced, removed from active training, or explicitly quarantined as non-training diagnostic code.

The goal is to make clean-label training safe by proving that:

1. Hidden-zone determinization does not leak opponent inkwell membership.
2. Full hidden-zone `World` specs are honored end-to-end.
3. Full ISMCTS uses one shared information-set tree, not per-world PIMC trees.
4. Neural leaf evaluation sees lane-local simulated public/perfect-recall history.
5. Perfect-recall keys preserve all action-relevant visible information.
6. Own hidden actions remain known to the acting player, while opponent hidden actions remain redacted.
7. Observed opponent actions update belief trackers during real self-play.
8. Sampler importance weights are mathematically correct and auditable.
9. Runtime diagnostics reflect the actual populated search tree.
10. Reproducibility does not fall back to unseeded `Math.random`.
11. All tests prove runtime behavior, not only isolated helper behavior.
12. No native TheCardGoats `lorcana-simulator` GameEngine source is modified.

Permitted TypeScript changes are limited to bot-owned bridge/server code, especially:

```text
lorcana-bot/engine/node_server/server.ts
```

Do not modify the underlying TheCardGoats GameEngine packages.

Clean-label training must remain blocked until every Tier A remediation phase below is implemented, tested, and verified.

---

# Current repo facts confirmed

The repository currently contains both a legacy `BeliefEvaluator` and a structured evaluator. The legacy evaluator samples only opponent hand plus deck IDs, excluding opponent hidden inkwell IDs, while `StructuredBeliefEvaluator` already exposes hand and inkwell probability channels over hand, deck, and inkwell pools.

Phase 0 now blocks active belief-guided sample writing before evaluator construction in `training/run.py`, `training/parallel.py`, `training/selfplay.py`, league training, and exploiter training. The legacy pooled search remains available only as `run_pimc_diagnostic()`.

`World` already has rich fields for opponent hand, opponent inkwell, opponent deck, self deck, seed, target/proposal logs, and rho, but active bridge paths still pass only hand IDs through determinization.

`EngineSimulator.begin_lane()` restores the snapshot and calls `engine.determinize(self_id, list(world.opponent_hand_ids), seed=...)`, ignoring sampled opponent inkwell, opponent deck, and self deck fields.

The Python bridge exposes `determinize(self_id, hand_instance_ids, seed)` only, and the TypeScript server RPC likewise accepts only `handInstanceIds`; this makes full-zone `World` specs impossible to honor through the current bridge contract.

The TypeScript wrapper’s determinizer repartitions opponent inkwell/deck from the remaining hidden pool and shuffles self deck, but it uses `Math.random` when no seed is supplied.

`sample_worlds()` still has the audited defects: zero-card/zero-hand returns empty default `World` objects, structured joint uniform currently sets `log_target == log_proposal`, and `rho` is rebased before storage even though the docstring says raw `b/q`.

`BeliefTracker` currently tracks hand-only particles as `frozenset`s, returns hand-only `World` objects, and persistent reweighting multiplies only included-card probabilities while omitting excluded-card likelihood terms.

`run_infoset()` calls the shared-tree implementation but discards its populated shared root and returns a newly created legacy `InfoSetNode`, which explains zero-visit diagnostics.

`history_event()` only stores actor seat, action family, and one card ID, while `candidateKey()` encodes the full action identity including targets, choices, costs, destinations, and mulligan selections.

`stepExact()` intentionally does not mutate server public history, so full ISMCTS must supply lane-local projected branch history to both the information-set key and neural leaf encoding.

The current root-invariance test samples only hand plus deck, not inkwell, so it cannot catch the inkwell leak.

---

# Layout confirmation and execution rules

This document is organized correctly for implementation work:

```text
purpose and safety boundary
  -> confirmed repository facts
  -> dependency-ordered remediation phases
  -> exact commit sequence
  -> final verification gate
```

The phases below describe logical ownership boundaries. The commit sequence near the end of this document is the authoritative implementation order. A phase may be split into smaller commits, but a downstream phase must not be activated before its prerequisites pass.

## Dependency graph

```text
Phase 0  fail-closed training guards
  |
  +--> Phase 13 baseline red regression probes
         |
         +--> Phase 1  canonical full World contract + deterministic seed contract
                |
                +--> Phase 2  sampler math fixes
                |      |
                |      +--> Phase 5  full-world tracker + persistent evidence transitions
                |
                +--> Phase 3  full-world bridge/server RPC
                       |
                       +--> Phase 7  EngineSimulator passes full worlds

Phase 8  complete actor-filtered info-set projection
  |
  +--> Phase 9  split lane-local public neural history from private perfect recall

Phase 5 + Phase 8
  |
  +--> Phase 10 observed-action correction in real self-play

Phase 2 + Phase 4 + Phase 5 + Phase 7 + Phase 9 + Phase 10 + Phase 14
  |
  +--> Phase 6 activate full ISMCTS as the only clean-label belief-search path

Phase 11 diagnostics + Phase 12 reproducibility verification
  |
  +--> final Tier A clean-label release gate
```

## Non-negotiable sequencing constraints

1. Add regression probes that fail on the audited implementation before changing production behavior.
2. Treat `World.seed` as part of the canonical `World` contract in Phase 1. Phase 12 verifies reproducibility end-to-end; it must not introduce seeding for the first time.
3. Do not activate clean-label `run_infoset()` training until `EngineSimulator.begin_lane()` honors complete `World` objects.
4. Keep neural public history and private perfect-recall history as separate projections. Do not expose own or opponent private recall events through a public-history field by accident.
5. Preserve observed-action evidence across tracker transitions or reseeds. Incrementing `action_updates` is not sufficient if the posterior update is discarded on the next decision.
6. Keep legacy PIMC available only under an explicitly diagnostic API that cannot write training samples.

---

# Dependency-ordered Tier A remediation plan

## Phase 0 — freeze unsafe clean-label training

### Status

Completed. Belief-guided sample-writing entrypoints fail closed through
`training/tier_a_guard.py`. Legacy root-pooled PIMC was renamed to
`run_pimc_diagnostic()` and is not called by self-play sample-writing loops.

### Problem

The active training paths can still generate contaminated labels. The code has partial Tier A components, but the live path still routes through legacy belief sampling and incomplete world determinization.

### Required remediation

Add a hard project rule:

```text
No clean-label self-play training is valid until all Tier A remediation tests pass.
```

In code, make the training entrypoints fail closed if they attempt to run belief-guided clean-label training through legacy PIMC or hand-only belief.

### Files/functions

```text
lorcana-bot/training/run.py
  selfplay_game()

lorcana-bot/training/parallel.py
  _worker()

lorcana-bot/search/ismcts.py
  run_belief()
  run_infoset()
```

### Required implementation behavior

For clean-label training:

```text
use_belief=True must use:
  StructuredBeliefEvaluator
  BeliefTracker with full hidden-zone World particles
  BISMCTS.run_infoset()
```

Do not allow:

```text
BeliefEvaluator
run_belief()
hand-only World particles
unseeded determinization
```

`run_belief()` may remain only as a renamed or explicitly marked diagnostic path, not as a training-label path. Since backward compatibility is not required, the cleanest remediation is:

```text
run_belief() -> rename to run_pimc_diagnostic()
training loops -> never call it
```

### Tests

Create:

```text
tests/test_tier_a_training_guards.py
```

Add:

```text
test_clean_label_training_uses_structured_belief_evaluator
test_clean_label_training_never_calls_run_belief
test_pimc_diagnostic_path_is_not_available_from_training_loop
```

### Completion criteria

The training loops cannot silently produce samples from legacy PIMC or hand-only belief.

---

## Phase 1 — define one canonical full hidden-zone `World` contract

### Problem

The `World` dataclass already contains rich fields, but the active engine path only honors `opponent_hand_ids`. This makes the current data model dishonest: the Python side can create a full world, but the TypeScript side ignores most of it.

### Required remediation

Make `World` the single canonical determinization contract.

Required fields:

```python
@dataclass(frozen=True)
class World:
    opponent_hand_ids: tuple[str, ...]
    opponent_inkwell_ids: tuple[str, ...]
    opponent_deck_ids: tuple[str, ...]
    self_deck_ids: tuple[str, ...]
    seed: str
    log_target: float
    log_proposal: float
    rho: float
    weight: float
```

Add explicit validation:

```python
World.validate_against_obs(obs)
```

It must prove:

```text
opponent_hand_ids count == obs.players.opp.handCount
opponent_inkwell_ids count == obs.players.opp.inkwell
opponent_deck_ids count == obs.players.opp.deckCount
no duplicate IDs across opponent hand/ink/deck
opponent hand + ink + deck equals the known hidden opponent pool
self_deck_ids count == obs.players.self.deckCount, when supplied
seed is non-empty for every world admitted to clean-label ISMCTS
```

`self_deck_ids` must have an explicit representation for “not supplied” versus “supplied and empty”. Prefer `tuple[str, ...] | None` rather than overloading `()`. If Python constructs the order, the server must honor it exactly. If the server constructs the order from the authoritative self-deck pool, it must use `World.seed`, return the realized order for auditability, and never use ambient randomness.

The source observation or bridge context must expose the complete opponent hidden pool to the search-only sampler:

```text
opponent hand IDs
opponent inkwell IDs
opponent deck IDs
public counts for each zone
```

These search-only identities must never enter the policy/value trunk.

### Files/functions

```text
lorcana-bot/search/determinize.py
  World
  sample_worlds()

lorcana-bot/search/belief_filter.py
  ParticleFilter
  BeliefTracker
```

### Affected downstream callers

```text
lorcana-bot/search/engine_sim.py
lorcana-bot/search/ismcts.py
lorcana-bot/engine/bridge.py
lorcana-bot/engine/node_server/server.ts
```

### Tests

Create or update:

```text
tests/test_world_contract.py
```

Add:

```text
test_world_partition_has_no_duplicates
test_world_partition_matches_hidden_pool
test_world_rejects_missing_inkwell_assignment
test_world_rejects_hand_count_mismatch
test_world_preserves_self_deck_when_known
test_world_rejects_empty_seed_for_clean_label_search
test_world_distinguishes_unsupplied_self_deck_from_empty_self_deck
```

### Completion criteria

No code path can call determinization with only hand IDs unless it is explicitly marked invalid for clean-label training.

---

## Phase 2 — fix the structured sampler and importance-weight semantics

### Problem

`sample_worlds()` currently violates its public contract in three ways:

1. Zero-hand or zero-card cases return default empty `World` objects, dropping inkwell/deck assignments.
2. Structured joint `proposal="uniform"` still sets `log_target == log_proposal`, forcing `rho == 1`.
3. `rho` is documented as raw `b/q`, but stored after rebasing by `log_rho.max()`.

### Required remediation

Split the sampler outputs into two separate concepts:

```python
rho_raw: exact b(world) / q(world)
weight: normalized/rebased weight for pooling
```

Recommended dataclass change:

```python
rho: float        # raw b/q, never rebased
weight: float     # normalized stable weight used for pooling
```

If numerical stability requires log rebasing, do it only when computing `weight`.

### Required sampler behavior

For `proposal="belief"`:

```text
log_target == log_proposal
rho == 1.0
weight == 1.0 or normalized equivalent
```

For `proposal="uniform"`:

```text
log_proposal = uniform assignment probability
log_target = structured target probability
rho = exp(log_target - log_proposal)
weight = normalized stable version of rho
```

For zero-hand cases:

```text
opponent_hand_ids == ()
opponent_inkwell_ids still assigned
opponent_deck_ids still assigned
self_deck_ids still assigned when available
```

A zero-card pool should only produce empty opponent zones if the actual hidden pool is empty.

### Files/functions

```text
lorcana-bot/search/determinize.py
  sample_worlds()
  _conditional_bernoulli()
  _joint_zone()
  World
```

### Tests

Update existing sampler tests and add:

```text
test_structured_zero_hand_preserves_inkwell_and_deck
test_structured_uniform_proposal_rho_is_not_forced_one
test_structured_uniform_weight_normalizes_but_rho_remains_raw
test_belief_proposal_keeps_exact_rho_one
test_joint_zone_uniform_target_probability_matches_bruteforce
test_world_seed_is_populated_for_every_sample
```

### Completion criteria

The sampler’s math, docs, and stored fields agree. A test must fail if `rho` is silently rebased or if uniform structured worlds force `rho == 1`.

---

## Phase 3 — extend bridge/server determinization to accept full `World` specs

### Problem

The Python bridge currently sends only `handInstanceIds`, and the TypeScript server accepts only `handInstanceIds`.

Even though the TypeScript determinizer randomizes inkwell/deck internally, that means the requested `World.opponent_inkwell_ids`, `World.opponent_deck_ids`, and `World.self_deck_ids` are discarded.

### Required remediation

Add a full-world determinization RPC without touching TheCardGoats GameEngine source.

### Python bridge change

Replace or overload:

```python
def determinize(self, self_id: str, hand_instance_ids: list[str], seed: str | None = None) -> dict:
```

with:

```python
def determinize_world(self, self_id: str, world: World) -> dict:
```

RPC payload:

```json
{
  "op": "determinize_world",
  "self": "<self_id>",
  "world": {
    "opponentHandIds": [],
    "opponentInkwellIds": [],
    "opponentDeckIds": [],
    "selfDeckIds": [],
    "seed": "..."
  }
}
```

### TypeScript server change

Add a bot-owned wrapper method:

```ts
determinizeWorld(selfId: string, world: DeterminizedWorldSpec): boolean
```

This must:

```text
1. clone the authoritative state
2. assign opponent hand exactly to world.opponentHandIds
3. assign opponent inkwell exactly to world.opponentInkwellIds
4. assign opponent deck exactly to world.opponentDeckIds
5. assign self deck exactly to world.selfDeckIds if supplied
6. validate counts and pool membership before loadState()
7. reject invalid specs rather than silently repairing them
8. never fall back to Math.random
9. return or expose the realized partition so Python can assert exact agreement
```

Keep the old random repartition method only if marked diagnostic/non-training.

For clean-label search, prefer a zero-randomness RPC: Python supplies the complete realized partition and the server validates and loads it. If server-side self-deck shuffling is retained, it must be a deterministic function of `World.seed` and the server must expose the realized order for verification.

### Files/functions

```text
lorcana-bot/engine/bridge.py
  determinize()
  add determinize_world()

lorcana-bot/engine/node_server/server.ts
  Session.determinize()
  add Session.determinizeWorld()
  handle(req) switch
```

### Required TypeScript boundary

Allowed:

```text
lorcana-bot/engine/node_server/server.ts
```

Forbidden:

```text
lorcana-simulator/packages/**
```

### Tests

Update/create:

```text
tests/test_engine_determinize_world.py
```

Add:

```text
test_determinize_world_honors_requested_opponent_inkwell
test_determinize_world_honors_requested_opponent_deck
test_determinize_world_honors_requested_self_deck
test_determinize_world_rejects_duplicate_hidden_ids
test_determinize_world_rejects_missing_hidden_pool_ids
test_determinize_world_is_reproducible_with_same_world_seed
test_determinize_world_never_uses_math_random_when_world_seed_present
```

### Completion criteria

A direct probe that supplies two different requested inkwell assignments must produce two different actual inkwell assignments matching the requested worlds.

---

## Phase 4 — replace active legacy belief evaluator wiring

### Problem

`training/run.py` and `training/parallel.py` instantiate `BeliefEvaluator`, not `StructuredBeliefEvaluator`.

`BeliefEvaluator` only returns hand/deck pool IDs and hand probabilities.

### Required remediation

Replace:

```python
from search.evaluator import NetEvaluator, BeliefEvaluator
belief = BeliefEvaluator(net) if use_belief else None
```

with:

```python
from search.evaluator import NetEvaluator, StructuredBeliefEvaluator
belief = StructuredBeliefEvaluator(net) if use_belief else None
```

Prepare this wiring behind the Phase 0 fail-closed guard. Do not resume sample-writing merely because the evaluator import has changed. Phase 6 is the only activation step.

Then update all callers expecting:

```python
pool_ids, probs = belief_evaluator(root_obs)
```

to expect:

```python
pool_ids, hand_probs, ink_probs = belief_evaluator(root_obs)
```

### Files/functions

```text
lorcana-bot/search/evaluator.py
  BeliefEvaluator
  StructuredBeliefEvaluator

lorcana-bot/search/ismcts.py
  run_infoset()
  run_belief() or renamed diagnostic equivalent

lorcana-bot/training/run.py
  selfplay_game()

lorcana-bot/training/parallel.py
  _worker()
```

### Required cleanup

Since backward compatibility is not required:

```text
BeliefEvaluator should be removed, renamed LegacyHandOnlyBeliefEvaluator, or made unavailable to training.
```

Do not leave both evaluators available in a way that training can accidentally pick the legacy one.

### Tests

Create/update:

```text
tests/test_structured_belief_wiring.py
```

Add:

```text
test_training_run_instantiates_structured_belief_evaluator
test_parallel_worker_instantiates_structured_belief_evaluator
test_run_infoset_rejects_hand_only_belief_evaluator
test_structured_belief_pool_includes_hand_deck_and_inkwell
```

### Completion criteria

No clean-label training path can produce worlds from a hand-only hidden pool.

---

## Phase 5 — redesign `BeliefTracker` around full `World` particles

### Problem

`ParticleFilter` stores only `frozenset` hand particles, and `sample_world()` returns `World(opponent_hand_ids=...)` only.

Persistent reweighting in `BeliefTracker.worlds()` multiplies only included-card probabilities and omits excluded-card `(1-p)` terms, so it is not a valid conditional-Bernoulli update.

### Required remediation

Replace hand-only particles with full-world particles.

Recommended model:

```python
@dataclass(frozen=True)
class WorldParticle:
    world: World
    log_weight: float
```

`ParticleFilter` should store:

```python
self.particles: list[World]
self.weights: np.ndarray
```

### Required tracker inputs

Change:

```python
worlds(pool_ids, probs, hand_count, n_worlds)
```

to:

```python
worlds(
    pool_ids: list[str],
    hand_probs: np.ndarray,
    ink_probs: np.ndarray,
    hand_count: int,
    ink_count: int,
    self_deck_ids: list[str],
    n_worlds: int,
) -> list[World]
```

### Remove invalid posterior-like reweighting

Do not repeatedly treat the current neural belief output as independent new evidence.

Replace the current included-card-only correction with one of these:

Preferred for Tier A correctness:

```text
Reseed when hidden pool/counts change.
Do not apply neural-belief reweighting on unchanged pools.
Use only observed-action correction as sequential evidence.
```

Acceptable if mathematically implemented and tested:

```text
Implement a defined likelihood model over full hand/ink/deck assignments:
  Π included zone probabilities
  Π excluded zone probabilities
  normalized under count constraints
```

For this remediation, the safer path is removal of the invalid reweighting.

### Preserve evidence across public state transitions

An action update is not useful if the next call to `worlds()` immediately reseeds and discards it. The tracker must define how posterior particles advance after observed public events.

Required Tier A behavior:

```text
revealed play/discard:
  reject inconsistent worlds
  remove the revealed instance from the hypothesized hand and hidden pool

face-down opponent inking:
  do not condition on hidden identity
  transition one latent hand card into latent inkwell according to the posterior

draw:
  transition one latent deck card into latent hand according to the posterior

public count-only change:
  preserve weights while resampling only latent assignments needed to satisfy counts

unsupported or ambiguous transition:
  reseed from the structured proposal, then replay retained public evidence
```

Implement either:

```text
A. explicit World-particle transition kernels, or
B. an evidence ledger replayed whenever a structured reseed is required
```

The evidence ledger may store only observer-visible projections. It must never retain opponent-private ink or mulligan instance identities.

### Files/functions

```text
lorcana-bot/search/belief_filter.py
  ParticleFilter
  BeliefTracker.worlds()
  BeliefTracker.sample_world()
  BeliefTracker.observe_opponent_action()
```

### Tests

Update/create:

```text
tests/test_belief_tracker_full_world.py
```

Add:

```text
test_tracker_particles_store_full_worlds
test_tracker_sample_world_returns_hand_ink_deck_self_deck
test_tracker_reseed_on_pool_change
test_tracker_reseed_on_hand_count_change
test_tracker_reseed_on_ink_count_change
test_tracker_does_not_apply_invalid_included_only_reweighting
test_tracker_action_update_changes_weights_and_increments_counter
test_tracker_worlds_are_seeded_reproducibly
test_tracker_revealed_play_transition_preserves_action_evidence
test_tracker_hidden_ink_transition_does_not_use_private_identity
test_tracker_draw_transition_preserves_weights
test_tracker_reseed_replays_public_evidence
```

### Completion criteria

`BeliefTracker.sample_world()` must never return a hand-only `World`, and an observed-action posterior update must still affect worlds sampled at the next decision.

---

## Phase 6 — make full ISMCTS the active clean-label search path

### Problem

`run_infoset()` exists, but the code still supports default PIMC-style `run_belief()`. `run_belief()` samples worlds, runs separate searches, and pools root statistics, which is exactly the strategy-fusion pattern Tier A was meant to remove.

### Required remediation

For clean-label training:

```text
BISMCTS.run_infoset() is mandatory.
BISMCTS.run_belief() is not allowed.
```

This phase is an activation step, not an early refactor. Land it only after the structured sampler, full-world tracker, full-world RPC, full-world lane adapter, lane-local histories, observed-action correction, and shared-node prior fix all pass their targeted tests.

Change `training/run.py` and `training/parallel.py` so belief training does not branch between PIMC and full ISMCTS.

Current pattern:

```python
if full_ismcts:
    res = mcts.run_infoset(...)
else:
    res = mcts.run_belief(...)
```

Required pattern:

```python
res = mcts.run_infoset(
    obs,
    structured_belief,
    tracker,
    total_simulations=cfg.simulations * n_worlds,
)
```

The CLI flag `--full-ismcts` should be removed, ignored with warning, or inverted into a diagnostic-only escape hatch that cannot write clean-label samples.

### Files/functions

```text
lorcana-bot/training/run.py
  selfplay_game()
  argparse options

lorcana-bot/training/parallel.py
  _worker()

lorcana-bot/search/ismcts.py
  run_infoset()
  run_belief()
```

### Tests

Create/update:

```text
tests/test_clean_label_search_mode.py
```

Add:

```text
test_belief_training_always_uses_run_infoset
test_full_ismcts_flag_no_longer_controls_correctness
test_run_belief_cannot_be_used_for_clean_label_samples
```

### Completion criteria

There is no config combination that accidentally sends belief-guided clean-label self-play through root-pooled PIMC.

---

## Phase 7 — pass full worlds through `EngineSimulator.begin_lane()`

### Problem

`EngineSimulator.begin_lane()` ignores full world fields and passes only opponent hand IDs.

### Required remediation

Change:

```python
obs = self.engine.determinize(self.self_id, list(world.opponent_hand_ids), seed=(world.seed or None))
```

to:

```python
obs = self.engine.determinize_world(self.self_id, world)
```

`begin_lane()` must validate that the returned observation matches the requested world.

### Files/functions

```text
lorcana-bot/search/engine_sim.py
  EngineSimulator.begin_lane()

lorcana-bot/engine/bridge.py
  determinize_world()

lorcana-bot/engine/node_server/server.ts
  determinize_world RPC
```

### Tests

Create/update:

```text
tests/test_engine_sim_full_world.py
```

Add:

```text
test_begin_lane_passes_full_world_to_engine
test_begin_lane_honors_two_distinct_inkwell_assignments
test_begin_lane_honors_two_distinct_self_deck_orders
test_begin_lane_rejects_engine_world_mismatch
```

### Completion criteria

The audit probe must now fail if `begin_lane()` ignores requested inkwell/deck assignments.

---

## Phase 8 — repair perfect-recall branch history and action keys

### Problem

The current Python branch event stores only:

```python
[actor_seat, family, maybe_card_id]
```

and `EngineSimulator.step()` builds that from only the first two colon-separated fields of the action key.

But TypeScript `candidateKey()` encodes many necessary distinctions:

```text
targets
choiceIndex
namedCard
resolveOptional
destinations
costs
discard/exert/banish costs
mulligan selections
```

These are part of perfect recall when visible to the actor.

### Required remediation

Replace `history_event(actor_seat, family, card_id)` with a full projection function.

Recommended API:

```python
def project_action_for_observer(
    stable_key: str,
    actor_seat: int,
    observer_seat: int,
    visibility_context: dict | None = None,
) -> dict:
```

The projected event must preserve:

```text
family
full stable key fields visible to the observer
actor seat
public action identity
public targets
public costs
public choices
public destinations
```

It must redact:

```text
opponent face-down inking card identity
opponent mulligan card identities
hidden hand choices not publicly revealed
```

It must preserve:

```text
own face-down inking card identity
own mulligan card identities
own hidden choices
```

Define two projections explicitly:

```python
project_public_action(stable_key, actor_seat, visibility_context) -> PublicHistoryEvent
project_action_for_observer(stable_key, actor_seat, observer_seat, visibility_context) -> RecallEvent
```

`project_public_action()` is suitable for neural `obs["history"]`. It must not contain any private identity. `project_action_for_observer()` is suitable for perfect-recall keying and may preserve the observer's own private action details.

### Required information-set key behavior

`info_set_key()` must include the projected branch history for the actor whose information set is being keyed.

The key must also use a complete canonical actor-visible state projection. The current hand-written visible card row omits fields such as visible instance ID, `cardType`, `ready`, and `keywords`. Replace it with an allowlisted canonical serializer that includes every actor-visible field capable of changing legal actions or future value while still excluding opponent-hidden identities and deck order.

It must not include:

```text
obs["hidden"] identities
opponent hidden card IDs
deck order
opponent private branch details
```

### Files/functions

```text
lorcana-bot/search/infoset.py
  history_event()
  info_set_key()

lorcana-bot/search/engine_sim.py
  EngineSimulator.step()
  EngineSimulator._with_key()

lorcana-bot/engine/node_server/server.ts
  candidateKey()
```

### Tests

Create/update:

```text
tests/test_infoset_perfect_recall.py
```

Add:

```text
test_history_preserves_full_public_action_key
test_history_does_not_truncate_targets_choices_costs_or_destinations
test_actor_knows_own_inked_card
test_observer_does_not_know_opponent_inked_card
test_actor_knows_own_mulligan_selection
test_observer_does_not_know_opponent_mulligan_selection
test_info_set_key_changes_for_visible_target_difference
test_info_set_key_changes_for_visible_choice_difference
test_info_set_key_does_not_change_for_opponent_hidden_ink_identity
test_info_set_key_changes_for_visible_instance_identity_when_action_relevant
test_info_set_key_changes_for_visible_keyword_or_status_difference
test_public_projection_never_contains_private_identity
```

### Completion criteria

Two actions with the same family/source but different visible targets, choices, costs, or destinations must produce different perfect-recall keys.

Two opponent inking actions that differ only by hidden inked card identity must produce the same key for the observer.

---

## Phase 9 — add lane-local projected history for neural leaf evaluation

### Problem

`stepExact()` intentionally does not mutate server public history.

That is correct for engine cleanliness, but neural leaves currently encode stale root history because the simulated branch history is only used for hashing, not for `obs["history"]`. `NetEvaluator` encodes whatever `obs` contains.

### Required remediation

`EngineSimulator` must maintain two lane-local branch-history views:

```text
public branch history:
  injected into obs["history"] before NetEvaluator()

per-observer perfect-recall history:
  passed into info_set_key() for the current actor
```

Recommended structure:

```python
self.branch_log: list[FullBranchEvent]
self.root_public_history: list[PublicHistoryEvent]
self.root_recall_history_by_seat: dict[int, list[RecallEvent]]
```

In `_with_key(obs)`:

```python
observer_seat = int(obs.get("selfIdx", 0))
public_branch = [
    project_public_action(event.stable_key, event.actor_seat, event.context)
    for event in self.branch_log
]
recall_branch = [
    project_action_for_observer(event.stable_key, event.actor_seat, observer_seat, event.context)
    for event in self.branch_log
]
obs = {
    **obs,
    "history": self.root_public_history + public_branch,
}
obs["infoSetKey"] = info_set_key(
    obs,
    self.root_recall_history_by_seat[observer_seat] + recall_branch,
)
```

Do not mutate server history. Do not inject private perfect-recall events into public neural history.

The real session must provide root recall histories for both seats. Add bot-owned server bookkeeping alongside the existing public `Session.history`:

```ts
private recallHistoryBySeat: Record<number, RecallEvent[]>
```

Real executed actions append one public event and one observer-projected recall event per seat. Search lanes copy the root histories and append hypothetical events locally. If any snapshot/restore path can mutate these histories, snapshot and restore them as a bundle with engine state.

### Files/functions

```text
lorcana-bot/search/engine_sim.py
  EngineSimulator.__init__()
  EngineSimulator.begin_lane()
  EngineSimulator.step()
  EngineSimulator._with_key()

lorcana-bot/search/infoset.py
  project_action_for_observer()
  info_set_key()

lorcana-bot/engine/serialization.py
  encode_obs()

lorcana-bot/engine/node_server/server.ts
  Session.history
  add Session.recallHistoryBySeat
  snapshot()/restore() if histories can be mutated in the lane path
```

`encode_obs()` must be audited to confirm it consumes `obs["history"]`. If it ignores history, then Tier A’s neural-history claim is false and serialization must be extended.

### Tests

Create/update:

```text
tests/test_lane_local_history.py
```

Add:

```text
test_step_exact_does_not_mutate_server_history
test_engine_sim_injects_branch_history_into_leaf_obs
test_net_evaluator_leaf_history_len_increases_after_simulated_step
test_root_history_len_less_than_leaf_history_len_after_branch
test_branch_history_projection_depends_on_observer
test_lane_history_resets_between_world_simulations
test_neural_history_uses_public_projection_only
test_infoset_key_uses_actor_private_recall_projection
```

### Completion criteria

The previous probe must invert:

```text
root_history_len < leaf_history_len
branch_events_len > 0
```

at simulated leaves.

---

## Phase 10 — integrate observed-opponent-action correction into real self-play

### Problem

`observe_opponent_action()` exists but is not called after real self-play actions. `action_updates` remains zero in real games.

### Required remediation

After every real executed action, update the tracker belonging to the non-acting observer seat.

Current loop:

```python
result = engine.step(legal[a]["stableKey"])
obs = result["obs"]
```

Required sequence:

```python
pre_action_obs = dec_obs
executed_key = result.get("executed") or legal[a]["stableKey"]

for observer_id, tracker in trackers.items():
    if observer_id != dec_obs["self"]:
        tracker.observe_opponent_action(
            event=project_action_for_observer(...),
            pre_action_obs=pre_action_obs,
            post_action_obs=obs,
            likelihood_fn=make_observed_action_likelihood(...),
        )
```

`observe_opponent_action()` must apply the likelihood to the pre-action particles and then advance those particles, or retained public evidence, into the post-action observation. Do not immediately erase the correction by reseeding on the next `worlds()` call.

### Critical leakage rule

Do not pass raw `executed_key` directly into observer belief correction if it contains hidden identity, especially:

```text
putCardIntoInkwell:<actual_card_id>
alterHand:<mulligan_card_ids>
```

Use projected public action events.

### Recommended likelihood model for Tier A

Start with a conservative, provable legality/visibility likelihood:

```text
playCard:<card_id>
  If the card was public after play:
    particles where opponent hand contained card before action -> likelihood 1.0
    others -> epsilon

putCardIntoInkwell:<hidden_card_id>
  Do not condition on identity for observer.
  Condition only on visible count transition if available.

quest/challenge/moveCharacterToLocation
  Usually public board actions.
  No hidden-hand correction unless action consumes hidden resources.

passTurn
  No hand identity correction.

resolveEffect / resolveBag
  Correct only visible revealed choices.
  Redact hidden choices.
```

Later policy-likelihood correction can be added, but Tier A should first implement a non-leaking legality/consistency correction.

### Files/functions

```text
lorcana-bot/search/belief_filter.py
  BeliefTracker.observe_opponent_action()

lorcana-bot/training/run.py
  selfplay_game()

lorcana-bot/training/parallel.py
  _worker()

lorcana-bot/search/infoset.py or new file:
  project_action_for_observer()
  public_action_likelihood()
```

### Tests

Create/update:

```text
tests/test_observed_action_correction_runtime.py
```

Add:

```text
test_run_selfplay_calls_observe_opponent_action_after_opponent_move
test_parallel_worker_calls_observe_opponent_action_after_opponent_move
test_action_updates_increments_during_real_selfplay
test_played_card_reweights_particles_containing_that_card
test_opponent_inking_does_not_reweight_by_hidden_card_identity
test_mulligan_does_not_leak_hidden_card_identity_to_observer
test_action_update_survives_next_tracker_worlds_call
```

### Completion criteria

A real self-play smoke test must show:

```text
tracker.action_updates > 0
```

after an opponent action is observed.

---

## Phase 11 — return populated full-ISMCTS diagnostics

### Problem

`BISMCTS.run_infoset()` gets a populated shared-tree result, then creates and returns a new empty `InfoSetNode`.

That corrupts diagnostics used by tracing and monitoring.

### Required remediation

Extend `SearchResult` so it can carry the real shared root.

Recommended change:

```python
@dataclass
class SearchResult:
    pi: np.ndarray
    value: float
    root: InfoSetNode | SharedInfoSetNode
    sims: int = 0
    stats: dict = field(default_factory=dict)
```

Then return:

```python
return SearchResult(
    pi=res.pi,
    value=res.value,
    root=res.root,
    sims=res.sims,
    stats={**res.stats, "mode": "infoset"},
)
```

Update `DecisionTracer` to handle both:

```text
InfoSetNode.N/W arrays
SharedInfoSetNode.edges dict
```

Update training monitoring to use actual results:

```text
mon.add(sims=res.sims, decisions=1)
parallel worker sims += res.sims
surface res.stats["invalid_sims"]
surface unique_infosets and deeper_transposition_hits
```

Do not report configured simulation counts when quarantined simulations reduce the number of successful backups. Track root revisits separately from deeper transposition hits so the shared-tree proof is not satisfied by root reuse alone.

### Files/functions

```text
lorcana-bot/search/ismcts.py
  SearchResult
  BISMCTS.run_infoset()

lorcana-bot/search/infoset_tree.py
  SharedInfoSetNode

lorcana-bot/training/trace.py
  DecisionTracer.log()
```

### Tests

Create/update:

```text
tests/test_infoset_diagnostics.py
```

Add:

```text
test_run_infoset_returns_populated_shared_root
test_returned_root_visit_sum_matches_policy_counts
test_decision_tracer_logs_nonzero_infoset_visits
test_infoset_q_values_are_reported_from_shared_edges
test_monitor_uses_actual_successful_sim_count
test_invalid_infoset_sims_surface_in_monitoring
test_deeper_transposition_hits_exclude_root_revisits
```

### Completion criteria

The reproduced audit probe must now show:

```text
returned_root_visit_sum > 0
```

for nontrivial searches.

---

## Phase 12 — make full ISMCTS reproducible

### Problem

Tracker worlds currently lack seeds, and TypeScript determinization falls back to `Math.random` when no seed is supplied.

### Required remediation

Verify the deterministic seed contract introduced with Phase 1 and carried through Phases 2, 3, 5, and 7. Every sampled `World` admitted to clean-label ISMCTS must contain a deterministic seed.

Recommended seed format:

```python
seed=f"{game_seed}:seat={seat}:decision={decision_idx}:sim={sim_idx}:particle={particle_idx}"
```

Do not allow full ISMCTS determinization to call server-side randomization without a seed. This phase is end-to-end hardening and verification, not the first point where seeds are added.

For `determinize_world`, no randomness should be needed if full assignments are supplied. If self deck must be shuffled by the server, the shuffled order must be generated in Python and passed as `self_deck_ids`.

### Files/functions

```text
lorcana-bot/search/determinize.py
  sample_worlds()

lorcana-bot/search/belief_filter.py
  ParticleFilter
  BeliefTracker

lorcana-bot/search/ismcts.py
  run_infoset()

lorcana-bot/engine/node_server/server.ts
  Session.determinize()
  Session.determinizeWorld()
```

### Tests

Create/update:

```text
tests/test_reproducibility_tier_a.py
```

Add:

```text
test_same_seed_same_world_sequence
test_same_seed_same_full_ismcts_policy
test_different_seed_can_change_world_sequence
test_determinize_world_does_not_call_math_random
test_tracker_particles_have_nonempty_seeds
```

### Completion criteria

Two identical full-ISMCTS runs with the same game seed and RNG seed produce the same sampled world sequence and root policy.

---

## Phase 13 — upgrade runtime tests so they catch the audited failures

### Problem

The current real-engine tests are too weak. The root-invariance test samples only hand plus deck, not inkwell.

### Required remediation

Add tests that directly reproduce every audited failure, then prove it is fixed. Introduce the initial regression file immediately after Phase 0 as `pytest.mark.xfail(strict=True)` probes or as a separate audit-only command. Convert each probe to a normal passing regression test as its owning phase lands. Do not defer all regression coverage until the end, and do not silently leave fixed behavior under `xfail`.

### Required test groups

```text
tests/test_tier_a_runtime_regressions.py
tests/test_engine_determinize_world.py
tests/test_lane_local_history.py
tests/test_observed_action_correction_runtime.py
tests/test_infoset_diagnostics.py
tests/test_reproducibility_tier_a.py
```

### Required regression probes

Add:

```text
test_active_determinization_pool_includes_inkwell
test_begin_lane_does_not_discard_world_inkwell
test_begin_lane_does_not_discard_world_deck
test_full_ismcts_uses_structured_world_sampler
test_action_updates_nonzero_in_real_selfplay
test_neural_leaf_receives_projected_branch_history
test_python_key_preserves_full_candidate_key_fields
test_python_key_actor_knows_own_inking
test_sampler_zero_hand_keeps_ink_and_deck
test_uniform_structured_sampler_has_nontrivial_rho
test_tracker_does_not_use_invalid_posterior_reweighting
test_infoset_diagnostics_nonzero_visits
test_full_ismcts_seed_reproducible
test_observed_action_evidence_survives_tracker_transition
test_deeper_shared_node_hit_is_not_only_root_reuse
```

### Completion criteria

The test suite must fail on the current audited implementation and pass only after the remediations are complete.

---

## Phase 14 — repair shared-node action-union priors and invariants

### Problem

`InfoSetTable.get_or_create()` adds newly observed actions to an existing node through `ensure_edges()`, but `_expand()` runs only when a node is first created. If another sampled world later exposes a previously unseen available action at the same information set, that edge can remain at a zero prior indefinitely.

The root invariant comment also says root action IDs are checked, while the implementation directly asserts only the key. Root actions are currently folded into the key, but a separate explicit action-ID assertion is clearer and catches accidental key refactors.

### Required remediation

When an existing shared node sees a legal-action availability signature that has not been evaluated:

```text
1. evaluate priors for the current actor-visible observation
2. merge priors by actionId only for actions in the observed subset
3. record that availability signature as evaluated
4. preserve root Dirichlet noise as a one-time root-only operation
```

Do not repeatedly count the same prior sample on every visit to the same availability signature.

At root lane initialization, explicitly assert:

```text
infoSetKey == root.key
set(root action IDs) == set(real root action IDs)
```

### Files/functions

```text
lorcana-bot/search/infoset_tree.py
  SharedInfoSetNode
  InfoSetTable.get_or_create()
  _expand()
  run_infoset()
```

### Tests

Create/update:

```text
tests/test_infoset_action_union.py
```

Add:

```text
test_existing_infoset_new_available_edge_receives_prior
test_same_availability_signature_does_not_double_count_prior
test_root_noise_applied_once_after_prior_merge
test_root_invariant_rejects_action_id_set_change
test_deeper_node_shared_across_distinct_worlds
```

### Completion criteria

Every available edge used by progressive widening has a prior derived from an actor-visible observation, and the real-engine shared-tree proof demonstrates at least one deeper shared node reached from distinct sampled worlds.

---

# Exact remediation sequence

Use this order. Keep clean-label training blocked until Commit 16 passes. Each commit adds or converts its owning regression tests; do not accumulate test work at the end.

## Commit 1 — freeze unsafe training

Implement Phase 0. Belief-guided clean-label training fails closed while remediation is in progress. Rename or mark PIMC as diagnostic-only.

## Commit 2 — add expected-failing audit probes

Start Phase 13. Add direct `xfail(strict=True)` or audit-only regressions for the audited failures before changing production behavior:

```text
inkwell excluded from active proposal pool
rich World ignored by lane adapter
structured uniform rho forced to one
zero-hand partition dropped
stale simulated leaf history
truncated recall event
action update not wired or not persistent
empty returned diagnostics root
unseeded lane determinization
new shared-node edge receives zero prior
```

## Commit 3 — canonical deterministic `World`

Implement Phase 1. Add strict partition validation, explicit optional self-deck semantics, and a required non-empty clean-label seed.

## Commit 4 — structured sampler math

Implement Phase 2. Fix structured zero-hand worlds, true uniform joint proposals, raw `rho`, normalized `weight`, and deterministic sampled-world seeds.

## Commit 5 — full-world bridge/server RPC

Implement Phase 3. Add `determinize_world`, validate exact partitions in bot-owned TypeScript, reject invalid worlds, and remove ambient randomness from the clean-label RPC.

## Commit 6 — full-world lane adapter

Implement Phase 7. Pass complete `World` objects through `EngineSimulator.begin_lane()` and assert the realized world matches the request.

## Commit 7 — structured evaluator contract

Implement Phase 4 in search-facing code. Structured evaluator output is the only accepted belief-search contract. Legacy hand-only evaluation remains diagnostic-only.

## Commit 8 — full-world tracker with persistent transitions

Implement Phase 5. Store full-world particles, remove invalid included-only neural reweighting, add public transition kernels or evidence replay, and prove observed evidence survives the next decision.

## Commit 9 — complete actor-filtered recall projection

Implement Phase 8. Preserve complete visible action identity and actor-visible state while redacting only opponent-private details.

## Commit 10 — lane-local public and recall histories

Implement Phase 9. Inject public hypothetical history into neural leaves and use per-observer recall projection for information-set keys.

## Commit 11 — shared-node action-union priors

Implement Phase 14. Evaluate newly seen availability signatures, merge priors by action ID, assert root action-ID invariance, and prove deeper sharing across distinct worlds.

## Commit 12 — observed-action correction

Implement Phase 10. Apply non-leaking opponent-action evidence in both self-play loops and prove the correction persists after tracker transition or replay.

## Commit 13 — activate full ISMCTS clean-label training

Implement Phase 6 only now. Remove the correctness-controlling `--full-ismcts` branch from sample-writing paths. Clean-label belief search always uses structured full-world shared-tree ISMCTS.

## Commit 14 — diagnostics and monitoring

Implement Phase 11. Return the populated shared root, teach tracers about shared edges, count actual successful simulations, surface invalid simulations, and distinguish deeper transpositions from root revisits.

## Commit 15 — reproducibility verification

Implement Phase 12. Prove identical seeds produce identical world sequences and root policies and that clean-label determinization never reaches `Math.random`.

## Commit 16 — close the runtime gate

Finish Phase 13. Convert all remaining expected-failing probes into passing regression tests, run the complete suite, and release the Phase 0 training guard only for the verified structured full-ISMCTS path.

---

# Final verification gate

Run:

```bash
../lorcana-bot-venv/bin/python -m pytest -q
git diff --check
../lorcana-bot-venv/bin/python -m compileall -q engine network search training tests
```

Add a Tier A-specific test command:

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
  tests/test_infoset_action_union.py \
  tests/test_reproducibility_tier_a.py \
  tests/test_tier_a_runtime_regressions.py
```

Tier A is complete only when all of these are true:

```text
No active clean-label training path imports or instantiates BeliefEvaluator.
No active clean-label training path calls run_belief/PIMC.
StructuredBeliefEvaluator is used for belief search.
sample_worlds() returns full hand/ink/deck worlds.
rho is raw b/q; weight is normalized/rebased separately.
zero-hand worlds preserve inkwell/deck assignments.
BeliefTracker stores and samples full World particles.
invalid included-only persistent reweighting is removed or replaced by a proven likelihood model.
engine.determinize_world() accepts and honors full World specs.
EngineSimulator.begin_lane() passes full worlds.
stepExact does not mutate server history.
EngineSimulator injects lane-local public branch history into neural leaf observations.
info_set_key preserves full visible action identity.
own inking/mulligan identities are retained for the actor.
opponent hidden inking/mulligan identities are redacted for the observer.
public neural history never receives private recall identities.
observe_opponent_action() is called during real self-play.
action_updates increments during real self-play.
observed-action evidence survives the next tracker transition or reseed.
newly available actions at an existing shared node receive evaluated priors.
root invariant explicitly checks the action-ID set.
deeper transposition metrics exclude root-only revisits.
run_infoset() returns the populated shared root.
DecisionTracer logs nonzero full-ISMCTS visits/Q.
full-ISMCTS determinization is seeded and reproducible.
runtime tests reproduce and close every audit finding.
no native lorcana-simulator/packages/** GameEngine source is modified.
```

Clean-label training remains blocked until this gate passes.
