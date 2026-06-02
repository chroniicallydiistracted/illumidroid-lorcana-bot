# Full ISMCTS End-to-End Implementation Specification

Date: 2026-06-01

Scope: replace the current root-pooled collection of per-world perfect-information
PUCT trees with one root-sampled information-set search graph. Sample one hidden
world per simulation, traverse only actions available in that sampled world, and
share action statistics at information-set nodes.

This is the implementation specification for audit finding **#2 full ISMCTS**.
It assumes audit findings **#5 valid world sampling / importance correction** and
**#4 per-seat persistent action-likelihood belief** land first.

## Implementation Checkpoint

This document is the research and end-state specification. The authoritative
dependency order and live remediation status are maintained in
`tier-a-belief-search-remediation-plan.md`.

Audited current checkpoint:

```text
Phase 0   complete — belief-guided clean-label sample writing fails closed
Phase 13  baseline installed — 6 passing probes (2 Phase-1 + 3 Phase-2 + 1 Phase-3) + 14 strict expected-red probes
Phase 1   complete — audited GO for the canonical full hidden-zone World boundary
Phase 2   complete — audited GO for exact structured sampling + importance semantics
Phase 3   IN REMEDIATION (NO-GO) — full-`World` RPC built + bounded findings fixed; protected-
          facts ledger partial (reveals guarded; static top-deck/pending/§15 open) + F3 unresolved
Next      close Phase 3 blockers before Phase 7
```

Sections labeled “current” below describe the pre-remediation audit baseline unless
an explicit checkpoint note says otherwise. Clean-label belief-guided training
remains blocked until the full Tier A release gate passes.

## 1. Executive Verdict

The pre-remediation `run_belief()` implementation, now quarantined as
`run_pimc_diagnostic()`, is not full ISMCTS. It is a determinized-UCT /
PIMC-style ensemble:

1. Sample a hidden world.
2. Build a new perfect-information PUCT tree for that world.
3. Search it independently.
4. Pool only root `N` and `W` statistics into a separate root node.

That is visible in `lorcana-bot/search/ismcts.py:125-198`, especially the new
`BISMCTS` subtree at line 160, the per-world loop at line 172, and root-only
pooling at lines 181-184.

The replacement must use one `InfoSetTable` for the whole root decision:

1. Sample one root-consistent world for each simulation.
2. Traverse the sampled world adaptively.
3. Resolve an information-set key after every exact transition.
4. Reuse the table node for that key across simulations and worlds.
5. Restrict selection to actions available in the current sampled world.
6. Back up the result into the shared node-edge statistics.

This removes the primary strategy-fusion bias called out by finding #2. It does
not, by itself, prove equilibrium play or eliminate every determinization
artifact. Section 15 separates the mandatory replacement from stricter
opponent-perspective hardening.

## 2. Research Basis

Primary sources:

1. Cowling, Powley, and Whitehouse, *Information Set Monte Carlo Tree Search*
   (2012):
   <https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf>
2. Silver and Veness, *Monte-Carlo Planning in Large POMDPs* (POMCP, 2010):
   <https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf>
3. Goodman, *Re-determinizing Information Set Monte Carlo Tree Search in Hanabi*
   (2019):
   <https://arxiv.org/pdf/1902.06075>
4. OpenSpiel's official IS-MCTS reference implementation:
   <https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/ismcts.py>

The required invariants taken from those sources are:

- Determinized UCT builds a separate tree for each determinization and combines
  root recommendations. Cowling et al. describe this as vulnerable to strategy
  fusion.
- ISMCTS samples a determinization on each simulation but builds one tree whose
  nodes represent information sets.
- Selection must only choose actions legal in the current determinization.
- When action availability differs across sampled worlds, each edge needs an
  availability count in addition to visit and value statistics.
- Root sampling can use a black-box simulator: sample latent state at the root,
  then let simulated transitions consume that state's chance outcomes.
- A root-sampled tree can still leak root-player private information into modeled
  opponent decisions. Re-determinization or a multi-tree variant is a separate,
  stricter extension.

## 3. Current Repo Audit

### 3.1 Root-only pooling is the #2 bug

Current code:

- `lorcana-bot/search/ismcts.py:142` creates a root `shared` node.
- `lorcana-bot/search/ismcts.py:160` creates a normal `BISMCTS` subtree searcher.
- `lorcana-bot/search/ismcts.py:172-180` determinizes one world and runs a fresh
  perfect-information tree.
- `lorcana-bot/search/ismcts.py:181-184` copies only root edge counts and values
  into `shared`.

No deeper node is shared between worlds. A world-specific tree can therefore
learn a different contingent plan for hidden states the acting player cannot
distinguish.

### 3.2 `InfoSetNode` is structurally a perfect-information node

Current `lorcana-bot/search/node.py:17-103` stores fixed arrays tied to the legal
list from one materialized observation:

```python
self.legal = obs.get("legal", [])
self.N = np.zeros(self.n_actions)
self.W = np.zeros(self.n_actions)
self.children: dict[int, InfoSetNode] = {}
```

That representation cannot support:

- node lookup by information-set key;
- an action union discovered across sampled worlds;
- per-world available-action subsets;
- edge availability counts;
- dynamic exact `stableKey` mapping per sampled lane;
- transpositions back into an already known information set.

### 3.3 `run_paths` cannot drive full ISMCTS by itself

Current search descends its Python child pointers before touching the engine,
then executes the already-selected stable-key path in one Bun call:

- `lorcana-bot/search/ismcts.py:305-351`
- `lorcana-bot/engine/node_server/server.ts:484-509`

That optimization assumes the in-memory path is valid independently of the
sampled hidden world. Full ISMCTS breaks that assumption at deeper nodes:

- legal actions can vary by sampled world;
- the next information-set key is only known after the exact transition;
- an existing structural child may have been materialized in another world;
- opponent private hands create legitimately different acting-player
  information sets.

Keep `run_paths` for plain search and regression tests. Full belief ISMCTS needs
an adaptive lane API described in section 9.

### 3.4 Search leaf history is stale

`lorcana-bot/engine/node_server/server.ts:356-358` explicitly keeps hypothetical
`runPaths()` actions out of `Session.history`. As a result, a deep leaf gets the
real root history rather than the root history plus simulated actions.

This is wrong for both:

- the neural history encoder at `lorcana-bot/engine/serialization.py:300-326`;
- a perfect-recall information-set key.

Branch-local hypothetical history is a required part of #2, not optional
telemetry.

### 3.5 The engine fingerprint is not an information-set key

The architecture document suggests reusing
`computeAutomatedActionStateFingerprint`. Do not do that directly.

`lorcana-simulator/packages/lorcana/lorcana-engine/src/automation/decision-trace.ts:66-70`
hashes the full authoritative state. It includes hidden truth. Using it as the
table key would split indistinguishable hidden worlds into separate nodes and
recreate the strategy-fusion problem.

Build a separate actor-filtered information-set fingerprint.

### 3.6 Snapshots currently omit history

`lorcana-bot/engine/node_server/server.ts:354` stores state references only.
`snapshot()` and `restore()` at lines 449-463 do not snapshot the external
`Session.history`.

Once simulated histories exist, snapshot handles must include:

- authoritative engine state reference;
- public neural history;
- per-observer perfect-recall history.

### 3.7 Root world contract checkpoint

Phase 1 has landed the canonical Python `World` boundary:

```python
@dataclass(frozen=True)
class World:
    opponent_hand_ids: tuple[str, ...]
    opponent_inkwell_ids: tuple[str, ...]
    opponent_deck_ids: tuple[str, ...]
    self_deck_ids: tuple[str, ...] | None
    seed: str
    log_target: float
    log_proposal: float
    rho: float
    weight: float
```

`World.opponent_hidden_pool()` and `World.validate_against_obs()` share a strict
source-witness validator. Missing or malformed zones, contradictory zone
cardinalities, non-string or empty instance IDs, duplicate source IDs, malformed
public counters, invalid partitions, and malformed clean-label seeds fail closed.

The remaining bridge gap is unchanged: the diagnostic Bun determinizer still
accepts hand IDs only and fills the remaining hidden zones internally. Phase 3
adds the full-world RPC; Phase 7 routes `EngineSimulator.begin_lane()` through it.

## 4. Required End-State Semantics

For one real root decision with root observation `I0`:

```text
table = one InfoSetTable()
root = table.get_or_create(info_key(I0))

repeat total_simulations times:
    w ~ posterior(hidden world | I0)
    state = determinize(root_snapshot, w)
    path = []

    repeat until terminal, new infoset leaf, or depth limit:
        obs = actor_filtered_observation(state, branch_history)
        key = info_key(obs)
        node = table.get_or_create(key)
        available = legal_action_ids(obs)

        if node is new:
            evaluate and expand node
            stop descent

        action = node.select(available)
        path.append((node, action, available))
        state = execute_exact(state, obs[action].stableKey)

    value = terminal_result or actor_relative_value_net(obs)
    backup(path, value)

return root visit policy aligned to the real root legal list
```

One table exists per root decision. It is an information-set DAG rather than a
strict object-child tree because multiple histories may safely transpose only
when their actor-filtered perfect-recall keys match.

## 5. Dependency Contract: Land #5 and #4 First

### 5.1 #5 world sampler contract

The preferred full-ISMCTS mode samples directly from the posterior:

```text
q(world | infoset) = b(world | infoset)
rho = b / q = 1
```

This gives unit simulation backups and keeps tree visit counts meaningful.

Do not carry the current Gumbel-top-k approximation into full ISMCTS. In
`lorcana-bot/search/determinize.py:64-70`, the code sets `logq = logb` by
assumption. Gumbel top-k with `log(p)` is not generally the same distribution as
a count-conditioned independent-Bernoulli subset.

Required Phase-2 continuation:

1. Predict hand and inkwell zone logits over every opponent hidden instance.
2. Sample a count-constrained zone assignment exactly with dynamic programming.
3. Assign exactly public `handCount` cards to hand.
4. Assign exactly public `inkwell` count cards to inkwell.
5. Assign the residual cards to deck and shuffle the residual deck.
6. Shuffle the searching actor's unknown deck order.
7. Return `log_target`, `log_proposal`, and `rho` for auditability even when
   direct posterior sampling makes `rho == 1`.

Use a joint three-zone model:

```text
zone(card) in {hand, inkwell, deck}
count(hand) = public opponent hand count
count(inkwell) = public opponent inkwell count
count(deck) = residual
```

A DP over `(card_index, hand_slots, ink_slots)` samples the exact constrained
categorical assignment. This is simpler and safer than computing the unordered
proposal probability of the current Gumbel-top-k sampler.

Canonical Python shape established by Phase 1:

```python
@dataclass(frozen=True)
class World:
    opponent_hand_ids: tuple[str, ...]
    opponent_inkwell_ids: tuple[str, ...]
    opponent_deck_ids: tuple[str, ...]
    self_deck_ids: tuple[str, ...] | None
    seed: str
    log_target: float = 0.0
    log_proposal: float = 0.0
    rho: float = 1.0
    weight: float = 1.0
```

If deck orders are generated server-side from `seed`, they can be omitted from
the wire payload but must be reproducible from `(root snapshot, world spec,
seed)`.

### 5.2 #4 per-seat tracker contract

The current default loop creates one `BeliefTracker` for an alternating-player
game at `lorcana-bot/training/run.py:57`. Full search must receive the tracker
for the current canonical seat:

```python
trackers = {
    "player_one": BeliefTracker(...),
    "player_two": BeliefTracker(...),
}
tracker = trackers[obs["self"]]
```

Each tracker represents that seat's posterior over its opponent hidden zones.
It must update after observed opponent actions using:

```text
weight(world) *= P(observed_action | world, observer_infoset)
```

The full ISMCTS planner consumes a sampler callback, not a static world list:

```python
world = belief_state.sample_world(rng)
```

Sampling occurs once per simulation, with replacement, from the current
posterior. The particle filter may resample internally when ESS collapses.

### 5.3 Inkwell channel must become active

`encode_belief()` already emits hidden inkwell labels at
`lorcana-bot/engine/serialization.py:390-423`, but:

- `LorcanaNet.belief_probs()` returns hand probabilities only
  (`lorcana-bot/network/model.py:84-102`);
- `BeliefEvaluator` omits inkwell instances from its search pool
  (`lorcana-bot/search/evaluator.py:65-74`).

Before #2, add a structured belief API returning both hand and inkwell logits or
probabilities over the complete hidden pool.

## 6. Information-Set Identity

### 6.1 Key rule

An information-set key may include only information available to the player
whose decision is being selected at that node.

Never include:

- `obs["hidden"]`;
- authoritative state fingerprints;
- opponent hidden instance identities;
- either deck's hidden draw order;
- hidden inkwell identities owned by the opponent;
- random server envelope metadata.

### 6.2 Key contents

Add `infoSetKey` to each actor-filtered observation. Compute it server-side from:

```text
actor id
self canonical seat index
turn, phase, step, status, forced flag
public player counts
actor-visible cards and derived stats, canonicalized
current actor-visible pending prompt semantics
sorted actor-visible logical action ids
observer-projected perfect-recall branch history
```

For hidden opponent cards, aggregate placeholders by `(owner, zone, hidden)`
rather than including instance IDs.

For visible cards, include the visible instance ID and gameplay fields needed to
distinguish states. The actor legitimately knows the identity of its own hand
and own inkwell cards.

Hash canonical JSON with a deterministic stable serializer. Keep an optional
debug payload or unhashed canonical string behind a test-only flag so collisions
and accidental hidden fields can be audited.

### 6.3 Perfect-recall history

Maintain two histories:

1. `publicHistory`: the existing leak-free sequence used by the neural trunk.
2. `infoHistoryByObserver`: a per-seat action-observation sequence used only for
   `infoSetKey`.

Each exact real or simulated action appends one observer-projected event for
each seat:

- public plays, quests, challenges, locations, and visible resolutions retain
  their visible details;
- face-down inking redacts the selected card for the opponent but retains it for
  the actor;
- hidden draws update visible counts, and the drawing actor's private projected
  state records the revealed own card;
- private prompt details are retained only for observers allowed to know them.

The current neural `history` can remain public-only. The stricter private
perfect-recall sequence belongs in the key, not necessarily in the trunk.

### 6.4 Do not merge on current board alone

Two paths that reach the same visible board can still be different information
sets when an observer remembers different past information. The key must include
projected history, not only the current projection.

### 6.5 Action identity and execution identity are different

Every legal action returned by the bridge needs:

```typescript
type SearchLegalAction = {
  actionId: string;   // logical actor-visible edge id for shared tree statistics
  stableKey: string;  // exact engine execution token for this lane and state
  family: string;
  // existing candidate descriptor fields
};
```

`stableKey` is an execution capability for one concrete state. `actionId` is the
shared information-set edge identity.

For most current Lorcana candidates they can initially be equal. Keep them
separate in the API now because:

- internal bag/effect IDs can be state-local;
- action descriptors can need visibility-aware redaction;
- sampled worlds can generate different exact instance tokens for logically
  equivalent visible choices;
- tree statistics must not depend on opaque runtime IDs.

Add an assertion: for the same `infoSetKey`, each `actionId` must resolve to at
most one currently executable `stableKey` per lane. Reject duplicate mappings.

## 7. Python Search Data Structures

Replace fixed action-index arrays in `lorcana-bot/search/node.py` for the new
belief path. The existing node can remain temporarily for legacy `run()`.

```python
@dataclass
class EdgeStats:
    action_id: str
    descriptor: dict
    prior_sum: float = 0.0
    prior_samples: int = 0
    visits: float = 0.0
    value_sum: float = 0.0
    availability: float = 0.0
    virtual_loss: float = 0.0
    virtual_availability: float = 0.0
    invalid_execs: int = 0

    @property
    def prior(self) -> float:
        return self.prior_sum / max(self.prior_samples, 1)

    @property
    def q(self) -> float:
        return self.value_sum / max(self.visits, 1e-8)


@dataclass
class SharedInfoSetNode:
    key: str
    actor: str | None
    exemplar_obs: dict
    terminal: bool
    winner: str | None
    edges: dict[str, EdgeStats]
    expanded: bool = False
    leaf_value: float = 0.0


class InfoSetTable:
    nodes: dict[str, SharedInfoSetNode]

    def get_or_create(self, obs: dict) -> tuple[SharedInfoSetNode, bool]:
        ...
```

Keep lane-local mappings out of shared nodes:

```python
@dataclass
class LaneObservation:
    lane_id: int
    obs: dict
    by_action_id: dict[str, dict]  # current actionId -> concrete legal descriptor
```

Do not store a structural `children[action_index]` pointer as the source of
truth. After every transition, resolve the next node from `obs["infoSetKey"]`.
An optional `next_keys` counter can remain for diagnostics.

## 8. Selection, Expansion, and Backup

### 8.1 Available-action subset

At a node visit, derive:

```python
available_ids = set(lane_obs.by_action_id)
```

Selection must only score those actions. Never choose from a node-global legal
array without intersecting the current lane's available set.

### 8.2 Edge availability

Record availability exposure for every currently available sibling action when
one action is selected from the node.

For a successful simulation backup:

```python
for action_id in available_ids:
    node.edges[action_id].availability += 1.0
node.edges[selected_id].visits += 1.0
node.edges[selected_id].value_sum += value_from_node_actor_pov
```

For an invalid exact execution, remove pending virtual statistics but do not add
real visits, values, or availability. Quarantine and report the lane.

### 8.3 Availability-aware PUCT

Cowling et al. replace the normal parent-visit term with action availability in
their subset-armed UCB rule. For this repo's PUCT variant, use:

```python
n_eff = edge.visits + edge.virtual_loss
a_eff = edge.availability + edge.virtual_availability
q = (edge.value_sum - edge.virtual_loss) / max(n_eff, 1e-8) if n_eff else fpu
u = c_puct * normalized_prior * sqrt(max(a_eff, 1.0)) / (1.0 + n_eff)
score = q + u
```

At a root where every action is always available, this reduces to the current
PUCT shape up to the usual parent-visit equivalence.

This availability-aware PUCT formula is a repo-specific adaptation of the
paper's availability-aware UCB rule. Keep it isolated and benchmark it against:

- plain parent-total PUCT with legal filtering;
- availability-aware PUCT;
- OpenSpiel-style inconsistent-action filtering.

### 8.4 Progressive widening

Apply progressive widening to the currently available subset:

```python
allowed = ceil(pw_c * (node_successful_visits + 1) ** pw_alpha)
candidate_ids = sort_available_by_prior(available_ids)[:allowed]
```

Register every newly observed edge in the node union, but only select open
actions. If a newly discovered available action has no stored prior, queue a
prior refresh for that observation before using it.

### 8.5 Priors under action-set variation

The current evaluator returns normalized legal priors, not raw logits. Store a
running prior mean per edge and renormalize across the current available subset
before selection:

```python
p = edge.prior
p_available = p / sum(node.edges[a].prior for a in available_ids)
```

Longer term, expose policy logits from `NetEvaluator` and store unnormalized
scores. That avoids normalization artifacts when legal subsets vary.

### 8.6 Root noise

Apply Dirichlet noise once to the real root node's action priors, keyed by
`actionId`. Do not independently perturb a separate root tree for each sampled
world.

Assert that all root determinizations resolve to the same root `infoSetKey` and
same root action-ID set. A failure is a hidden-information leak or key bug.

### 8.7 Value backup

Keep the existing two-player zero-sum actor-relative value conversion from
`lorcana-bot/search/ismcts.py:33-42`.

For each selected path edge:

```python
v = value_for(node.actor, leaf, leaf_value)
edge.visits += 1.0
edge.value_sum += v
```

Terminal values use the winner. Non-terminal values use the leaf actor's neural
value and flip sign when the backed-up node actor differs.

### 8.8 Importance-weighted fallback mode

Preferred mode is exact posterior sampling with `rho == 1`.

If an experimental proposal uses `q != b`, do not overwrite UCT visit meaning
with one float counter. Track both:

```python
raw_visits += 1
weighted_mass += rho
weighted_value_sum += rho * value
raw_availability += 1
```

Use raw counts for exploration, weighted sums for `Q`, and report ESS. Treat
weighted root policy targets as experimental until calibrated against direct
posterior sampling.

## 9. Bun Bridge: Adaptive Search Lanes

### 9.1 Why lanes are required

Python must choose each next action after seeing the sampled lane's current
filtered observation. A precomputed `run_paths(root, paths)` call cannot do
that.

The correctness-first bridge keeps batched IPC at each depth:

```text
begin B sampled lanes       1 IPC
step active lanes depth 0   1 IPC
step active lanes depth 1   1 IPC
...
evaluate new leaves         1 batched network call
drop B lanes                1 IPC
```

This is slower than current precomputed path execution but correct. Section 16
describes the production optimization path.

### 9.2 Server lane state

Add to `Session`:

```typescript
type SnapshotBundle = {
  state: any;
  publicHistory: any[];
  infoHistoryByObserver: Record<string, any[]>;
};

type SearchLane = {
  id: number;
  state: any;
  publicHistory: any[];
  infoHistoryByObserver: Record<string, any[]>;
  worldId: string;
};

private snapshots = new Map<number, SnapshotBundle>();
private searchLanes = new Map<number, SearchLane>();
private laneCounter = 0;
```

### 9.3 New RPCs

Add:

```text
begin_ismcts_wave(rootSnapshot, rootSelf, worlds[])
  -> lanes[{laneId, worldId, obs}]

step_ismcts_lanes(steps[{laneId, stableKey}])
  -> lanes[{laneId, obs, success, matched, invalidPath, failedKey}]

drop_ismcts_lanes(laneIds[])
  -> ok
```

`begin_ismcts_wave`:

1. Load the bundled root snapshot.
2. Clone its histories.
3. Apply one explicit world determinization.
4. Store the resulting authoritative state reference and branch histories in a
   lane.
5. Return actor-filtered observation with `infoSetKey` and `actionId`s.

`step_ismcts_lanes`:

1. Load each lane's state with `shallowProtect`.
2. Execute exactly the requested `stableKey`.
3. Reject any fallback or mismatch.
4. Append branch-local public and per-observer projected history.
5. Store the new state reference back into the lane.
6. Return the new actor-filtered observation.

`drop_ismcts_lanes` removes all lane state references in `finally` blocks.

### 9.4 Refactor observation

Change:

```typescript
observe()
```

to accept history overrides:

```typescript
observe(
  publicHistory = this.publicHistory,
  infoHistoryByObserver = this.infoHistoryByObserver,
)
```

Compute `infoSetKey` from the current actor's filtered board and that actor's
projected history.

### 9.5 Refactor snapshots

Make `snapshot()` and `restore()` round-trip the whole `SnapshotBundle`, not
only engine state. Update `runPaths()` to use branch-local cloned histories so
plain search leaf evaluations also see hypothetical public events.

### 9.6 Explicit world validation

The Bun determinizer must reject malformed worlds:

- duplicate instance IDs;
- IDs absent from the hidden pool;
- wrong hand count;
- wrong inkwell count;
- hidden-pool conservation failure;
- non-string, empty, duplicate, or unknown supplied self-deck IDs;
- self-deck conservation failure.

Return a structured error with `worldId`. Never silently filter IDs as current
`server.ts:430` does.

## 10. Python Planner Flow

Add a new entry point:

```python
def run_infoset(
    self,
    root_obs: dict,
    belief_state,
    total_simulations: int | None = None,
) -> SearchResult:
    ...
```

The legacy wrapper has already been renamed `run_pimc_diagnostic()` and
quarantined from clean-label sample writing. Migrate future clean-label callers
only to the structured full-world `run_infoset()` path.

Wave algorithm:

```python
def run_infoset(...):
    table = InfoSetTable()
    root = table.get_or_create(root_obs)
    expand_root_and_add_noise(root)
    root_snap = engine.snapshot()

    try:
        while sims_done < total:
            worlds = [belief_state.sample_world(rng) for _ in range(wave_size)]
            lanes = engine.begin_ismcts_wave(root_snap, root_obs["self"], worlds)
            active = [LaneTrace.from_obs(lane) for lane in lanes]

            try:
                for depth in range(depth_limit + 1):
                    leaves = []
                    steps = []

                    for lane in active:
                        node, created = table.get_or_create(lane.obs)
                        lane.current_node = node

                        if terminal_or_depth_cut(node, depth):
                            leaves.append(lane)
                            continue

                        if created or needs_prior_refresh(node, lane.obs):
                            leaves.append(lane)
                            continue

                        action_id = node.select(lane.available_action_ids, cfg)
                        lane.record_pending(node, action_id, lane.available_action_ids)
                        steps.append({
                            "laneId": lane.id,
                            "stableKey": lane.stable_key(action_id),
                        })

                    evaluate_and_expand_new_leaves_in_one_batch(leaves)
                    backup_finished_lanes(leaves)
                    active = engine.step_ismcts_lanes(steps)
                    quarantine_invalid_lanes(active)

                    if not active:
                        break
            finally:
                engine.drop_ismcts_lanes([lane.id for lane in lanes])
    finally:
        engine.restore(root_snap)
        engine.drop_snapshot(root_snap)

    return result_from_real_root_order(root_obs, table)
```

Implementation details:

- A lane stops at its first new node, terminal node, invalid execution, or depth
  limit.
- Existing known depth-limit nodes may use their stored leaf value.
- New non-terminal leaves are batch-evaluated with `NetEvaluator.batch_eval`.
- Materialize and merge edge priors by `actionId`.
- Remove virtual loss in every success, invalid, exception, and cleanup path.
- Root `pi` is emitted in the exact order of `root_obs["legal"]`.

## 11. Configuration and Caller Migration

### 11.1 Change simulation-budget semantics

Current caller semantics are:

```text
total work = n_worlds * sims_per_world
```

Full ISMCTS semantics are:

```text
total work = total_simulations
one posterior world sampled per simulation
```

Change `SearchConfig`:

```python
@dataclass
class SearchConfig:
    simulations: int = 64        # total root-sampled simulations
    batch_size: int = 8          # sampled lanes per wave
    depth_limit: int = 8
    ...
```

Remove `n_determinizations`; it is not used meaningfully today.

During migration, preserve compute parity explicitly:

```python
total_simulations = old_n_worlds * old_sims_per_world
```

Then change CLI help and defaults intentionally. Do not silently reduce search
budget by a factor of `n_worlds`.

### 11.2 Change monitoring

Current training code estimates sims by multiplication:

- `lorcana-bot/training/run.py:66`
- `lorcana-bot/training/parallel.py:47`

Use actual results:

```python
mon.add(sims=res.sims, decisions=1)
```

Aggregate:

- started simulations;
- completed simulations;
- invalid lanes;
- determinization rejects;
- unique infosets;
- transposition hits;
- average available actions;
- root posterior ESS;
- max evaluation batch;
- lane RPC count;
- state transitions per decision.

### 11.3 Migrate every caller

Update:

- `lorcana-bot/training/run.py`
- `lorcana-bot/training/parallel.py`
- `lorcana-bot/training/selfplay.py`
- `lorcana-bot/training/league.py`
- `lorcana-bot/training/league_train.py`
- `lorcana-bot/training/exploitability.py`
- README and `CLAUDE.md`

League parity finding #7 still applies: use per-seat persistent trackers,
clean exact-execution filtering, trivial-decision filtering, stuck guards,
turn caps, and auxiliary targets consistently.

## 12. File-by-File Implementation Checklist

### `lorcana-bot/search/determinize.py`

- **DONE (Phase 1 audited GO):** define the canonical full reproducible `World`
  contract, strict opponent hidden-pool witness validation, and clean-label seed
  admission checks.
- **DONE (Phase 2 audited GO):** add log-domain exact count-constrained hand and
  hand/inkwell/deck assignment samplers from #5, raw `rho`, normalized `weight`,
  complete assignment audit logs, structured zero-ink handling, and fail-closed
  sampler input validation.
- Keep `rho`, `log_target`, and `log_proposal` diagnostics.
- **PARTIAL:** Phase 1 world-contract and Phase 2 sampler-math tests landed; Phase 12
  adds end-to-end deterministic replay proofs.

### `lorcana-bot/search/belief_filter.py`

- Maintain one tracker per observing seat in callers.
- Track joint hand/inkwell particles or a structured world sampler.
- Add action-likelihood correction hook.
- Expose `sample_world(rng)`, not only `to_worlds(n)`.
- Report ESS and reseed/resample counters.

### `lorcana-bot/search/evaluator.py`

- Add structured hand+inkwell belief evaluation.
- Optionally expose raw policy logits for robust prior pooling.
- Keep batch leaf evaluation.

### `lorcana-bot/search/node.py`

- Add `EdgeStats`, `SharedInfoSetNode`, and `InfoSetTable`.
- Key actions by `actionId`, not array position.
- Add availability counts and virtual availability.
- Add availability-aware PUCT and progressive widening over current availability.
- Keep or isolate legacy `InfoSetNode` until plain `run()` migrates.

### `lorcana-bot/search/ismcts.py`

- Add `run_infoset()`.
- Sample one world per simulation.
- Replace structural child-pointer descent with adaptive lane observation lookup.
- Batch lane transitions by depth.
- Batch-evaluate new leaves.
- Apply root noise once.
- Back up shared node-edge statistics.
- Align returned root policy to real root legal order.
- Quarantine invalid executions without polluting shared nodes.
- Return detailed search stats.

### `lorcana-bot/engine/node_server/server.ts`

- Split public history from per-observer key history.
- Add observer-projected branch-history appenders.
- Add actor-filtered `infoSetKey`.
- Add logical `actionId`.
- Bundle histories into snapshots.
- Validate explicit world determinizations.
- Add adaptive lane map and three lane RPCs.
- Make `runPaths()` branch-history aware for legacy plain search.
- Drop all lane references on cleanup.

### `lorcana-bot/engine/bridge.py`

- Add typed wrappers for `begin_ismcts_wave`, `step_ismcts_lanes`, and
  `drop_ismcts_lanes`.
- Serialize `World` specs.
- Validate response lengths and lane IDs.
- Preserve `finally` cleanup on RPC errors.

### Training callers

- Keep one belief tracker per canonical seat.
- Pass current seat tracker into full ISMCTS.
- Count actual simulations from `SearchResult`.
- Log new correctness and performance metrics.

### Documentation

- Replace claims that Phase 2 is already full B-ISMCTS.
- Explain that `run_belief()` used root-pooled per-world trees before #2.
- Document total-simulation semantics.
- Document baseline full ISMCTS versus optional opponent-fair hardening.

## 13. Required Tests

### 13.1 Pure node tests

Add tests for:

1. Same `infoSetKey` merges statistics from two observations.
2. Different hidden truth with identical actor-visible observation yields the
   same key.
3. Different actor-known hand yields a different key.
4. Different public or observer-visible history yields a different key.
5. Opponent face-down ink identity does not change the observer's key.
6. Own face-down ink identity does change the owner's key.
7. Selection never emits an action absent from the current available subset.
8. Availability increments every available sibling, not only the selected edge.
9. Rarely available actions are not penalized as if they were always available.
10. Virtual availability and virtual loss are removed on success and failure.
11. Progressive widening only opens currently available actions.
12. Root output policy aligns by `actionId`, never array index.

### 13.2 Bridge tests

Add tests for:

1. Snapshot/restore round-trips public and observer histories.
2. A lane determinization cannot change real session history.
3. Lane A and lane B evolve independently.
4. `step_ismcts_lanes` executes exactly the requested lane stable key.
5. Invalid lane stable key returns a mismatch and does not mutate another lane.
6. Dropped lanes cannot be stepped.
7. Explicit malformed worlds are rejected.
8. Full hidden-pool conservation holds.
9. Same root snapshot plus same world seed produces the same trajectory.
10. Root determinizations preserve root `infoSetKey` and root action-ID set.

### 13.3 Search integration tests

Build a small synthetic imperfect-information game fixture if possible. It
should force the distinguishing behavior:

```text
Hidden world A: action X is best.
Hidden world B: action Y is best.
The acting player cannot distinguish A from B.
```

Assert:

1. Legacy per-world root pooling exhibits contingent-world preference.
2. Shared ISMCTS stores one root node.
3. Shared deeper infoset statistics receive visits from both worlds.
4. Selected policy is based on one executable information-set strategy.
5. No unavailable action executes.
6. Engine returns to the real root after search.
7. Branch-local neural history contains hypothetical search actions.
8. `res.sims` equals completed root-sampled simulations.

Also retain and update:

- `lorcana-bot/tests/test_search.py`
- `lorcana-bot/tests/test_search_belief.py`
- `lorcana-bot/tests/test_bridge.py`
- `lorcana-bot/tests/test_determinize.py`

### 13.4 Leak tests

Add mutation tests:

1. Shuffle opponent hidden hand identities while preserving visible state:
   root info key unchanged.
2. Shuffle opponent hidden inkwell identities: observer info key unchanged.
3. Shuffle either deck order: observer info key unchanged.
4. Change own known hand: own info key changes.
5. Scramble `obs["hidden"]`: policy/value encoding, info key, and root legal
   action IDs remain unchanged.

### 13.5 Statistical tests

For exact posterior sampling:

1. Empirical hand/inkwell marginals match DP target marginals.
2. Every sample satisfies public counts.
3. Every direct-posterior sample has `rho == 1`.
4. Same RNG seed reproduces sampled worlds.
5. Particle-filter sampling frequencies match normalized particle weights.

### 13.6 Performance tests

Track before and after:

- decisions per second;
- completed simulations per second;
- Bun RPCs per decision;
- exact engine transitions per decision;
- average search depth;
- unique nodes per decision;
- transposition hit rate;
- net leaf batch size;
- lane memory high-water mark.

Correctness lands first. Do not claim performance parity with `run_paths` until
measured.

## 14. Acceptance Gates

The #2 work is complete only when all of these are true:

1. `run_infoset()` uses one `InfoSetTable` per root decision.
2. One posterior world is sampled per simulation.
3. No per-world subtree `BISMCTS.run()` call remains in the belief path.
4. Every deeper transition resolves the next node by actor-filtered
   `infoSetKey`.
5. Selection intersects shared node edges with current lane availability.
6. Every edge tracks availability separately from visits.
7. Root noise is applied once.
8. Branch-local hypothetical history reaches both leaf network observation and
   info-set key computation.
9. Snapshot restore includes histories.
10. Root policy is aligned by logical `actionId`.
11. Exact-execution mismatches quarantine simulations instead of creating shared
    neutral child nodes.
12. Belief self-play uses per-seat trackers.
13. Search uses the active hand+inkwell world model.
14. Unit, bridge, leak, statistical, and integration tests pass.
15. Documentation no longer calls root-only pooling full ISMCTS.

## 15. Opponent-Perspective Hardening After Baseline #2

### 15.1 Why this is separate

Root sampling fixes the primary PIMC strategy-fusion bug: the root player must
choose one strategy across indistinguishable root worlds.

However, classic root-sampled ISMCTS can still model an opponent as though that
opponent knew root-player private information. Goodman discusses this leakage
when motivating re-determinizing ISMCTS.

For Lorcana, this matters when an opponent action's downstream value depends on
the root player's hidden hand, hidden inkwell identities, or future deck order.
An opponent search node keyed without that private information can still learn
values only inside the root player's actual hidden context during one search.

### 15.2 Required baseline stance

Ship and benchmark baseline #2 first:

- root-consistent world sampled once per simulation;
- actor-filtered shared info-set nodes;
- no actor-switch re-determinization.

Label it accurately as root-sampled shared-tree ISMCTS. It removes the current
audit finding and has a clear conformance target.

### 15.3 Optional stricter experiment

Add an experimental `opponent_model="redeterminized"` mode:

1. At an actor switch, sample hidden zones unknown to the new actor from that
   actor's posterior.
2. Preserve everything the new actor can observe and everything publicly
   committed by the trajectory.
3. Let the actor choose from its actor-perspective re-determinized state.
4. Execute the chosen logical `actionId` against the root-consistent trajectory
   state when compatible.
5. Resample or fall back to compatible available actions if the action cannot
   execute in the trajectory state.
6. Measure incompatible-action rate and exploitability proxy.

This dual-state selection/trajectory design is an engineering inference for
this repo, not a verbatim algorithm from the papers. It avoids silently backing
up impossible fake trajectories, one of the known hazards of naive
re-determinization.

Do not mix this experiment into the first #2 patch. It needs separate tests and
ablation results.

## 16. Performance Roadmap

### 16.1 Correctness-first implementation

Use Bun adaptive lanes with one batched transition RPC per active depth. This is
the smallest auditable change that preserves the Python policy/value stack and
server-authoritative rules.

### 16.2 Optimization after correctness

Once conformance is green, reduce boundary overhead in this order:

1. Allow `step_ismcts_lanes` to execute short server-side continuations when the
   Python table proves the continuation has one available action.
2. Replace JSON lane payloads with a compact binary or shared-memory format.
3. Move info-key hashing and action-ID canonicalization fully into Bun.
4. Mirror the shared search table into Bun so selection and transitions stay
   server-side, returning only new leaves for batched PyTorch evaluation.
5. Long term, follow the architecture document's in-process WASM/kernel-host
   direction so only leaf inference crosses a process boundary.

Do not optimize by returning to preselected cross-world `run_paths`. That would
reintroduce the correctness bug.

## 17. Suggested Commit Sequence

Current checkpoint: the canonical `World` portion of item 1 is complete and
audited. The next implementation step is the remaining Phase-2 sampler math.

1. **#5 sampler**: exact joint hand/inkwell world sampling; the reproducible
   canonical `World` contract is already established by Phase 1.
2. **#4 trackers**: per-seat persistent belief and observed-action likelihood.
3. **History correctness**: bundled snapshots, branch-local public history,
   per-observer key history.
4. **Identity contract**: actor-filtered `infoSetKey`, logical `actionId`, leak
   tests.
5. **Bridge lanes**: begin/step/drop adaptive lane RPCs and validation tests.
6. **Shared node table**: keyed edges, availability counts, PUCT, pure tests.
7. **Planner**: `run_infoset()`, lane waves, batched leaves, root result mapping.
8. **Caller migration**: total-simulation semantics, per-seat trackers,
   monitoring, league parity.
9. **Docs and ablation**: update claims, compare legacy pooling against full
   shared-tree ISMCTS.
10. **Optional hardening**: separate re-determinized opponent-model experiment.

## 18. Review Checklist for Each Patch

Before merging any patch in this sequence, ask:

1. Can any authoritative hidden identity reach policy/value tensors?
2. Can any authoritative hidden identity reach `infoSetKey`?
3. Can one sampled lane mutate another lane or the real game?
4. Can action statistics be indexed by positional legal-array order?
5. Can selection choose an edge unavailable in the current sampled lane?
6. Can an exact-execution mismatch enter shared value statistics?
7. Can snapshot restore leave hypothetical history behind?
8. Did a CLI or monitor retain the old `n_worlds * sims_per_world` assumption?
9. Are direct posterior samples unit-weighted?
10. Is a performance optimization preserving adaptive per-world selection?

If any answer is wrong or uncertain, #2 is not complete.
