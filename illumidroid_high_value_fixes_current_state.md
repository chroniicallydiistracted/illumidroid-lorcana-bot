# illumidroid-lorcana-bot Current-State Findings and High-Value Fix Backlog

## Executive Summary

The current illumidroid build is substantially stronger than our earlier snapshot.

Major previously identified weaknesses have been fixed:

- Python `hash()` card identity instability has been replaced by a committed compact card vocabulary plus stable `blake2b` OOV hashing.
- The old collapsed action representation has been replaced by a rich action candidate encoder.
- The policy head is now a candidate-token transformer rather than a simple category/source/target scorer.
- Public history encoding has been added.
- Hidden inkwell belief labels and belief logits have been added.
- Belief-world pooling now maps by exact stable key rather than action index.
- Persistent belief tracking is integrated into default and parallel self-play.
- KL anti-collapse is active in the default self-play trainer.
- `train.sh` now exposes both default self-play and Phase 3 league training.
- The stale `training/distributed.py` issue appears resolved by removal.

Current high-level comparison:


---

## Current illumidroid Strengths

### 1. Rich Lorcana-shaped neural action model

Current `lorcana-bot/engine/serialization.py` no longer encodes actions only as family/source/first-target. It now emits:

- `cand_feats`
- `cand_tok_pos`
- `cand_tok_role`
- `cand_tok_mask`
- `cand_named_id`
- legacy `action_src`, `action_tgt`, `action_tgt2`, and `action_choice` for compatibility

The candidate descriptor covers:

- source
- all targets up to cap
- singers
- shift target
- banish costs
- discard costs
- exert costs
- named-card choice
- destination zones
- choice index
- optional resolution
- cost type and ink cost

Primary files:

- `lorcana-bot/engine/serialization.py`
- `lorcana-bot/engine/node_server/server.ts`
- `lorcana-bot/network/heads.py`
- `lorcana-bot/network/model.py`

This is a major Lorcana-relevant improvement because many Lorcana candidates are only distinguishable by costs, singers, shift targets, optional choices, and multi-target structure.

### 2. Candidate-transformer policy head

Current `lorcana-bot/network/heads.py` uses a candidate-transformer policy head. Each legal candidate becomes a small token set:

- candidate CLS token
- optional named-card token
- role-tagged pointers into card-token representations

This allows the policy to distinguish candidates that previously collapsed to the same features.

Primary file:

- `lorcana-bot/network/heads.py`

### 3. Deterministic card vocabulary

Current `lorcana-bot/engine/serialization.py` uses:

- `PAD_ID = 0`
- `HIDDEN_ID = 1`
- committed card vocabulary map
- dedicated known-card rows
- stable `blake2b` OOV hashing

This fixes the previous cross-worker Python `hash()` instability issue.

Primary file:

- `lorcana-bot/engine/serialization.py`

### 4. Public history encoding

Current trunk has a `HistoryEncoder` that summarizes a rolling window of public events and fuses it into the global token.

Primary files:

- `lorcana-bot/network/trunk.py`
- `lorcana-bot/engine/node_server/server.ts`
- `lorcana-bot/engine/serialization.py`

This helps the bot infer gameplay context from visible sequences: played cards, hand-count changes, ink-count changes, lore pressure, and turn timing.

### 5. Hand and inkwell belief heads

Current belief encoding includes hidden hand, hidden deck, and hidden inkwell. The belief head emits two channels:

- probability the card is in opponent hand
- probability the card is in opponent inkwell

Primary files:

- `lorcana-bot/engine/serialization.py`
- `lorcana-bot/network/heads.py`
- `lorcana-bot/network/model.py`
- `lorcana-bot/training/learner.py`

This is a strong Lorcana-specific improvement because face-down ink choices are strategically meaningful.

### 6. Stable-key belief pooling

Current `run_belief()` pools root statistics by exact stable key instead of assuming action-index alignment.

Primary file:

- `lorcana-bot/search/ismcts.py`

This fixes a major correctness risk where determinized worlds with reordered legal actions could assign visits and values to the wrong action.

### 7. Persistent belief tracker in default and parallel self-play

Current default self-play creates a `BeliefTracker` per game and passes it into `run_belief()`.

Primary files:

- `lorcana-bot/training/run.py`
- `lorcana-bot/training/parallel.py`
- `lorcana-bot/search/belief_filter.py`

This makes belief worlds temporally coherent instead of independently resampled every decision.

### 8. Auxiliary race/clock learning

Current model includes an auxiliary head predicting:

- final self lore
- final opponent lore
- turns to game end

Primary files:

- `lorcana-bot/engine/serialization.py`
- `lorcana-bot/network/heads.py`
- `lorcana-bot/network/model.py`
- `lorcana-bot/training/learner.py`
- `lorcana-bot/training/run.py`
- `lorcana-bot/training/parallel.py`

This directly targets Lorcana race-clock reasoning.

---

# High-Value Fix Backlog

## 1. Make MCTS path execution exact, not fallback-tolerant

### Current logic

Real game steps return:

- `executed`
- `requested`
- `matched`
- `success`

Self-play filters recorded samples using these fields.

However, MCTS search uses `runPaths()` in `lorcana-bot/engine/node_server/server.ts`. `runPaths()` executes each key using `executeKey(key)` and ignores whether the exact requested stable key executed.

`executeKey()` uses `chooseKeyStrategy(key)`. `chooseKeyStrategy()` sorts the requested key first but leaves every other candidate behind it.

### Why this is bad

If the requested stable key fails, Lorcanito may execute a different fallback candidate. MCTS then evaluates a leaf that does not correspond to the path selected by the tree.

This can corrupt:

- leaf values
- root visit policies
- MCTS-backed training targets
- belief-world evaluations
- action priors learned from search

In Lorcana terms, the bot may think it evaluated “challenge this character,” but the engine may have executed a different fallback action.

### How to fix it

Implement an exact-only execution mode for search.

In `server.ts`:

1. Add an exact strategy that returns only the matching candidate, not the rest of the candidate list.
2. Replace or supplement `executeKey()` with `executeKeyDetailed()`:

```ts
type ExecuteKeyResult = {
  success: boolean;
  requested: string;
  executed: string | null;
  matched: boolean;
};

private executeKeyDetailed(key: string): ExecuteKeyResult {
  // exact-only implementation
}
```

3. Update `runPaths()` to stop immediately when a path step fails or mismatches.
4. Return diagnostics for invalid search paths:

```ts
{
  obs: this.observe(),
  invalidPath: true,
  requested,
  executed,
  failedAtDepth
}
```

In Python search:

1. Detect invalid path results in `_simulate()` and `run_batched()`.
2. Remove virtual loss if applicable.
3. Do not expand the child as a valid leaf.
4. Optionally penalize or mask that edge for the current root tree.
5. Add regression tests.

### What it achieves

This makes search trustworthy. Every MCTS leaf value will correspond to the exact stable-key sequence chosen by the tree.

Priority: **Critical**

---

## 2. Bring the league trainer up to parity with the default self-play trainer

### Current logic

`training/run.py` and `training/parallel.py` include:

- persistent `BeliefTracker`
- clean `matched/success` filtering
- forced/single-action filtering
- auxiliary race/clock targets

`training/league.py` currently lags behind. `NetPlayer.act()`:

- calls `run_belief()` without a persistent tracker
- executes `engine.step(...)`
- ignores `matched/success`
- records samples without clean filtering
- records no aux target metadata

`training/match.py` then converts raw records into `Sample` objects without aux targets.

### Why this is bad

The Phase 3 league is supposed to be the strongest training path, but it currently trains from lower-quality samples than the default self-play runner.

This weakens:

- PFSP training
- Elo-based population training
- frozen past-main training
- exploitability proxy training

### How to fix it

Update `NetPlayer` and `play_match()`.

Raw records should include:

```python
(enc, pi, belief, actor, self_idx, turn)
```

Then `play_match()` should create:

```python
Sample(
    enc=enc,
    pi=pi,
    z=outcome_from_actor_pov,
    belief=belief,
    aux=aux_targets(final_obs, self_idx, turn),
)
```

Add a per-game tracker lifecycle:

```python
class Player:
    def begin_game(self) -> None:
        pass
```

`NetPlayer.begin_game()` should reset its `BeliefTracker`.

`NetPlayer.act()` should pass the tracker into `run_belief()`.

Also enforce the same sample filtering used by `run.py`:

```python
clean = result.get("success", True) and (
    result.get("matched", True) or result.get("executed") == "passTurn"
)
trivial = len(legal) <= 1 or obs.get("forced")
```

Only record when:

```python
clean and not trivial and actor is not None and len(res.pi) == len(legal)
```

### What it achieves

The league trainer becomes the strongest training path instead of a lower-quality side path.

Priority: **Critical**

---

## 3. Add Botcana-style checkpoint provenance and promotion gates

### Current logic

illumidroid saves checkpoints mainly as:

```python
{
    "model": ...,
    "d_model": d_model,
    "layers": layers,
}
```

Botcana2.0 has a stronger provenance model. It tracks fair/oracle policy, promotion eligibility, oracle contamination, and rejects unsafe promotion artifacts.

### Why this is bad

Without checkpoint metadata, you cannot prove whether a model is competition-legal or training-clean.

A checkpoint should record:

- whether oracle information was ever used
- whether hidden truth was used only for belief labels
- deck pool identity
- card vocabulary identity
- Lorcanito version
- illumidroid version
- search caps
- training mode
- promotion eligibility

### How to fix it

Add checkpoint metadata:

```python
metadata = {
    "training_mode": "selfplay",
    "information_policy": "fair",
    "oracle_allowed": False,
    "saw_oracle": False,
    "belief_supervision_used": True,
    "hidden_truth_used_only_for_labels": True,
    "search_caps": SEARCH_CAPS,
    "deck_pool_hash": deck_pool_hash,
    "card_vocab_hash": card_vocab_hash,
    "lorcanito_commit": lorcanito_commit,
    "illumidroid_commit": illumidroid_commit,
    "created_at": timestamp,
    "promotion_eligible": True,
}
```

Add load-time gates:

- `--promotion` rejects missing metadata.
- `--promotion` rejects `saw_oracle=true`.
- `--promotion` rejects unknown or mixed information policy.
- `--promotion` rejects incompatible card vocab.
- `--promotion` rejects incompatible deck pool.
- `--promotion` rejects pre-action-encoding-fix checkpoints.

### What it achieves

This makes training artifacts auditable and prevents accidental hidden-information contamination.

Priority: **Critical**

---

## 4. Use inkwell belief inside search, not only as an auxiliary loss

### Current logic

The bot learns inkwell belief:

- `belief_label_ink`
- `belief_logits_ink`
- inkwell belief loss

But `BeliefEvaluator` only returns hidden hand + hidden deck IDs and probabilities for current hand membership.

### Why this is bad

Inkwell identity matters in Lorcana. Knowing what the opponent likely inked changes how the bot should play around:

- removal
- sweepers
- songs
- late-game threats
- combo pieces
- uninkable cards
- matchup-specific power cards

The model learns this signal, but search does not currently consume it.

### How to fix it

Extend belief evaluation to return both channels:

```python
BeliefState(
    pool_ids=...,
    p_hand=...,
    p_ink=...,
)
```

Extend determinization to sample a full hidden partition:

```text
hand / deck / inkwell
```

Update server-side `determinize()` to accept:

```ts
handInstanceIds: string[]
inkwellInstanceIds: string[]
```

Then repartition opponent hidden zones:

- hand = sampled hand IDs
- inkwell = sampled inkwell IDs
- deck = remaining hidden deck IDs

### What it achieves

It turns hidden inkwell belief into active gameplay strength.

Priority: **High**

---

## 5. Make belief tracking action-likelihood aware

### Current logic

`BeliefTracker` reuses carried particles and reweights them by current neural belief probabilities. It does not yet evaluate how likely the opponent’s observed action was under each hidden-hand world.

### Why this is bad

Opponent actions are evidence.

A strong Lorcana player infers hidden hand from behavior:

- opponent did not play a curve card
- opponent inked instead of singing
- opponent avoided removing a threat
- opponent challenged instead of questing
- opponent sequenced in a way that implies a follow-up

The current tracker is temporally coherent but not fully action-conditioned.

### How to fix it

After each observed opponent action:

1. For each particle/world:
   - determinize to that world
   - enumerate legal actions
   - evaluate the opponent policy prior
   - find the probability assigned to the observed stable key
2. Reweight:

```python
particle_weight *= P(observed_action | particle_world)
```

3. Resample when ESS drops below threshold.

For efficiency:

- only update after opponent decisions
- cap particles
- use policy-only forward, not full MCTS
- cache evaluations where possible
- optionally approximate likelihood with legality plus neural prior

### What it achieves

The bot starts inferring hidden hand from opponent behavior like a real high-level Lorcana player.

Priority: **High**

---

## 6. Move from automation-candidate caps toward full Lorcanito move grammar

### Current logic

The bridge exposes Lorcanito automation candidates as the legal action set.

It widens search caps:

```ts
targetPool: 24,
targetCombinationsPerFamily: 48,
choiceIndices: 16,
singerCombinations: 32,
```

The bridge comment explicitly says full `getAvailableMoves` grammar is a larger project.

### Why this is bad

Automation candidates may not cover the entire raw legal move grammar.

Future or complex cards may require:

- many target combinations
- ordered destination choices
- nested optional choices
- name-a-card inputs
- unusual payment combinations
- new mechanics

Wider caps reduce but do not eliminate this risk.

### How to fix it

Phase A:

- expose diagnostics for candidate cap pressure
- log when families are capped
- record raw counts before and after dedupe where possible

Phase B:

- replace automation expansion for high-value families with raw move-option expansion:
  - resolve effect targets
  - sing / sing-together costs
  - activated ability costs
  - destination choices

Phase C:

- make the neural policy rank canonical raw engine candidate objects, not only automation summaries

### What it achieves

Moves illumidroid from a strong automation-layer bot toward a true rules-complete Lorcanito bot.

Priority: **High**

---

## 7. Add search-time invalid-action regression tests

### Current logic

Real-step sample filtering exists, but search-path execution integrity is not sufficiently protected.

### Why this is bad

The dangerous failure mode is silent. A fallback path can return a valid observation, causing tests to pass while MCTS evaluates the wrong line.

### How to fix it

Add tests for:

1. Exact path execution:
   - given a root stable key, `run_paths()` executes exactly that key

2. Missing key rejection:
   - pass a stable key not present in legal actions
   - `run_paths()` must not execute a fallback action

3. MCTS no-fallback expansion:
   - mock invalid path result
   - prove MCTS does not expand that child as a valid leaf

4. Real-step sample filtering:
   - force mismatch
   - prove self-play does not store a sample

### What it achieves

Protects the integrity of MCTS and all MCTS-derived training targets.

Priority: **High**

---

## 8. Add league-path parity tests

### Current logic

Default self-play and league training differ.

### Why this is bad

The codebase can drift so that the strongest conceptual trainer has weaker sample hygiene.

### How to fix it

Add tests proving league path has:

- persistent belief tracker
- clean `matched/success` filtering
- forced/single-action filtering
- aux target emission
- correct `--no-belief` propagation
- no sample recording on mismatch

### What it achieves

Prevents PFSP/league training from silently lagging behind default self-play.

Priority: **High**

---

## 9. Upgrade value learning beyond terminal-only win/loss

### Current logic

The value head is trained on terminal outcome `z`. The aux head adds final lore and turns-to-end, but the main value head still uses only terminal result.

### Why this is incomplete

Terminal win/loss is sparse and noisy.

Many good Lorcana moves improve:

- board stability
- lore clock
- hand resources
- future singer availability
- threat density
- opponent pressure

without immediately changing terminal outcome.

### How to fix it

Add auxiliary heads or targets for:

- next-turn lore delta
- board material delta
- hand-size delta
- expected opponent lore next turn
- lethal threat next turn
- can-win-next-turn binary target

Longer term:

- train n-step bootstrapped value targets
- train value from MCTS improved estimates, not only final game result

### What it achieves

Improves non-terminal position evaluation, which directly improves search quality.

Priority: **High**

---

## 10. Add matchup/deck identity conditioning

### Current logic

The bot sees current cards and public history. It does not appear to have an explicit self-deck or matchup conditioning vector.

### Why this is bad

Lorcana decisions are matchup-dependent.

Good lines differ against:

- evasive aggro
- control
- ramp
- discard
- Steelsong
- item engines
- locations
- midrange value decks

The bot can infer some of this from public cards, but early mulligan and inking decisions need stronger priors.

### How to fix it

Add legal conditioning features:

- own deck ID embedding
- own deck histogram embedding
- own ink-color distribution
- public opponent archetype inference from revealed cards
- opponent deck ID only in closed-decklist settings where it is legally known

Do not feed hidden opponent decklist in competition-legal training.

### What it achieves

Improves mulligan, inking, early sequencing, and matchup-specific plans.

Priority: **High**

---

## 11. Add mulligan-specific training

### Current logic

Mulligan is treated as a normal candidate family.

### Why this is bad

Mulligan is a special high-leverage Lorcana decision:

- no board exists
- rewards are delayed
- action space is combinatorial
- correct choice depends on curve, inkability, colors, deck role, and matchup

General self-play learns this slowly.

### How to fix it

Create a dedicated mulligan training phase or head.

Features:

- curve after keep
- inkable count
- uninkable count
- early-play density
- color access
- duplicate penalties
- key synergy pieces
- matchup role

Data:

- opening hands from real decks
- rollout labels
- baseline comparisons
- search-improved keep/mulligan alternatives

Runtime:

- either use a mulligan-specific module for `alterHand`
- or enrich candidate features specifically for `alterHand`

### What it achieves

Improves one of the highest-impact decisions in every game.

Priority: **High**

---

## 12. Add Botcana-style counterfactual rollout labels

### Current logic

Botcana snapshots a decision, forces top-N candidates, verifies exact execution, rolls out the game, and labels the best candidate.

illumidroid mostly trains from MCTS/self-play targets.

### Why this matters

Counterfactual rollout data gives direct per-decision supervision:

> In this exact state, candidate A produced a better continuation than B/C/D.

That is cleaner than noisy full-game policy gradients or pure terminal outcomes.

### How to fix it

Port the concept into illumidroid:

1. Sample decision states.
2. Select top-N legal candidates by current policy, MCTS, or baseline.
3. For each candidate:
   - restore snapshot
   - force exact candidate
   - verify exact execution
   - continue with fair baseline or current net
   - record outcome and aux targets
4. Train policy from:
   - best-candidate one-hot
   - softmax over rollout values
   - advantage-weighted labels

### What it achieves

Gives the neural policy cleaner candidate-level learning data.

Priority: **High**

---

## 13. Add adaptive search budgets

### Current logic

Search budgets are mostly static:

- simulations
- depth
- batch
- number of belief worlds

### Why this is bad

Not every Lorcana decision deserves equal compute.

Examples:

- forced bag resolution needs no search
- obvious pass needs no search
- lethal turns need more search
- high-branch multi-target effects need more search
- high belief uncertainty needs more worlds
- low policy confidence needs more simulations

### How to fix it

Compute per-decision diagnostics:

- legal count
- policy entropy
- top-1 vs top-2 gap
- value uncertainty
- lore clock distance
- belief entropy
- terminal threat proximity

Scale:

- `simulations`
- `n_worlds`
- `depth_limit`

based on those diagnostics.

### What it achieves

More strength per CPU second, which directly addresses engine-throughput constraints.

Priority: **Medium-High**

---

## 14. Add search/evaluation caching

### Current logic

Batched search reduces IPC and batches neural inference, but there is no obvious transposition/value cache.

### Why this is bad

Search may revisit equivalent states through different action orders, especially around forced effects, pass sequences, or commutative choices.

### How to fix it

Add a public-info state fingerprint:

```python
(state_fingerprint, actor, legal_stable_keys_hash)
```

Cache:

```python
(priors, value)
```

Use it in:

- `NetEvaluator.batch_eval()`
- MCTS leaf expansion
- belief-world evaluation when visible state is identical

### What it achieves

Higher search throughput and fewer redundant engine/net evaluations.

Priority: **Medium-High**

---

## 15. Return full belief probabilities from the evaluator

### Current logic

The model can output hand and inkwell logits, but `belief_probs()` returns only hand probabilities.

### Why this is bad

The inkwell channel is trained but underused.

### How to fix it

Add:

```python
belief_probs_full(...)
```

Return:

```python
{
    "pool_ids": ...,
    "p_hand": ...,
    "p_ink": ...,
}
```

Then use this structured belief state in:

- determinization
- belief tracker
- diagnostics
- trace logs

### What it achieves

Connects the second belief channel to gameplay and debugging.

Priority: **Medium-High**

---

## 16. Make search caps observable and trainable

### Current logic

Search caps are widened but not exposed as part of observation diagnostics.

### Why this is bad

If a decision is capped, the bot may train as though missing legal options did not exist.

### How to fix it

Expose in observations:

```ts
legalDiagnostics: {
    searchCaps,
    familyCounts,
    cappedFamilies,
    rawCandidateCountBeforeDedup,
    finalCandidateCount
}
```

Training policy:

- skip capped decisions
- lower sample weight
- dynamically increase caps
- log cap pressure by family/card

### What it achieves

Prevents silent training on incomplete legal sets.

Priority: **Medium-High**

---

## 17. Add card-vocab regeneration tests

### Current logic

The bot uses committed `engine/card_vocab.json`.

### Why this can still go wrong

If decks change and vocab is stale, cards fall into the OOV band.

### How to fix it

Add a test:

```text
build_vocab(current decks) == committed card_vocab.json
```

Fail with a clear instruction:

```text
Run python -m engine.build_vocab
```

Also store:

- deck pool hash
- known card count
- card set range
- vocab build timestamp

### What it achieves

Keeps card identity learning collision-free as deck pools evolve.

Priority: **Medium**

---

## 18. Add real Lorcana scenario tests

### Current logic

Most tests validate mechanics, shapes, belief, bridge, and search sanity.

### Why this is not enough

A bot can pass shape tests while making bad Lorcana decisions.

### How to fix it

Add scenario tests for:

1. Lethal quest
2. Must-challenge-to-prevent-lethal
3. Do-not-overcommit-into-likely-removal
4. Singer preservation
5. Ink discipline
6. Evasive race
7. Shift line distinction
8. Multi-target effect distinction

### What it achieves

Ties development to actual Lorcana competence.

Priority: **Medium**

---

## 19. Add official paired-seed evaluation gauntlet

### Current logic

Default evaluation uses small anchor gauntlets.

### Why this is bad

Small gauntlets are noisy. Results can be distorted by:

- seat advantage
- matchup variance
- seed variance
- draw variance

### How to fix it

Add:

```bash
./train.sh eval --checkpoint X --games 224 --paired-seeds --seat-swap --promotion
```

Evaluation should include:

- fixed deck suite
- archetype coverage
- paired seed seat swap
- fair-only opponents
- confidence intervals
- per-archetype win rates
- promotion metadata validation

Promotion threshold should use a lower confidence bound, not raw winrate only.

### What it achieves

Tells you whether the bot is actually stronger, not just lucky.

Priority: **Medium**

---

## 20. Centralize fair/oracle information policy

### Current logic

illumidroid has fair strategy registration and oracle isolation in the bridge, but the policy logic is distributed.

### Why this is bad

Distributed fairness logic is easier to break.

### How to fix it

Create:

```python
training/information_policy.py
```

with:

```python
resolve_information_policy(
    mode: "train" | "eval" | "promotion" | "research",
    requested_strategy: str,
    allow_oracle: bool,
)
```

Rules:

- training defaults fair
- promotion rejects oracle
- oracle requires `--oracle-research`
- checkpoint metadata records policy
- checkpoint load validates provenance

### What it achieves

Prevents hidden-information contamination.

Priority: **Medium**

---

## 21. Add hidden-information leak tests

### Current logic

The architecture intends hidden card identities to be used only for belief supervision and determinization, not policy/value inputs.

### Why this needs testing

The system now has:

- public history
- hidden labels
- inkwell labels
- candidate descriptors
- bridge observations
- belief heads

A future change could accidentally leak hidden identities into policy/value inputs.

### How to fix it

Add tests:

1. Build two observations differing only in opponent hidden hand identity.
2. `encode_obs()` must be identical.
3. `encode_belief()` may differ.
4. Policy/value outputs must be identical when only hidden labels differ.
5. Hidden inkwell identity must not appear in public history as `defId`.

### What it achieves

Keeps the bot competition-legal.

Priority: **Medium**

---

## 22. Make search depth Lorcana-turn-aware

### Current logic

Search depth is measured in decision plies.

### Why this is incomplete

A Lorcana turn can contain many decision plies:

- ink
- play
- quest
- challenge
- resolve effects
- pass

A fixed depth may stop before the opponent meaningfully responds.

### How to fix it

Add semantic depth controls:

```python
max_decision_plies
max_full_turns
must_include_opponent_response
```

Track whether:

- current player passed
- turn changed
- opponent got a meaningful action
- lore clock changed
- terminal threat appeared

### What it achieves

Search evaluates lines in a way that better matches Lorcana turn structure.

Priority: **Medium**

---

## 23. Add action-family-specific diagnostics

### Current logic

Policy loss and sample metrics are aggregated.

### Why this is bad

You need to know whether the bot is bad at:

- mulligan
- inking
- questing
- challenging
- playing cards
- singing
- targeting
- resolving effects

Aggregate loss hides the failure mode.

### How to fix it

Track by action family:

- policy CE
- entropy
- sample count
- visit concentration
- top policy vs top MCTS disagreement
- chosen family distribution
- winrate impact

Trace logs should include top candidates by:

- prior
- visit count
- Q value
- stable key
- family

### What it achieves

Makes training failures diagnosable.

Priority: **Medium**

---

## 24. Add model distillation for faster self-play

### Current logic

The strongest actor uses search, which is expensive.

### Why this matters

The project is engine-throughput constrained.

### How to fix it

Periodically distill MCTS behavior into a fast policy checkpoint.

Use the distilled policy for:

- rollout continuations
- low-stakes decisions
- action-likelihood belief tracking
- scripted replacement opponents
- cheap evaluation

Hybrid actor:

- full search for high-uncertainty decisions
- distilled policy for obvious decisions

### What it achieves

More games per hour without using full search everywhere.

Priority: **Medium**

---

## 25. Add checkpoint population management

### Current logic

League can freeze past mains and track Elo.

### Why this needs improvement

A strong population needs curation. Too many weak agents waste training. Too few agents cause overfitting.

### How to fix it

Add rules:

- freeze only if new main exceeds threshold
- keep diverse archetype specialists
- keep exploiters that expose weaknesses
- prune dominated agents
- tag checkpoints by matchup strengths
- sample opponents by exploitability and diversity, not only win/loss

### What it achieves

Better self-play curriculum and less overfitting.

Priority: **Medium**

---

# Recommended Implementation Order

1. Exact `run_paths` execution integrity
2. League path parity with default self-play
3. Checkpoint provenance and promotion gates
4. Use inkwell belief in determinization
5. Action-likelihood belief tracking
6. Official paired-seed eval gauntlet
7. Search cap diagnostics
8. Counterfactual rollout labels
9. Mulligan-specific training
10. Adaptive search budgets
11. Search/evaluation cache
12. Full belief probability API
13. Card-vocab regeneration tests
14. Real Lorcana scenario tests
15. Centralized fair/oracle policy resolver
16. Hidden-information leak tests
17. Lorcana-turn-aware search depth
18. Action-family diagnostics
19. Distillation for fast self-play
20. Population management

---

# Final Judgment

The current illumidroid build is now a serious champion-level ML bot architecture.

The most important fixes are no longer basic representation problems. Those have largely been addressed. The remaining high-value work is about:

- making search exact and trustworthy,
- making league training as clean as default self-play,
- preventing contaminated checkpoints,
- converting learned belief into stronger gameplay,
- improving evaluation rigor,
- and scaling training efficiently.

Botcana2.0 remains the cleaner Lorcanito-native safety baseline, but illumidroid now has the stronger long-term ML ceiling.
