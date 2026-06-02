# Audit Remediation Status

Response to the 18-finding audit. Verdict accepted: trunk is leak-free, **search was not**.
The training-blocking and bounded true-bugs are fixed + tested this pass; the
research-scale items are sequenced below (honest effort, not hand-waved).

## FIXED this pass (with tests)

| # | Finding | Fix | Test |
|---|---|---|---|
| **1** | **Search used privileged hidden info** (real opp inkwell, real own/opp deck order) | `server.ts determinize` now randomizes ALL hidden zones from the actor's info set: opp hand = belief sample, **opp inkwell = resampled (count-consistent)**, opp deck = shuffled, **self deck = shuffled**. Self hand/inkwell legitimately known. | `test_determinize_is_leak_free` (conservation + inkwell randomized) |
| **6** | Belief search forced `dirichlet_eps=0` (no root exploration) | sub-search inherits configured noise | engine suite |
| **3** | Belief worlds kept stale hand size after a draw | `BeliefTracker` reseeds on hand-count change, not just pool | `test_belief_tracker_seeds_reweights_reseeds` |
| **8** | PFSP treated draws as wins for the 2nd id | track `wins_a/wins_b`; draws score **0.5** both sides | `test_pfsp_draws_counted_as_half` |
| **16** | stderr undrained → pipe can block the engine | daemon drain thread + ring buffer | engine suite |
| **18** | turn-cap/stuck games trained value/aux from a non-terminal cutoff (inconsistent single vs parallel) | both paths **discard** turn-capped/stuck games (no cutoff labels) | smoke |
| — | (regression from the lighter MLP head) candidates differing by an extra target could collapse | head pools tokens by **mean + sum** (count/identity-sensitive) | `test_candidates_..._distinct_logits` (seeded) |

Verification: fast 22/22, engine 24/24.

## Tier A — belief-search correctness — IN PROGRESS

The earlier completion claim was disproven by the follow-up audit.
`illumidroid_full_ismcts_implementation_spec.md` remains the #2 design spec.
`tier-a-belief-search-remediation-plan.md` is the authoritative implementation
sequence.

Current audited checkpoint:

```text
Phase 0   complete — belief-guided clean-label training fails closed
Phase 13  baseline installed — 5 passing probes (2 Phase-1 + 3 Phase-2) + 15 strict expected-red probes
Phase 1   complete — audited GO for the canonical full hidden-zone World boundary
Phase 2   complete — audited GO for exact structured sampling + importance semantics
Phase 3   IN REMEDIATION (NO-GO) — full-`World` RPC built + bounded findings fixed (cardIndex/
          cardMeta reconcile, owner/controller preserve, summary invariance, seat==currentActor,
          exact-order agreement, metadata allowlist); protected-facts ledger PARTIAL (reveals
          guarded; static top-deck/pending-effect-refs/§15 open) + obs hidden-ID sanitization (F3)
          unresolved
Next      close Phase 3 blockers before Phase 7
```

Phase 1 now provides a strict `World.validate_against_obs()` contract and a shared
`World.opponent_hidden_pool()` / `_witness_pool()` boundary. The source witness
rejects missing or malformed zones, non-integer counters, non-string or empty IDs,
duplicate IDs, and hidden-zone cardinalities that disagree with public counts.
`World.seed` is enforced as a non-empty string for clean-label admission.
Belief-guided clean-label training remains blocked while the downstream phases land.

Phase 2 closes audit finding 6 at the sampler boundary. `sample_worlds()` now uses
log-domain exact dynamic programs for the count-conditioned hand target and the
structured hand/inkwell/deck target; stores raw `rho = b/q` separately from normalized
pooling `weight`; keeps structured mode active when `ink_count == 0`; includes the
residual uniform inkwell-partition factor in full-world audit logs; and rejects malformed
proposal modes, pools, probability vectors, counts, and supplied base seeds fail-closed.

### Historical partial implementation snapshot

The following table records the earlier isolated algorithm work. It is not a
release-complete Tier A claim; the follow-up findings below still govern the active
remediation.

| # | Finding | Fix | Tests |
|---|---|---|---|
| **5** | `determinize.py` Gumbel-top-k forced `ρ=1` *assuming* `q≈b` (Gumbel-top-k with log p is NOT the count-conditioned Bernoulli) | **Exact** count-conditioned Bernoulli sampler (suffix-DP elementary-symmetric mass) → `q` known in closed form, `ρ` exact (==1 for the belief proposal, varies for uniform). Added an **exact joint 3-zone (hand/ink/deck) DP sampler** (§5.1) + a rich reproducible `World` spec (opp hand/ink/deck + self deck + `log_target`/`log_proposal`/`rho`). Structured hand+inkwell belief API (`model.belief_probs_structured`, `StructuredBeliefEvaluator`, §5.3). | `test_5_conditional_bernoulli_marginals_match_exact_target`, `_proposal_pmf_is_exact_rho_one`, `_joint_zone_marginals_match_brute_force`, `_world_spec_partitions_pool_and_satisfies_counts`, `_uniform_proposal_rho_tracks_target_and_normalizes`, `_same_seed_reproduces_worlds`, `_structured_belief_evaluator_hand_and_ink_channels` |
| **4** | one shared `BeliefTracker` overwritten across seats; no observed-action correction | **One persistent SIR tracker PER SEAT** in both self-play loops (keyed by `obs["self"]`, never overwritten); `observe_opponent_action(likelihood_fn)` action-likelihood correction; `sample_world(rng)` (one posterior world/sim, ρ==1); ESS + counters | `test_4_tracker_sample_world_and_action_likelihood`, per-seat wiring in `run.py`/`parallel.py` |
| **2** | root-pooled **per-world perfect-info trees** (PIMC determinized-UCT) → strategy fusion (Cowling et al. 2012) | **`run_infoset`**: ONE shared `InfoSetTable` per root decision; one posterior world sampled per simulation; deeper nodes resolved + shared by a **leak-free actor-filtered `infoSetKey`** (never reads `obs["hidden"]`; face-down inking redacted; perfect-recall branch history); **availability-aware PUCT** + PW over the current world's available subset; root noise once; invalid execs quarantined; root π aligned by `actionId`. Real-engine driver `EngineSimulator` over existing bridge + new non-history-polluting `step_exact`; opt-in `--full-ismcts`. | `test_2_full_ismcts_removes_strategy_fusion` (synthetic IIG, §13.3), `_key_ignores_opponent_hidden_identities_and_deck_order`, `_key_changes_on_visible_state_and_history`, `_history_redacts_face_down_inking`, `_table_merges_same_key_and_counts_transpositions`, `_availability_increments_every_available_sibling`, `_select_never_returns_unavailable_action`, `_value_for_flips_by_actor`, `test_run_infoset_full_ismcts_real_engine`, `_root_key_is_world_invariant` |

**Scope honesty (#2).** This is the *correctness-first* baseline the spec mandates
(§16.1): the shared-tree algorithm is proven both on a synthetic imperfect-information
fixture (strategy fusion removed: gambles average to ≈0 instead of the PIMC ~+1
over-valuation) and end-to-end against the real engine (transposition hits >0, 0 invalid
sims, root restored, history un-polluted). It is **root-sampled shared-tree ISMCTS** — NOT
yet the optional opponent-perspective *re-determinization* hardening (§15, separate
experiment), and the lane driver is one IPC/transition (the batched `begin/step/drop_ismcts`
RPC optimization, §9/§16, is deferred. Legacy PIMC now remains available only as
`run_pimc_diagnostic()` and cannot write belief-guided training labels. The info-set key is computed
leak-free in Python from the filtered obs (server-side key + per-observer history projection,
§6.3, is a later refinement).

### Historical follow-up audit findings

**Audit result at remediation start**
Tier A was not complete end-to-end. The isolated algorithms were promising and
64 tests passed at that snapshot, but active training paths still produced
contaminated or incomplete search labels. Phase 0, Phase 13 baseline setup, Phase 1,
and Phase 2 have since landed; the unresolved downstream findings remain sequenced in
the authoritative remediation plan.

**Findings**
1. **Critical: active determinization leaks opponent inkwell membership.**  
[BeliefEvaluator](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/evaluator.py:65) samples only authoritative hand + deck IDs, excluding hidden inkwell IDs. Although the server repartitions all hidden zones in [server.ts](/home/andre/illumidroid-lorcana-bot/lorcana-bot/engine/node_server/server.ts:426), an actual inkwell card can never be hypothesized as a hand card. Both default PIMC and opt-in ISMCTS use this legacy evaluator in [run.py](/home/andre/illumidroid-lorcana-bot/lorcana-bot/training/run.py:56) and [parallel.py](/home/andre/illumidroid-lorcana-bot/lorcana-bot/training/parallel.py:61).

2. **Critical: the rich `World` partition is discarded by full ISMCTS.**  
[EngineSimulator.begin_lane()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/engine_sim.py:37) passes only `opponent_hand_ids`. It ignores sampled inkwell, opponent deck, and self deck fields. A direct probe supplied two different requested inkwell assignments and received the same actual inkwell assignment both times.

3. **High: action-likelihood correction is not integrated.**  
[observe_opponent_action()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/belief_filter.py:158) exists and has an isolated test, but neither gameplay loop calls it after [run.py](/home/andre/illumidroid-lorcana-bot/lorcana-bot/training/run.py:87) or [parallel.py](/home/andre/illumidroid-lorcana-bot/lorcana-bot/training/parallel.py:90). Per-seat trackers are present, but `action_updates` remains zero during real self-play.

4. **High: neural leaf evaluation receives stale simulated history.**  
[stepExact()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/engine/node_server/server.ts:511) intentionally leaves public history unchanged. [EngineSimulator](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/engine_sim.py:44) appends a branch event only for hashing. [NetEvaluator](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/evaluator.py:32) therefore encodes root history at deeper simulated leaves. A probe confirmed `root_history_len == leaf_history_len` while `branch_events_len == 1`.

5. **High: the Python key is not a full perfect-recall key.**  
[EngineSimulator](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/engine_sim.py:47) stores only the first two colon-separated action-key fields. This drops targets, choices, costs, destinations, and mulligan selections encoded by [candidateKey()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/engine/node_server/server.ts:171). [history_event()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/infoset.py:54) also redacts inking for the acting player, who knows their own inked card. The deferred per-observer history work is required for the report’s perfect-recall claim.

6. **Medium: structured sampler edge cases violated its public contract — REMEDIATED IN PHASE 2.**
[sample_worlds()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/determinize.py:461) previously had three defects:
- Structured `proposal="uniform"` still sets `log_target == log_proposal`, so every `rho` is `1`.
- A zero-card hand returns empty worlds, dropping inkwell and deck assignments.
- `rho` was documented as raw `b/q`, but rebased before storage.

Phase 2 additionally hardened the exactness boundary discovered during re-review:
both DPs are log-domain, structured `ink_count == 0` worlds still use the deck channel,
unstructured residual zone factors remain in both audit logs, and malformed sampler
inputs reject instead of being silently repaired.

7. **Medium: persistent reweighting is not a valid conditional-Bernoulli update.**  
[BeliefTracker.worlds()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/belief_filter.py:146) multiplies only included-card probabilities. It omits excluded-card `(1-p)` terms and repeatedly treats the current posterior-like prediction as new evidence. This needs a defined likelihood model or removal.

8. **Medium: full-ISMCTS diagnostics report zero visits.**  
[BISMCTS.run_infoset()](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/ismcts.py:258) discards the populated shared root and returns a new empty legacy `InfoSetNode`. [DecisionTracer](/home/andre/illumidroid-lorcana-bot/lorcana-bot/training/trace.py:57) and live logging then show zero visits and Q values. A probe reproduced `returned_root_visit_sum: 0.0` with a non-uniform policy.

9. **Medium: full-ISMCTS reproducibility is incomplete.**  
Tracker worlds have no seeds in [belief_filter.py](/home/andre/illumidroid-lorcana-bot/lorcana-bot/search/belief_filter.py:95). The adapter passes `None`, causing [server.ts](/home/andre/illumidroid-lorcana-bot/lorcana-bot/engine/node_server/server.ts:414) to use `Math.random`.

10. **Low: current tests miss the runtime failures.**  
The root-invariance test samples only hand + deck in [test_search_belief.py](/home/andre/illumidroid-lorcana-bot/lorcana-bot/tests/test_search_belief.py:104), so it cannot catch the inkwell leak. The real-engine smoke test does not assert deeper sharing. `transposition_hits > 0` alone would also be weak because root reuse increments it.

**Confirmed Working**
The exact conditional-Bernoulli sampler works in its tested path. Per-seat trackers are instantiated correctly. The shared `InfoSetTable`, action-ID alignment, invalid-path quarantine, and synthetic strategy-fusion fixture are real improvements.

**Verification**
Latest verification (Phase-3 NO-GO #3 closure findings addressed + tested; awaiting re-audit):
```text
../lorcana-bot-venv/bin/python -m pytest -q   # 166 passed / 2 skipped / 14 xfailed (182 collected)
git diff --check
python -m compileall -q engine network search training tests
```

## REMAINING — sequenced (recommended order)

### Tier B — training hygiene + real evaluation
- **#7 league parity** (MODERATE): give `league.py`/`match.py` the default trainer's per-game belief tracker, matched/forced/trivial filtering, stuck-guard, turn-cap, and aux targets.
- **#12 legacy path** (SMALL): deprecate/align `selfplay.py`; fix `README` to point at `run.py`.
- **#10 checkpoint provenance + resumability** (MODERATE): save info-policy, oracle flag, vocab/deck hashes, commits, search config, optimizer+RNG+round+replay state, promotion-eligibility; validate on load.
- **#11 real promotion gate** (MODERATE-LARGE): held-out deck metagame, paired/mirrored seats, confidence intervals, rejection threshold; stop saving unconditionally.

### Tier C — representation & rules coverage
- **#13 state-rep expansion** (LARGE): ink color, inkability, classifications, structured effects, explicit phase/step, own-deck identity, matchup embedding; regrow vocab beyond the current 155.
- **#14 nested `resolveBag`/`resolveEffect`** (SMALL-MOD, engine): the measured ~0.1% gap — targeted hybrid so those states are played, not discarded.
- **#9 mulligan grammar** (LARGE, engine): planner exposes only 3 mulligan plans; expose the full legal opening-hand subset space.

### Tier D — engine quality & repro
- **#15 engine type-clean + unsupported mechanics** (LARGE, vendored engine): fix `check-types` failures (`zone-operations.ts`, a phase0 test) and unsupported TODOs (non-ink Shift costs).
- **#17 lock dependencies** (SMALL): pin exact versions in `requirements.txt`/`pyproject.toml`.

## Training guidance
Belief-guided clean-label training is blocked while Tier-A remediation is in
progress. The legacy root-pooled PIMC path is now diagnostic-only as
`run_pimc_diagnostic()`. Non-belief diagnostics may continue with `--no-belief`,
but do not use them as evidence that Tier-A belief-search labels are clean.

Resume belief-guided clean-label self-play only after the release gate in
`tier-a-belief-search-remediation-plan.md` passes.
Remaining before "competition-promotable": Tier B (real held-out promotion gate #11,
league parity #7, provenance/resume #10) — a checkpoint is not champion-level until a
real evaluation gate exists. Tier C/D are representation/engine depth.
