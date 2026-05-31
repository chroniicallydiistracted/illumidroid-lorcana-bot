# Phase 2 — Plan: Belief net + importance-weighted determinization + Bayesian filtering

**Goal (architecture doc §2.4 / §3.2 / §5.3):** the strength inflection point.
Replace Phase 1's single realized-world search (N=1) with a search that (a) has a
learned **belief** over the opponent's hidden cards, (b) **samples worlds from
that belief** and pools statistics across them with **importance weights**, and
(c) **sharpens the belief between turns** with a Bayesian particle filter.

## Design decisions (pinned to verified engine facts)

- **Leak-free belief.** The trunk is built only from the acting player's filtered
  view (no opponent identities). The belief head predicts, for each card in the
  opponent's *known pool* (self-play conditioning, §6 deck-conditioning),
  `P(card currently in opp hand)`, count-consistent to the public hand size.
  Candidate pool tokens are scored against the trunk but never fed *into* it, so
  the policy/value heads can't cheat.
- **Truth is free in self-play.** The bridge exposes the opponent's true hidden
  zones (authoritative state) as supervised belief targets — zero labeling cost.
- **Determinization by state surgery (verified feasible).** Clone the
  authoritative state, repartition the opponent's hidden instances into hand/deck
  per a belief sample, fix zone summaries, `loadState`. The engine accepts the
  edited world and keeps playing (probe `DET_OK`).
- **Importance weighting.** Sample N worlds `w_i ~ q`, weight `ρ_i = b(w_i)/q(w_i)`;
  pool `Q(I,a) = Σ ρ_i v_i / Σ ρ_i`. Default proposal `q = b` (ρ≈1); a uniform
  proposal is available to exercise the weighting.
- **Real ISMCTS node pooling.** The root information set is shared across worlds
  (same filtered view), so child statistics accumulate across determinizations —
  the genuine ISMCTS upgrade over Phase 1's per-path tree.

## Workstreams

**A. Belief network (centerpiece)**
- A1 Bridge: `observe` returns a `hidden` block — opponent hand+deck cards
  (instanceId, definitionId, zone) from the authoritative state (truth/pool).
- A2 Serialization: `encode_belief` → candidate pool tokens + per-card in-hand
  label + known hand count. Kept separate from policy/value tensors (no leak).
- A3 Network: `BeliefHead` scoring each candidate from (trunk ⊕ candidate id
  embedding); `LorcanaNet` becomes three-headed. `belief()` infer helper +
  count-normalization at inference.
- A4 Learner: `L_belief` (BCE + count-consistency reg), weight `c_belief`;
  calibration metric (predicted-in-hand vs realized).

**B. Importance-weighted determinization**
- B1 Bridge: `determinize(hand_instance_ids, seed)` op (clone → repartition opp
  hidden zones → loadState → snapshot handle).
- B2 `search/determinize.py`: sample N hand-assignments from the belief; compute
  importance weights; pluggable proposal (belief / uniform).
- B3 `search/ismcts.py`: `run_belief()` — for each world: determinize, run PUCT
  sims; pool stats at shared infoset nodes weighted by ρ_i. Pure pooling math
  factored out for unit testing.

**C. Bayesian filtering between turns**
- C1 `search/belief_filter.py`: particle filter over opponent-hand hypotheses;
  reweight by observed-action likelihood after each opponent move; SIR resample.
  Neural belief = proposal; observed action = correction.

**D. Training + tests + docs**
- D1 `training/bootstrap.py` / `selfplay.py`: log truth, train belief jointly.
- D2 Tests: belief shapes + **leak-freeness**, determinize round-trip,
  importance-pooling math, particle-filter reweight/resample, calibration improves
  after a short train.
- D3 Phase 2 completion report; update CLAUDE.md + memory.

## Honest scope note
With the current homogeneous placeholder fixture decks (each card a distinct
*synthetic* id, no real effects), the belief net learns base-rate / public-signal
structure rather than rich card semantics, and determinization permutes inert
cards. The **machinery is correct and identical** for real decks; wiring the real
`BEST_AI_DECK_DOSSIERS` card catalog is the orthogonal follow-on that makes the
belief strategically potent. Phase 2 here delivers the correct, tested machinery
and the leak-free training loop.
