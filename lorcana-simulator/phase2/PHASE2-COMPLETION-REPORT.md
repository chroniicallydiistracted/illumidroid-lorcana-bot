# Phase 2 — COMPLETION REPORT (belief net + importance-weighted determinization + Bayesian filtering)

**Stack:** Python 3.12 · PyTorch 2.12 (CPU) · Bun 1.3.14 (engine host)
**Verdict:** ✅ **Phase 2 complete.** The agent now maintains a learned,
leak-free belief over the opponent's hidden hand, samples worlds from that
belief and searches them with importance weights, and sharpens the belief
between turns with a particle filter. **27/27 tests pass.** All code in
`lorcana-bot/`. Plan: `PHASE2-PLAN.md`.

---

## 0. Exit-gate scorecard

| Phase 2 component | Status | Evidence |
|---|---|---|
| Belief net `b(world\|I)` (3rd head) | ✅ | `network/heads.py:BeliefHead`; predicts P(card∈opp hand), count-consistent |
| **Leak-free** (no hidden-info bleed into policy/value) | ✅ | `test_belief.py::test_belief_is_leak_free` — scrambling belief inputs leaves policy/value bit-identical |
| Truth supervision (free in self-play) | ✅ | bridge `hidden` block (authoritative opp zones) → belief labels |
| Importance-weighted determinization | ✅ | `search/determinize.py` (ρ=b/q); `bridge.determinize` state surgery |
| Pooled multi-world ISMCTS | ✅ | `BISMCTS.run_belief` pools per-world searches at the shared root |
| Bayesian filtering (SIR particle filter) | ✅ | `search/belief_filter.py`; reweight + ESS-resample tested |
| Joint 3-head training | ✅ | bootstrap: policy 0.73→0.48, value 2.03→0.52, **belief 0.365→0.071** |

---

## 1. Belief network (A)

The trunk is built **only** from the acting player's filtered view, so it never
sees opponent identities. `BeliefHead` scores each card in the opponent's known
pool (self-play deck-conditioning, doc §6) against the trunk vector:
`P(in opp hand) = σ(MLP(trunk ⊕ id_embed(card)))`, sharing the trunk's identity
embedding. Candidate tokens never enter the trunk ⇒ **no leak**, verified by a
test asserting policy/value logits are unchanged when belief candidate ids are
scrambled (while the belief output does change).

- **Truth is free:** the bridge `observe` now returns a `hidden` block — the
  opponent's true hand/deck (instanceId, definitionId) from the authoritative
  state — used only as supervised labels / determinization pool, never fed to
  policy/value.
- **Count-consistency:** at inference, per-card probabilities are scaled to sum
  to the public hand size (`BeliefHead.normalize_to_count`); training adds an L1
  count penalty. Result: count-MAE 5.0 → 0.16.
- **Loss (doc §5.3 belief term):** per-candidate BCE + count penalty, added to
  the Phase-1 policy+value loss. Belief BCE 0.365 → 0.071; in-hand/out-of-hand
  separation −0.004 → 0.241 over 6 bootstrap epochs.

## 2. Importance-weighted determinization (B)

- **State surgery (verified):** `bridge.determinize(self, hand_instance_ids)`
  clones the authoritative state, repartitions the opponent's hidden cards into
  hand = the given ids (deck = the rest, seed-shuffled), fixes zone summaries,
  and `loadState`s — producing a playable determinized world. The engine accepts
  it and keeps playing.
- **World sampling:** `search/determinize.sample_worlds` draws N opponent-hand
  worlds. Proposal `q="belief"` samples ∝ belief (Gumbel-top-k, ρ≈1, the usual
  ISMCTS choice); `q="uniform"` gives ρ ∝ b (reweights toward belief-likely
  worlds). Weights self-normalized so Σρ = N. Belief-proposal sampling provably
  concentrates on high-probability cards (test).
- **Pooled ISMCTS (`run_belief`):** for each world, restore the true root,
  determinize, and run a full Phase-1 PUCT search; pool the **root** visit/value
  statistics across worlds weighted by ρ_i. **Correctness key:** determinization
  only changes the opponent's *hidden* cards, so the acting player's filtered
  view — and hence the root legal-action set — is identical across worlds,
  making index-aligned pooling exact. The engine is restored to the true root on
  return (tested).

## 3. Bayesian filtering (C)

`search/belief_filter.ParticleFilter` — a sequential-importance-resampling
filter over opponent-hand hypotheses (doc §2.4):
- seeds particles from the neural belief (proposal),
- `reweight(likelihood_fn)` multiplies particle weights by the observed action's
  likelihood under each world (correction),
- `maybe_resample` does SIR when effective sample size drops below a threshold,
- `marginal()` exposes the sharpened per-card belief; `to_worlds()` feeds it back
  into determinization.
Tested: seeding matches base rate, evidence sharpens the marginal toward the
observed card, resampling preserves particle count and restores ESS.

## 4. Tests (27/27)

New Phase 2 tests: `test_belief.py` (shapes, count-consistency, **leak-free**,
learning), `test_determinize.py` (world sampling + weights + particle filter,
engine-free), `test_search_belief.py` (hidden-truth exposure, determinize
round-trip + legal-set invariance, `run_belief` pooling + root restoration).
Phase 1's 14 tests still pass unchanged.

## 5. Real decks (now wired)

25 real tournament-winning decklists (`lorcana-bot/decks/*.json`, fixture shape,
provenance in `decks/docs/`) are resolved against the full card catalog
(`all001..all012Cards` + `resolveLorcanaDeckListTextFromPool`) and used by the
bridge **by default**. `reset(seed, deck_p1, deck_p2)` selects a pair (or picks
one deterministically from the seed); `list_decks` enumerates; bootstrap and
self-play sample distinct pairs for metagame diversity (§6). All 25 resolve with
**0 unresolved cards**; same-seed determinism is preserved (instance ids are the
test engine's deterministic `t0000xx`, not the `Math.random` path).

**Honest findings from the real-deck switch:**
- Games now **terminate via real win conditions** (≈70–170 decisions/game) instead
  of the placeholders' 242-step durdle — real lethal/lore races actually resolve.
- **Leak-free verified on real cards:** opponent hand cards in the filtered
  projection that feeds the trunk are `hidden, definitionId=null`; real
  identities appear only in the authoritative truth block used for belief labels.
- The belief head learns **count-consistency** well (count-MAE 9.0 → 1.27) but
  per-card in-hand **separation is small (~0.02)** — predicting *which* cards are
  in hand from public info alone is genuinely hard. (Note: the placeholders'
  higher "0.24" separation was largely an artifact — their synthetic ids encoded
  the zone name, a naming leak that real shared identities remove.) Sharpening
  this is exactly what the §2.3 public-history encoder, §2.4 archetype priors,
  and the revealed/played-card hard constraints add — Phase 3 enrichments, not
  algorithm changes here.

## 6. Carry-forward into Phase 3 (League / PSRO + exploitability)

1. The belief + filter give the importance-weighted determinization the league
   will search over; PFSP opponent sampling plugs in at the self-play actor.
2. `BeliefEvaluator` / `ParticleFilter` are ready to drive between-turn belief
   updates once the actor loop observes opponent actions.
3. The remaining strategic-potency unlock is real decks (catalog wiring), which
   also benefits Phase 3's deck-diversity anti-collapse requirement.
