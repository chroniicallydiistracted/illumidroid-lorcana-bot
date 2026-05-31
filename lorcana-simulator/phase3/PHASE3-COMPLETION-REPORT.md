# Phase 3 — COMPLETION REPORT (League / PSRO self-play + exploitability gating)

**Stack:** Python 3.12 · PyTorch 2.12 (CPU) · Bun 1.3.14 (engine host, 25 real decks)
**Verdict:** ✅ **Phase 3 machinery complete.** The agent now trains against a
*population* (scripted anchors + frozen past mains) with PFSP opponent sampling,
KL-trust-region + entropy anti-collapse, periodic freezing into the league, Elo
tracking, and an exploitability proxy. All in `lorcana-bot/training/`. Plan:
`PHASE3-PLAN.md`.

---

## 0. Exit-gate scorecard

| Phase 3 component | Status | Evidence |
|---|---|---|
| League population (main / past / anchors / exploiters) | ✅ | `training/league.py:League` + `Player`/`ScriptedPlayer`/`NetPlayer` |
| PFSP opponent sampling | ✅ | `League.pfsp_weights/_sample`; test: favors hard opponents |
| Elo ratings + results matrix | ✅ | `League.record_result/_update_elo/win_prob`; tested |
| Match routing (two policies, one game) | ✅ | `training/match.py:play_match`; per-seat samples, z by winner |
| KL trust-region anti-collapse | ✅ | `learner.py` (`c_kl`, `set_reference`); test: KL=0 to self, >0 after drift |
| Entropy bonus + deck diversity | ✅ | `c_entropy`; Phase-2 deck-pair sampling across the 25-deck metagame |
| Exploitability proxy (main exploiter) | ✅ | `exploitability.py:exploitability_proxy` |
| Gauntlet evaluation + Elo gating | ✅ | `exploitability.py:gauntlet`; orchestrator gating step |
| Orchestrator end-to-end | ✅ | `training/league_train.py` — generate → train → freeze → gate |

---

## 1. League & players (`league.py`)

`Player.act(engine, obs)` advances one decision and returns a training record
only for *learning* seats. `ScriptedPlayer` drives the shipped automation (the
permanent anchors `best`/`deckAware`); `NetPlayer` acts via B-ISMCTS
(`run`/`run_belief`) and records (I, π_MCTS, belief, actor) when learning. The
`League` registry tracks players by kind (main / past / anchor / exploiter), a
pairwise results matrix, **Elo**, and **PFSP**:

- **PFSP (`mode="hard"`)** weights each opponent by `(1 − P(main beats it))` with
  a floor, so search budget concentrates on opponents the main loses to or is
  even with — the anti-collapse pressure of §5.1. (`mode="even"` = `p(1−p)`,
  closest matchups.)
- **Elo** updates per game (K=32); `win_prob` is the empirical (prior-smoothed)
  matchup estimate driving PFSP.

## 2. Match routing (`match.py`)

`play_match` resets a game (real deck pair) and routes each decision to the seat
to move — the observation is already that actor's information set, so each policy
reasons from its own POV. Learning seats' decisions become `Sample`s with the
outcome z assigned from that seat's perspective. Seats are alternated by callers
to remove first-player bias.

## 3. Anti-collapse (`learner.py`)

Added a **KL trust-region / proximal term** `c_kl · KL(π_current ‖ π_ref)` to a
frozen reference policy captured at each round's start (`Learner.set_reference`),
so updates can't lurch away from a working policy (no catastrophic forgetting,
§5.1). Illegal/padded actions contribute 0 to the KL by construction. Combined
with the existing entropy bonus and Phase-2 deck-diversity sampling. (NeuRD /
regret-matching regularization is noted as a later refinement.)

## 4. Exploitability + gauntlet (`exploitability.py`)

- `gauntlet(main, opponents)` — seat-alternated win-rate vs a fixed set
  (anchors + frozen checkpoints across decks); the promotion-gate signal.
- `exploitability_proxy` — trains a **main exploiter** (a net optimized *only* to
  beat the frozen main) and reports its win-rate. This is the AlphaStar-style,
  engine-native stand-in for full local-best-response/exploitability (which would
  otherwise need OpenSpiel machinery). A high exploiter win-rate flags an
  exploitable main even when it beats the league on average.

## 5. Orchestrator (`league_train.py`)

Each iteration: freeze the KL reference → generate games (main vs PFSP-sampled
opponents, alternating seats, across deck pairs) → train the main (policy +
value + belief + KL + entropy) → periodically **freeze** the main into the league
as a new `past_k` → update Elo. A final **gauntlet** vs the anchors provides the
promotion-gate signal; `--exploit` additionally runs the exploitability proxy.

## 6. End-to-end proof

A tiny run (`--init bc_realdecks.pt --iterations 2 --games 4 --sims 6`,
warm-started from the real-deck bootstrap) exercises the whole loop:

```
it0 g0: main(player_one) vs anchor_deckaware -> winner=player_two (+55)
it0 g1: main(player_two) vs anchor_best      -> winner=player_two (+49)
it0 g2: main(player_one) vs past_0           -> winner=player_two (+36)
it0 g3: main(player_two) vs anchor_deckaware -> winner=player_one (+22)
[league] iter 1/2: winrate=0.25 loss=1.460 kl=0.047 elo_main=970.1 league=5
it1 ... vs anchor_deckaware / past_0 / past_1 / past_1
[league] iter 2/2: winrate=0.75 loss=1.328 kl=0.020 elo_main=1007.1 league=6
[league] gauntlet vs anchors: {anchor_best: 0.0, anchor_deckaware: 0.5, _overall: 0.25}
[league] saved -> checkpoints/league.pt
```

Confirmed end-to-end: **PFSP** sampled a varied opponent mix (scripted anchors +
frozen `past_0`/`past_1`); **freezing** grew the league (5 → 6 players); **Elo**
moved (970 → 1007); the **KL trust region** was active and small (0.047 → 0.020,
anti-collapse); training reduced loss; and the **gauntlet** gating evaluation ran.
The numbers are noisy at this scale (4 games/iter) — this proves the *machinery*,
not convergence (which is a `--iterations/--games/--sims` compute knob).

## 6b. Information-policy fairness (training-data hygiene)

A correctness audit found the engine planner **defaults an unset
`informationPolicy` to `"oracle"`** (`planner.ts:3088`; oracle ⇒
`opponentKnowledgeSource="full-deck"` — the strategy decides using the
opponent's hidden cards). Only `bestDeckAwareLoreRaceAutomatedActionStrategy` is
explicitly fair; the previously-used `"best"` (oracle) and `"deckAware"` (oracle)
were cheating. **Two vectors:** observation features are *not* leaked (obs is
always `getBoard(viewFor(actor))` fog-filtered, and belief labels only supervise
the leak-free belief head — re-verified), but **targets were**: BC cloned an
oracle teacher and value z came from oracle games.

**Fixed:** the bridge is now **fair-by-default** — a `fair()` wrapper forces
`informationPolicy:"fair"`; `best/fairBest/fairDefault/fairControl/fairAggro/
deckAware` are all fair; `oracle`/`oracleDeckAware` are isolated as **eval-only**.
`step_auto` returns `policy`; `bootstrap.generate_game` **raises** on oracle play;
league anchors are fair (`fairBest/fairAggro/fairControl`). A fair-data bootstrap
(`bc_fair.pt`) trains cleanly (policy 1.23→0.56, value 2.79→0.71, belief
count-MAE 11.4→1.89). Pre-fix checkpoints (`bc.pt`, `bc_realdecks.pt`,
`league.pt`) are oracle-tainted at the target level and superseded by fair runs.

## 7. Tests

`test_league.py`: Elo movement, win-prob symmetry/prior, **PFSP favors hard
opponents**, frozen-copy independence, **KL = 0 to self / > 0 after drift**, and
match routing + outcome assignment (engine-free fake engine), plus a real-engine
`NetPlayer` vs `ScriptedPlayer` smoke. Phases 1–2 (27 tests) still pass.

## 8. Scope realism & carry-forward

The engine runs ~hundreds of ms per searched decision, so league *convergence* is
a compute job (scale `--iterations/--games/--sims`), not new code. The machinery
is complete and correct. Natural next steps (Phase 4, doc §5.4/§3.2):
ReBeL/PBS subgame solving at key decisions, distributional/auxiliary value-head
enrichments, the §2.3 public-history encoder to sharpen the belief, and true
LBR/exploitability via OpenSpiel as an external check.
