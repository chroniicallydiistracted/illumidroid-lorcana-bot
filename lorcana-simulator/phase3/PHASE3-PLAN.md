# Phase 3 — Plan: League / PSRO self-play + exploitability gating

**Goal (architecture doc §5.1 / §6):** stop policy collapse in the non-transitive
Lorcana metagame and make the agent *hard to exploit*, not just good against a
mirror. Train a **population** (league), sample opponents by **PFSP**, gate
promotions on **Elo + exploitability**, and add anti-collapse regularization.

## Why (the §5.1 argument)
Plain self-play against the latest net collapses (aggro→ramp→control→aggro is a
rock-paper-scissors cycle). Win-rate-vs-self can rise while *exploitability*
also rises. So: train main agents against a league of frozen pasts + scripted
anchors + exploiters, weight opponents by how informative they are (PFSP), and
track an explicit exploitability proxy as the north-star metric.

## Components

**A. League players (`training/league.py`)**
- `Player` interface: `act(engine, obs) -> (new_obs, sample|None)` — chooses and
  steps one decision; returns a training sample iff it is a *learning* seat.
- `ScriptedPlayer(strategy)` — drives via `engine.step_auto` (the shipped
  automation; permanent league anchors).
- `NetPlayer(net, cfg, record, use_belief, sims, n_worlds, temperature)` — acts
  via B-ISMCTS (`run` / `run_belief`); records (I, π_MCTS, z, belief) when learning.
- `League` — registry of players (main, frozen pasts, scripted anchors,
  exploiters), a pairwise results matrix, **PFSP** opponent sampling, and **Elo**.

**B. Match routing (`training/match.py`)**
- `play_match(engine, seat_players, deck_pair, seed)` runs one game where each
  seat is a (possibly different) Player; routes each decision to the seat whose
  turn it is (obs is already from the actor's POV); returns the winner and each
  learning seat's samples (z assigned by winner vs that seat).

**C. Exploitability + evaluation (`training/exploitability.py`)**
- `gauntlet(main, opponents, n_games)` — win-rate of `main` vs a fixed gauntlet
  (scripted anchors + frozen checkpoints across archetypes/decks).
- `exploitability_proxy` — the league's **main-exploiter** mechanism: a trainable
  net trained *only* to beat the frozen main; its converged win-rate is the
  exploitability estimate (AlphaStar-style; a tractable stand-in for full LBR).

**D. Anti-collapse in the learner (`training/learner.py`)**
- Add a **KL-trust-region / proximal** term `c_kl · KL(π_current ‖ π_ref)` to a
  frozen reference policy (no catastrophic forgetting), alongside the existing
  entropy bonus. Deck diversity is already on (Phase-2 deck sampling). NeuRD /
  regret regularization noted as a later refinement.

**E. Orchestrator (`training/league_train.py`)**
- Iterate: (1) generate games — main vs PFSP-sampled league opponents across
  deck pairs; (2) train main (KL + entropy); (3) periodically **freeze** main
  into the league and update Elo; (4) **gate**: keep the new main only on
  Elo non-regression vs the league. Optional exploiter training round.

**F. Tests + proof + docs**
- Fast/pure: PFSP weighting, Elo update, league freeze/promote, KL term.
- Engine smoke: a short `play_match` (Net vs Scripted) terminates with a winner
  and yields the learning seat's samples; one tiny league iteration runs.
- Phase 3 completion report; update CLAUDE.md + memory.

## Scope realism
The engine runs ~hundreds of ms/decision under search, so full league
*convergence* is a compute job, not a single session. This phase delivers the
correct, tested **machinery** (league, PFSP, Elo, match routing, exploitability,
gating, anti-collapse) and proves it runs end-to-end at tiny scale; scaling up is
a knob (iterations × games × sims), not new code.
