# Phase 1 — COMPLETION REPORT (plain neural-ISMCTS, end-to-end)

**Stack:** Python 3.12 · PyTorch 2.12 (CPU) · NumPy 2.x · Bun 1.3.14 (engine host)
**Engine:** `@tcg/lorcana-engine` driven via `LorcanaMultiplayerTestEngine`
**Verdict:** ✅ **Phase 1 complete.** A working end-to-end self-play loop:
Python search drives the real Lorcanito kernel, a dual-headed net supplies
priors + values, behaviour-cloning from the scripted oracle gives a competent
prior, and the self-play actor→learner loop runs. All Phase 1 code lives in
`lorcana-bot/` alongside the simulator (per CLAUDE.md).

---

## 0. Exit-gate scorecard

| Phase 1 sub-step | Status | Evidence |
|---|---|---|
| 1A Bridge — kernel callable from Python | ✅ | `engine/bridge.py` + `engine/node_server/server.ts`; full games play across the boundary |
| 1A Legal-action mask + state serialization | ✅ | `engine/serialization.py`; `obs["legal"]` is the engine-authoritative action set |
| 1A `LorcanaEnv` gym wrapper `step→(obs,mask,reward,done,info)` | ✅ | `test_bridge.py::test_env_plays_full_random_game` |
| 1B Set-transformer trunk + factored policy + distributional value | ✅ | `network/{trunk,heads,model}.py`; `test_network.py` (4 tests) |
| 1C B-ISMCTS: PUCT, depth-limited neural leaves, progressive widening | ✅ | `search/{node,ismcts}.py`; `test_search.py` (4 tests) |
| 1D Behaviour cloning from automation | ✅ | `training/bootstrap.py`; **loss 2.55 → 0.49** over 6 epochs |
| 1D Self-play actor + learner loop | ✅ | `training/selfplay.py`; warm-starts from BC, generates MCTS targets, trains |
| Test suite green | ✅ | **14 / 14 pass** (`pytest lorcana-bot/tests`) |
| Cross-boundary determinism | ✅ | same seed ⇒ identical scripted trajectory; diff seed ⇒ divergent action sequence |

---

## 1. The bridge (1A)

**Transport.** Option B from CLAUDE.md: a Bun process
(`engine/node_server/server.ts`) hosts the headless kernel and exchanges
newline-delimited JSON over stdio — **one round-trip per game step**, not per
MCTS node. `engine/bridge.py` owns the subprocess (`LorcanaEngine`) and exposes
`reset / observe / step / step_auto / snapshot / restore`. `LorcanaEnv` is the
gym-style wrapper.

**Action model — the key design choice.** The engine's automation candidate
enumeration (`enumerateAutomatedActionsForCurrentActor`) *is* the legal-action
set. Each `AutomatedActionCandidate` carries a `family` (→ policy category),
source/target card ids (→ policy source/target), and we reconstruct the engine's
exact `getCandidateKey` to get a stable id. **Executing a chosen action** is done
by handing the planner a one-shot strategy that orders the chosen candidate first
(the planner executes the first candidate that succeeds); a synthetic `passTurn`
action maps to the planner's pass fallback (a strategy that returns no
candidates). Verified exact: random-policy games show **0 key mismatches** —
the requested action is the one executed.

**Fog of war.** Observations are taken from the acting player's filtered
projection (`getBoard("playerOne"|"playerTwo")`), so the encoder physically
cannot see hidden zones (confirmed: opponent hand cards arrive `hidden:true`
with no definition).

**Tree branching.** Added server-side `snapshot`/`restore` (clone of the
authoritative `MatchState`, kept Node-side; only an integer handle crosses the
wire). The search snapshots the root once and restores it between simulations.

**Determinism.** Same seed ⇒ byte-identical scripted trajectory across two
independent subprocesses; different seeds diverge at the executed-action level
(different shuffles → different cards played by step 4). Hidden-RNG divergence
is the Phase-0 guarantee this builds on.

## 2. The network (1B)

`LorcanaNet` = shared **set-transformer trunk** + two heads (belief head deferred
to Phase 2, per the plan).

* **Trunk** (`trunk.py`): per-card token = numeric features + learned identity
  embedding (hashed `definitionId` vocab); a learned **NULL** token (card-position
  0, the pointer target for action-less sources/targets) and a **global CLS**
  token built from scalar features are concatenated; `n_layers` of pre-norm
  self-attention with key-padding masks. CLS output = trunk vector; null+card
  outputs = per-token reps.
* **Policy** (`heads.py`): the factored grammar (category × source × target) is
  realized as a **pointer head over enumerated legal actions** — each action's
  score combines its category embedding with the gathered source/target token
  reps and the trunk. Softmax over legal actions only; illegal entries are
  `-inf` (legality stays in the engine). Verified: illegal actions receive 0
  probability.
* **Value** (`heads.py`): **distributional C51**, 51 atoms on [-1,1], trained on
  outcome `z` via a two-hot target (verified roundtrip-exact). `infer()` returns
  numpy priors + scalar values — the only tensor work the actor does per leaf.

## 3. The search (1C)

`BISMCTS` (`ismcts.py`) over `InfoSetNode` (`node.py`):

* **PUCT** selection (AlphaZero form) with **progressive widening**
  (`k·(N+1)^α` actions opened, ranked by prior).
* **Depth-limited**: leaves are evaluated by the value head, never rolled to
  terminal (unless the game actually ends).
* **Zero-sum perspective handling**: values are stored per node from that node's
  own actor's POV; backup flips sign by whether a node's actor matches the
  player a leaf value is expressed for (`value_for`). Lorcana's many consecutive
  same-player decisions are handled correctly (not naïve ply-alternation).
* **Determinization N=1** over the realized world (the engine's true state).
  The multi-determinization / belief-weighted seam is explicit in
  `SearchConfig.n_determinizations` and `_simulate` — Phase 2 fills it in.
* Outputs the visit-count policy `π` (the AZ training target) and a root value.
* **Bug caught + fixed during bring-up:** `run()` now restores the engine to the
  root decision state before returning, so the caller steps the chosen action
  from the correct position (previously left on the last simulation's leaf).

## 4. Training (1D)

* **Bootstrap / behaviour cloning** (`bootstrap.py`): the scripted
  `bestDeckAwareOracleLoreRaceAutomatedActionStrategy` plays both seats; at each
  decision we log the information set, the chosen action (one-hot policy
  target), and the eventual outcome `z` from that actor's POV. Supervised result
  (8 games, 1936 samples, 6 epochs):

  | epoch | loss | policy | value | H(π) |
  |---|---|---|---|---|
  | 1 | 2.546 | 0.716 | 1.830 | 0.782 |
  | 3 | 1.383 | 0.518 | 0.865 | 0.540 |
  | 6 | 0.489 | 0.474 | 0.014 | 0.489 |

* **Self-play** (`selfplay.py`): both seats driven by B-ISMCTS + `NetEvaluator`;
  logs `(I, π_MCTS, z)` and trains via the same `Learner`. Warm-starts from the
  BC checkpoint. AlphaZero-style policy improvement loop, single-process.
* **Learner** (`learner.py`): replay buffer + masked policy cross-entropy +
  two-hot value cross-entropy + optional entropy bonus; grad-clipped Adam.

## 5. Deliverables (`lorcana-bot/`)

```
engine/    node_server/server.ts, bridge.py, serialization.py
network/   trunk.py, heads.py, model.py
search/    node.py, ismcts.py, evaluator.py
training/  learner.py, bootstrap.py, selfplay.py
tests/     test_bridge.py, test_network.py, test_search.py   (14 pass)
README.md, requirements.txt, pyproject.toml, conftest.py
```

## 6. Performance notes (not blockers; addressed in later phases)

* ~8 engine steps/s through the JSON bridge; ~0.3 MCTS-decisions/s at 16 sims.
  The architecture's in-process WASM kernel (Option A) and the batched-inference
  Sebulba split (§4.2) are the throughput path — both promotable behind the same
  `bridge.py` / `Evaluator` interfaces with no API changes here.

## 7. Carry-forward into Phase 2

1. The determinization seam (`n_determinizations`, `_simulate`) is where the
   belief net + importance-weighted determinization + Bayesian filtering plug in.
2. The trunk already emits per-token reps suitable for a **belief head**
   (opponent hand / inkwell) on the shared trunk — the third head from the doc.
3. `serialization.py` exposes hidden/visible flags per card token, so belief
   supervision targets (true hidden world, free in self-play) are one logging
   hook away.
