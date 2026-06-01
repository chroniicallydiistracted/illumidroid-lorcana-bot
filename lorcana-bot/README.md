# lorcana-bot — B-ISMCTS agent (Phases 0–3 complete)

Python ML stack that drives the Lorcanito TypeScript engine to play Disney
Lorcana. Implemented: bridge → set-transformer net (policy + distributional
value + **belief** heads) → B-ISMCTS (PUCT, depth-limited, **belief-guided
importance-weighted determinization**, leaf-batched GPU inference) → training
(BC bootstrap, self-play, **league/PSRO + exploitability**). Plays the real
25-deck tournament metagame. Phases 0–3 done; Phase 4 (ReBeL/PBS, richer value
heads) is future.

**Run it:** `../train.sh` (parallel self-play) or `../train.sh league` (PFSP).
See `../CLAUDE.md` for current state + commands, `../lorcana-simulator/lorcana-bot-architecture.md`
for the design, `PORT-AUDIT.md` for the engine-port analysis, and the phase
reports in `../lorcana-simulator/phase{1,2,3}/`.

## Layout

```
engine/
  node_server/server.ts   Bun process hosting the headless kernel (JSON-lines over stdio)
  bridge.py               LorcanaEngine subprocess client + gym-style LorcanaEnv
  serialization.py        obs -> card-token / global / factored-action tensors
network/
  trunk.py                set-transformer encoder over card tokens
  heads.py                factored pointer policy + distributional value + belief (P2)
  model.py                LorcanaNet 3 heads (+ numpy infer / belief_probs)
search/
  node.py                 InfoSetNode: PUCT + progressive widening
  ismcts.py               B-ISMCTS planner: run() (P1) + run_belief() (P2)
  determinize.py          belief-sampled worlds + importance weights (P2)
  belief_filter.py        SIR particle filter / Bayesian filtering (P2)
  evaluator.py            Uniform/Net evaluators + BeliefEvaluator (P2)
training/
  learner.py              ReplayBuffer + policy/value/belief loss + gradient steps
  bootstrap.py            behaviour cloning from the scripted automation
  selfplay.py             AlphaZero-style self-play actor loop (belief-guided)
tests/                    test_bridge / test_network / test_search
                          + test_belief / test_determinize / test_search_belief (P2)
```

## Setup

```bash
python3 -m venv ../lorcana-bot-venv
../lorcana-bot-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
../lorcana-bot-venv/bin/pip install numpy pytest
# Bun must be on PATH (the engine runs under Bun); the simulator workspace must
# have its node_modules installed (see ../CLAUDE.md dev commands).
```

## Quick start

```python
from engine.bridge import LorcanaEnv
env = LorcanaEnv()
obs, mask = env.reset("seed-0")
obs, mask, reward, done, info = env.step(0)   # action = index into obs["legal"]
env.close()
```

Search with a (random-init) net:

```python
from engine.bridge import LorcanaEngine
from network.model import LorcanaNet
from search.ismcts import BISMCTS, SearchConfig
from search.evaluator import NetEvaluator

eng = LorcanaEngine(); obs = eng.reset("seed-0")
res = BISMCTS(eng, NetEvaluator(LorcanaNet()), SearchConfig(simulations=64)).run(obs)
action = int(res.pi.argmax())          # search-improved policy over obs["legal"]
obs = eng.step(obs["legal"][action]["stableKey"])["obs"]
```

## Training

```bash
# Behaviour-clone the scripted oracle (Stage 0 prior):
python -m training.bootstrap --games 20 --epochs 5 --out checkpoints/bc.pt
# Self-play, warm-started from the BC prior:
python -m training.selfplay --init checkpoints/bc.pt --iterations 3 --games 4 --sims 32
```

## Tests

```bash
pytest tests/test_network.py          # fast, no engine
pytest tests/test_bridge.py tests/test_search.py   # spawn the Bun engine (slower)
```

## Action model

The kernel's automation enumeration is the legal-action set. Each action is an
`AutomatedActionCandidate` carrying a `family` (policy category), source/target
card ids (policy source/target), and a stable key. A chosen action executes by
handing the planner a one-shot strategy that orders that candidate first; a
synthetic `passTurn` action maps to the planner's pass fallback. Legality stays
authoritative in the engine — the network only scores legal continuations.

## Phase 2 — belief net + importance-weighted determinization + Bayesian filtering

Done. `BISMCTS.run_belief(obs, BeliefEvaluator(net), n_worlds=N)` samples N
opponent-hand worlds from the (leak-free) belief head, searches each determinized
world with Phase-1 PUCT, and pools root statistics with importance weights.
`search/belief_filter.ParticleFilter` sharpens the belief between turns. See
`../lorcana-simulator/phase2/PHASE2-COMPLETION-REPORT.md`.

## Real decks

25 real tournament-winning decklists live in `decks/*.json` (fixture shape
`{name, cards}`; provenance under `decks/docs/`). The bridge resolves them
against the full card catalog and uses them by default:

```python
eng = LorcanaEngine()
eng.list_decks()                                   # [{id, name}, ...]  (25)
eng.reset("seed", deck_p1="01_cloudy_ae_aggro", deck_p2="18_yurple")
# omit deck ids -> a pair is picked deterministically from the seed
# deck_p1="placeholder" -> old synthetic fallback
```

Bootstrap/self-play sample distinct deck pairs per game for metagame diversity.
This makes the belief net and determinization strategically meaningful (real,
shared card identities across games).

## Known scope limits (addressed later)

* **Transport** is one JSON round-trip per game step (Option B). The in-process
  WASM path (Option A) is a throughput option (note: QuickJS-WASM is interpreted
  and ~20–100× slower than the JIT'd Bun subprocess — see the WASM viability
  research); the real throughput levers are transport-batching + batched
  inference (Sebulba), promotable behind the same `bridge.py` interface.
* Single-process actor/learner; the distributed Sebulba split and Phase 3
  league/PSRO are the next milestones.
