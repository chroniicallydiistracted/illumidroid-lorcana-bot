# lorcana-bot — Phase 1 (plain neural-ISMCTS)

Python ML stack that drives the Lorcanito TypeScript engine to play Disney
Lorcana. Phase 1 is the end-to-end self-play loop: bridge → network → search →
bootstrap. No belief net, no league yet (Phases 2–4).

See `../CLAUDE.md` for project memory and `../lorcana-simulator/lorcana-bot-architecture.md`
for the full design. Phase 1 deliverable write-up:
`../lorcana-simulator/phase1/PHASE1-COMPLETION-REPORT.md`.

## Layout

```
engine/
  node_server/server.ts   Bun process hosting the headless kernel (JSON-lines over stdio)
  bridge.py               LorcanaEngine subprocess client + gym-style LorcanaEnv
  serialization.py        obs -> card-token / global / factored-action tensors
network/
  trunk.py                set-transformer encoder over card tokens
  heads.py                factored pointer policy + distributional (C51) value
  model.py                LorcanaNet (+ numpy infer for the actor)
search/
  node.py                 InfoSetNode: PUCT + progressive widening
  ismcts.py               B-ISMCTS planner (depth-limited, neural leaves)
  evaluator.py            UniformEvaluator / NetEvaluator
training/
  learner.py              ReplayBuffer + policy/value loss + gradient steps
  bootstrap.py            behaviour cloning from the scripted automation
  selfplay.py             AlphaZero-style self-play actor loop
tests/                    test_bridge / test_network / test_search
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

## Known Phase-1 scope limits (addressed later)

* **Determinization N=1** over the realized world (the engine's true state). The
  multi-determinization / belief-weighted sampling seam is in `SearchConfig`
  (`n_determinizations`) and `ismcts._simulate`; Phase 2 fills it in.
* **Transport** is one JSON round-trip per game step (Option B). The in-process
  WASM path (Option A) can be promoted behind the same `bridge.py` interface.
* Single-process actor/learner; the distributed Sebulba split is a later perf
  task, not a correctness one.
