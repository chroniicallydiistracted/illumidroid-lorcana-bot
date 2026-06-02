# lorcana-bot — B-ISMCTS agent

Python ML stack that drives the Lorcanito TypeScript engine to play Disney
Lorcana. Implemented: bridge → set-transformer net (policy + distributional
value + **belief** heads) → B-ISMCTS (PUCT, depth-limited, **belief-guided
importance-weighted determinization**, leaf-batched GPU inference) → training
(BC bootstrap, self-play, **league/PSRO + exploitability**). Plays the real
25-deck tournament metagame.

**Tier-A remediation is active:** belief-guided clean-label training is
fail-closed until the full hidden-zone shared-tree release gate passes. Run
`../train.sh --no-belief` for non-belief diagnostics only.
Audited checkpoint: Phase 0 guard complete; Phase 13 baseline installed
(2 passing Phase-1 probes + 18 strict expected-red probes); Phase 1 canonical
full hidden-zone `World` boundary complete and audited GO. Phase 2 structured
sampler math is next.
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
  ismcts.py               B-ISMCTS planner: run() + diagnostic PIMC + shared-tree prototype
  determinize.py          belief-sampled worlds + importance weights (P2)
  belief_filter.py        SIR particle filter / Bayesian filtering (P2)
  evaluator.py            Uniform/Net evaluators + legacy/structured belief evaluators
training/
  learner.py              ReplayBuffer + policy/value/belief loss + gradient steps
  bootstrap.py            behaviour cloning from the scripted automation
  selfplay.py             AlphaZero-style self-play actor loop (belief labels guarded)
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
# Non-belief diagnostic self-play remains available during Tier-A remediation.
# Belief-guided clean-label training is intentionally blocked.
python -m training.selfplay --no-belief --init checkpoints/bc.pt --iterations 3 --games 4 --sims 32
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

## Tier-A belief-search remediation

`BISMCTS.run_pimc_diagnostic(...)` preserves the legacy root-pooled PIMC search
for diagnostics only. It must not produce training labels. Belief-guided
clean-label self-play remains blocked until the structured full-world,
shared-tree path passes the Tier-A release gate in
`../tier-a-belief-search-remediation-plan.md`.

Phase 1 of that remediation is complete: `search/determinize.py` now defines the
canonical reproducible `World`, validates the complete opponent hidden-pool
witness fail-closed, and enforces clean-label seed admission. The next active
remediation step is Phase 2 sampler math.

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
