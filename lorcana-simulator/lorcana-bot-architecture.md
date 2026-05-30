# Champion-Level Lorcana TCG Agent — Systems Design Document

**Status:** Architecture Proposal (v1.0)
**Scope:** Full-stack design for a self-play–trained agent that plays Disney Lorcana at champion level, using the Lorcanito TypeScript engine as the authoritative rules oracle.
**Design thesis:** The hard problem is not "evaluate a board." It is *acting near-optimally under a belief over hidden worlds whose support changes every turn*. Every component below exists to maintain, sharpen, and search over that belief efficiently.

---

## 0. Executive Summary & Algorithmic Verdict

| Concern | Decision |
|---|---|
| Primary planner | **Neural-guided Information-Set MCTS (ISMCTS) with PUCT selection** — an AlphaZero policy/value net *inside* an information-set search, not a perfect-information AZ. |
| Belief handling | **Learned belief network** producing `P(world ∣ public infoset)`, used for **importance-weighted determinization** rather than uniform sampling. |
| Equilibrium pressure | **ReBeL-style public-belief-state (PBS) value head + league/PSRO self-play** to bound exploitability in this non-transitive metagame. |
| Engine bridge | **Extract the headless `@tcg/core` kernel and run it in-process (WASM) inside each search actor.** The engine is a Mutative-patch reducer with built-in fog-of-war projection and legal-move generation — per-node JS↔Python IPC is eliminated, not optimized. |
| Inference | **Central GPU inference server with dynamic batching** (Sebulba/Podracer actor–learner split). |

The rest of this document defends and details each row.

---

## 0.5. Engine Ground Truth (verified against `TheCardGoat/lorcana-simulator`)

Before any ML design, the actual engine contract. Lorcanito is **not** a loose simulator — it is a clean monorepo (`pnpm`/`nx`) whose rules kernel sits in `packages/lorcana/`, built on a generic `@tcg/core` `MatchRuntime` framework (boardgame.io-lineage). Every assumption in this document is now pinned to a real API.

**Package map (`packages/lorcana/`):**

| Package | Role for us |
|---|---|
| `@tcg/lorcana-engine` | The rules kernel + `@tcg/core` runtime. **This is what the search drives.** |
| `@tcg/lorcana-types` | State/card/ability/targeting types (the schema to serialize). |
| `@tcg/lorcana-cards` | Card definitions + parser + catalog data (the card vocabulary + structured effects of §2.2). |
| `@tcg/lorcana-interaction` | Interaction surface types. |
| `@tcg/lorcana-server-adapter` | Server wiring. |
| `@tcg/lorcana-simulator` | The SvelteKit **app** (routes/static) — UI only, **separable and discardable** for our purposes. |

**The runtime contract (this is the whole bridge problem, solved by the engine's own shape):**

- **State:** `LorcanaMatchState = MatchState = { G: LorcanaG, ctx: TCGCtx }`. `ctx` carries `zones` (first-class zone runtime), `random.seed`, `status`, `matchID/gameID`, priority, time.
- **Transition (the reducer):** `executeCommand(...)` applies a move and **returns `patches: Patch[]`** — Mutative (Immer-lineage) patches. State is immutable with **structural sharing**, so cloning a node for search is copy-on-write cheap, and every transition is **revertible via inverse patches**. This is the single most important fact for MCTS performance (see §4).
- **Legal-action generation (the action mask):** the two-layer `available-moves` API — `getAvailableMoves()` (Layer 1: which `AvailableMoveId` categories are legal and which `selectableCardIds` start them) then `getMoveOptions()` (Layer 2: given a first selection, the legal targets/costs). Move categories: `playCard, singCard, shiftCard, putCardIntoInkwell, quest, challenge, moveCharacterToLocation, activateAbility, passTurn, questWithAll, chooseWhoGoesFirst, concede`. **This hierarchy *is* our action grammar** (§2.2) and dictates a factored policy head.
- **Information set (fog of war, engine-enforced):** `filterMatchView(state, {role, playerID}, zoneRegistry) → FilteredMatchView`. Zones declare `visibility: "public" | "private" | "secret"` plus `ordered`, `ownerScoped`, `faceDown`. `filterRandom` strips the RNG seed from client views. The doc-comment is explicit: *filtering is a runtime guarantee, not a UI convention.* Mapping:
  - `public` → board, discard, lore, ink **count** → fully observed.
  - `private` (`ownerScoped`) → **opponent hand** → the hidden variable the belief net models.
  - `secret`/`ordered` → **deck ordering** → the top-deck distribution.
  - `faceDown + ownerScoped` → **inkwell**: count public, identities hidden → a *distinct* belief target (what did they choose to ink?).
- **Structured projection for the encoder:** `engine/projection.ts` already emits `EngineProjectionSnapshot` / `EngineBoardProjection` / `Engine{Active,Pending}EffectProjection`. **Use these as the feature source** for the state encoder instead of re-deriving features from raw `G`.
- **Determinism & restore:** seeded RNG (`ctx.random.seed`) + `RuntimeSnapshot` / `MatchSnapshot{stateID}` with `saveSnapshot`/`getSnapshotAtStateID`. Determinizations are reproducible; tree nodes are restorable.
- **Free bootstrap assets:** `@tcg/lorcana-engine` already ships an `automation/` layer — `defaultLoreRaceAutomatedActionStrategy`, `deckAwareLoreRaceAutomatedActionStrategy`, `bestDeckAwareOracleLoreRaceAutomatedActionStrategy`, `BEST_AI_DECK_DOSSIERS`, plus `createAutomatedActionBoardSnapshot` and `computeAutomatedActionStateFingerprint`. These give us **scripted league anchors**, a **deck distribution**, a **ready board-feature snapshot**, and a **transposition-table hash** for free.

**Net effect on the design:** the "build a faithful reducer + legal-move generator + fog-of-war projection + serialization + RNG seeding" work that would normally be P0's biggest risk is *already done and tested* (the repo carries extensive `e2e/` suites per keyword: shift, evasive, ward, bodyguard, support, challenger-resist, reckless-rush, vanish, boost). Our P0 collapses to "extract this kernel from the SvelteKit app and make it callable at scale" (§4).

---

## 1. Core Architecture Framework

### 1.1 Why not the obvious candidates (and what we steal from each)

**Pure AlphaZero (perfect-information MCTS).** Disqualified at the root: AZ assumes the searching player can enumerate the true state. In Lorcana the true state includes opponent hand, opponent inkwell contents, and both deck orderings. A naïve AZ that searches the *true* state is a cheater — it will learn lines that are only correct because it "saw" the top-deck. We keep AZ's **dual-headed network + MCTS-as-policy-improvement** loop, but never let the network condition on hidden ground truth.

**Pure Counterfactual Regret Minimization (CFR / Deep CFR).** CFR is the gold standard for *exploitability guarantees* in imperfect-information games, and it is why poker fell. But CFR's natural habitat is games with a **small, flat action abstraction** (fold/call/raise-buckets). Lorcana's per-turn decision is a *deeply sequential combinatorial program*: ink a card → play a card → trigger an ability with sub-targets → quest with character A → challenge with B into C → sing a song → pass. A single turn is itself a tree of dozens of dependent sub-decisions, and the full game tree's information sets number far beyond what tabular or even Deep CFR traversal can cover at championship quality. We **do not** run CFR over the whole game. We borrow its **regret-matching update and exploitability metric** for evaluation and for local subgame solving.

**Plain ISMCTS.** This is the correct *spine*. ISMCTS searches information sets directly: at each visit it samples a *determinization* (a concrete consistent world), expands the tree using legal moves of that world, but **pools visit/value statistics at shared information-set nodes** so that statistics aggregate across many sampled worlds. It handles (a) hidden information, (b) stochasticity (draws are chance nodes), and (c) the gigantic structured action space (via neural priors + progressive widening). Its weakness — it is heuristic and can be exploitable — is patched by the belief network and the PBS value head below.

### 1.2 The chosen system: **B-ISMCTS** (Belief-guided, Neural ISMCTS with PBS leaves)

```
                          ┌──────────────────────────────────────────┐
                          │            B-ISMCTS PLANNER                │
                          │                                            │
   public infoset I  ───► │  1. Belief net  b(world | I) ──► sample    │
                          │     N determinizations w_1..w_N            │
                          │                                            │
                          │  2. For each w_i: run PUCT-ISMCTS,         │
                          │     priors & leaf values from the          │
                          │     dual-headed net f_θ                    │
                          │                                            │
                          │  3. Pool stats at infoset nodes →          │
                          │     improved policy π(I)                   │
                          │                                            │
                          │  4. Leaves NOT rolled to terminal; cut at  │
                          │     depth d, evaluated by PBS value head   │
                          └───────────────┬────────────────────────────┘
                                          │  π(I), v(I)
                                          ▼
                                    action a ~ π(I)
```

**Selection rule (PUCT, AlphaZero form):**

```
a* = argmax_a [ Q(I,a) + c_puct · P(a|I) · √(Σ_b N(I,b)) / (1 + N(I,a)) ]
```

where `P(a|I)` is the network policy prior (legal-masked), `N` are visit counts pooled across determinizations, and `Q` is the mean pooled value. Add **Dirichlet noise** at the root for exploration during self-play.

**Why this handles *future-state probability modeling* best for a TCG:**

- Hidden variables are handled at their natural granularity — the **information set** — and the belief net makes the determinization sampling *informed* rather than uniform, so search budget concentrates on plausible futures (the opponent probably holds removal given they ran two ink of Ruby/Amethyst and have passed on tempo, etc.).
- Re-drawing every turn is exactly a **chance node**; ISMCTS already models this. Across determinizations the unknown draws are re-sampled, so the value at a node is an expectation over futures, not a single lucky line.
- The PBS value leaf gives the search a *strategically calibrated* horizon: instead of rolling a 14-turn game to terminal (compute death + high variance), we cut at depth `d` and ask a value head trained on outcomes "what is the win-probability of this belief state."

### 1.3 Network topology `f_θ` (the shared trunk)

Three heads on one trunk (multi-task, shared representation):

```
                        ┌────────► Policy head  p(a | I)   (legal-masked logits over the action grammar)
   I ─► Trunk (set/   ──┤
        transformer)    ├────────► Value head   v(I) ∈ [-1,1]  (categorical / two-hot, see §5.4)
                        │
                        └────────► Belief head  b(world | I)  (factored distribution over hidden zones)
```

The belief head can live on the **shared trunk** (cheap, regularizes the trunk with ground-truth supervision available in self-play) or as a **separate network** if you want to decouple its training cadence. Recommendation: shared trunk + detached gradient option, so belief supervision improves the representation without destabilizing the policy.

---

## 2. State Embedding & Probability Layers

### 2.1 Representation principle: the board is a *set of typed entities*, not an image

Do **not** flatten the state into a fixed CNN grid. Lorcana zones are variable-length multisets. Use a **permutation-invariant set/transformer encoder** over *card tokens*. This generalizes across board sizes and is essential for the belief head.

> **Feed it the engine's projection, not raw `G`.** Build tokens from `EngineBoardProjection` / `EngineProjectionSnapshot` (from `engine/projection.ts`), which already resolves derived stats, active/continuous effects, and pending effects. And build it from the **`FilteredMatchView` for the acting player** (`filterMatchView(state,{role,playerID})`) — never from authoritative server state — so the encoder physically *cannot* see hidden zones. The zone `visibility` flag tells you which tokens are real (`public`) vs. must be supplied by the belief net (`private`/`secret`).

**Per-card token = concat of:**

| Field | Encoding | Notes |
|---|---|---|
| Card identity | learned embedding `E[card_id]` (vocabulary over the printed card pool) | the workhorse signal |
| Cost / Strength / Willpower / Lore | scalar → small MLP / Fourier features | numeric stats |
| Ink color | 6-way one-hot (Amber, Amethyst, Emerald, Ruby, Sapphire, Steel) | also feeds the belief prior |
| Card type | {Character, Action, Song, Item, Location} one-hot | |
| Keyword multi-hot | Shift, Evasive, Rush, Bodyguard, Ward, Resist, Reckless, Challenger, Support, Singer, … | see §2.2 |
| Dynamic state | damage counters, ready/exerted, "ink wet" (played this turn), attached items/locations | the part that changes turn-to-turn |
| Zone embedding | {hand, board-ready, board-exerted, inkwell, discard, deck-known, in-play-location} | positional analogue |
| Ownership | self / opponent | |

Then: `tokens → Transformer encoder (with zone-segment embeddings) → mean/attention pool per zone → global state vector`. Append **global scalars**: lore totals (self/opp), ink available, turn number, phase, cards-in-hand counts, deck sizes.

### 2.2 Encoding card *text* / keywords without brittleness

Two-layer scheme, because new sets ship constantly and you want zero-shot-ish generalization:

1. **Structured effect schema (primary).** `@tcg/lorcana-cards` already implements each card's behavior as structured definitions (`src/cards/`, with a `parser/` and a `data/` catalog) — not free text. Parse *that* into a **typed effect vector / small effect-graph**: trigger condition (on-play, on-quest, on-challenge, activated, static), targets (self/opp/all/chosen), magnitude, duration, resource cost. Keywords (Shift, Evasive, Rush, …) get **learned keyword embeddings** summed into the token — let the net learn that Evasive interacts with the challenge legality mask, that Rush removes the "ink wet" restriction for challenging, etc., as *relational* facts rather than opaque flags. The card catalog is also your **vocabulary** for `E[card_id]`.
2. **Oracle-text encoder (fallback / generalization).** A small frozen text encoder (e.g. a distilled sentence encoder) over the printed rules text, projected into the token space. This gives a usable embedding for cards the structured parser hasn't been hand-mapped for yet, smoothing new-set rollouts. Blend: `card_token = E[card_id] (if seen) ⊕ structured_effect_vec ⊕ α·text_embed`.

> Design note: keep the **legality mask authoritative in the engine**, never in the net. Use the two-layer `available-moves` API directly: Layer 1 (`getAvailableMoves`) gives the legal move categories + selectable cards; Layer 2 (`getMoveOptions`) gives legal targets/costs for a chosen first selection. The net learns *preferences* over this masked structure; Lorcanito decides *legality* (can this Evasive character be challenged? is Bodyguard forcing a target? is the song singable / sing-together total met?). This cleanly separates "what's allowed" (rules) from "what's wise" (policy) — and the two layers map directly to a **factored policy head** (§2.5).

### 2.3 Preserving chronological / "future" information

"Future information" in an imperfect-info game is really **trajectory-conditioned belief**: what the opponent has done tells you what they likely hold and will do. Two mechanisms:

- **Public-history encoder.** Maintain a sequence of *public events* (cards played, inked, quested, challenged, songs sung, passes, mulligan info). Encode with a causal Transformer / GRU into a `history_context` vector concatenated to the state. This is what lets the agent reason "they have passed on developing tempo twice — they are holding reactive cards / a big swing."
- **Turn-phase + horizon features.** Explicit turn counter, who is "on the play," ink-curve trajectory, and a learned **horizon embedding** so the value head can distinguish "ahead now but out of gas" from "behind now with an inevitable engine."

### 2.4 The Opponent Range / Hand-Modeling sub-network (the centerpiece)

This is what turns generic ISMCTS into a Lorcana specialist. We model the opponent's hidden state as a distribution and *sample from it* during search.

**Inputs:**
- `history_context` (public play sequence).
- Opponent's **declared ink identity** — inferred from cards they have inked/played; in a 2-ink format this massively constrains the legal card pool.
- **Meta prior:** a Dirichlet over archetypes `θ_arch` fit from tournament/ladder decklists (Ruby/Amethyst aggro, Amber/Steel control, Sapphire ramp, etc.). Card-conditional priors `P(card ∣ archetype)`.
- Cards already **revealed/played/discarded** (removes them from the live distribution — a hard constraint applied post-hoc).

**Output (factored to stay tractable):**
- `P(card_c ∈ opp_hand)` for each card `c` in the inferred pool (multi-label head), and
- a **count-consistent** distribution over the multiset of the hand (hand size is public via the `private` zone's count), enforced by a differentiable normalization / optimal-transport projection so the marginals sum to the known hand size.
- a **separate inkwell-identity belief** — the inkwell is `faceDown + ownerScoped` (count public, identities `secret`), so *which* cards the opponent chose to ink is itself hidden information worth modeling (it reveals what they were willing to spend vs. hold).
- Optionally `P(next_draw = c)` from the residual deck distribution.

> **Ground truth is free in self-play.** The server holds the authoritative `MatchState`; each actor's policy only ever consumes its `FilteredMatchView`. Logging both gives exact supervised targets for the belief head — the true contents of every `private`/`secret` zone — at zero labeling cost (see §5.3).

**How it plugs into search — importance-weighted determinization:**

```
world w_i  ~  b(world | I)        # sample a concrete opp hand + deck order consistent with belief
weight   ρ_i = b(w_i | I) / q(w_i)  # if proposal q ≠ b, importance-correct
node value  Q(I,a) = Σ_i ρ_i · v_i(a)  /  Σ_i ρ_i
```

Replacing **uniform** determinization with **belief-weighted** determinization is the single highest-leverage modeling choice in this document: it focuses the (expensive) tree search on the worlds that actually have probability mass, which is exactly "managing a massive future decision space while accounting for hidden variables."

**Bayesian filtering between turns.** After each opponent action, update the belief by conditioning on the observed action's likelihood under each hypothesized world (a particle-filter / sequential-importance-resampling step). The neural belief net is the *proposal*; the observed legality + play likelihood is the *correction*. This keeps the belief sharp as hidden info is revealed.

### 2.5 Factored policy head (mirrors the engine's two-layer action API)

Don't emit a flat softmax over a combinatorial action space — it would be enormous and mostly illegal. Mirror `available-moves`:

```
Policy head =  π_cat(category | I)              # over the 12 AvailableMoveId categories
            ×  π_src(card | category, I)         # over Layer-1 selectableCardIds
            ×  π_tgt(target/cost | card, I)      # over Layer-2 getMoveOptions (targets, costs,
                                                 #   sing-together singers, ability index, location)
```

Each factor is masked by the corresponding engine output, so the net only ever scores legal continuations. During search this is a natural **progressive-widening** boundary (expand categories first, then sources, then targets) and a clean place to apply **action abstraction** (§3.2) — e.g. collapse interchangeable `putCardIntoInkwell` choices. The composite log-prob `log π = log π_cat + log π_src + log π_tgt` is what PUCT consumes as the prior `P(a|I)`.

---

## 3. Future-Lookahead Search & Rollout Engine

### 3.1 Looking ahead when info is revealed every turn

The core trick: **determinize, but re-sample the unknown future at chance boundaries.**

- **Within a determinization `w_i`:** the opponent hand and deck order are *fixed*, so the next few turns are a (large but) well-defined game tree. Draw steps consume the fixed deck order of `w_i`.
- **Across determinizations:** because we sample many `w_i ~ b(world|I)`, the *aggregate* value at an information-set node is an expectation over all plausible draws and opponent holdings. Top-decks are handled honestly: a line that only wins under one lucky draw gets averaged down across the determinization ensemble.

```
                 INFORMATION-SET NODE (statistics pooled here)
                        │
        ┌───────────────┼───────────────┐         each child edge = an action in the
        ▼               ▼               ▼          ACTION GRAMMAR (ink/play/quest/
   det. w_1        det. w_2   ...   det. w_N        challenge/sing/ability/pass)
   (opp hand A,    (opp hand B,
    deck order A)   deck order B)
        │               │
   chance: draw     chance: draw
   from deckA        from deckB
        ▼               ▼
   ... deeper game tree, cut at depth d ...
        ▼
   PBS VALUE LEAF  v(I_leaf)   ← NOT a terminal rollout
```

### 3.2 Determinization vs. Belief States — and how to not blow the compute budget

You will use **both**, at different layers:

- **Determinization (sampling)** is the cheap, scalable Monte-Carlo layer. **Concretely:** take the acting player's `FilteredMatchView`, then *fill in* the `private` zone (opp hand) and order the `secret` zone (decks) by sampling from the belief net, producing a complete authoritative `MatchState`; set `ctx.random.seed` so future draws are reproducible. Budget knob: `N` determinizations × `S` simulations each. Importance weighting (§2.4) makes each sample worth more, so you need fewer. Tree traversal uses **Mutative inverse patches** to undo `executeCommand` rather than deep-copying state at every node (copy-on-write structural sharing makes this nearly free).
- **Belief state / PBS (ReBeL-style)** is the principled layer used at the **root subgame** when exploitability matters (e.g., key combat-trade decisions, or the closing turns). Represent the *public belief state* — the public board plus the joint belief over both players' private states — and run **depth-limited subgame solving** with regret-matching, terminating at the **PBS value net**. This gives a much less exploitable policy than raw ISMCTS for the decisions that decide games.

**Compute-budget controls (all of these ship):**

1. **Depth limiting + neural leaf value.** Never roll to terminal. Cut at depth `d` (e.g., 1–2 full turn cycles) and call the value head. This is the dominant cost saver.
2. **Progressive widening** on the branching factor: only expand `k·N(I)^α` children of a node, ordered by policy prior, so the combinatorial within-turn action sequence doesn't explode the tree.
3. **Action abstraction / macro-grouping.** Collapse strategically-equivalent micro-actions (e.g., "ink *this* spare land-equivalent" when two inkable cards are interchangeable; "quest with all idle non-attackers") into single search actions, expanded to concrete sequences only at execution. Huge branching reduction.
4. **Sub-turn factoring.** Solve the ink/play sub-decision and the combat sub-decision as *staged* subproblems where independence holds, rather than one flat product space.
5. **Transposition table** keyed by a canonicalized infoset hash — reuse the engine's `computeAutomatedActionStateFingerprint` (zones are sets — it already canonicalizes) to share statistics across move reorderings.
6. **Batched leaf evaluation** — collect leaves across determinizations/actors and evaluate the net in big GPU batches (see §4).

### 3.3 Macro-strategy vs. micro-tactics

This is a temporal-abstraction problem; solve it by **division of labor between the value head and the search**:

- **Micro (this turn):** the depth-limited search resolves *exactly* — challenge math (who trades, who survives given Strength/Willpower/Resist/damage), questing lines, ability sequencing, singing math, ink decisions. Because this is shallow and exact, the agent gets the tactical arithmetic right.
- **Macro (the win plan):** carried implicitly by the **value head**, which is trained on game *outcomes* (§5), so a leaf state that looks lore-behind but is set up for a turn-7 lethal swing evaluates high. The macro plan is not a separate symbolic planner; it is the value gradient that the shallow search climbs.
- **Optional explicit macro layer:** an **options / subgoal head** that proposes a strategic intent ("race to 20," "stabilize and grind," "assemble Shift package") and biases the policy prior. Useful for interpretability and for league diversity, but the value-head-as-plan approach is sufficient and simpler — add this only if you observe myopic closing lines in eval.

> Practical rule: *deepen the search near decision points that the value head is most uncertain about* (use value-head ensemble variance or policy entropy to allocate extra simulations). Don't search uniformly.

---

## 4. API & Bridge Topology (TypeScript engine ↔ Python ML)

### 4.1 The non-negotiable constraint

MCTS at championship strength issues **10^5–10^7 state transitions per move decision** (determinizations × simulations × depth). A per-transition round-trip across a JS↔Python process boundary — REST, or even gRPC — adds tens of microseconds to milliseconds *each*, which dominates everything and caps you at hobby strength. **The boundary crossing must be removed from the inner loop, not optimized.** There are exactly two production-grade ways to do that.

### 4.2 Recommended: extract the `@tcg/core` kernel and run it *in-process* in each actor

The verified contract (§0.5) makes this clean. The rules kernel — `@tcg/lorcana-engine` + `@tcg/lorcana-cards` + the `@tcg/core` runtime — has **no DOM/Svelte dependency**; only the `@tcg/lorcana-simulator` app does. So you extract a headless kernel that exposes exactly four operations the search needs:

```
init(deckA, deckB, seed)                 → MatchState              (createLorcanaServerGame)
executeCommand(state, command)           → { state', patches[] }   (the reducer)
getAvailableMoves(state, playerID)       → legal action mask       (Layer 1 + Layer 2)
filterMatchView(state, {role,playerID})  → FilteredMatchView        (the information set)
        + snapshot()/restore(stateID), seed control                (RuntimeSnapshot)
```

Two ways to make those calls IPC-free inside the search worker, in order of preference:

1. **Compile the extracted kernel to WebAssembly** and load it in the search process (Python via `wasmtime`/`wasmer`, or a C++/Rust search host). Because the kernel is pure TS over Mutative state with no DOM, **QuickJS→WASM (Javy)** compiles it faithfully with modest effort; a later hand-port of the hottest paths (combat resolution, `getAvailableMoves`) to Rust/AssemblyScript buys more throughput. `executeCommand`/`getAvailableMoves` become **in-process microsecond calls**.
2. **Node embedded as a co-process per actor** using the engine's existing `in-memory-transport.ts`, exchanging only **Mutative `Patch[]`** (the runtime already emits them) over shared memory — never full states, never JSON. Lower effort, higher per-call cost than WASM; acceptable if porting is deferred.

Either way the inner MCTS loop never crosses to Python except to **batch leaf states to the inference server**:

```
┌──────────────────────── SELF-PLAY ACTOR PROCESS (×K, one per core) ───────────────────────┐
│                                                                                            │
│   B-ISMCTS search loop  ──in-process calls──►  @tcg/core KERNEL (WASM)                      │
│         │                                       executeCommand → Patch[]  (revert via       │
│         │                                       inverse patches; no deep copy)              │
│         │                                       getAvailableMoves → action mask             │
│         │                                       filterMatchView → information set           │
│         │                                                                                  │
│         └── batches leaf FilteredMatchViews ──► local queue ──┐                             │
└────────────────────────────────────────────────────────────────┼──────────────────────────┘
                                                                  │  (FlatBuffers, zero-copy)
                                                                  ▼
                                       ┌──────────────────────────────────────┐
                                       │   GPU INFERENCE SERVER (Triton/custom)│
                                       │   dynamic batching of f_θ forward     │
                                       └───────────────┬──────────────────────┘
                                                       ▼  priors / values / beliefs
                                       ┌──────────────────────────────────────┐
                                       │   REPLAY BUFFER (Reverb)              │
                                       └───────────────┬──────────────────────┘
                                                       ▼
                                       ┌──────────────────────────────────────┐
                                       │   LEARNER (PyTorch/JAX) ──► weights ──►│ inference + actors
                                       └──────────────────────────────────────┘
```

This is the **Sebulba / Podracer / EnvPool** topology. The only surviving "bridge" is **batched leaf serialization to the inference server** — coarse-grained (one message per batch of leaves), not per-transition.

### 4.3 If you must keep Node in the loop (fallback)

Run a **pool of Node worker processes** hosting the headless kernel behind the engine's own `in-memory-transport.ts`, exchanging **Mutative `Patch[]`** (already emitted by `executeCommand`) plus `FilteredMatchView`s over shared memory / Unix-domain sockets — never JSON in the hot path. Crucially, **vectorize**: Python sends a *batch* of `(stateRef, command)` and gets a *batch* of patch-sets back, amortizing the crossing. Slower than WASM-in-process and harder to scale across cores, but avoids the port. Use gRPC/`websocket-transport.ts` only for control-plane messages (init game, snapshot, fetch log), never per-node transitions.

### 4.4 Wire format

- **Information set / action / mask:** FlatBuffers schema (zero-copy reads) generated from `@tcg/lorcana-types`. Action = typed union mirroring `AvailableMoveId` (`PlayCard{cardRef, costType, targets[]}`, `SingCard`, `ShiftCard`, `PutCardIntoInkwell`, `Quest{charRef}`, `Challenge{attacker, defender}`, `MoveCharacterToLocation`, `ActivateAbility{cardRef, abilityIndex, targets[]}`, `PassTurn`, `Concede`). Legal mask as a bitset over the factored head (§2.5). After the initial snapshot, transmit **patches**, not full views.
- **Determinism:** pass `ctx.random.seed` explicitly; a determinized world is then exactly reproducible and the learner can replay any logged game via `RuntimeSnapshot` + the move log.

### 4.5 Conformance harness (critical, often skipped)

If you compile or port the kernel, **continuously prove it matches the source engine.** Run **differential testing** against the in-repo `e2e/` and `specs/` suites (the repo already has per-keyword e2e: shift, evasive, ward, bodyguard, support, challenger-resist, reckless-rush, vanish, boost) plus random legal action sequences: feed both the original TS engine and your WASM kernel identical command streams and assert identical resulting `MatchState` **and** identical `getAvailableMoves` output at every step. Pin the engine commit and the `@tcg/lorcana-cards` catalog version together (Lorcana rules are versioned — e.g. v2.0.1, Feb 2026), and re-run conformance on every set release. A silent divergence here teaches the agent illegal or wrong lines.

---

## 5. Training Regimen for Long-Term Strategy

### 5.1 Curriculum & opponent generation (preventing policy collapse)

Plain self-play against the latest agent **collapses** in a non-transitive metagame (Lorcana has rock-paper-scissors archetype dynamics: aggro beats ramp, ramp beats control, control beats aggro). You must train against a *population*, not a mirror.

**Stage 0 — Bootstrap.** Two free sources of a competent prior: (a) behavior-clone from human game logs (Lorcanito/duels.ink logs are abundant); (b) **distill the engine's shipped `automation/` strategies** — `bestDeckAwareOracleLoreRaceAutomatedActionStrategy` and friends are far above random and make ideal initial teachers and permanent **league anchors**. Either skips the brutal random-play phase and gives the value head a sane starting signal. Use `BEST_AI_DECK_DOSSIERS` as the initial deck distribution.

**Stage 1 — League / PSRO self-play** (AlphaStar-style league):
- **Main agents** — optimize win-rate against the whole league.
- **Main exploiters** — train *only* to beat the current main agents; they surface and punish exploitable habits, then their lessons are folded back.
- **League exploiters** — train to beat the entire historical population; they prevent forgetting and cyclic regression.
- Opponents are sampled by **prioritized fictitious self-play** (PFSP): weight opponents by how informative/hard they are, not uniformly.

**Anti-collapse instrumentation (all on):**
- **Entropy bonus** in the policy loss (keep `π` from collapsing to a single line).
- **KL trust region / proximal update** to the previous policy (no catastrophic forgetting).
- **NeuRD / regret-style regularization** on the policy update — in imperfect-information games, naïve policy-gradient self-play cycles; regret-based updates have convergence pressure toward equilibrium.
- **Deck diversity:** train across a *distribution of decklists spanning the metagame* (condition the net on its own decklist as input). A champion bot must be deck-robust, and metagame coverage is itself an anti-collapse force.
- **Exploitability tracking as the north-star eval metric:** periodically compute a **local best response (LBR)** / approximate best response against the frozen agent. Win-rate-vs-self can rise while exploitability also rises — only LBR/exploitability tells you you're actually getting harder to beat. (OpenSpiel provides these tools.)

### 5.2 The self-play data → learner loop

AlphaZero-style policy improvement, adapted for infosets:
- Actors play full games with B-ISMCTS; at each decision they log `(I, π_MCTS, eventual_outcome z, true_hidden_world)`.
- `π_MCTS` is the **pooled visit-count distribution** at the root infoset — the search-improved policy that the network policy head is trained to imitate.
- `z` is the terminal result (see §5.4 for the target design).
- `true_hidden_world` is **known in self-play** — free, perfect supervision for the belief head.

### 5.3 Loss functions

```
L(θ) =      L_value
     + c1 · L_policy        (cross-entropy to MCTS visit distribution, legal-masked)
     + c2 · L_belief        (supervised belief vs. revealed true hidden world)
     + c3 · ||θ||²          (L2)
     − c4 · H(p)            (entropy bonus, anti-collapse)
```

- **Policy:** `L_policy = − Σ_a π_MCTS(a|I) · log p_θ(a|I)` over legal actions only.
- **Value:** `L_value = CE( z , v_θ(I) )` with a **categorical / two-hot distributional value** (see §5.4), not plain MSE — variance from top-decks makes a distribution far better-calibrated than a point estimate.
- **Belief:** `L_belief = CE( true_world , b_θ(·|I) )`, count-consistency–projected. This is the supervision that makes the determinization proposals actually accurate.
- For the **ReBeL/PBS leaf value**, train a separate value target on public-belief states using the subgame-solve outputs as bootstrapped targets.

### 5.4 Training the value head for *equity*, not myopic lore

This is the subtlety the question rightly fixates on. A value head trained only to predict "current lore difference" will play greedily and lose to setup decks.

**Do these:**

1. **Train on the game outcome `z`, not on lore.** Primary target is win/loss (`+1 / −1`), optionally a small margin term. Because `z` is the *eventual* result, the head is forced to learn that a lore-behind-but-inevitable board is high-value. This is the single most important choice for non-myopia.
2. **Bootstrap with the search root value (AlphaZero target).** Use the MCTS-improved root value as an auxiliary value target — it's lower-variance than raw `z` and propagates lookahead into the head.
3. **Distributional value (C51 / two-hot).** Predict a distribution over outcome, not a scalar. Top-decks and singer/removal swings create genuinely bimodal futures; a distribution captures "70% inevitable win / 30% they top-deck the answer" far better than a mean, and downstream the search can be risk-aware.
4. **TD(λ) / n-step returns** to blend bootstrap and Monte-Carlo, trading bias for variance along trajectories.
5. **Auxiliary equity heads (representation shaping, not the objective):** predict future-lore-differential at horizon `h`, predict whether you'll have board control in 2 turns, predict opponent archetype. These auxiliary targets enrich the trunk and improve sample efficiency — but the *decision* value remains `P(win)`. Keep them as side heads with small weights so they never turn the agent greedy.

> Calibration check in eval: bucket value-head outputs and verify realized win-rate matches predicted win-probability (reliability diagram). A champion value head must be *calibrated*, not just accurate on average — the search trusts it as a leaf oracle.

---

## 6. Cross-Cutting Concerns & Build Phasing

**Deck-conditioning.** Feed the agent's own 60-card decklist as a set-encoded input. A single set of weights then plays *any* deck, and you can train across the metagame with one model — essential for both strength and anti-collapse.

**Evaluation suite.** (1) Win-rate vs. a fixed gauntlet of scripted + frozen-checkpoint opponents across archetypes; (2) **LBR/exploitability** (the real metric); (3) value-head calibration; (4) head-to-head ladder vs. prior best (gate promotions on Elo + non-regression vs. exploiters); (5) human expert review of closing-turn lines (myopia detector).

**Recommended libraries / tooling**

| Layer | Recommendation |
|---|---|
| Algorithm reference + exploitability/LBR + ISMCTS/CFR baselines | **OpenSpiel** (prototype the whole algorithm here first) |
| Batched MCTS on accelerators | **`mctx`** (DeepMind, JAX) — purpose-built for exactly this |
| NN framework | **PyTorch** (fastest to iterate) or **JAX + Flax/Haiku** (best for TPU-pod-scale self-play with `mctx`) |
| Replay buffer | **Reverb** |
| Distributed orchestration | **Launchpad**, or Ray for a more general cluster |
| Batched inference serving | **NVIDIA Triton** (dynamic batching) or a custom asyncio batcher |
| Engine embedding | Extract the `@tcg/core` kernel; **`wasmtime`/`wasmer`** to run it in-process; **QuickJS/Javy** (TS→WASM, faithful) with optional Rust/AssemblyScript hot-path port; fallback via the engine's `in-memory-transport.ts` |
| Wire format | **FlatBuffers** or **Cap'n Proto** (zero-copy; no JSON in hot path); transmit Mutative **`Patch[]`** after initial snapshot |
| Card data / rules pinning | `@tcg/lorcana-cards` catalog pinned to the engine commit (cross-check vs **LorcanaJSON / Lorcast**); re-run conformance per set release |

**Phased delivery**

1. **P0 — Headless kernel + bridge + conformance.** The rules engine, legal-move generation, fog-of-war projection, serialization, RNG seeding, and a scripted-AI baseline **already exist and are e2e-tested**. P0 is therefore *not* "write a simulator" — it is: extract the `@tcg/core` kernel from the SvelteKit app, compile it to WASM (or host via `in-memory-transport`), and stand up the differential-conformance harness (§4.5). De-risks everything downstream.
2. **P1 — Plain neural-ISMCTS + value/policy net.** No belief net yet (uniform determinization). Get the actor–inference–learner loop and behavior-cloning bootstrap working end-to-end.
3. **P2 — Belief net + importance-weighted determinization + Bayesian filtering.** This is the strength inflection point.
4. **P3 — League/PSRO self-play + exploitability gating.** Where it becomes hard to beat.
5. **P4 — ReBeL/PBS subgame solving at key decisions + distributional/auxiliary value heads.** Where it becomes champion-level and low-exploitability.

---

### Appendix A — One-line mental model

> Maintain a calibrated **belief over hidden worlds**, **sample the worlds that matter** (belief-weighted determinization), **search shallow but exact** over them with a neural prior, **terminate at a win-probability value head trained on outcomes** (so the plan is long-horizon), and **train against a league** so the policy is robust and hard to exploit. Everything else is plumbing in service of those five verbs.
