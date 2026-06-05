# Lorcana Headless Port Status

| Step | Name | Status | Branch | PR | Oracle hash | Tests | Codex verdict | Notes |
|---:|---|---|---|---|---|---|---|---|
| 0 | Freeze TypeScript oracle and card catalog hash | done (entire oracle vendored into `oracle/source/`, byte-exact + hashed; RNG golden vectors added) | port/step-00-freeze-oracle | | ruleset `b15fd6d0…5f60aca4` @ git `89af4d5` | `bun test oracle/tools/freeze.test.ts` → 16 pass | | Full closure (types/engine/cards/shared/typescript-config + glue), 8889 files/~79MB; `oracle/golden/rng-golden-vectors.json` (seedrandom 3.0.5); strict `--vs-upstream` + local-path allowlist; replay/snapshot are format contracts owned by Steps 6/11/13/32/33 |
| 1 | Port `lorcana-types` schemas | done (approved after 3 review rounds; workspace + `lorcana-schema` crate; closed-set discriminator enums + lossless DSL nodes for card/ability/condition/effect/cost/target/keyword/deck schema) | port/step-01-types-schema | | ruleset `b15fd6d0…5f60aca4` @ git `89af4d5` | `cargo test -p lorcana-schema` → 48 pass (9 round-trip / 30 fail-closed / 6 discriminator-coverage / 3 oracle-corpus); full 2754-card catalog deserializes losslessly via `LORCANA_CARD_CORPUS` | REJECT (1st, 2nd, 3rd) → fixed → APPROVE | Created `lorcana-rs/` workspace (edition 2024, Rust 1.96). Recursive DSL nodes use a typed `type` discriminator + flattened order-preserving `IndexMap` rest (blueprint §5.2); full per-field typing deferred to consuming steps. Integers `i64` for byte-exact round-trip. **Audit fixes (2026-06-04):** (1) optional `field?: T` now rejects explicit `null` (`serde_util::optional_non_null`), `actionSubtype` keeps `"song"\|null`; (2) closed unions typed (`banishItem`→`BoolOrNumber`, `discardCardType`→`CardTypeOrSong`, `cardCopyLimit`→`CardCopyLimit`); (3) **parent-kind-aware** recursive nested-DSL discriminator validation in all tagged nodes + `Trigger.event` — covers the direct `Effect`/`Effect[]` containers (`steps`/`options`/`choices`/`then`/`else`/`ifTrue`/`ifFalse`) plus the nested-in-container fields `reveal-and-route.routes[*].condition`/`sideEffects` and `create-triggered-ability.ability.trigger`/`condition`/`effect`; union-typed `Effect\|"prevent"\|…` fields (`with`/`replacement`/`cost`) intentionally excluded to avoid over-rejection; de-risked against the full 2754-card catalog (0 rejected, 0 lossy); (4) bidirectional oracle↔Rust discriminator set-equality for all 39 closed categories; (5) `Cargo.lock` un-ignored (must be committed). Coverage test reads the frozen oracle. Deck-validation split: see Step 2 note. No gameplay logic; no TS oracle modified. |
| 2 | Port deck-validation rules | done (DAC format-validation algorithm + `LORCANA_FORMATS` ported in `lorcana-schema/src/deck_format.rs`); **residual gap**: no `DeckStats`/standard-rules validator (no oracle algorithm exists) | port/step-02-deck-validation | | ruleset `b15fd6d0…5f60aca4` @ git `89af4d5` | `cargo test -p lorcana-schema` → 84 pass (36 in `tests/deck_format.rs`); messages cross-checked byte-identical vs the TS oracle via bun differential | APPROVE | Ported `validateDeckForFormat`/`validateDeck`/`getDeckFormats` + `groupByCanonical`/`resolveMaxCopies` + `LORCANA_FORMATS` from `decks/validate-deck.ts`, mirroring `validate-deck.test.ts`. Injected `lookup` = `&dyn Fn(&str) -> Option<CardFormatData>`. Preserves rule order, format/failure/ink insertion order, verbatim messages, `"no-limit"`→unlimited, canonical-grouped copy limits, and unknown-lookup skip/fallback behavior. **Residual decision:** `cards/deck-validation.ts` declares `DeckStats` + standard-rules `DeckValidationError` shapes, but `lorcana-types` contains **no producing algorithm**, so none was invented (the DAC path is the only deck-validation algorithm in `lorcana-types`). Revisit if the oracle/engine adds one. No gameplay logic; no TS oracle modified. |
| 3 | Port static resource and card instance registry | not started | | | | | | |
| 4 | Port `MatchState`, `TCGCtx`, `LorcanaG`, and card metadata defaults | not started | | | | | | |
| 5 | Port zone configuration | not started | | | | | | |
| 6 | Port zone query API | not started | | | | | | |
| 7 | Port zone mutation API | not started | | | | | | |
| 8 | Port RNG and shuffle exactly | not started | | | | | | |
| 9 | Port card runtime query API | not started | | | | | | |
| 10 | Port framework read/write context | not started | | | | | | |
| 11 | Replace global static registry behavior with engine-owned cache hooks | not started | | | | | | |
| 12 | Port command envelope and validation skeleton | not started | | | | | | |
| 13 | Port command reducer | not started | | | | | | |
| 14 | Port game setup and board setup | not started | | | | | | |
| 15 | Port phase/flow config | not started | | | | | | |
| 16 | Exclude or gate debug/manual moves | not started | | | | | | |
| 17 | Port serialization and replay snapshot format | not started | | | | | | |
| 18 | Port primitive mutations and turn metrics | not started | | | | | | |
| 19 | Port condition evaluator | not started | | | | | | |
| 20 | Port static-effect registry | not started | | | | | | |
| 21 | Port derived-card projection | not started | | | | | | |
| 22 | Port targeting resolver, target analysis, target availability, and slotted targets | not started | | | | | | |
| 23 | Port cost helpers: ink, exert, Shift, Sing, Sing Together | not started | | | | | | |
| 24 | Port setup moves: choose first player and mulligan | not started | | | | | | |
| 25 | Port put-card-into-inkwell | not started | | | | | | |
| 26 | Port quest | not started | | | | | | |
| 27 | Port move-character-to-location | not started | | | | | | |
| 28 | Port challenge legality | not started | | | | | | |
| 29 | Port challenge damage and lethal sweep | not started | | | | | | |
| 30 | Port primitive action-effect resolver families | not started | | | | | | |
| 31 | Port pending effect suspension/resumption | not started | | | | | | |
| 32 | Port triggered ability capture | not started | | | | | | |
| 33 | Port Bag resolution | not started | | | | | | |
| 34 | Port replacement/prevention effects | not started | | | | | | |
| 35 | Port full `playCard` | not started | | | | | | |
| 36 | Port activated abilities | not started | | | | | | |
| 37 | Port pass-turn and turn transition | not started | | | | | | |
| 38 | Port delayed/floating effects | not started | | | | | | |
| 39 | Port win/loss conditions | not started | | | | | | |
| 40 | Integrate card catalog through generated/normalized IR | not started | | | | | | |
| 41 | Hand-port card helpers that cannot be represented in generic IR | not started | | | | | | |
| 42 | Build production legal-action generator | not started | | | | | | |
| 43 | Build player observation API for hidden-information ML | not started | | | | | | |
| 44 | Build deterministic replay runner | not started | | | | | | |
| 45 | Build TypeScript-vs-port differential harness | not started | | | | | | |
| 46 | Run full unit, card, replay, legal-action, and full-game simulation parity tests | not started | | | | | | |
| 47 | Redesign internals for performance only after parity | blocked until parity | | | | | | |
