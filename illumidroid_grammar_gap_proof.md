# Action-Grammar Gap — Proof (measured, not assumed)

**Question:** should illumidroid replace the Lorcanito *automation candidate*
enumeration with the engine's *full move grammar* (`getAvailableMoves` /
`getMoveOptions`)? The premise was that the automation function lacks the full
grammar/features.

**Method:** instrumented the bridge (`grammar_probe` op + `tools/grammar_gap_probe.py`)
to enumerate **every real decision three ways** on the *same* state and read the
engine's own diagnostics:
- **capped automation** — `enumerateAutomatedActions({searchCaps: 24/48/16/32})` (what the bot uses)
- **uncapped automation** — same with effectively-infinite caps (Botcana-style full enumeration)
- **engine diagnostics** — `unsupported-shape` (a legal move the automation *cannot represent*),
  `overflow-skip` (combinations dropped by a cap), `validation-reject` (illegal anyway)

Driven by the deterministic fair `best` strategy over real tournament decks.

## Results (24 games, 2308 decisions)

| Question | Result |
|---|---|
| **Caps truncate the action space?** | **0.1%** — 3/2308 decisions (only complex `resolveBag` states; uncapped added ≤58 candidates). Negligible. |
| **Grammar gap (`unsupported-shape`)?** | **0.1%** — 3/2308 decisions, **100% of them**: `resolveBag: "Nested branching exceeds the v1 automation support matrix"`. |
| **Zero-candidate states** | 16.5% (381/2308). **All but 3 are genuine "pass" states** (only legal action is to pass). The 3 exceptions == the nested-bag gap above. |

(A smaller 12-game / 1110-decision run showed **0** of each — the gap is ~0.1%, so it needs a larger sample to observe.)

## Conclusion

1. **There is no broad grammar gap.** In **99.9%** of decisions the automation
   enumeration is byte-for-byte the full engine-legal action set — it never reports
   an unsupported move in any *real action* family (playCard/quest/challenge/ink/
   ability/move). This confirms Botcana2.0's own architecture doc (Finding 1:
   "Lorcanito already enumerates the full, engine-validated legal action space")
   and contradicts the premise. A wholesale `getAvailableMoves` rewrite would
   change nothing for 99.9% of play and is **not** justified.

2. **The caps are not the bottleneck.** illumidroid already caps *higher* than
   Botcana (24/48/16/32 vs the engine default 8/16/8/16) and they bind in 0.1% of
   decisions. Raising them is a 1-line option, not a project.

3. **There is ONE real, narrow gap: nested triggered-ability (bag/effect)
   resolution with branching.** ~0.1% of decisions enter a `resolveBag` state the
   automation's "v1 support matrix" can't enumerate → **zero candidates → the bot
   can't act → the stuck/loop game** you observed. This is the genuine residual
   (Botcana's doc acknowledges the same class).

## Recommendation

Do **not** do the full `getAvailableMoves` rewrite. Instead, the surgical fix
matched to the evidence:

- **Now (already in place):** the stuck-guard aborts+discards these rare games, so
  they no longer stall a run. (`game_fingerprint` + `stuck_step`.)
- **Targeted fix (the real work):** handle the **nested-branching resolveBag/
  resolveEffect** states via the engine's resolution API when the automation
  enumerates nothing — i.e., a hybrid only where it's actually needed (0.1%), not a
  wholesale replacement. This recovers those games instead of forfeiting them.
- **Optional:** raise `SEARCH_CAPS` (or drop to engine defaults to match Botcana) —
  immaterial either way per the data.

## Reproduce / regress

- Measure: `python -m tools.grammar_gap_probe [n_games]` (from `lorcana-bot/`).
- Regression test: `tests/test_bridge.py::test_automation_grammar_gap_is_only_known_residual`
  — fails if a NEW `unsupported-shape` appears in any real action family (the only
  situation that would justify the raw grammar).
