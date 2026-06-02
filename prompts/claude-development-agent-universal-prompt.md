# Claude Development Agent Universal Prompt

You are working on the illumidroid-lorcana-bot headless GameEngine port.

Your job is not to redesign the game, improve the rules, simplify behavior, or optimize before correctness. Your job is to port the current Lorcanito / TheCardGoats TypeScript GameEngine behavior into the new Rust headless engine with 1:1 behavioral parity against the frozen TypeScript oracle.

Read these files before doing any implementation work:

1. AGENTS.md
2. CLAUDE.md, if present
3. docs/port/headless_lorcana_engine_porting_blueprint.md
4. docs/port/headless_lorcana_engine_porting_symbol_registry.md
5. Any nested AGENTS.md or CLAUDE.md in the package/crate you touch

Before editing implementation files, report:

1. Exact blueprint step.
2. TypeScript oracle files/functions being mirrored.
3. Current oracle behavior and invariants.
4. Rust files/functions/modules being created or changed.
5. Completed upstream dependencies this work relies on.
6. Downstream systems affected.
7. Conformance tests that will prove parity.
8. Negative/fail-closed tests that will catch unsafe behavior.
9. Symbol registry entries that must be added or updated.
10. Whether legacy/debug behavior must be removed, gated, or test-only.

Now perform only the requested implementation step. If the requested work violates dependency order, stop and produce a dependency-blocker report instead of implementing downstream behavior early.
