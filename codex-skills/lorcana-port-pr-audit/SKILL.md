# Lorcana Port PR Audit

Use this skill to audit PRs for the headless Lorcana Rust port.

Do not implement. Do not patch. Audit only.

Read:
1. AGENTS.md
2. CLAUDE.md
3. docs/port/headless_lorcana_engine_porting_blueprint.md
4. docs/port/headless_lorcana_engine_porting_symbol_registry.md
5. PR diff
6. Claimed TypeScript oracle files

Audit for:
- wrong blueprint step
- dependency-order violation
- insufficient oracle inspection
- TypeScript oracle behavior changes
- missing conformance tests
- tests that only prove execution rather than parity
- invalid input not failing closed
- serialization/replay drift
- hidden information leakage
- debug/manual moves in production paths
- host RNG / rand::StdRng / Math.random in deterministic paths
- HashMap/HashSet iteration order in rules-visible behavior
- process-global rule caches
- missing symbol registry updates
- performance optimization before parity
- undocumented uncertainty

Output:
1. Verdict: approve / reject / needs human decision.
2. Blocking findings.
3. Non-blocking findings.
4. Missing tests.
5. Missing registry updates.
6. Commands that must be run.
7. Questions requiring human decision.
8. Exact files and lines where possible.
