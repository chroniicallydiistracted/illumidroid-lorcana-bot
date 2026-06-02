@codex review

Audit this PR against AGENTS.md, CLAUDE.md, the blueprint, and the symbol registry.
Focus on parity drift, dependency-order violations, missing conformance tests,
missing registry updates, deterministic-order bugs, and any change to the TypeScript oracle.
Do not suggest style-only changes unless they hide correctness risk.

Return:
1. Verdict: approve / reject / needs human decision.
2. Blocking findings.
3. Non-blocking findings.
4. Missing tests.
5. Missing registry updates.
6. Required verification commands.
7. Human decisions required.
