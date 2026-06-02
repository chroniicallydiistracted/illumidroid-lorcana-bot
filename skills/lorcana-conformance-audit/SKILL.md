# Lorcana Conformance Review

Audit conformance evidence for a port step.

Do not implement.

Required checks:
- Oracle source hash present.
- Card catalog hash present where relevant.
- Ruleset hash present where relevant.
- TypeScript oracle files inspected.
- Rust behavior compared against oracle for the layer.
- Comparison happens after every command for replay/differential tests.
- Snapshot normalization is explicitly documented.
- Missing fields, undefined/null shape, order, and hidden state are considered.
- Tests cover valid and invalid inputs.
- Tests run real runtime path, not only mocks.

Output:
1. Conformance verdict.
2. Evidence accepted.
3. Evidence rejected.
4. Missing comparisons.
5. Snapshot/replay risks.
6. Required next tests.
