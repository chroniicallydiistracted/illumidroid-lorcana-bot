# Lorcana Audit Integrity

Purpose: force skeptical review quality for Codex or any QA agent.

Use this skill before approving any implementation, test, conformance claim, benchmark, or dependency-step completion.

## Anti-rationalization rules

Reject these rationalizations:

| Rationalization | Required response |
|---|---|
| "Close enough to oracle" | Block. Parity must be exact or discrepancy documented. |
| "This test proves it runs" | Block. Tests must prove behavior. |
| "Performance requires this change" | Block unless relevant layer already has conformance. |
| "UI behavior suggests this" | Block. UI is not source of truth. |
| "HashMap order probably does not matter" | Block or require proof/order-independent test. |
| "Registry update is documentation only" | Block. Registry is required continuity state. |
| "Snapshot changed because implementation is better" | Block unless oracle intentionally changed. |

## Self-critique gate

Before verdict, answer:

1. What would make this PR unsafe despite passing tests?
2. Which oracle file proves the behavior?
3. Which test would fail if the implementation were wrong?
4. What state field/order could drift silently?
5. What changed symbol is missing from the registry?

If confidence is under 8/10, verdict cannot be approve.
