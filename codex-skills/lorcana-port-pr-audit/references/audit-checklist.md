# Audit Checklist

## Blockers

- PR does not identify blueprint step.
- PR touches downstream systems early.
- PR changes TypeScript oracle behavior without explicit authorization.
- PR changes symbols but not symbol registry.
- PR uses host RNG in deterministic engine path.
- PR uses HashMap/HashSet iteration for observable rules behavior.
- PR introduces process-global rule cache.
- PR exposes hidden information in player observation.
- PR includes debug/manual moves in production legal actions.
- PR claims parity without oracle hash.
- PR lacks fail-closed invalid-input tests.
- PR lacks relevant conformance tests.

## Required output

- Approve / reject / human decision.
- Blocking findings.
- Non-blocking findings.
- Missing tests.
- Missing registry updates.
- Commands to run.
