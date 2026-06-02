# Lorcana Determinism Audit

Audit deterministic behavior risks.

Do not implement.

Search for:
- host RNG
- rand::StdRng in engine path
- thread_rng
- Math.random
- unordered HashMap/HashSet iteration in observable logic
- process-global caches
- async runtime in lorcana-core
- timestamp-dependent state in conformance output
- debug/manual moves in production legal actions
- snapshot normalization that hides real drift

Output:
1. Blocking determinism risks.
2. Review-required risks.
3. Allowed test-only/fuzz-only uses.
4. Required tests.
