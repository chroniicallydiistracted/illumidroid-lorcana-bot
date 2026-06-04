# oracle/snapshot-schema — format contract

Canonical serialized-state schema fixtures exported from the frozen TypeScript
oracle, used to lock the JSON shape the Rust port must reproduce (blueprint
Steps 11 & 13).

**Status:** format contract only. Content is populated by the steps that own the
serialization exporters (initial-state snapshots → Step 11; full
serialization/replay round-trip schema → Step 13). Do not author snapshots by
hand before those steps.

## Oracle source of the schema

```
lorcana-engine/src/core/runtime/serialization.ts   (SerializedMatchState, RuntimeSnapshot)
lorcana-engine/src/serialization.ts                 (LorcanaServerAuthoritativeSnapshot, undo stack)
lorcana-engine/src/core/runtime/types.ts            (MatchState { G, ctx })
lorcana-engine/src/types/runtime-state.ts           (LorcanaG, LorcanaCardMeta)
```

## Required fixture shape

```jsonc
{
  "rulesetHash": "<oracle/ruleset-hash.txt at capture time>",  // REQUIRED
  "label": "human-readable scenario id",
  "snapshot": { /* exact oracle serialized state, byte-stable JSON */ }
}
```

## Parity rules (blueprint §4.1, §5.2)

- Preserve the exact oracle JSON shape first, including explicit `undefined` /
  missing-field semantics in `LorcanaCardMeta` (`createDefaultCardMeta`).
  Normalize only after parity is proven, never before.
- Snapshots must be byte-stable: a fixed input game produces an identical
  snapshot. Insertion order is observable and must be preserved.
- A snapshot whose `rulesetHash` differs from the current
  `oracle/ruleset-hash.txt` is stale and must be regenerated.
