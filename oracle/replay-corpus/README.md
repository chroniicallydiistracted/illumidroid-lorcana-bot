# oracle/replay-corpus — format contract

Deterministic replay fixtures captured from the frozen TypeScript oracle, used
for lockstep differential validation of the Rust port (blueprint Steps 32–33).

**Status:** format contract only. Content is populated by the steps that own the
exporters (random legal-action streams → Step 32; full differential replay →
Step 33). Do not hand-author behavioral fixtures here before those steps.

## Required fixture shape

Each fixture is a directory or JSON file capturing one full game as a command
stream plus per-command authoritative snapshots:

```jsonc
{
  "rulesetHash": "<oracle/ruleset-hash.txt at capture time>",  // REQUIRED
  "seed": "<ctx.random.seed>",
  "config": { /* NewGameConfig: decks, player ids, time mode = "none" */ },
  "steps": [
    {
      "command": { /* exact command envelope sent to executeCommand */ },
      "validation": { /* validation result */ },
      "post": { /* canonical snapshot AFTER this command (see snapshot-schema) */ }
    }
  ],
  "outcome": { "winner": "...", "ended": true }
}
```

## Comparison contract (blueprint §6 Step 33, CLAUDE.md)

The differential harness must compare after **every** command, not only the
final winner:

```
state.G, state.ctx, zones, card metadata, deck order, discard order, Bag,
pending effects, replacement state, triggered state, priority, logs, events,
legal actions, winner/outcome
```

Any fixture whose `rulesetHash` does not match the current `oracle/ruleset-hash.txt`
is stale and must be regenerated, not silently accepted.
