# Frozen TypeScript Oracle (blueprint Step 0)

This directory freezes the **entire** TypeScript Lorcanito / TheCardGoats
Lorcana GameEngine as the behavioral oracle for the Rust port. The complete
oracle dependency closure is vendored byte-for-byte into **`oracle/source/`**,
so the oracle of record is self-contained and tamper-evident, independent of the
mutable `lorcana-simulator/` working tree.

Every parity and performance claim later in the port MUST cite the `ruleset-hash`
recorded here (blueprint §0, §10). A claim with no oracle hash is not trusted.

## What is frozen (`oracle/source/`)

The full oracle dependency closure, vendored as whole package directories — no
file sub-selection:

| Package | Why included |
|---|---|
| `packages/lorcana/lorcana-types` (`@tcg/lorcana-types`) | Card/ability/condition/cost/effect/target schemas |
| `packages/lorcana/lorcana-engine` (`@tcg/lorcana-engine`) | Runtime, rules, moves, effects, triggers, serialization |
| `packages/lorcana/lorcana-cards` (`@tcg/lorcana-cards`) | The card catalog + generated card data |
| `packages/shared` (`@tcg/shared`) | Transitive dep of the cards package |
| `packages/typescript-config` (`@tcg/typescript-config`) | tsconfig base extended by `shared` |

Plus workspace glue: `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`,
`.npmrc`.

**Excluded** (non-source, install-state, or regenerable only — never the
oracle): `node_modules`, `dist`, `.turbo`, `.nx`, `coverage`, `.cache`,
`*.tsbuildinfo`, `*.xml`. Symlinks are skipped. Current freeze: **8889 files**
(643 source + 8246 card catalog), ~79 MB.

`oracle/source/` is generated output. **Do not hand-edit it** — regenerate via
the writer below.

## Artifacts

| File | Meaning |
|---|---|
| `source-hash.txt` | sha256 over the non-card partition of `oracle/source/`. |
| `card-catalog-hash.txt` | sha256 over the card catalog (`…/lorcana-cards/`). |
| `ruleset-hash.txt` | Content fingerprint of the whole frozen oracle (see below). |
| `freeze-manifest.json` | Full detail + provenance (git commit, resolved/declared deps, counts, closure, excludes, hashing spec). |
| `file-hashes/{source,cards}.sha256` | Per-file `sha256sum` manifests; pinpoint any drifted file. |
| `golden/rng-golden-vectors.json` | Frozen-oracle RNG/shuffle golden vectors (Step 0 export; targets for the Rust RNG at Step 6). |
| `replay-corpus/` | Deterministic command-stream parity fixtures (format contract; content produced by Steps 32–33). |
| `snapshot-schema/` | Canonical state/snapshot schema fixtures (format contract; content produced by Steps 11 & 13). |

## Hashing algorithm (so a future Rust `lorcana-cli oracle-freeze` reproduces it)

1. Enumerate every regular file under `oracle/source/` (symlinks ignored).
2. `sha256` each file's **raw bytes** (no normalization).
3. Partition: card catalog = paths under `packages/lorcana/lorcana-cards/`;
   source = everything else.
4. Per-partition manifest: lines `"<sha256>  <relpath>"`, POSIX-normalized
   paths, **sorted ascending by path**, joined with `\n` + trailing `\n`.
5. `source-hash = sha256(sourceManifest)`, `card-catalog-hash = sha256(cardManifest)`.
6. `ruleset-hash = sha256(canonicalJson({ formatVersion, sourceHash,
   cardCatalogHash, deps }))`, keys sorted recursively. `deps` are the
   **resolved (locked)** `seedrandom` / `@types/seedrandom` versions.

The `ruleset-hash` is a pure content fingerprint: it excludes wall-clock time
and git commit, so identical frozen bytes always yield an identical id. The git
commit is provenance only, in `freeze-manifest.json`.

## Determinism dependency frozen here

The oracle RNG is `seedrandom@3.0.5` used as
`seedrandom(`${seed}:${draws}`)()` with `draws` incremented before each call
(`…/lorcana-engine/src/core/runtime/match-runtime.random-apis.ts`). The Rust
port must reproduce this exactly (blueprint §4.3, §13.5); it must NOT use
`rand::StdRng`. The freeze tests fail closed if this pin changes.

## RNG/shuffle golden vectors

`golden/rng-golden-vectors.json` exports the exact outputs of the frozen oracle
RNG so the Rust seedrandom-compatible RNG (Step 6) can be proven bit-for-bit.
It is generated from the pinned, installed `seedrandom@3.0.5` (imported by
explicit path; generation fails closed if the installed version differs) and
records the `ruleset-hash` it was produced against. Semantics it locks:

- `random()`: the draw counter is incremented **before** computing; the value is
  `seedrandom(`${seed}:${draws}`)()`.
- `shuffle()`: Fisher–Yates, `j = floor(random()*(i+1))` for `i=len-1..1`;
  consumes `len-1` draws.

The freeze tests regenerate the vectors and assert byte-equality, so they cannot
silently drift.

## Vendored local-path exceptions (allowlist)

Two vendored upstream docs contain absolute `/Users/...` developer-machine path
strings from the original oracle authors. They are part of the byte-exact freeze
and are **not edited**:

- `oracle/source/packages/lorcana/lorcana-cards/CARD_MIGRATION_LOG.md`
- `oracle/source/packages/lorcana/lorcana-engine/docs/choice-effects-implementation.md`

The freeze tests scan the entire frozen copy for local-path markers
(`/Users/`, `/home/`, `C:\Users\`) and assert the set of matching files equals
exactly this allowlist (`LOCAL_PATH_ALLOWLIST` in `freeze-config.ts`). Any new
local path appearing in `oracle/source/` fails closed.

## Commands

Run from the repo root (requires `bun` + `rsync`):

```bash
bun run oracle/tools/freeze-oracle.ts                # (re)vendor + (re)hash + regen golden vectors
bun run oracle/tools/gen-rng-golden.ts               # regen only the RNG golden vectors
bun run oracle/tools/verify-freeze.ts                # fail-closed integrity of frozen copy
bun run oracle/tools/verify-freeze.ts --vs-upstream  # FAIL CLOSED on any frozen-vs-working-tree drift
bun run oracle/tools/verify-freeze.ts --upstream-report  # informational drift report (exit 0)
bun test oracle/tools/freeze.test.ts                 # faithful/complete/reproducible/fail-closed
```

`verify-freeze.ts` exits non-zero if `oracle/source/` no longer matches the
recorded hashes. `--vs-upstream` additionally **fails closed** on any
missing/extra/changed file between the frozen copy and the working tree; use
`--upstream-report` for a non-failing informational diff. The frozen copy is the
reference of record. Re-freeze **only** on an intentional oracle change (which,
per `CLAUDE.md`, happens only when the user explicitly asks for an oracle-side
fix), then update parity claims to the new `ruleset-hash`.

The agent audit guards (`scripts/agent/check_forbidden_paths.sh`,
`check_dependency_step.py`) exempt `oracle/source/**` — the frozen copy is a
byte-exact reference, not port-authored code, and its integrity is policed by the
freeze tests/verifier above.

## Parity-corpus scope

`replay-corpus/` and `snapshot-schema/` carry format contracts now; their
behavioral content is produced by the later steps that own the exporters (RNG /
shuffle golden vectors → Step 6; initial-state / serialization snapshots →
Steps 11 & 13; random legal-action replay streams → Steps 32–33). Each populated
fixture must record the `ruleset-hash` it was generated against.
