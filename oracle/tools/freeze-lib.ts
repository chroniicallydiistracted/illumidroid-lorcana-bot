/**
 * Oracle freeze — pure hashing library.
 *
 * Blueprint Step 0 ("Freeze the TypeScript oracle"). The freeze vendors the
 * ENTIRE TypeScript oracle (its full dependency closure) into `oracle/source/`
 * and hashes that frozen copy. This module is pure: it partitions and hashes an
 * already-collected file list. Filesystem walking + exclude policy live in
 * `freeze-config.ts` so this stays trivially testable and side-effect free.
 *
 * It does NOT import or execute any oracle gameplay code.
 *
 * Hashing algorithm (documented so a future Rust `lorcana-cli oracle-freeze`
 * reproduces it bit-for-bit):
 *
 *   1. Collect every regular file under the frozen oracle (symlinks ignored),
 *      as { path, sha256(rawBytes) } with POSIX-normalized paths relative to
 *      `oracle/source/`.
 *   2. Partition into the card catalog (path under
 *      `packages/lorcana/lorcana-cards/`) and everything else ("source").
 *   3. Per-partition manifest: lines "<sha256>  <relpath>", sorted ascending by
 *      path, joined by "\n" with a trailing "\n" (sha256sum-compatible).
 *   4. source-hash = sha256(sourceManifest); card-catalog-hash = sha256(cardManifest).
 *   5. ruleset-hash = sha256(canonicalJson({ formatVersion, sourceHash,
 *      cardCatalogHash, deps })). Pure content fingerprint: excludes wall-clock
 *      and git commit so identical frozen bytes always yield an identical id.
 */

import { createHash } from "node:crypto";

/** Bumped to 2.x: the freeze now vendors the entire oracle closure. */
export const FREEZE_FORMAT_VERSION = "2.0.0";

/** Files under this prefix (relative to oracle/source/) are the card catalog. */
export const CARDS_PATH_PREFIX = "packages/lorcana/lorcana-cards/";

export interface FileHash {
  /** POSIX-normalized path relative to the frozen oracle root. */
  path: string;
  sha256: string;
}

export interface DepVersions {
  seedrandom: string;
  "@types/seedrandom": string;
}

export function sha256Hex(data: Buffer | string): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Deterministic JSON with recursively sorted object keys. */
export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortKeys(value));
}

function sortKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v as Record<string, unknown>).sort()) {
      out[k] = sortKeys((v as Record<string, unknown>)[k]);
    }
    return out;
  }
  return v;
}

/** sha256sum-compatible manifest text for an already-sorted file list. */
export function manifestText(files: FileHash[]): string {
  return `${files.map((f) => `${f.sha256}  ${f.path}`).join("\n")}\n`;
}

export function sortFiles(files: FileHash[]): FileHash[] {
  return [...files].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

export function computeRulesetHash(input: {
  sourceHash: string;
  cardCatalogHash: string;
  deps: DepVersions;
}): string {
  return sha256Hex(
    canonicalJson({
      formatVersion: FREEZE_FORMAT_VERSION,
      sourceHash: input.sourceHash,
      cardCatalogHash: input.cardCatalogHash,
      deps: input.deps,
    }),
  );
}

export interface OracleFreeze {
  formatVersion: string;
  /** All frozen files, sorted by path. */
  files: FileHash[];
  /** Card-catalog partition, sorted. */
  cardFiles: FileHash[];
  /** Everything-else partition, sorted. */
  sourceFiles: FileHash[];
  sourceManifest: string;
  cardManifest: string;
  sourceHash: string;
  cardCatalogHash: string;
  rulesetHash: string;
}

/**
 * Partition + hash the vendored oracle file list. Fails closed if the list is
 * empty or either partition is empty (an incomplete freeze must never produce a
 * silently "valid" hash).
 */
export function hashOracleFiles(files: FileHash[], deps: DepVersions): OracleFreeze {
  if (files.length === 0) {
    throw new Error("oracle freeze: no files (refusing empty hash)");
  }
  const sorted = sortFiles(files);
  const cardFiles = sorted.filter((f) => f.path.startsWith(CARDS_PATH_PREFIX));
  const sourceFiles = sorted.filter((f) => !f.path.startsWith(CARDS_PATH_PREFIX));
  if (cardFiles.length === 0) {
    throw new Error(`oracle freeze: no card catalog files under ${CARDS_PATH_PREFIX}`);
  }
  if (sourceFiles.length === 0) {
    throw new Error("oracle freeze: no engine/types/source files outside the card catalog");
  }

  const sourceManifest = manifestText(sourceFiles);
  const cardManifest = manifestText(cardFiles);
  const sourceHash = sha256Hex(sourceManifest);
  const cardCatalogHash = sha256Hex(cardManifest);
  const rulesetHash = computeRulesetHash({ sourceHash, cardCatalogHash, deps });

  return {
    formatVersion: FREEZE_FORMAT_VERSION,
    files: sorted,
    cardFiles,
    sourceFiles,
    sourceManifest,
    cardManifest,
    sourceHash,
    cardCatalogHash,
    rulesetHash,
  };
}

export interface TreeDiff {
  /** In `a` but not `b`. */
  missing: string[];
  /** In `b` but not `a`. */
  extra: string[];
  /** Present in both but content differs. */
  changed: string[];
}

/** Compare two file lists by path + content hash (used for faithful-copy checks). */
export function diffFiles(a: FileHash[], b: FileHash[]): TreeDiff {
  const ma = new Map(a.map((f) => [f.path, f.sha256]));
  const mb = new Map(b.map((f) => [f.path, f.sha256]));
  const missing: string[] = [];
  const changed: string[] = [];
  for (const [p, h] of ma) {
    if (!mb.has(p)) missing.push(p);
    else if (mb.get(p) !== h) changed.push(p);
  }
  const extra: string[] = [];
  for (const p of mb.keys()) if (!ma.has(p)) extra.push(p);
  return { missing: missing.sort(), extra: extra.sort(), changed: changed.sort() };
}

export function diffIsEmpty(d: TreeDiff): boolean {
  return d.missing.length === 0 && d.extra.length === 0 && d.changed.length === 0;
}

/**
 * Fail-closed drift assertion: throws if `frozen` and `current` differ at all.
 * `aLabel`/`bLabel` name the two sides in the error (e.g. "frozen" vs "working
 * tree"). Used by `verify-freeze --vs-upstream` and tested directly.
 */
export function assertNoDrift(
  frozen: FileHash[],
  current: FileHash[],
  aLabel = "frozen",
  bLabel = "current",
): void {
  const d = diffFiles(frozen, current);
  if (diffIsEmpty(d)) return;
  const sample = (xs: string[]) => xs.slice(0, 10).map((p) => `    ${p}`).join("\n");
  const parts: string[] = [
    `oracle drift detected (${aLabel} vs ${bLabel}): ` +
      `missing=${d.missing.length} extra=${d.extra.length} changed=${d.changed.length}`,
  ];
  if (d.changed.length) parts.push(`  changed:\n${sample(d.changed)}`);
  if (d.missing.length) parts.push(`  only-in-${aLabel}:\n${sample(d.missing)}`);
  if (d.extra.length) parts.push(`  only-in-${bLabel}:\n${sample(d.extra)}`);
  throw new Error(parts.join("\n"));
}
