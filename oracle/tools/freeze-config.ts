/**
 * Oracle freeze — shared paths, closure definition, exclude policy, file walking.
 *
 * Defines the ENTIRE TypeScript oracle closure that gets vendored into
 * `oracle/source/`, the (non-source) exclude policy, and a deterministic,
 * exclude-aware file walker used by the writer, verifier, and tests so they all
 * agree on exactly which files are frozen.
 *
 * No oracle gameplay code is imported here.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { type DepVersions, type FileHash, sha256Hex, sortFiles } from "./freeze-lib";

const here = dirname(fileURLToPath(import.meta.url)); // <repo>/oracle/tools
export const REPO_ROOT = join(here, "..", "..");
export const ORACLE_DIR = join(REPO_ROOT, "oracle");
/** The vendored, frozen, byte-exact copy of the entire TypeScript oracle. */
export const VENDOR_DIR = join(ORACLE_DIR, "source");
export const FILE_HASHES_DIR = join(ORACLE_DIR, "file-hashes");
export const SIM_ROOT = join(REPO_ROOT, "lorcana-simulator");

/**
 * The complete oracle dependency closure (relative to the workspace root),
 * vendored as full package directories — no file sub-selection.
 *
 *   lorcana-types    : @tcg/lorcana-types  (no workspace deps)
 *   lorcana-engine   : @tcg/lorcana-engine -> @tcg/lorcana-types
 *   lorcana-cards    : @tcg/lorcana-cards  -> engine, types, @tcg/shared
 *   shared           : @tcg/shared         -> @tcg/typescript-config
 *   typescript-config: @tcg/typescript-config (tsconfig base extended by shared)
 */
export const CLOSURE_DIRS = [
  "packages/lorcana/lorcana-types",
  "packages/lorcana/lorcana-engine",
  "packages/lorcana/lorcana-cards",
  "packages/shared",
  "packages/typescript-config",
] as const;

/** Workspace glue needed to identify/reproduce the oracle deterministically. */
export const CLOSURE_FILES = [
  "package.json",
  "pnpm-workspace.yaml",
  "pnpm-lock.yaml",
  ".npmrc",
] as const;

/** Non-source, regenerable, or install-state artifacts that are NOT the oracle. */
export const EXCLUDE_DIR_NAMES = new Set([
  "node_modules",
  "dist",
  ".turbo",
  ".nx",
  "coverage",
  ".cache",
]);
export const EXCLUDE_FILE_SUFFIXES = [".tsbuildinfo", ".xml"];

/** rsync exclude args mirroring the policy above (used by the writer). */
export const RSYNC_EXCLUDES = [
  ...[...EXCLUDE_DIR_NAMES].map((d) => `--exclude=${d}`),
  ...EXCLUDE_FILE_SUFFIXES.map((s) => `--exclude=*${s}`),
];

export const GOLDEN_DIR = join(ORACLE_DIR, "golden");

export const ARTIFACTS = {
  sourceHash: join(ORACLE_DIR, "source-hash.txt"),
  cardCatalogHash: join(ORACLE_DIR, "card-catalog-hash.txt"),
  rulesetHash: join(ORACLE_DIR, "ruleset-hash.txt"),
  manifest: join(ORACLE_DIR, "freeze-manifest.json"),
  fileHashes: {
    source: join(FILE_HASHES_DIR, "source.sha256"),
    cards: join(FILE_HASHES_DIR, "cards.sha256"),
  },
  rngGolden: join(GOLDEN_DIR, "rng-golden-vectors.json"),
} as const;

/**
 * Allowlisted upstream files that legitimately contain absolute `/Users/...`
 * developer-machine path strings (historical doc references from the original
 * oracle authors). They are part of the byte-exact freeze and must NOT be
 * edited. The freeze tests fail closed if any OTHER frozen file contains a
 * `/Users/` path, so no new local path can sneak into the vendored oracle.
 * Paths are relative to `oracle/source/`.
 */
export const LOCAL_PATH_ALLOWLIST: readonly string[] = [
  "packages/lorcana/lorcana-cards/CARD_MIGRATION_LOG.md",
  "packages/lorcana/lorcana-engine/docs/choice-effects-implementation.md",
];

/** Substrings that indicate a leaked absolute developer-machine path. */
export const LOCAL_PATH_MARKERS: readonly string[] = ["/Users/", "/home/", "C:\\Users\\"];

/**
 * Scan the vendored oracle for files containing any local-path marker. Returns
 * sorted relative paths. Skips files that aren't valid UTF-8 text (none of the
 * oracle's text/JSON sources are binary, but this stays robust).
 */
export function scanFrozenLocalPaths(baseDir: string = VENDOR_DIR): string[] {
  const hits: string[] = [];
  for (const f of listClosureFiles(baseDir)) {
    let text: string;
    try {
      text = readFileSync(join(baseDir, f.path), "utf8");
    } catch {
      continue;
    }
    if (LOCAL_PATH_MARKERS.some((m) => text.includes(m))) hits.push(f.path);
  }
  return hits.sort();
}

// Pinned seedrandom (the frozen oracle RNG). Imported by explicit path so RNG
// golden vectors are generated with the exact oracle version, never an
// auto-installed substitute.
export const EXPECTED_SEEDRANDOM_VERSION = "3.0.5";
export const SEEDRANDOM_MODULE = join(SIM_ROOT, "node_modules/seedrandom/index.js");
export const SEEDRANDOM_PKG_JSON = join(SIM_ROOT, "node_modules/seedrandom/package.json");

export function installedSeedrandomVersion(): string {
  return (JSON.parse(readFileSync(SEEDRANDOM_PKG_JSON, "utf8")) as { version: string }).version;
}

// Resolve deps / RNG pin from the frozen copy by default (source of record).
export const VENDOR_LOCK = join(VENDOR_DIR, "pnpm-lock.yaml");
export const VENDOR_ENGINE_PKG = join(VENDOR_DIR, "packages/lorcana/lorcana-engine/package.json");
export const UPSTREAM_LOCK = join(SIM_ROOT, "pnpm-lock.yaml");

function isExcludedFile(name: string): boolean {
  return EXCLUDE_FILE_SUFFIXES.some((s) => name.endsWith(s));
}

/**
 * Recursively collect regular files under `dir`, applying the exclude policy.
 * Symlinks are ignored (deterministic, never follows into excluded trees).
 */
function collectFiltered(dir: string): string[] {
  const out: string[] = [];
  const stack = [dir];
  while (stack.length > 0) {
    const cur = stack.pop() as string;
    for (const e of readdirSync(cur, { withFileTypes: true })) {
      if (e.isSymbolicLink()) continue;
      const full = join(cur, e.name);
      if (e.isDirectory()) {
        if (!EXCLUDE_DIR_NAMES.has(e.name)) stack.push(full);
      } else if (e.isFile() && !isExcludedFile(e.name)) {
        out.push(full);
      }
    }
  }
  return out;
}

/**
 * List the oracle closure under `baseDir` as { path, sha256 } with paths
 * relative to `baseDir`. Used for both the vendored copy (baseDir = VENDOR_DIR)
 * and the live workspace (baseDir = SIM_ROOT) so faithful-copy comparison is
 * apples-to-apples. Fails closed if any closure directory or file is missing.
 */
export function listClosureFiles(baseDir: string): FileHash[] {
  const abs: string[] = [];
  for (const d of CLOSURE_DIRS) {
    const full = join(baseDir, d);
    if (!existsSync(full) || !statSync(full).isDirectory()) {
      throw new Error(`oracle freeze: missing closure directory: ${full}`);
    }
    abs.push(...collectFiltered(full));
  }
  for (const f of CLOSURE_FILES) {
    const full = join(baseDir, f);
    if (!existsSync(full) || !statSync(full).isFile()) {
      throw new Error(`oracle freeze: missing closure file: ${full}`);
    }
    abs.push(full);
  }
  const files = abs.map((a) => ({
    path: relative(baseDir, a).split(sep).join("/"),
    sha256: sha256Hex(readFileSync(a)),
  }));
  return sortFiles(files);
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
}

/** Resolve the single locked version of a dependency from a pnpm-lock.yaml. */
export function resolveLockedVersion(name: string, lockPath: string = VENDOR_LOCK): string {
  const lock = readFileSync(lockPath, "utf8");
  const re = new RegExp(`^\\s+'?${escapeRegExp(name)}@([0-9][A-Za-z0-9.\\-]*)'?:`, "gm");
  const versions = new Set<string>();
  for (const m of lock.matchAll(re)) versions.add(m[1]);
  if (versions.size === 0) {
    throw new Error(`oracle freeze: dependency not found in ${lockPath}: ${name}`);
  }
  if (versions.size > 1) {
    throw new Error(
      `oracle freeze: dependency '${name}' resolves to multiple versions ` +
        `[${[...versions].sort().join(", ")}]; determinism requires exactly one`,
    );
  }
  return [...versions][0];
}

export function declaredRange(
  field: "dependencies" | "devDependencies",
  name: string,
  pkgPath: string = VENDOR_ENGINE_PKG,
): string {
  const pkg = JSON.parse(readFileSync(pkgPath, "utf8")) as Record<string, Record<string, string>>;
  const range = pkg[field]?.[name];
  if (!range) throw new Error(`oracle freeze: ${name} not declared in engine ${field} (${pkgPath})`);
  return range;
}

export function resolveDeps(lockPath: string = VENDOR_LOCK): DepVersions {
  return {
    seedrandom: resolveLockedVersion("seedrandom", lockPath),
    "@types/seedrandom": resolveLockedVersion("@types/seedrandom", lockPath),
  };
}

export function currentGitCommit(): string {
  return execFileSync("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT }).toString().trim();
}
