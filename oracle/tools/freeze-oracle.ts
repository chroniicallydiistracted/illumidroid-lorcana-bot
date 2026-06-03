/**
 * Oracle freeze — writer CLI.
 *
 * Blueprint Step 0. Vendors the ENTIRE TypeScript oracle (its full dependency
 * closure) into `oracle/source/` as a byte-exact, self-contained frozen copy,
 * then hashes that copy and writes the recorded artifacts. Run from repo root:
 *
 *   bun run oracle/tools/freeze-oracle.ts
 *
 * Reads oracle files only; never executes or modifies oracle gameplay code.
 * `oracle/source/` is generated output — do not hand-edit it.
 */

import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  ARTIFACTS,
  CLOSURE_DIRS,
  CLOSURE_FILES,
  FILE_HASHES_DIR,
  ORACLE_DIR,
  RSYNC_EXCLUDES,
  SIM_ROOT,
  VENDOR_DIR,
  currentGitCommit,
  declaredRange,
  listClosureFiles,
  resolveDeps,
} from "./freeze-config";
import { hashOracleFiles } from "./freeze-lib";
import { generateRngGolden } from "./gen-rng-golden";

function write(path: string, text: string): void {
  writeFileSync(path, text);
  console.log(`  wrote ${path}`);
}

function vendorOracle(): void {
  console.log(`vendoring entire oracle closure into ${VENDOR_DIR} ...`);
  rmSync(VENDOR_DIR, { recursive: true, force: true });
  mkdirSync(VENDOR_DIR, { recursive: true });

  // Copy full package directories with the non-source exclude policy. `-rpt`
  // (no -l) copies regular files + dirs and skips symlinks, matching the
  // hashing policy. `-R` recreates the workspace-relative path under VENDOR_DIR.
  for (const d of CLOSURE_DIRS) {
    execFileSync("rsync", ["-rptR", ...RSYNC_EXCLUDES, d, `${VENDOR_DIR}/`], {
      cwd: SIM_ROOT,
      stdio: "inherit",
    });
  }
  // Copy workspace glue files verbatim.
  for (const f of CLOSURE_FILES) {
    const dest = join(VENDOR_DIR, f);
    mkdirSync(dirname(dest), { recursive: true });
    cpSync(join(SIM_ROOT, f), dest);
  }
}

async function main(): Promise<void> {
  vendorOracle();

  const files = listClosureFiles(VENDOR_DIR);
  const deps = resolveDeps(); // from the frozen lockfile
  const gitCommit = currentGitCommit();
  const freeze = hashOracleFiles(files, deps);

  rmSync(FILE_HASHES_DIR, { recursive: true, force: true });
  mkdirSync(FILE_HASHES_DIR, { recursive: true });
  mkdirSync(`${ORACLE_DIR}/replay-corpus`, { recursive: true });
  mkdirSync(`${ORACLE_DIR}/snapshot-schema`, { recursive: true });

  write(ARTIFACTS.sourceHash, `${freeze.sourceHash}\n`);
  write(ARTIFACTS.cardCatalogHash, `${freeze.cardCatalogHash}\n`);
  write(ARTIFACTS.rulesetHash, `${freeze.rulesetHash}\n`);
  write(ARTIFACTS.fileHashes.source, freeze.sourceManifest);
  write(ARTIFACTS.fileHashes.cards, freeze.cardManifest);

  const manifest = {
    formatVersion: freeze.formatVersion,
    description:
      "Frozen byte-exact copy of the ENTIRE TypeScript Lorcana oracle closure " +
      "(blueprint Step 0), vendored under oracle/source/. Hashes are content " +
      "fingerprints; gitCommit is provenance only and is NOT part of any hash. " +
      "See oracle/README.md for the exact algorithm.",
    gitCommit,
    vendoredFrom: "lorcana-simulator/ (workspace)",
    closure: { dirs: [...CLOSURE_DIRS], files: [...CLOSURE_FILES] },
    excludes: { dirNames: RSYNC_EXCLUDES, note: "non-source / install-state / regenerable only" },
    hashingSpec: {
      algorithm: "sha256",
      fileBytes: "raw (no normalization)",
      manifestLine: "<sha256>  <relpath>",
      manifestSort: "ascending by POSIX-normalized relative path",
      partition: "card catalog = paths under packages/lorcana/lorcana-cards/; source = the rest",
      rulesetHash:
        "sha256(canonicalJson({ formatVersion, sourceHash, cardCatalogHash, deps }))",
      symlinks: "ignored",
    },
    deps: {
      resolved: deps,
      declared: {
        seedrandom: declaredRange("dependencies", "seedrandom"),
        "@types/seedrandom": declaredRange("devDependencies", "@types/seedrandom"),
      },
    },
    counts: {
      totalFiles: freeze.files.length,
      sourceFiles: freeze.sourceFiles.length,
      cardFiles: freeze.cardFiles.length,
    },
    sourceHash: freeze.sourceHash,
    cardCatalogHash: freeze.cardCatalogHash,
    rulesetHash: freeze.rulesetHash,
  };
  write(ARTIFACTS.manifest, `${JSON.stringify(manifest, null, 2)}\n`);

  await generateRngGolden();

  console.log("\noracle freeze complete (entire oracle vendored):");
  console.log(`  source-hash       ${freeze.sourceHash}`);
  console.log(`  card-catalog-hash ${freeze.cardCatalogHash}`);
  console.log(`  ruleset-hash      ${freeze.rulesetHash}`);
  console.log(
    `  files             total=${freeze.files.length} ` +
      `source=${freeze.sourceFiles.length} cards=${freeze.cardFiles.length}`,
  );
}

await main();
