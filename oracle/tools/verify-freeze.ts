/**
 * Oracle freeze — verifier CLI (fail-closed).
 *
 * Default: re-hash the vendored frozen oracle (`oracle/source/`) and compare to
 * the recorded artifacts. Exits non-zero on any drift (tamper-evidence of the
 * frozen copy). A parity/performance claim is only valid against the
 * `ruleset-hash` recorded here.
 *
 *   bun run oracle/tools/verify-freeze.ts
 *   bun run oracle/tools/verify-freeze.ts --vs-upstream     # FAIL CLOSED on any
 *                                                            # frozen-vs-working-tree drift
 *   bun run oracle/tools/verify-freeze.ts --upstream-report # informational only (exit 0)
 */

import { existsSync, readFileSync } from "node:fs";
import { ARTIFACTS, SIM_ROOT, VENDOR_DIR, listClosureFiles, resolveDeps } from "./freeze-config";
import { assertNoDrift, diffFiles, diffIsEmpty, hashOracleFiles } from "./freeze-lib";

function readHash(path: string): string {
  if (!existsSync(path)) {
    throw new Error(`oracle verify: missing artifact ${path}; run freeze-oracle.ts first`);
  }
  return readFileSync(path, "utf8").trim();
}

function main(): void {
  const freeze = hashOracleFiles(listClosureFiles(VENDOR_DIR), resolveDeps());
  const checks: Array<[string, string, string]> = [
    ["source-hash", freeze.sourceHash, readHash(ARTIFACTS.sourceHash)],
    ["card-catalog-hash", freeze.cardCatalogHash, readHash(ARTIFACTS.cardCatalogHash)],
    ["ruleset-hash", freeze.rulesetHash, readHash(ARTIFACTS.rulesetHash)],
  ];

  let ok = true;
  for (const [name, computed, recorded] of checks) {
    if (computed === recorded) console.log(`  OK    ${name} ${computed}`);
    else {
      ok = false;
      console.error(`  DRIFT ${name}\n    recorded: ${recorded}\n    computed: ${computed}`);
    }
  }
  if (!ok) {
    console.error(
      "\noracle verify: FAILED — the frozen oracle copy changed (or freeze is stale).\n" +
        "Inspect oracle/file-hashes/*.sha256 to find the drifted files.",
    );
    process.exit(1);
  }
  console.log("\noracle verify: OK — frozen oracle copy matches recorded hashes.");

  const strict = process.argv.includes("--vs-upstream");
  const report = process.argv.includes("--upstream-report");
  if (!strict && !report) return;

  const upstream = listClosureFiles(SIM_ROOT);
  if (strict) {
    // Fail closed on ANY missing/extra/changed file.
    assertNoDrift(freeze.files, upstream, "frozen", "working tree");
    console.log("oracle verify: --vs-upstream OK — frozen copy is identical to the working tree.");
    return;
  }

  // --upstream-report: informational only, never fails.
  const d = diffFiles(freeze.files, upstream);
  if (diffIsEmpty(d)) {
    console.log("oracle verify: frozen copy is identical to the working tree.");
    return;
  }
  console.log(
    `\noracle verify (report): frozen copy DIFFERS from working tree ` +
      `(missing=${d.missing.length} extra=${d.extra.length} changed=${d.changed.length}).`,
  );
  for (const p of d.changed.slice(0, 20)) console.log(`  changed ${p}`);
  for (const p of d.missing.slice(0, 20)) console.log(`  only-in-frozen ${p}`);
  for (const p of d.extra.slice(0, 20)) console.log(`  only-in-working ${p}`);
  console.log(
    "(informational: the frozen copy is the reference of record; re-freeze only on an " +
      "intentional oracle change.)",
  );
}

main();
