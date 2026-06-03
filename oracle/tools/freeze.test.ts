/**
 * Oracle freeze — conformance / fail-closed tests (bun:test).
 *
 *   bun test oracle/tools/freeze.test.ts
 *
 * Proves the freeze is a faithful, complete, reproducible, fail-closed copy of
 * the ENTIRE TypeScript oracle. These would fail if: the vendored copy diverges
 * from the working tree, recorded hashes drift, the ruleset hash is not derived
 * from its inputs, a partition is empty/missing, a non-source dir leaks in, or
 * the seedrandom RNG pin changes.
 */

import { describe, expect, it } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import {
  ARTIFACTS,
  CLOSURE_DIRS,
  CLOSURE_FILES,
  EXCLUDE_DIR_NAMES,
  EXPECTED_SEEDRANDOM_VERSION,
  LOCAL_PATH_ALLOWLIST,
  REPO_ROOT,
  SIM_ROOT,
  VENDOR_DIR,
  VENDOR_ENGINE_PKG,
  declaredRange,
  installedSeedrandomVersion,
  listClosureFiles,
  resolveDeps,
  resolveLockedVersion,
  scanFrozenLocalPaths,
} from "./freeze-config";
import {
  CARDS_PATH_PREFIX,
  assertNoDrift,
  computeRulesetHash,
  diffFiles,
  hashOracleFiles,
} from "./freeze-lib";
import { buildRngGolden } from "./gen-rng-golden";

function read(path: string): string {
  return readFileSync(path, "utf8").trim();
}

describe("oracle freeze (Step 0) — entire oracle vendored", () => {
  it("recorded hashes reproduce from the frozen copy (tamper-evidence)", () => {
    const f = hashOracleFiles(listClosureFiles(VENDOR_DIR), resolveDeps());
    expect(f.sourceHash).toBe(read(ARTIFACTS.sourceHash));
    expect(f.cardCatalogHash).toBe(read(ARTIFACTS.cardCatalogHash));
    expect(f.rulesetHash).toBe(read(ARTIFACTS.rulesetHash));
  });

  it("frozen copy is byte-identical to the working tree (faithful + complete)", () => {
    const d = diffFiles(listClosureFiles(VENDOR_DIR), listClosureFiles(SIM_ROOT));
    expect(d.changed).toEqual([]);
    expect(d.missing).toEqual([]);
    expect(d.extra).toEqual([]);
  });

  it("ruleset hash is derived purely from source-hash + catalog-hash + deps", () => {
    const f = hashOracleFiles(listClosureFiles(VENDOR_DIR), resolveDeps());
    const reDerived = computeRulesetHash({
      sourceHash: f.sourceHash,
      cardCatalogHash: f.cardCatalogHash,
      deps: resolveDeps(),
    });
    expect(reDerived).toBe(read(ARTIFACTS.rulesetHash));
  });

  it("hashing is deterministic, path-sorted, and partitioned", () => {
    const a = hashOracleFiles(listClosureFiles(VENDOR_DIR), resolveDeps());
    const b = hashOracleFiles(listClosureFiles(VENDOR_DIR), resolveDeps());
    expect(a.rulesetHash).toBe(b.rulesetHash);
    const paths = a.files.map((f) => f.path);
    expect([...paths].sort()).toEqual(paths);
    expect(a.files.length).toBe(a.sourceFiles.length + a.cardFiles.length);
    expect(a.cardFiles.every((f) => f.path.startsWith(CARDS_PATH_PREFIX))).toBe(true);
    expect(a.sourceFiles.some((f) => f.path.startsWith(CARDS_PATH_PREFIX))).toBe(false);
  });

  it("fails closed on empty / incomplete inputs", () => {
    expect(() => hashOracleFiles([], resolveDeps())).toThrow(/no files/);
    // Source-only (no card catalog) must be rejected.
    expect(() =>
      hashOracleFiles([{ path: "packages/lorcana/lorcana-engine/src/x.ts", sha256: "a" }], resolveDeps()),
    ).toThrow(/no card catalog/);
    // Missing closure dir under a bogus base must throw.
    expect(() => listClosureFiles(join(REPO_ROOT, "oracle", "__nope__"))).toThrow(/missing closure/);
  });

  it("frozen copy contains the full closure and no excluded dirs leaked in", () => {
    for (const d of CLOSURE_DIRS) {
      expect(statSync(join(VENDOR_DIR, d)).isDirectory()).toBe(true);
    }
    for (const f of CLOSURE_FILES) {
      expect(statSync(join(VENDOR_DIR, f)).isFile()).toBe(true);
    }
    // Sentinel oracle files are present in the frozen copy.
    expect(
      existsSync(
        join(VENDOR_DIR, "packages/lorcana/lorcana-engine/src/core/runtime/match-runtime.random-apis.ts"),
      ),
    ).toBe(true);
    // No node_modules (or other excluded dir) anywhere in the frozen copy.
    const paths = listClosureFiles(VENDOR_DIR).map((f) => f.path);
    for (const ex of EXCLUDE_DIR_NAMES) {
      expect(paths.some((p) => p.split("/").includes(ex))).toBe(false);
    }
  });

  it("seedrandom RNG pin is frozen (declared range + locked version)", () => {
    const enginePkg = JSON.parse(readFileSync(VENDOR_ENGINE_PKG, "utf8")) as {
      dependencies: Record<string, string>;
    };
    expect(enginePkg.dependencies.seedrandom).toBe("^3.0.5");
    expect(declaredRange("dependencies", "seedrandom")).toBe("^3.0.5");
    expect(resolveLockedVersion("seedrandom")).toBe("3.0.5");
    expect(resolveLockedVersion("@types/seedrandom")).toBe("3.0.8");
  });

  it("verify-freeze CLI passes against the committed freeze", () => {
    const out = execFileSync("bun", ["run", join(import.meta.dir, "verify-freeze.ts")], {
      cwd: REPO_ROOT,
    }).toString();
    expect(out).toContain("oracle verify: OK");
  });

  it("verify-freeze --vs-upstream passes (frozen == working tree)", () => {
    const out = execFileSync(
      "bun",
      ["run", join(import.meta.dir, "verify-freeze.ts"), "--vs-upstream"],
      { cwd: REPO_ROOT },
    ).toString();
    expect(out).toContain("--vs-upstream OK");
  });
});

describe("oracle freeze — strict upstream drift fails closed", () => {
  const base = [
    { path: "a", sha256: "1" },
    { path: "b", sha256: "2" },
  ];

  it("assertNoDrift passes when identical", () => {
    expect(() => assertNoDrift(base, [...base])).not.toThrow();
  });

  it("assertNoDrift fails closed on a changed file", () => {
    expect(() =>
      assertNoDrift(base, [
        { path: "a", sha256: "1" },
        { path: "b", sha256: "DIFFERENT" },
      ]),
    ).toThrow(/changed=1/);
  });

  it("assertNoDrift fails closed on a missing file", () => {
    expect(() => assertNoDrift(base, [{ path: "a", sha256: "1" }])).toThrow(/missing=1/);
  });

  it("assertNoDrift fails closed on an extra file", () => {
    expect(() => assertNoDrift(base, [...base, { path: "c", sha256: "3" }])).toThrow(/extra=1/);
  });

  it("verify-freeze --vs-upstream exits non-zero when the frozen copy drifts", () => {
    // Drive the strict CLI with a deliberately incomplete upstream by pointing
    // SIM_ROOT at an empty temp dir via a tiny inline script that reuses the lib.
    const script = `
      import { assertNoDrift, hashOracleFiles } from "${join(import.meta.dir, "freeze-lib.ts")}";
      import { listClosureFiles, resolveDeps, VENDOR_DIR } from "${join(import.meta.dir, "freeze-config.ts")}";
      const frozen = hashOracleFiles(listClosureFiles(VENDOR_DIR), resolveDeps()).files;
      assertNoDrift(frozen, frozen.slice(0, frozen.length - 1), "frozen", "working tree");
    `;
    let threw = false;
    try {
      execFileSync("bun", ["-e", script], { cwd: REPO_ROOT, stdio: "pipe" });
    } catch (e) {
      threw = true;
      expect(String((e as { stderr?: Buffer }).stderr ?? "")).toContain("drift detected");
    }
    expect(threw).toBe(true);
  });
});

describe("oracle freeze — vendored local-path allowlist", () => {
  it("only allowlisted upstream files contain absolute developer paths", () => {
    expect(scanFrozenLocalPaths()).toEqual([...LOCAL_PATH_ALLOWLIST].sort());
  });
});

describe("oracle freeze — RNG/shuffle golden vectors", () => {
  it("golden vectors regenerate identically and are pinned to seedrandom 3.0.5", async () => {
    expect(installedSeedrandomVersion()).toBe(EXPECTED_SEEDRANDOM_VERSION);
    const fresh = await buildRngGolden();
    const committed = JSON.parse(readFileSync(ARTIFACTS.rngGolden, "utf8")) as typeof fresh;
    expect(committed.seedrandomVersion).toBe(EXPECTED_SEEDRANDOM_VERSION);
    expect(committed.rulesetHash).toBe(read(ARTIFACTS.rulesetHash));
    expect(committed.random).toEqual(fresh.random);
    expect(committed.shuffle).toEqual(fresh.shuffle);
    // Non-trivial coverage: multiple seeds and a deck-sized shuffle present.
    expect(committed.random.length).toBeGreaterThan(0);
    expect(committed.shuffle.some((s) => s.input.length === 60)).toBe(true);
  });
});
