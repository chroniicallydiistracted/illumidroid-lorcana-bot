#!/usr/bin/env python3
"""Resolve the internal import closure of the Lorcana engine kernel.

Starts from entry files, follows static relative + '#core' imports, treats
bare specifiers as external leaves. Emits the reachable internal .ts file set.
"""
import os, re, sys, json

ROOT = "/home/claude/lorcana-simulator/packages/lorcana"
ENGINE = os.path.join(ROOT, "lorcana-engine")
TYPES = os.path.join(ROOT, "lorcana-types")

# package.json "imports" subpath map for the engine
SUBPATH = {
    "#core": os.path.join(ENGINE, "src/core/index.ts"),
    "#core/testing": os.path.join(ENGINE, "src/core/testing/index.ts"),
}

IMPORT_RE = re.compile(
    r'(?:import|export)\s+(?:type\s+)?[^"\']*?from\s*["\']([^"\']+)["\']'
    r'|import\s*\(\s*["\']([^"\']+)["\']\s*\)'
    r'|import\s+["\']([^"\']+)["\']'
)

def resolve_relative(importer, spec):
    base = os.path.dirname(importer)
    # strip trailing .js (NodeNext-style) -> try .ts
    cand_specs = [spec]
    if spec.endswith(".js"):
        cand_specs.append(spec[:-3])
    target = os.path.normpath(os.path.join(base, spec))
    targets = [os.path.normpath(os.path.join(base, s)) for s in cand_specs]
    for t in targets:
        for ext in (".ts", ".tsx", ".d.ts"):
            if os.path.isfile(t + ext):
                return t + ext
        for idx in ("index.ts", "index.tsx"):
            p = os.path.join(t, idx)
            if os.path.isfile(p):
                return p
        if os.path.isfile(t):
            return t
    return None

def resolve(importer, spec):
    if spec.startswith("#"):
        return SUBPATH.get(spec)
    if spec.startswith("."):
        return resolve_relative(importer, spec)
    # bare specifier -> external leaf
    return None

def imports_of(path):
    out = []
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return out
    # crude block-comment strip to avoid the serialization.ts comment etc.
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    for m in IMPORT_RE.finditer(src):
        spec = m.group(1) or m.group(2) or m.group(3)
        if spec:
            out.append(spec)
    return out

def main(entries):
    seen = set()
    external = {}
    stack = list(entries)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        for spec in imports_of(f):
            r = resolve(f, spec)
            if r is None:
                if not spec.startswith((".", "#")):
                    external[spec] = external.get(spec, 0) + 1
            elif r not in seen:
                stack.append(r)
    internal = sorted(s for s in seen if s.endswith((".ts", ".tsx")))
    return internal, external

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "full":
        entries = [os.path.join(ENGINE, "src/index.ts")]
    else:  # 'kernel' -- the minimal search surface
        entries = [
            os.path.join(ENGINE, "src/lorcana-server.ts"),
            os.path.join(ENGINE, "src/available-moves.ts"),
            os.path.join(ENGINE, "src/serialization.ts"),
            os.path.join(ENGINE, "src/core/runtime/view-filter.ts"),
            os.path.join(ENGINE, "src/core/engine/projection.ts"),
        ]
    internal, external = main(entries)
    # split internal by test vs not
    nontest = [f for f in internal if not f.endswith((".test.ts", ".spec.ts"))]
    out = {
        "mode": mode,
        "internal_files": len(internal),
        "internal_nontest": len(nontest),
        "external_specifiers": external,
        "files": [os.path.relpath(f, ROOT) for f in nontest],
    }
    json.dump(out, open(f"/home/claude/probe/closure_{mode}.json", "w"), indent=2)
    print(f"[{mode}] reachable internal files: {len(internal)}  (non-test: {len(nontest)})")
    print(f"[{mode}] external specifiers: {json.dumps(external)}")
