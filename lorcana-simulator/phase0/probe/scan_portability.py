#!/usr/bin/env python3
"""Scan the reachable kernel closure for non-portable / non-deterministic APIs.

Categories:
  BLOCKER      - DOM / network / fs / process-spawn: breaks a QuickJS-WASM build if reached
  DETERMINISM  - nondeterministic sources that must be seeded/shimmed regardless of host
  SHIMMABLE    - host globals with trivial polyfills
  LOGGING      - logging side-channel; isolate or no-op
We record file:line evidence and whether the hit is at module top-level
(eval-time, harder) vs inside a function body (call-time, only matters if invoked).
"""
import os, re, json

ROOT = "/home/claude/lorcana-simulator/packages/lorcana"
closure = json.load(open("/home/claude/probe/closure_full.json"))
files = [os.path.join(ROOT, f) for f in closure["files"]]

# pattern -> (category, label).  Word-boundaried, fairly conservative.
PATTERNS = {
    r"\bdocument\b": ("BLOCKER", "DOM:document"),
    r"\bwindow\b": ("BLOCKER", "DOM:window"),
    r"\bnavigator\b": ("BLOCKER", "DOM:navigator"),
    r"\blocalStorage\b|\bsessionStorage\b": ("BLOCKER", "DOM:webStorage"),
    r"\bXMLHttpRequest\b": ("BLOCKER", "net:XHR"),
    r"\bWebSocket\b": ("BLOCKER", "net:WebSocket"),
    r"\bfetch\s*\(": ("BLOCKER", "net:fetch"),
    r"from\s*[\"']node:(fs|child_process|net|http|https|os|worker_threads|cluster|dgram|tls)[\"']": ("BLOCKER", "node:builtin"),
    r"require\(\s*[\"'](fs|child_process|net|http|https|os|worker_threads)[\"']": ("BLOCKER", "node:require"),
    r"\b__dirname\b|\b__filename\b": ("BLOCKER", "node:dirglobals"),

    r"\bMath\.random\s*\(": ("DETERMINISM", "Math.random"),
    r"\bDate\.now\s*\(": ("DETERMINISM", "Date.now"),
    r"\bnew\s+Date\s*\(\s*\)": ("DETERMINISM", "new Date()"),
    r"\bperformance\.now\s*\(": ("DETERMINISM", "performance.now"),
    r"\bcrypto\.getRandomValues\b|\bgetRandomValues\b": ("DETERMINISM", "crypto.getRandomValues"),
    r"\bnanoid\b": ("DETERMINISM", "nanoid"),
    r"\bprocess\.hrtime\b": ("DETERMINISM", "process.hrtime"),

    r"\bsetTimeout\s*\(|\bsetInterval\s*\(|\bsetImmediate\s*\(": ("SHIMMABLE", "timers"),
    r"\bqueueMicrotask\s*\(": ("SHIMMABLE", "queueMicrotask"),
    r"\bprocess\.env\b": ("SHIMMABLE", "process.env"),
    r"\bprocess\.nextTick\b": ("SHIMMABLE", "process.nextTick"),
    r"\bBuffer\b": ("SHIMMABLE", "Buffer"),
    r"\bTextEncoder\b|\bTextDecoder\b": ("SHIMMABLE", "TextEncoder"),
    r"\beval\s*\(|new\s+Function\s*\(": ("SHIMMABLE", "eval/Function"),
    r"\bstructuredClone\s*\(": ("SHIMMABLE", "structuredClone"),

    r"@logtape/": ("LOGGING", "logtape-import"),
    r"\bconsole\.": ("LOGGING", "console"),
}
COMPILED = [(re.compile(p), c, l) for p, (c, l) in PATTERNS.items()]

# crude brace-depth tracker to tell top-level from in-function
def scan(path):
    hits = []
    try:
        lines = open(path, encoding="utf-8", errors="ignore").read().split("\n")
    except Exception:
        return hits
    depth = 0
    in_block_comment = False
    for i, raw in enumerate(lines, 1):
        line = raw
        # strip line comments and block comments (approx)
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]; in_block_comment = False
            else:
                continue
        line = re.sub(r"//.*$", "", line)
        if "/*" in line:
            if "*/" in line:
                line = re.sub(r"/\*.*?\*/", "", line)
            else:
                line = line.split("/*", 1)[0]; in_block_comment = True
        for rx, cat, label in COMPILED:
            if rx.search(line):
                hits.append((i, cat, label, depth == 0, raw.strip()[:120]))
        depth += line.count("{") - line.count("}")
        if depth < 0: depth = 0
    return hits

summary = {}          # label -> {count, files:set, toplevel:int}
by_cat = {}
evidence = {}         # label -> list of (relpath:line, snippet)
for f in files:
    rel = os.path.relpath(f, ROOT)
    for (ln, cat, label, toplevel, snip) in scan(f):
        s = summary.setdefault(label, {"category": cat, "count": 0, "files": set(), "toplevel": 0})
        s["count"] += 1; s["files"].add(rel)
        if toplevel: s["toplevel"] += 1
        by_cat[cat] = by_cat.get(cat, 0) + 1
        evidence.setdefault(label, [])
        if len(evidence[label]) < 6:
            evidence[label].append(f"{rel}:{ln}  |  {snip}")

print("="*78)
print("REACHABLE-CLOSURE PORTABILITY SCAN  ({} files)".format(len(files)))
print("="*78)
order = ["BLOCKER", "DETERMINISM", "SHIMMABLE", "LOGGING"]
for cat in order:
    labels = sorted([l for l, s in summary.items() if s["category"] == cat],
                    key=lambda l: -summary[l]["count"])
    print(f"\n### {cat}  (total hits: {by_cat.get(cat,0)})")
    if not labels:
        print("   — none —"); continue
    for l in labels:
        s = summary[l]
        print(f"   {l:24s} hits={s['count']:4d}  files={len(s['files']):3d}  top-level={s['toplevel']}")
# dump evidence for the categories that matter most
print("\n" + "="*78); print("EVIDENCE (first hits)"); print("="*78)
for cat in ["BLOCKER", "DETERMINISM"]:
    for l, s in summary.items():
        if s["category"] == cat:
            print(f"\n-- {l} --")
            for e in evidence[l]:
                print("   " + e)
