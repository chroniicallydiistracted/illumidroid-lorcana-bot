#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

BASE = os.environ.get("AGENT_BASE_REF", "HEAD")
REGISTRY = "docs/port/headless_lorcana_engine_porting_symbol_registry.md"


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()


def changed_files() -> list[str]:
    try:
        out = run(["git", "diff", "--name-only", BASE])
    except Exception:
        try:
            out = run(["git", "diff", "--name-only", "--cached"])
        except Exception:
            out = ""
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    if os.environ.get("ALLOW_NO_SYMBOL_REGISTRY") == "1":
        print("[registry] ALLOW_NO_SYMBOL_REGISTRY=1 set; skipping.")
        return 0

    files = changed_files()
    if not files:
        print("[registry] no changed files detected.")
        return 0

    registry_changed = REGISTRY in files

    triggers = []
    trigger_prefixes = (
        "lorcana-rs/",
        "oracle/",
        "crates/",
    )
    trigger_suffixes = (
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "pyproject.toml",
    )

    for f in files:
        if f == REGISTRY:
            continue
        if f.startswith(trigger_prefixes) or f.endswith(trigger_suffixes):
            triggers.append(f)
        if f.endswith((".rs", ".toml", ".json", ".yaml", ".yml")) and (
            "fixture" in f or "fixtures" in f or "snapshot" in f or "oracle" in f
        ):
            triggers.append(f)

    if triggers and not registry_changed:
        print("[registry] BLOCKER: changed files likely require symbol registry updates:")
        for f in sorted(set(triggers)):
            print(f"  - {f}")
        print(f"[registry] Update {REGISTRY}, or rerun with ALLOW_NO_SYMBOL_REGISTRY=1 only for a justified no-symbol-change diff.")
        return 1

    print("[registry] passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
