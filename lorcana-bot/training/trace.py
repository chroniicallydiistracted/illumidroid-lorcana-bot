"""Decision-level tracing for self-play.

Verbose, per-decision JSONL so we can see *what* was chosen, *what the
alternatives were*, and *why* (priors / visits / Q) — drift detection, instead of
only result-based logging. Mirrors the engine's `AutomatedActionDecisionTrace`
shape ([automation/types.ts]).

Gated by `--trace`; one file per process (the run, or each parallel worker), so
there is no cross-process contention. Every record carries `game_id` so a single
file can be filtered per game.

IMPORTANT (fairness): we log only the FAIR, structural candidate fields. The
engine summary's `heuristics`/`contributors`/`opponentKnowledgeSource` are NEVER
recorded here or fed to the net — they can embody oracle (full-deck) knowledge.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

# candidate fields worth recording (fair + structural); skip empty/sentinel values
_DESC_KEYS = ("src", "tgt", "tgt2", "choice", "targets", "singers", "costType",
              "namedCard", "resolveOptional", "abilityIndex", "destinationZones")


def _descriptor(act: dict) -> dict:
    d: dict[str, Any] = {"family": act.get("family")}
    for k in _DESC_KEYS:
        v = act.get(k)
        if v not in (None, -1, "", [], {}):
            d[k] = v
    return d


class DecisionTracer:
    """Append-only JSONL writer for per-decision traces. A no-op when disabled."""

    def __init__(self, out_dir: str, run_id: str, top_k: int = 8, enabled: bool = True) -> None:
        self.enabled = enabled
        self.top_k = top_k
        self._fh = None
        self.path = None
        if enabled:
            os.makedirs(out_dir, exist_ok=True)
            self.path = os.path.join(out_dir, f"trace-{run_id}.jsonl")
            self._fh = open(self.path, "a")

    def log(self, obs: dict, res, chosen_idx: int, game_id: str = "",
            executed: dict | None = None, belief_sep: float | None = None) -> None:
        if not self.enabled or self._fh is None:
            return
        root = res.root
        legal = root.legal
        P = np.asarray(root.P, dtype=float)
        N = np.asarray(root.N, dtype=float)
        Q = np.asarray(root.q_values(), dtype=float)
        pi = np.asarray(res.pi, dtype=float)

        order = np.argsort(-N) if N.size else np.arange(len(legal))
        cands = []
        for a in order[: self.top_k]:
            a = int(a)
            cands.append({
                "idx": a,
                "stableKey": legal[a].get("stableKey") if a < len(legal) else None,
                "descriptor": _descriptor(legal[a]) if a < len(legal) else {},
                "P": round(float(P[a]), 4) if a < P.size else None,
                "N": int(N[a]) if a < N.size else 0,
                "Q": round(float(Q[a]), 4) if a < Q.size else None,
                "pi": round(float(pi[a]), 4) if a < pi.size else None,
                "chosen": a == int(chosen_idx),
            })

        rec: dict[str, Any] = {
            "t": round(time.time(), 3),
            "game_id": game_id,
            "turn": obs.get("turn"),
            "phase": obs.get("phase"),
            "actor": obs.get("actor"),
            "forced": obs.get("forced"),
            "n_legal": len(legal),
            "root_value": round(float(res.value), 4),
            "root_visits": int(N.sum()) if N.size else 0,
            "chosen_idx": int(chosen_idx),
            "chosen_key": legal[chosen_idx].get("stableKey")
            if 0 <= int(chosen_idx) < len(legal) else None,
            "candidates": cands,
        }
        if belief_sep is not None:
            rec["belief_sep"] = round(float(belief_sep), 4)
        if executed is not None:
            rec["executed"] = {k: executed.get(k)
                               for k in ("success", "matched", "executed") if k in executed}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --- human-readable live play-by-play ----------------------------------------

def _short(name: str | None, fallback: str = "?") -> str:
    if not name:
        return fallback
    return name.split(" - ")[0][:18]   # drop the version subtitle, cap length


def _describe(act: dict, names: dict) -> str:
    """One-line readable description of a chosen action."""
    fam = act.get("family", "?")
    src = _short(names.get(act.get("src")), act.get("src") or "")
    tgts = [_short(names.get(t), t) for t in (act.get("targets") or []) if t]
    ct = act.get("costType")
    if fam == "playCard":
        s = f'play {src}'
        if ct and ct not in ("none", "standard"):
            s += f'[{ct}]'
        if act.get("singers"):
            s += f' (sing x{len(act["singers"])})'
        if tgts:
            s += " → " + ", ".join(tgts)
        return s
    if fam == "putCardIntoInkwell":
        return f'ink {src}'
    if fam == "quest":
        return f'quest {src}'
    if fam == "challenge":
        return f'challenge {src} → {tgts[0] if tgts else "?"}'
    if fam == "moveCharacterToLocation":
        return f'move {src} → {tgts[0] if tgts else "?"}'
    if fam == "activateAbility":
        return f'ability {src}#{act.get("abilityIndex", "")}'
    if fam in ("resolveBag", "resolveEffect"):
        extra = f' choice={act["choice"]}' if act.get("choice", -1) >= 0 else ""
        return f'{fam}{extra}'
    if fam == "alterHand":
        return "mulligan"
    if fam == "passTurn":
        return "pass"
    return fam


class GameLogger:
    """Verbose, human-readable play-by-play with three independent outputs, so
    EVERY game can be captured even when only one worker streams to the terminal:

      * `to_stdout` — print live (single-process clean narrative);
      * `sink(line)` — ship the line elsewhere (parallel worker 0 -> main queue);
      * `file_path`  — append every line to a per-worker file (ALL workers), so a
        full record of every game survives regardless of what the terminal shows.

    `tail -f <file_path>` lets you watch any individual worker's narrative."""

    def __init__(self, enabled: bool = True, to_stdout: bool = False,
                 sink=None, file_path: str | None = None) -> None:
        self.enabled = enabled
        self._stdout = to_stdout
        self._sink = sink
        self._file = None
        if enabled and file_path:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            self._file = open(file_path, "a")

    def _emit(self, line: str) -> None:
        if not self.enabled:
            return
        if self._stdout:
            print(line, flush=True)
        if self._sink is not None:
            self._sink(line)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def game_start(self, gid: str, p1: str, p2: str) -> None:
        self._emit(f"\n┌─ game {gid}  |  P1 {p1}  vs  P2 {p2}")

    def game_end(self, winner, turn) -> None:
        w = "draw" if winner is None else str(winner)
        self._emit(f"└─ result: {w}  (turn {turn})")

    def decision(self, obs: dict, res, chosen_idx: int, result: dict) -> None:
        if not self.enabled:
            return
        names = {c.get("id"): c.get("name") for c in obs.get("cards", []) or []}
        legal = obs.get("legal", [])
        if not (0 <= chosen_idx < len(legal)):
            return
        act = legal[chosen_idx]
        plabel = f"P{int(obs.get('selfIdx', 0)) + 1}"
        turn, phase = obs.get("turn", "?"), (obs.get("phase") or "")[:5]
        # post-action public counts from the event the bridge just appended
        post = (result.get("obs", {}) or {}).get("history") or []
        ev = post[-1] if post else {}
        lore = ev.get("lore", ["?", "?"]); ink = ev.get("ink", ["?", "?"]); hand = ev.get("hand", ["?", "?"])
        import numpy as _np
        N = int(_np.asarray(res.root.N)[chosen_idx]) if res.root.N.size > chosen_idx else 0
        Q = float(_np.asarray(res.root.q_values())[chosen_idx]) if res.root.N.size > chosen_idx else 0.0
        pi = float(res.pi[chosen_idx]) if res.pi.size > chosen_idx else 0.0
        ng = "" if result.get("matched", True) else " ⚠fallback"
        self._emit(f"  T{turn:<2} {phase:<5} {plabel} │ {_describe(act, names):<34} │ "
                   f"lore {lore[0]}:{lore[1]} ink {ink[0]}:{ink[1]} hand {hand[0]}:{hand[1]} │ "
                   f"π{pi:.2f} N{N} Q{Q:+.2f}{ng}")
