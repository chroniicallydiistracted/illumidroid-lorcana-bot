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
