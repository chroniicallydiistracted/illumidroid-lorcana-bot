"""Phase 1A bridge — Python client for the Bun-hosted Lorcana kernel.

`LorcanaEngine` owns a `bun run server.ts` subprocess and exchanges
newline-delimited JSON (one request/response per game step). `LorcanaEnv`
wraps it as a gym-style environment: `reset() -> (obs, mask)` and
`step(action) -> (obs, mask, reward, done, info)`.

The kernel is the authority for legality: every observation carries the list
of legal actions (`obs["legal"]`), so the policy never has to guess. See
CLAUDE.md (Phase 1A) and lorcana-bot-architecture.md §4 for the design.
"""

from __future__ import annotations

import collections
import json
import os
import select
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
SIM_ROOT = REPO_ROOT / "lorcana-simulator"
SERVER_TS = _THIS.parent / "node_server" / "server.ts"

PASS_KEY = "passTurn"


class BridgeError(RuntimeError):
    pass


class LorcanaEngine:
    """One Bun subprocess hosting one game session."""

    def __init__(self, bun: str = "bun", timeout: float = 120.0) -> None:
        if not SERVER_TS.exists():
            raise BridgeError(f"server not found: {SERVER_TS}")
        self.timeout = timeout
        self._proc = subprocess.Popen(
            [bun, "run", str(SERVER_TS)],
            cwd=str(SIM_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain stderr continuously: an undrained PIPE fills (~64KB) and blocks the
        # Bun process mid-write. A daemon thread keeps it empty and retains the last
        # lines for error reporting.
        self._stderr_lines: collections.deque = collections.deque(maxlen=200)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        ready = self._read()
        if not ready.get("ready"):
            raise BridgeError(f"server failed to start: {ready}")

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            for line in self._proc.stderr:        # blocks per-line; keeps the pipe empty
                self._stderr_lines.append(line)
        except (ValueError, OSError):
            pass                                   # pipe closed on shutdown

    # -- low-level rpc --------------------------------------------------------
    def _read(self) -> dict:
        assert self._proc.stdout is not None
        # Bound the wait: a hung Bun engine (pathological game state / effect loop)
        # would otherwise block readline() forever, leaving the worker alive-but-
        # stuck so the round never completes (observed: 0 sims/s for 30+ min).
        ready, _, _ = select.select([self._proc.stdout], [], [], self.timeout)
        if not ready:
            self._proc.kill()   # don't leave a wedged subprocess around
            raise BridgeError(f"engine RPC timed out after {self.timeout}s (engine hung)")
        line = self._proc.stdout.readline()
        if not line:
            err = "".join(self._stderr_lines)      # drained by the background thread
            raise BridgeError(f"server closed unexpectedly. stderr:\n{err}")
        return json.loads(line)

    def _rpc(self, req: dict) -> dict:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        resp = self._read()
        if not resp.get("ok", False):
            raise BridgeError(f"rpc {req.get('op')} failed: {resp.get('error')}")
        return resp

    # -- high-level api -------------------------------------------------------
    def reset(self, seed: str = "seed-0", deck_p1: str | None = None,
              deck_p2: str | None = None) -> dict:
        """Start a game. With deck ids (see `list_decks`), uses those real
        tournament decks; omit them to deterministically pick a pair from the
        seed; pass deck_p1="placeholder" to force the synthetic fallback."""
        req = {"op": "reset", "seed": seed}
        if deck_p1 is not None:
            req["deckP1"] = deck_p1
        if deck_p2 is not None:
            req["deckP2"] = deck_p2
        resp = self._rpc(req)
        self.last_decks = resp.get("decks")
        return resp["obs"]

    def list_decks(self) -> list[dict]:
        """Available real decks: [{id, name}, ...]."""
        return self._rpc({"op": "list_decks"})["decks"]

    def observe(self) -> dict:
        return self._rpc({"op": "observe"})["obs"]

    def grammar_probe(self) -> dict:
        """Grammar-gap proof: capped vs uncapped automation vs raw legal moves."""
        return self._rpc({"op": "grammar_probe"})["probe"]

    def step(self, stable_key: str) -> dict:
        """Execute a chosen legal action. Returns the full step result."""
        return self._rpc({"op": "step", "stableKey": stable_key})

    def step_auto(self, strategy: str = "best") -> dict:
        """Let the scripted strategy choose+execute (bootstrap data gen)."""
        return self._rpc({"op": "step_auto", "strategy": strategy})

    def snapshot(self) -> int:
        """Checkpoint authoritative state server-side; returns a handle id."""
        return int(self._rpc({"op": "snapshot"})["id"])

    def restore(self, snap_id: int) -> dict:
        """Restore a snapshot; returns the observation at that state."""
        return self._rpc({"op": "restore", "id": snap_id})["obs"]

    def drop_snapshot(self, snap_id: int) -> None:
        self._rpc({"op": "drop_snapshot", "id": snap_id})

    def run_paths(self, root_snap: int, paths: list[list[str]]) -> list[dict]:
        """Execute a batch of descent paths (lists of stableKeys) in-process from
        the root snapshot; return each path's leaf observation. One round-trip
        replaces (paths x path-length) per-step calls — the in-process search."""
        return self._rpc({"op": "run_paths", "root": root_snap, "paths": paths})["obs"]

    def step_exact(self, stable_key: str) -> dict:
        """Execute exactly `stable_key` from the CURRENT (lane) state WITHOUT mutating
        public history; returns the next obs (flags `invalidPath` on mismatch). The
        adaptive single-step transition used by full ISMCTS (`run_infoset`)."""
        return self._rpc({"op": "step_exact", "stableKey": stable_key})["obs"]

    def determinize(self, self_id: str, hand_instance_ids: list[str],
                    seed: str | None = None) -> dict:
        """Repartition the opponent's hidden cards into hand=hand_instance_ids
        (deck = the rest), producing a determinized world. Operates on the
        current state — restore the true root first. Returns the world's obs."""
        req = {"op": "determinize", "self": self_id, "handInstanceIds": hand_instance_ids}
        if seed is not None:
            req["seed"] = seed
        return self._rpc(req)["obs"]

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._rpc({"op": "close"})
            except Exception:
                pass
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def __enter__(self) -> "LorcanaEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def winner_reward(obs: dict, perspective: str) -> float:
    """+1 / -1 / 0 from `perspective`'s point of view at a terminal obs."""
    if not obs.get("done"):
        return 0.0
    w = obs.get("winner")
    if w is None:
        return 0.0
    return 1.0 if str(w) == str(perspective) else -1.0


class LorcanaEnv:
    """Gym-style single-agent-per-decision wrapper.

    Each `step` advances the game by one decision of *whichever* player is to
    move; observations are always from that actor's information set. For
    self-play both seats are driven by the same policy, so this single-stream
    view is correct. `info["actor"]` tells you whose decision it was.
    """

    def __init__(self, engine: Optional[LorcanaEngine] = None, max_steps: int = 400) -> None:
        self.engine = engine or LorcanaEngine()
        self.max_steps = max_steps
        self._obs: dict = {}
        self._steps = 0
        self._owns_engine = engine is None

    def reset(self, seed: str = "seed-0") -> tuple[dict, np.ndarray]:
        self._obs = self.engine.reset(seed)
        self._steps = 0
        return self._obs, self.action_mask()

    def action_mask(self) -> np.ndarray:
        n = len(self._obs.get("legal", []))
        return np.ones(n, dtype=bool)

    @property
    def legal(self) -> list[dict]:
        return self._obs.get("legal", [])

    @property
    def obs(self) -> dict:
        return self._obs

    def step(self, action: int) -> tuple[dict, np.ndarray, float, bool, dict]:
        legal = self.legal
        if not legal:
            raise BridgeError("no legal actions but game not over")
        if not (0 <= action < len(legal)):
            raise BridgeError(f"action {action} out of range (n={len(legal)})")
        actor = self._obs.get("actor")
        key = legal[action]["stableKey"]
        result = self.engine.step(key)
        self._obs = result["obs"]
        self._steps += 1
        done = bool(self._obs.get("done")) or self._steps >= self.max_steps
        reward = winner_reward(self._obs, actor) if done else 0.0
        info = {
            "actor": actor,
            "executed": result.get("executed"),
            "matched": result.get("matched"),
            "success": result.get("success"),
            "winner": self._obs.get("winner"),
            "steps": self._steps,
            "truncated": self._steps >= self.max_steps and not self._obs.get("done"),
        }
        return self._obs, self.action_mask(), reward, done, info

    def close(self) -> None:
        if self._owns_engine:
            self.engine.close()
