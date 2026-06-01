"""Live training monitor — a background heartbeat so the terminal never looks
hung. A daemon thread prints a status line every `interval` seconds with
throughput, progress, per-head losses, GPU memory and ETA, regardless of where
the training loop currently is. The loop pushes counters via `add()/update()`
and discrete events via `event()`.

Accuracy notes (these were bugs before):
  * Rolling sims/s & dec/s advance ONLY on the heartbeat tick, and only when at
    least `interval/2` has elapsed, so a fast second caller can't divide by a
    near-zero dt and print a garbage spike. The last good rate is reused instead.
  * Rendering has no side effects. `stop()` prints a final SUMMARY with *lifetime*
    averages (total/elapsed), not an instantaneous near-zero-dt sample.
  * The full learner stats dict is surfaced (policy/value/belief/belief_ink/kl/
    entropy + separation diagnostics), not just the scalar total.
"""

from __future__ import annotations

import threading
import time


class LiveMonitor:
    def __init__(self, interval: float = 3.0, total_games: int | None = None) -> None:
        self.interval = interval
        self.total_games = total_games
        self._c: dict = {"games": 0, "decisions": 0, "sims": 0, "buffer": 0, "round": 0,
                         "loss": None, "losses": None, "winrate": None, "wr_round": None,
                         "last": "", "phase": "init"}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.time()
        self._prev = (self._t0, 0, 0)      # (t, sims, decisions) for rolling rates
        self._last_rate = (0.0, 0.0)       # last good (sims/s, dec/s)
        self._thread: threading.Thread | None = None

    # -- counter API (called from the training loop) -------------------------
    def update(self, **kv) -> None:
        with self._lock:
            self._c.update(kv)

    def add(self, **kv) -> None:
        with self._lock:
            for k, v in kv.items():
                self._c[k] = self._c.get(k, 0) + v

    def event(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] • {msg}", flush=True)

    # -- rendering helpers (no side effects) ---------------------------------
    @staticmethod
    def _gpu_mem() -> str:
        try:
            import torch
            if not torch.cuda.is_available():
                return "-"
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = 0.0
            try:
                _, total_b = torch.cuda.mem_get_info()
                total = total_b / 1e9
            except Exception:
                pass
            util = ""
            try:
                util = f" {torch.cuda.utilization()}%"
            except Exception:
                pass
            cap = f"/{total:.1f}" if total else ""
            return f"{reserved:.1f}{cap}GB pk{peak:.1f}{util}"
        except Exception:
            return "-"

    @staticmethod
    def _fmt_losses(ls: dict | None) -> str:
        if not ls:
            return ""
        parts = []
        if "policy" in ls:
            parts.append(f"p {ls['policy']:.2f}")
        if "value" in ls:
            parts.append(f"v {ls['value']:.2f}")
        if "belief" in ls:
            sep = ls.get("belief_sep")
            parts.append(f"bel {ls['belief']:.2f}" + (f"/{sep:+.2f}" if sep is not None else ""))
        if "belief_ink" in ls:
            sep = ls.get("belief_ink_sep")
            parts.append(f"ink {ls['belief_ink']:.2f}" + (f"/{sep:+.2f}" if sep is not None else ""))
        if "kl" in ls:
            parts.append(f"kl {ls['kl']:.3f}")
        if "entropy" in ls:
            parts.append(f"H {ls['entropy']:.2f}")
        return "  L[" + " ".join(parts) + "]" if parts else ""

    def _render(self, sps: float, dps: float, gph: float, eta: str, c: dict, el: float) -> str:
        loss = f"{c['loss']:.3f}" if c.get("loss") is not None else "-"
        if c.get("winrate") is not None:
            wr = f"{c['winrate']:.0%}" + (f"@r{c['wr_round']}" if c.get("wr_round") else "")
        else:
            wr = "-"
        games = f"{c['games']}" + (f"/{self.total_games}" if self.total_games else "")
        return (f"[{int(el // 60):02d}:{int(el % 60):02d}] {c['phase']:<10} "
                f"r{c['round']} g{games} | {sps:5.0f} sims/s {dps:4.1f} dec/s "
                f"{gph:4.0f} games/h | buf {c['buffer']} | loss {loss} wr {wr} | "
                f"gpu {self._gpu_mem()}{eta}{self._fmt_losses(c.get('losses'))} | {c['last']}")

    def _eta(self, gph: float, games: int) -> str:
        if self.total_games and gph > 0 and games < self.total_games:
            rem = (self.total_games - games) / gph * 3600
            return f" eta {int(rem // 60)}m{int(rem % 60):02d}s"
        return ""

    def _heartbeat_line(self) -> str:
        with self._lock:
            c = dict(self._c)
        now = time.time()
        el = max(now - self._t0, 1e-6)
        # rolling rate — advance the window only on a real tick (dt >= interval/2),
        # otherwise reuse the last good rate so we never divide by a tiny dt.
        pt, ps, pd = self._prev
        dt = now - pt
        if dt >= self.interval * 0.5:
            self._last_rate = ((c["sims"] - ps) / dt, (c["decisions"] - pd) / dt)
            self._prev = (now, c["sims"], c["decisions"])
        sps, dps = self._last_rate
        gph = c["games"] / el * 3600
        return self._render(sps, dps, gph, self._eta(gph, c["games"]), c, el)

    def _summary_line(self) -> str:
        with self._lock:
            c = dict(self._c)
        el = max(time.time() - self._t0, 1e-6)
        # lifetime averages (not an instantaneous sample) for the final line
        sps, dps = c["sims"] / el, c["decisions"] / el
        gph = c["games"] / el * 3600
        return ("SUMMARY " + self._render(sps, dps, gph, "", c, el)
                + f"  | totals: {c['sims']} sims {c['decisions']} dec {c['games']} games")

    # -- lifecycle -----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            print(self._heartbeat_line(), flush=True)

    def start(self) -> "LiveMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(self._summary_line(), flush=True)
