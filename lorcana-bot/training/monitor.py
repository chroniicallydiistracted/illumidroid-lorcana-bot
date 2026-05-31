"""Live training monitor — a background heartbeat so the terminal never looks
hung. A daemon thread prints a status line every `interval` seconds with
throughput, progress, loss, GPU memory and ETA, regardless of where the training
loop currently is. The loop pushes counters via `add()/update()` and discrete
events via `event()`.
"""

from __future__ import annotations

import threading
import time


class LiveMonitor:
    def __init__(self, interval: float = 3.0, total_games: int | None = None) -> None:
        self.interval = interval
        self.total_games = total_games
        self._c: dict = {"games": 0, "decisions": 0, "sims": 0, "buffer": 0, "round": 0,
                         "loss": None, "winrate": None, "last": "", "phase": "init"}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.time()
        self._prev = (time.time(), 0, 0)  # (t, sims, decisions) for rolling rates
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

    # -- rendering -----------------------------------------------------------
    @staticmethod
    def _gpu_mem() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e6
                util = ""
                try:
                    util = f"/{torch.cuda.utilization()}%"
                except Exception:
                    pass
                return f"{used:.0f}MB{util}"
        except Exception:
            pass
        return "-"

    def _line(self) -> str:
        with self._lock:
            c = dict(self._c)
        now = time.time()
        el = now - self._t0
        pt, ps, pd = self._prev
        dt = max(now - pt, 1e-6)
        sps = (c["sims"] - ps) / dt
        dps = (c["decisions"] - pd) / dt
        self._prev = (now, c["sims"], c["decisions"])
        gph = c["games"] / el * 3600 if el > 0 else 0.0
        eta = ""
        if self.total_games and gph > 0 and c["games"] < self.total_games:
            rem = (self.total_games - c["games"]) / gph * 3600
            eta = f" eta {int(rem // 60)}m{int(rem % 60):02d}s"
        loss = f"{c['loss']:.3f}" if c.get("loss") is not None else "-"
        wr = f"{c['winrate']:.0%}" if c.get("winrate") is not None else "-"
        games = f"{c['games']}" + (f"/{self.total_games}" if self.total_games else "")
        return (f"[{int(el // 60):02d}:{int(el % 60):02d}] {c['phase']:<10} "
                f"r{c['round']} g{games} | {sps:5.0f} sims/s {dps:4.1f} dec/s "
                f"{gph:4.0f} games/h | buf {c['buffer']} | loss {loss} wr {wr} | "
                f"gpu {self._gpu_mem()}{eta} | {c['last']}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            print(self._line(), flush=True)

    def start(self) -> "LiveMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(self._line(), flush=True)
