"""Parallel self-play workers that STREAM progress to the live monitor.

K spawn-isolated CPU worker processes each own a Bun engine and play
belief-guided batched self-play. They push (a) frequent progress stats and (b)
finished-game samples to a shared queue, so the main process updates the
LiveMonitor *during* the round (never a blocked/silent wait) and the GPU learner
trains on the pooled samples at round end. Fresh workers per round reload the
latest weights from a checkpoint (spawn keeps them CUDA-free; only the main
process touches the GPU).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time

import numpy as np

from training.tier_a_guard import require_tier_a_clean_label_belief_training_ready


def _worker(cfg: dict, q) -> None:
    wid = cfg["wid"]
    sims = decs = games = 0
    try:
        require_tier_a_clean_label_belief_training_ready(
            use_belief=bool(cfg["use_belief"]), context="training.parallel._worker")
        import sys
        sys.path.insert(0, cfg["bot_root"])
        import torch
        torch.set_num_threads(1)
        from engine.bridge import LorcanaEngine
        from engine.serialization import (encode_obs, encode_belief, aux_targets,
                                           game_fingerprint, stuck_step)
        from network.model import LorcanaNet
        from search.ismcts import BISMCTS, SearchConfig
        from search.evaluator import NetEvaluator
        from training.trace import DecisionTracer, GameLogger

        ck = torch.load(cfg["ckpt"], map_location="cpu")
        net = LorcanaNet(d_model=ck["d_model"], n_layers=ck["layers"])
        net.load_state_dict(ck["model"])
        net.eval()
        rng = np.random.default_rng(int(cfg["seed"]))   # int seed (numpy requires int/None)
        sc = SearchConfig(simulations=cfg["sims"], depth_limit=cfg.get("depth", 10), temperature=1.0,
                          dirichlet_eps=0.25, batch_size=cfg["batch"])
        use_belief, n_worlds = cfg["use_belief"], cfg["n_worlds"]
        per = sc.simulations * (n_worlds if use_belief else 1)
        tracer = DecisionTracer(cfg.get("trace_dir", ""), f"r{cfg.get('round', 0)}-w{wid}",
                                enabled=cfg.get("trace", False))
        # EVERY worker logs its full play-by-play to its own file (so all games
        # are captured); only worker 0 ALSO streams live to the main process so
        # the terminal stays one coherent narrative.
        _tdir = cfg.get("trace_dir") or ""
        glog = GameLogger(enabled=cfg.get("watch", False),
                          file_path=os.path.join(_tdir, f"game-w{wid}.log") if _tdir else None,
                          sink=(lambda line: q.put(("log", line))) if wid == 0 else None)
        with LorcanaEngine(timeout=300) as engine:
            deck_ids = [d["id"] for d in engine.list_decks()]
            mcts = BISMCTS(engine, NetEvaluator(net), sc, rng)
            for g in range(cfg["games"]):
                i, j = rng.choice(len(deck_ids), 2, replace=False)
                p1, p2 = deck_ids[int(i)], deck_ids[int(j)]
                obs = engine.reset(f"{cfg['seed']}-{g}", p1, p2)
                glog.game_start(f"{cfg['seed']}-{g}", p1, p2)
                pending = []
                steps = 0
                stuck = same = 0
                last_fp = None
                aborted = False
                while not obs.get("done") and steps < 400:
                    legal = obs.get("legal", [])
                    actor = obs.get("actor")
                    if not legal:
                        break
                    if use_belief:
                        raise AssertionError("Tier-A guard must block belief-guided sample writing")
                    else:
                        res = mcts.run(obs)
                    dec_obs, dec_actor = obs, actor
                    a = int(rng.choice(len(res.pi), p=res.pi)) if res.pi.sum() > 0 else 0
                    result = engine.step(legal[a]["stableKey"])
                    obs = result["obs"]
                    if tracer.enabled:
                        tracer.log(dec_obs, res, a, game_id=f"{cfg['seed']}-{g}", executed=result)
                    if glog.enabled:
                        glog.decision(dec_obs, res, a, result)
                    clean = result.get("success", True) and (
                        result.get("matched", True) or result.get("executed") == "passTurn")
                    # skip mundane decisions (forced / single legal action): no
                    # policy signal, only buffer dilution.
                    trivial = len(legal) <= 1 or dec_obs.get("forced")
                    if clean and not trivial and dec_actor is not None and len(res.pi) == len(legal):
                        pending.append((encode_obs(dec_obs), res.pi.copy(), dec_actor, encode_belief(dec_obs),
                                        int(dec_obs.get("selfIdx", 0)), int(dec_obs.get("turn", 0))))
                    steps += 1
                    sims += per
                    decs += 1
                    if decs % 5 == 0:
                        q.put(("stat", wid, sims, decs, games))
                    # stuck-game guard (see run.py): abort + discard non-advancing games
                    fp = game_fingerprint(obs)
                    stuck, same, abort = stuck_step(stuck, same, fp == last_fp,
                                                    result.get("success", True))
                    last_fp = fp
                    if abort:
                        aborted = True
                        q.put(("error", wid, f"game {cfg['seed']}-{g} aborted: stuck at "
                                             f"turn {obs.get('turn')} (no progress) — discarded"))
                        break
                    if obs.get("turn", 0) > 35:   # turn cap: not a real terminal -> discard
                        aborted = True
                        q.put(("error", wid, f"game {cfg['seed']}-{g}: turn cap (turn>35) — discarded"))
                        break
                winner = obs.get("winner")
                glog.game_end("stuck" if aborted else winner, obs.get("turn"))
                games += 1
                samples = [] if aborted else [
                    (e, pi, (0.0 if winner is None else (1.0 if str(winner) == str(ac) else -1.0)),
                     b, aux_targets(obs, si, tn))
                    for (e, pi, ac, b, si, tn) in pending]
                q.put(("samples", samples))
                q.put(("game", wid, winner, p1, p2, mcts._invalid_leaves))
                q.put(("stat", wid, sims, decs, games))
    except Exception as e:  # surface worker crashes instead of hanging the round
        q.put(("error", wid, str(e)[:200]))
    finally:
        for _c in ("tracer", "glog"):
            try:
                locals()[_c].close()
            except (NameError, AttributeError, KeyError):
                pass
        q.put(("done", wid))


def generate_round_parallel(ckpt_path: str, n_actors: int, worker_cfg: dict,
                            monitor, base: dict, round_idx: int, base_buffer: int = 0):
    """Spawn n_actors workers for one round; stream progress to `monitor`; return
    (samples_as_tuples, round_sims, round_decs, round_games). `base` holds the
    monotonic cumulative totals across previous rounds (for live rate display);
    `base_buffer` is the persistent ReplayBuffer size so the live `buffer` figure
    reflects the cumulative buffer, not just this round's fresh samples."""
    require_tier_a_clean_label_belief_training_ready(
        use_belief=bool(worker_cfg["use_belief"]),
        context="training.parallel.generate_round_parallel")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for wid in range(n_actors):
        cfg = dict(worker_cfg)
        cfg["wid"] = wid
        cfg["seed"] = round_idx * 1000 + wid   # int (numpy default_rng needs int/None)
        cfg["ckpt"] = ckpt_path
        p = ctx.Process(target=_worker, args=(cfg, q), daemon=True)
        p.start()
        procs.append(p)

    per_worker: dict[int, tuple] = {}     # wid -> (sims, decs, games)
    invalid_by_worker: dict[int, int] = {}  # wid -> cumulative search invalid leaves
    samples: list = []
    done: set = set()
    stall_timeout = max(2 * int(worker_cfg.get("engine_timeout", 300)), 600)
    last_progress = time.time()
    try:
        while len(done) < n_actors:
            try:
                msg = q.get(timeout=1.0)
            except _queue.Empty:
                # liveness guard: a crashed worker can't hang the round forever
                if not any(p.is_alive() for p in procs):
                    break
                # watchdog: a worker hung *outside* an IPC (the bridge's own RPC
                # timeout handles in-IPC hangs) must not stall the round forever.
                if time.time() - last_progress > stall_timeout:
                    monitor.event(f"round watchdog: no progress for {stall_timeout}s — "
                                  f"terminating {sum(p.is_alive() for p in procs)} stuck worker(s)")
                    break
                continue
            last_progress = time.time()
            kind = msg[0]
            if kind == "stat":
                _, wid, s, d, g = msg
                per_worker[wid] = (s, d, g)
            elif kind == "samples":
                samples.extend(msg[1])
            elif kind == "game":
                _, wid, winner, p1, p2, invalid = msg
                invalid_by_worker[wid] = invalid
                monitor.update(last=f"w{wid} {p1[:10]} v {p2[:10]} -> {winner or 'draw'}")
            elif kind == "error":
                monitor.event(f"worker {msg[1]} error: {msg[2]}")
            elif kind == "log":           # live play-by-play from the focus worker
                print(msg[1], flush=True)
            elif kind == "done":
                done.add(msg[1])
            rs = sum(v[0] for v in per_worker.values())
            rd = sum(v[1] for v in per_worker.values())
            rg = sum(v[2] for v in per_worker.values())
            monitor.update(sims=base["sims"] + rs, decisions=base["decisions"] + rd,
                           games=base["games"] + rg, buffer=base_buffer + len(samples))
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    invalid_total = sum(invalid_by_worker.values())
    if invalid_total:
        monitor.event(f"search exact-exec mismatches this round: {invalid_total} "
                      f"(invalid leaves dropped — should be ~0)")
    rs = sum(v[0] for v in per_worker.values())
    rd = sum(v[1] for v in per_worker.values())
    rg = sum(v[2] for v in per_worker.values())
    return samples, rs, rd, rg
