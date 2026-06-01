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

import numpy as np


def _worker(cfg: dict, q) -> None:
    wid = cfg["wid"]
    sims = decs = games = 0
    try:
        import sys
        sys.path.insert(0, cfg["bot_root"])
        import torch
        torch.set_num_threads(1)
        from engine.bridge import LorcanaEngine
        from engine.serialization import encode_obs, encode_belief
        from network.model import LorcanaNet
        from search.ismcts import BISMCTS, SearchConfig
        from search.evaluator import NetEvaluator, BeliefEvaluator
        from search.belief_filter import BeliefTracker
        from training.trace import DecisionTracer

        ck = torch.load(cfg["ckpt"], map_location="cpu")
        net = LorcanaNet(d_model=ck["d_model"], n_layers=ck["layers"])
        net.load_state_dict(ck["model"])
        net.eval()
        rng = np.random.default_rng(int(cfg["seed"]))   # int seed (numpy requires int/None)
        sc = SearchConfig(simulations=cfg["sims"], depth_limit=4, temperature=1.0,
                          dirichlet_eps=0.25, batch_size=cfg["batch"])
        use_belief, n_worlds = cfg["use_belief"], cfg["n_worlds"]
        per = sc.simulations * (n_worlds if use_belief else 1)
        tracer = DecisionTracer(cfg.get("trace_dir", ""), f"r{cfg.get('round', 0)}-w{wid}",
                                enabled=cfg.get("trace", False))
        with LorcanaEngine(timeout=300) as engine:
            deck_ids = [d["id"] for d in engine.list_decks()]
            mcts = BISMCTS(engine, NetEvaluator(net), sc, rng)
            belief = BeliefEvaluator(net) if use_belief else None
            for g in range(cfg["games"]):
                i, j = rng.choice(len(deck_ids), 2, replace=False)
                p1, p2 = deck_ids[int(i)], deck_ids[int(j)]
                obs = engine.reset(f"{cfg['seed']}-{g}", p1, p2)
                tracker = BeliefTracker(rng=rng) if use_belief else None
                pending = []
                steps = 0
                while not obs.get("done") and steps < 400:
                    legal = obs.get("legal", [])
                    actor = obs.get("actor")
                    if not legal:
                        break
                    res = (mcts.run_belief(obs, belief, n_worlds=n_worlds,
                                           sims_per_world=sc.simulations, tracker=tracker)
                           if use_belief else mcts.run(obs))
                    dec_obs, dec_actor = obs, actor
                    a = int(rng.choice(len(res.pi), p=res.pi)) if res.pi.sum() > 0 else 0
                    result = engine.step(legal[a]["stableKey"])
                    obs = result["obs"]
                    if tracer.enabled:
                        tracer.log(dec_obs, res, a, game_id=f"{cfg['seed']}-{g}", executed=result)
                    clean = result.get("success", True) and (
                        result.get("matched", True) or result.get("executed") == "passTurn")
                    # skip mundane decisions (forced / single legal action): no
                    # policy signal, only buffer dilution.
                    trivial = len(legal) <= 1 or dec_obs.get("forced")
                    if clean and not trivial and dec_actor is not None and len(res.pi) == len(legal):
                        pending.append((encode_obs(dec_obs), res.pi.copy(), dec_actor, encode_belief(dec_obs)))
                    steps += 1
                    sims += per
                    decs += 1
                    if decs % 5 == 0:
                        q.put(("stat", wid, sims, decs, games))
                winner = obs.get("winner")
                games += 1
                samples = [(e, pi, (0.0 if winner is None else (1.0 if str(winner) == str(ac) else -1.0)), b)
                           for (e, pi, ac, b) in pending]
                q.put(("samples", samples))
                q.put(("game", wid, winner, p1, p2))
                q.put(("stat", wid, sims, decs, games))
    except Exception as e:  # surface worker crashes instead of hanging the round
        q.put(("error", wid, str(e)[:200]))
    finally:
        try:
            tracer.close()
        except (NameError, AttributeError):
            pass
        q.put(("done", wid))


def generate_round_parallel(ckpt_path: str, n_actors: int, worker_cfg: dict,
                            monitor, base: dict, round_idx: int, base_buffer: int = 0):
    """Spawn n_actors workers for one round; stream progress to `monitor`; return
    (samples_as_tuples, round_sims, round_decs, round_games). `base` holds the
    monotonic cumulative totals across previous rounds (for live rate display);
    `base_buffer` is the persistent ReplayBuffer size so the live `buffer` figure
    reflects the cumulative buffer, not just this round's fresh samples."""
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
    samples: list = []
    done: set = set()
    try:
        while len(done) < n_actors:
            try:
                msg = q.get(timeout=1.0)
            except _queue.Empty:
                # liveness guard: a crashed worker can't hang the round forever
                if not any(p.is_alive() for p in procs):
                    break
                continue
            kind = msg[0]
            if kind == "stat":
                _, wid, s, d, g = msg
                per_worker[wid] = (s, d, g)
            elif kind == "samples":
                samples.extend(msg[1])
            elif kind == "game":
                _, wid, winner, p1, p2 = msg
                monitor.update(last=f"w{wid} {p1[:10]} v {p2[:10]} -> {winner or 'draw'}")
            elif kind == "error":
                monitor.event(f"worker {msg[1]} error: {msg[2]}")
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
    rs = sum(v[0] for v in per_worker.values())
    rd = sum(v[1] for v in per_worker.values())
    rg = sum(v[2] for v in per_worker.values())
    return samples, rs, rd, rg
