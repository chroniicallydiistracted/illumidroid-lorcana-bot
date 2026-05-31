"""Throughput — parallel self-play actors + GPU learner (Podracer-synchronous).

Why this shape: an actor is *engine-bound* (~100+ ms per Bun game step) while a
net forward is ~1 ms, so a central GPU inference server would barely speed actors
up — the real win is running K actors across the CPU cores. The GPU accelerates
the LEARNER (large batches / fast epochs) and is where bigger nets would pay off.

Each round: spawn K actor processes (CUDA-free; local CPU net loaded from the
latest checkpoint) → they self-play games and return samples → the learner trains
on GPU → save checkpoint → next round (actors reload it). A central batched GPU
inference server (RemoteEvaluator) is the documented next step if/when the net
grows enough to make inference the bottleneck.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np


# ---- actor worker (runs in its own process; CPU only, owns a Bun engine) ----
def _actor_worker(cfg: dict) -> list:
    # IMPORTANT: stay off CUDA in actors. Net runs on CPU; inference is cheap
    # relative to the engine. Torch threads pinned to 1 so K actors don't fight.
    import torch
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0, cfg["bot_root"])
    from engine.bridge import LorcanaEngine
    from network.model import LorcanaNet
    from search.ismcts import SearchConfig
    from training.selfplay import play_game

    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    net = LorcanaNet(d_model=ckpt["d_model"], n_layers=ckpt["layers"])
    net.load_state_dict(ckpt["model"])
    net.eval()

    rng = np.random.default_rng(cfg["seed"])
    sc = SearchConfig(simulations=cfg["sims"], depth_limit=cfg["depth"], temperature=1.0,
                      dirichlet_eps=0.25)
    samples: list = []
    decisions = 0
    with LorcanaEngine(timeout=300) as engine:
        deck_ids = [d["id"] for d in engine.list_decks()]
        for g in range(cfg["games"]):
            p1 = p2 = None
            if len(deck_ids) >= 2:
                p1, p2 = (deck_ids[int(i)] for i in rng.choice(len(deck_ids), 2, replace=False))
            gs = play_game(engine, net, sc, seed=f"{cfg['seed']}-{g}", rng=rng,
                           use_belief=cfg["use_belief"], n_worlds=cfg["n_worlds"],
                           deck_p1=p1, deck_p2=p2)
            samples.extend(gs)
            decisions += len(gs)
    return [(s.enc, s.pi, s.z, s.belief) for s in samples]  # picklable tuples


def _spawn_round(pool_ctx, n_actors: int, cfg_base: dict, round_idx: int) -> list:
    cfgs = []
    for a in range(n_actors):
        c = dict(cfg_base)
        c["seed"] = f"r{round_idx}a{a}"
        cfgs.append(c)
    with pool_ctx.Pool(processes=n_actors) as pool:
        results = pool.map(_actor_worker, cfgs)
    flat = []
    for r in results:
        flat.extend(r)
    return flat


def run_distributed(init_ckpt: str, rounds: int = 3, actors: int = 8,
                    games_per_actor: int = 1, sims: int = 8, depth: int = 4,
                    n_worlds: int = 4, use_belief: bool = True, epochs: int = 2,
                    out: str = "lorcana-bot/checkpoints/dist.pt", verbose: bool = True) -> dict:
    import torch
    from network.model import LorcanaNet
    from training.learner import Learner, ReplayBuffer, Sample

    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # working checkpoint actors read each round
    ck = torch.load(init_ckpt, map_location="cpu")
    d_model, layers = ck["d_model"], ck["layers"]
    work_ckpt = os.path.join(bot_root, "checkpoints", "_dist_work.pt")
    os.makedirs(os.path.dirname(work_ckpt), exist_ok=True)
    torch.save(ck, work_ckpt)

    net = LorcanaNet(d_model=d_model, n_layers=layers).to(device)
    net.load_state_dict(ck["model"])
    learner = Learner(net, lr=1e-3, c_kl=0.5, c_entropy=0.01, device=torch.device(device))

    ctx = mp.get_context("spawn")
    cfg_base = {"bot_root": bot_root, "ckpt": work_ckpt, "games": games_per_actor,
                "sims": sims, "depth": depth, "n_worlds": n_worlds,
                "use_belief": use_belief}
    history = []
    for r in range(rounds):
        learner.set_reference()
        t0 = time.time()
        raw = _spawn_round(ctx, actors, cfg_base, r)
        gen_dt = time.time() - t0
        buf = ReplayBuffer()
        for enc, pi, z, belief in raw:
            buf.add(Sample(enc=enc, pi=pi, z=z, belief=belief))
        stats = {}
        t1 = time.time()
        for _ in range(epochs):
            if len(buf):
                stats = learner.train_epoch(buf, batch_size=256)
        train_dt = time.time() - t1
        # broadcast new weights to actors (CPU state dict)
        torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()},
                    "d_model": d_model, "layers": layers}, work_ckpt)
        dps = len(buf) / gen_dt if gen_dt > 0 else 0.0
        rec = {"round": r, "samples": len(buf), "gen_s": round(gen_dt, 1),
               "train_s": round(train_dt, 1), "decisions_per_s": round(dps, 2),
               "loss": stats.get("loss")}
        history.append(rec)
        if verbose:
            print(f"[dist] round {r+1}/{rounds}: {len(buf)} samples in {gen_dt:.1f}s "
                  f"({dps:.2f} dec/s, {actors} actors) | train {train_dt:.1f}s "
                  f"loss={rec['loss']}", flush=True)

    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()},
                "d_model": d_model, "layers": layers}, out)
    if verbose:
        print(f"[dist] saved -> {out} (device={device})", flush=True)
    return {"history": history, "device": device, "actors": actors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="lorcana-bot/checkpoints/bc_fair.pt")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--actors", type=int, default=8)
    ap.add_argument("--games-per-actor", type=int, default=1)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--n-worlds", type=int, default=4)
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--out", type=str, default="lorcana-bot/checkpoints/dist.pt")
    args = ap.parse_args()
    run_distributed(init_ckpt=args.init, rounds=args.rounds, actors=args.actors,
                    games_per_actor=args.games_per_actor, sims=args.sims,
                    n_worlds=args.n_worlds, use_belief=not args.no_belief, out=args.out)


if __name__ == "__main__":
    main()
