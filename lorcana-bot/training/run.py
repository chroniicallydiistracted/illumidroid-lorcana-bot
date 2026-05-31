"""All-in-one training entrypoint: bigger net on GPU + batched inference +
belief-guided self-play + a live monitor + periodic eval — one command.

    python lorcana-bot/training/run.py            # sensible defaults (~13M net)
    python lorcana-bot/training/run.py --help

Runs single-process (one Bun engine, net on GPU) so the GPU is actually used for
both batched leaf inference and the learner. Ctrl-C saves and exits cleanly.
"""

from __future__ import annotations

import os
import sys

# make `engine`, `network`, `search`, `training` importable from any cwd
_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT not in sys.path:
    sys.path.insert(0, _BOT)

import argparse
import time

import numpy as np
import torch

from engine.bridge import LorcanaEngine
from engine.serialization import encode_obs, encode_belief
from network.model import LorcanaNet
from search.ismcts import BISMCTS, SearchConfig
from search.evaluator import NetEvaluator, BeliefEvaluator
from training.learner import Learner, ReplayBuffer, Sample
from training.monitor import LiveMonitor
from training.exploitability import deck_pair, gauntlet
from training.league import NetPlayer, ScriptedPlayer


def _outcome(winner, actor) -> float:
    if winner is None:
        return 0.0
    return 1.0 if str(winner) == str(actor) else -1.0


def selfplay_game(engine, net, cfg, seed, rng, mon, deck_ids, use_belief, n_worlds):
    mcts = BISMCTS(engine, NetEvaluator(net), cfg, rng)
    belief = BeliefEvaluator(net) if use_belief else None
    p1, p2 = deck_pair(rng, deck_ids)
    obs = engine.reset(seed, p1, p2)
    pending: list[tuple] = []
    steps = 0
    per_decision = cfg.simulations * (n_worlds if use_belief else 1)
    while not obs.get("done") and steps < 400:
        legal = obs.get("legal", [])
        actor = obs.get("actor")
        if not legal:
            break
        if use_belief:
            res = mcts.run_belief(obs, belief, n_worlds=n_worlds, sims_per_world=cfg.simulations)
        else:
            res = mcts.run(obs)
        if actor is not None and len(res.pi) == len(legal):
            pending.append((encode_obs(obs), res.pi.copy(), actor, encode_belief(obs)))
        a = int(rng.choice(len(res.pi), p=res.pi)) if res.pi.sum() > 0 else 0
        obs = engine.step(legal[a]["stableKey"])["obs"]
        steps += 1
        mon.add(sims=per_decision, decisions=1)
    winner = obs.get("winner")
    samples = [Sample(enc=e, pi=pi, z=_outcome(winner, ac), belief=b) for (e, pi, ac, b) in pending]
    return samples, winner, p1, p2


def main() -> None:
    ap = argparse.ArgumentParser(description="Lorcana bot self-play + training (GPU, batched, monitored)")
    ap.add_argument("--init", type=str, default="", help="resume from checkpoint (defines arch)")
    ap.add_argument("--d-model", type=int, default=512, help="trunk width (default 512 ~13M params)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--sims", type=int, default=16, help="MCTS sims per (world)")
    ap.add_argument("--batch", type=int, default=32, help="leaf-batch / virtual-loss wave size")
    ap.add_argument("--n-worlds", type=int, default=4, help="belief determinizations per decision")
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--actors", type=int, default=6,
                    help="parallel self-play worker processes (1 = single-process)")
    ap.add_argument("--games-per-actor", type=int, default=1, help="games each actor plays per round")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=0, help="0 = run forever (Ctrl-C to stop)")
    ap.add_argument("--bc-games", type=int, default=6, help="fair behaviour-clone warmup games (0 to skip)")
    ap.add_argument("--eval-every", type=int, default=5, help="rounds between gauntlet evals (0 = never)")
    ap.add_argument("--out", type=str, default=os.path.join(_BOT, "checkpoints", "run.pt"))
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    use_belief = not args.no_belief

    # net: resume defines arch; otherwise build the requested net
    d_model, layers = args.d_model, args.layers
    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location="cpu")
        d_model, layers = ck.get("d_model", d_model), ck.get("layers", layers)
        net = LorcanaNet(d_model=d_model, n_layers=layers)
        net.load_state_dict(ck["model"])
        resumed = True
    else:
        net = LorcanaNet(d_model=d_model, n_layers=layers)
        resumed = False
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())

    learner = Learner(net, lr=1e-3, c_kl=0.5, c_entropy=0.01, device=device)
    cfg = SearchConfig(simulations=args.sims, depth_limit=4, temperature=1.0,
                       dirichlet_eps=0.25, batch_size=args.batch)
    games_per_round = max(1, args.actors) * args.games_per_actor
    mon = LiveMonitor(interval=3.0, total_games=(args.rounds * games_per_round or None)).start()
    mon.event(f"net d{d_model}x{layers} = {n_params/1e6:.1f}M params on {device} | "
              f"actors={args.actors} sims={args.sims} batch={args.batch} "
              f"worlds={args.n_worlds if use_belief else 0} belief={use_belief} | "
              f"{'resumed' if resumed else 'fresh'}")
    work_ckpt = os.path.join(os.path.dirname(args.out), "_work.pt")
    base = {"sims": 0, "decisions": 0, "games": 0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng()
    buf = ReplayBuffer()

    def save_to(path):
        torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()},
                    "d_model": d_model, "layers": layers}, path)

    def save():
        save_to(args.out)

    try:
        with LorcanaEngine(timeout=300) as engine:
            deck_ids = [d["id"] for d in engine.list_decks()]
            mon.event(f"engine up: {len(deck_ids)} real decks")

            # --- optional fair behaviour-clone warmup (gives a non-random prior) ---
            if not resumed and args.bc_games > 0:
                from training.bootstrap import generate_game
                mon.update(phase="bootstrap")
                for g in range(args.bc_games):
                    p1, p2 = deck_pair(rng, deck_ids)
                    s = generate_game(engine, f"bc-{g}", "best", 400, p1, p2)  # 'best' is FAIR
                    buf.extend(s)
                    mon.add(games=1, decisions=len(s))
                    mon.update(buffer=len(buf), last=f"bc {g+1}/{args.bc_games}")
                st = {}
                for _ in range(4):
                    if len(buf):
                        st = learner.train_epoch(buf, batch_size=256)
                        mon.update(loss=st.get("loss"))
                mon.event(f"bootstrap done: {len(buf)} samples, loss={st.get('loss', 0):.3f}")
                save()

            # --- self-play + train loop ---
            rounds_done = 0
            while args.rounds == 0 or rounds_done < args.rounds:
                mon.update(phase="selfplay", round=rounds_done + 1)
                if args.actors > 1:
                    # parallel: workers reload these weights, stream progress live
                    save_to(work_ckpt)
                    from training.parallel import generate_round_parallel
                    wcfg = {"bot_root": _BOT, "sims": args.sims, "batch": args.batch,
                            "n_worlds": args.n_worlds, "use_belief": use_belief,
                            "games": args.games_per_actor}
                    raw, rs, rd, rg = generate_round_parallel(work_ckpt, args.actors, wcfg,
                                                              mon, base, rounds_done)
                    base["sims"] += rs; base["decisions"] += rd; base["games"] += rg
                    for (e, pi, z, b) in raw:
                        buf.add(Sample(enc=e, pi=pi, z=z, belief=b))
                    mon.update(buffer=len(buf))
                else:
                    for g in range(args.games_per_actor):
                        s, winner, p1, p2 = selfplay_game(engine, net, cfg, f"sp{rounds_done}-{g}",
                                                          rng, mon, deck_ids, use_belief, args.n_worlds)
                        buf.extend(s)
                        mon.add(games=1)
                        mon.update(buffer=len(buf), last=f"{p1[:12]} v {p2[:12]} -> {winner or 'draw'}")
                mon.update(phase="train")
                st = {}
                for _ in range(args.epochs):
                    if len(buf):
                        st = learner.train_epoch(buf, batch_size=256)
                mon.update(loss=st.get("loss"))
                rounds_done += 1
                save()
                if args.eval_every and rounds_done % args.eval_every == 0:
                    mon.update(phase="eval")
                    evalp = NetPlayer("main", net, SearchConfig(simulations=args.sims, depth_limit=3,
                                      batch_size=args.batch), record=False, use_belief=use_belief,
                                      n_worlds=args.n_worlds, greedy=True, rng=rng)
                    anchors = [ScriptedPlayer("fairBest", "fairBest"),
                               ScriptedPlayer("fairAggro", "fairAggro")]
                    wr = gauntlet(engine, evalp, anchors, n_games_per=2, rng=rng,
                                  seed_prefix=f"eval{rounds_done}", deck_ids=deck_ids)
                    mon.update(winrate=wr["_overall"])
                    mon.event(f"round {rounds_done} gauntlet vs fair anchors: "
                              f"overall {wr['_overall']:.0%} {dict((k, round(v,2)) for k,v in wr.items() if k!='_overall')}")
    except KeyboardInterrupt:
        mon.event("interrupted — saving checkpoint")
    finally:
        save()
        mon.stop()
        print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
