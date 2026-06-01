"""All-in-one training entrypoint: bigger net on GPU + batched inference +
belief-guided self-play + a live monitor + periodic eval — one command.

    python lorcana-bot/training/run.py            # sensible defaults (~13M net)
    python lorcana-bot/training/run.py --help

Parallel by default (`--actors > 1`): spawn-isolated CPU workers (training/parallel)
run batched belief self-play and stream samples to the GPU learner here in the
main process. `--actors 1` runs everything single-process. Ctrl-C saves cleanly.
This is the throughput-oriented self-play trainer; `training/league_train.py` is
the Phase-3 PFSP/league trainer (`./train.sh league ...`).
"""

from __future__ import annotations

import os
import sys

# Reduce CUDA allocator fragmentation BEFORE importing torch: on long runs the
# reserved pool crept to the 8GB cap (8.5/8.6GB) and spilled to slow host memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# make `engine`, `network`, `search`, `training` importable from any cwd
_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT not in sys.path:
    sys.path.insert(0, _BOT)

import argparse
import time

import numpy as np
import torch

from engine.bridge import LorcanaEngine
from engine.serialization import encode_obs, encode_belief, aux_targets, game_fingerprint, stuck_step
from network.model import LorcanaNet
from search.ismcts import BISMCTS, SearchConfig
from search.evaluator import NetEvaluator
from training.learner import Learner, ReplayBuffer, Sample
from training.monitor import LiveMonitor
from training.trace import DecisionTracer, GameLogger
from training.exploitability import deck_pair, gauntlet
from training.league import NetPlayer, ScriptedPlayer
from training.tier_a_guard import require_tier_a_clean_label_belief_training_ready


def _outcome(winner, actor) -> float:
    if winner is None:
        return 0.0
    return 1.0 if str(winner) == str(actor) else -1.0


def selfplay_game(engine, net, cfg, seed, rng, mon, deck_ids, use_belief, n_worlds,
                  tracer=None, logger=None, full_ismcts=False):
    require_tier_a_clean_label_belief_training_ready(
        use_belief=use_belief, context="training.run.selfplay_game")
    mcts = BISMCTS(engine, NetEvaluator(net), cfg, rng)
    p1, p2 = deck_pair(rng, deck_ids)
    obs = engine.reset(seed, p1, p2)
    if logger is not None:
        logger.game_start(str(seed), p1, p2)
    pending: list[tuple] = []
    steps = 0
    stuck = same = 0
    last_fp = None
    per_decision = cfg.simulations * (n_worlds if use_belief else 1)
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
        if tracer is not None:
            tracer.log(dec_obs, res, a, game_id=str(seed), executed=result)
        if logger is not None:
            logger.decision(dec_obs, res, a, result)
        steps += 1
        mon.add(sims=per_decision, decisions=1)
        # stuck-game guard: a state that won't advance (zero-candidate / pass
        # rejected) would otherwise loop to the 400-step cap, stalling the worker
        # and poisoning the buffer. Trip on repeated FAILED steps on a frozen
        # state (precise — won't fire on legit successful sequences), with a high
        # pure-repeat backstop for the rare "succeeds but doesn't advance" case.
        fp = game_fingerprint(obs)
        stuck, same, abort = stuck_step(stuck, same, fp == last_fp, result.get("success", True))
        last_fp = fp
        if abort:
            mon.event(f"game {seed} aborted: no progress (stuck at turn "
                      f"{obs.get('turn')} {obs.get('phase')}) — discarding")
            return [], "stuck", p1, p2
        # turn cap: a game neither side can close (grinds to turn 40+) wastes huge
        # compute and stalls the round. It is NOT a real terminal, so discard it
        # rather than train value/aux from the cutoff (consistent with stuck games).
        if obs.get("turn", 0) > 35:
            mon.event(f"game {seed}: turn cap (turn>35, no side closed) — discarding")
            return [], "turncap", p1, p2
        # only record the sample if the engine executed exactly the chosen action
        # (a failed/fallback execution would attach the target to a different line)
        clean = result.get("success", True) and (
            result.get("matched", True) or result.get("executed") == "passTurn")
        # skip mundane decisions: a forced state or a single legal action teaches
        # the policy nothing (the mask already determines the move) and only
        # dilutes the buffer / biases the prior toward trivial lines.
        trivial = len(legal) <= 1 or dec_obs.get("forced")
        if clean and not trivial and dec_actor is not None and len(res.pi) == len(legal):
            pending.append((encode_obs(dec_obs), res.pi.copy(), dec_actor, encode_belief(dec_obs),
                            int(dec_obs.get("selfIdx", 0)), int(dec_obs.get("turn", 0))))
    winner = obs.get("winner")
    if logger is not None:
        logger.game_end(winner, obs.get("turn"))
    if mcts._invalid_leaves:
        mon.event(f"search exact-exec mismatches this game: {mcts._invalid_leaves} (should be ~0)")
    samples = [Sample(enc=e, pi=pi, z=_outcome(winner, ac), belief=b,
                      aux=aux_targets(obs, si, tn)) for (e, pi, ac, b, si, tn) in pending]
    return samples, winner, p1, p2


def main() -> None:
    ap = argparse.ArgumentParser(description="Lorcana bot self-play + training (GPU, batched, monitored)")
    ap.add_argument("--init", type=str, default="", help="resume from checkpoint (defines arch)")
    ap.add_argument("--d-model", type=int, default=256,
                    help="trunk width. 256x4 (~3M) is CPU-actor-tractable; the old 512x8 (16M) "
                         "made each decision ~14s and starved learning.")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=8, help="MCTS sims per (world)")
    ap.add_argument("--depth", type=int, default=4,
                    help="search horizon in decision plies. Higher = stronger but slower.")
    ap.add_argument("--batch", type=int, default=16, help="leaf-batch / virtual-loss wave size")
    ap.add_argument("--n-worlds", type=int, default=2, help="belief determinizations per decision")
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--full-ismcts", action="store_true",
                    help="Reserved during Tier-A remediation. Belief-guided clean-label "
                         "training remains blocked until the release gate passes.")
    ap.add_argument("--actors", type=int, default=6,
                    help="parallel self-play worker processes (1 = single-process)")
    ap.add_argument("--games-per-actor", type=int, default=1, help="games each actor plays per round")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-batch", type=int, default=64,
                    help="learner minibatch (8GB-safe; the candidate transformer is row-chunked "
                         "+ gradient-checkpointed so high-candidate states don't OOM)")
    ap.add_argument("--rounds", type=int, default=0, help="0 = run forever (Ctrl-C to stop)")
    ap.add_argument("--bc-games", type=int, default=6, help="fair behaviour-clone warmup games (0 to skip)")
    ap.add_argument("--eval-every", type=int, default=5, help="rounds between gauntlet evals (0 = never)")
    ap.add_argument("--trace", action="store_true",
                    help="write per-decision JSONL traces (candidate priors/visits/Q) to <out>/traces/")
    ap.add_argument("--watch", action="store_true",
                    help="stream a live, human-readable play-by-play (actions/turns/decisions). "
                         "Cleanest with --actors 1; in parallel only worker 0's game is shown.")
    ap.add_argument("--out", type=str, default=os.path.join(_BOT, "checkpoints", "run.pt"))
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    use_belief = not args.no_belief
    require_tier_a_clean_label_belief_training_ready(
        use_belief=use_belief, context="training.run.main")

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
    cfg = SearchConfig(simulations=args.sims, depth_limit=args.depth, temperature=1.0,
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
    trace_dir = os.path.join(os.path.dirname(args.out), "traces")
    # single-process tracer; parallel workers each open their own (see parallel.py)
    tracer = DecisionTracer(trace_dir, f"run{int(time.time())}",
                            enabled=args.trace and args.actors <= 1)
    if args.trace:
        mon.event(f"decision tracing -> {trace_dir}/")
    # live play-by-play. Single-process prints + logs to a file; parallel workers
    # each log to their own file (ALL games captured) and worker 0 also streams live.
    glog = GameLogger(enabled=args.watch and args.actors <= 1, to_stdout=True,
                      file_path=os.path.join(trace_dir, "game-main.log"))
    if args.watch:
        mon.event(f"watch mode: live play-by-play (worker 0) + per-worker logs in {trace_dir}/game-w*.log")

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
                        st = learner.train_epoch(buf, batch_size=args.train_batch)
                        mon.update(loss=st.get("loss"), losses=st)
                mon.event(f"bootstrap done: {len(buf)} samples, loss={st.get('loss', 0):.3f}")
                save()

            # --- self-play + train loop ---
            rounds_done = 0
            while args.rounds == 0 or rounds_done < args.rounds:
                mon.update(phase="selfplay", round=rounds_done + 1)
                # anchor the KL trust region to the round-start policy (activates c_kl)
                learner.set_reference()
                if args.actors > 1:
                    # parallel: workers reload these weights, stream progress live
                    save_to(work_ckpt)
                    from training.parallel import generate_round_parallel
                    wcfg = {"bot_root": _BOT, "sims": args.sims, "batch": args.batch,
                            "n_worlds": args.n_worlds, "use_belief": use_belief,
                            "games": args.games_per_actor, "depth": args.depth,
                            "trace": args.trace, "trace_dir": trace_dir, "round": rounds_done,
                            "watch": args.watch, "full_ismcts": args.full_ismcts}
                    raw, rs, rd, rg = generate_round_parallel(work_ckpt, args.actors, wcfg,
                                                              mon, base, rounds_done,
                                                              base_buffer=len(buf))
                    base["sims"] += rs; base["decisions"] += rd; base["games"] += rg
                    for (e, pi, z, b, aux) in raw:
                        buf.add(Sample(enc=e, pi=pi, z=z, belief=b, aux=aux))
                    mon.update(buffer=len(buf))
                else:
                    for g in range(args.games_per_actor):
                        s, winner, p1, p2 = selfplay_game(engine, net, cfg, f"sp{rounds_done}-{g}",
                                                          rng, mon, deck_ids, use_belief, args.n_worlds,
                                                          tracer=tracer, logger=glog,
                                                          full_ismcts=args.full_ismcts)
                        buf.extend(s)
                        mon.add(games=1)
                        mon.update(buffer=len(buf), last=f"{p1[:12]} v {p2[:12]} -> {winner or 'draw'}")
                mon.update(phase="train")
                st = {}
                for _ in range(args.epochs):
                    if len(buf):
                        st = learner.train_epoch(buf, batch_size=args.train_batch)
                mon.update(loss=st.get("loss"), losses=st or None)
                rounds_done += 1
                save()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()   # release per-round cached blocks (anti-creep)
                if args.eval_every and rounds_done % args.eval_every == 0:
                    mon.update(phase="eval")
                    # release cached training blocks so the eval search (belief, GPU)
                    # doesn't stack on top of them and spill to slow host memory.
                    learner._reference = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    evalp = NetPlayer("main", net, SearchConfig(simulations=args.sims, depth_limit=3,
                                      batch_size=min(args.batch, 16)), record=False, use_belief=use_belief,
                                      n_worlds=args.n_worlds, greedy=True, rng=rng)
                    anchors = [ScriptedPlayer("fairBest", "fairBest"),
                               ScriptedPlayer("fairAggro", "fairAggro")]
                    wr = gauntlet(engine, evalp, anchors, n_games_per=2, rng=rng,
                                  seed_prefix=f"eval{rounds_done}", deck_ids=deck_ids)
                    mon.update(winrate=wr["_overall"], wr_round=rounds_done)
                    mon.event(f"round {rounds_done} gauntlet vs fair anchors: "
                              f"overall {wr['_overall']:.0%} {dict((k, round(v,2)) for k,v in wr.items() if k!='_overall')}")
    except KeyboardInterrupt:
        mon.event("interrupted — saving checkpoint")
    finally:
        tracer.close()
        glog.close()
        save()
        mon.stop()
        print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
