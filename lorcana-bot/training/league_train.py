"""Phase 3 — league / PSRO training orchestrator.

Trains a main agent against a population (scripted anchors + frozen past mains)
with PFSP opponent sampling, KL-trust-region + entropy anti-collapse, periodic
freezing into the league, and Elo tracking. Optionally runs an exploitability
proxy (a trained main-exploiter). See architecture doc §5.1.

Engine throughput makes full convergence a compute job; the defaults here run a
small end-to-end proof. Scale via --iterations/--games/--sims.
"""

from __future__ import annotations

import argparse
import os
import sys

# make engine/network/search/training importable when run as a script
_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT not in sys.path:
    sys.path.insert(0, _BOT)

import numpy as np

from engine.bridge import LorcanaEngine
from network.model import LorcanaNet
from search.ismcts import SearchConfig
from training.learner import Learner, ReplayBuffer
from training.league import League, NetPlayer, ScriptedPlayer, frozen_net_copy, SEAT_ONE, SEAT_TWO
from training.match import play_match
from training.exploitability import gauntlet, exploitability_proxy, deck_pair
from training.tier_a_guard import require_tier_a_clean_label_belief_training_ready


def build_main_net(init: str, d_model: int, layers: int):
    import torch
    if init and os.path.exists(init):
        ckpt = torch.load(init, map_location="cpu")
        net = LorcanaNet(d_model=ckpt.get("d_model", d_model), n_layers=ckpt.get("layers", layers))
        net.load_state_dict(ckpt["model"])
        return net, ckpt.get("d_model", d_model), ckpt.get("layers", layers)
    return LorcanaNet(d_model=d_model, n_layers=layers), d_model, layers


def run_league(init: str = "", iterations: int = 3, games_per_iter: int = 6,
               epochs: int = 2, freeze_every: int = 2, sims: int = 12,
               n_worlds: int = 4, d_model: int = 96, layers: int = 2,
               c_kl: float = 0.5, c_entropy: float = 0.01, use_belief: bool = True,
               run_exploit: bool = False, out: str = "lorcana-bot/checkpoints/league.pt",
               verbose: bool = True) -> dict:
    require_tier_a_clean_label_belief_training_ready(
        use_belief=use_belief, context="training.league_train.run_league")
    rng = np.random.default_rng(0)
    net, d_model, layers = build_main_net(init, d_model, layers)
    learner = Learner(net, lr=1e-3, c_kl=c_kl, c_entropy=c_entropy)
    cfg = SearchConfig(simulations=sims, depth_limit=4, temperature=1.0)
    cfg_opp = SearchConfig(simulations=max(4, sims // 2), depth_limit=3, temperature=0.5)

    league = League(rng=rng)
    main_id = "main"
    main = NetPlayer(main_id, net, cfg, record=True, use_belief=use_belief,
                     n_worlds=n_worlds, rng=rng)
    league.register(main, "main")
    # FAIR anchors only — oracle strategies would make the main learn against /
    # from cheating play (skewed value targets). Archetype-diverse for §5.1.
    league.register(ScriptedPlayer("anchor_fair_best", "fairBest"), "anchor")
    league.register(ScriptedPlayer("anchor_fair_aggro", "fairAggro"), "anchor")
    league.register(ScriptedPlayer("anchor_fair_control", "fairControl"), "anchor")
    # seed the past-main pool with the initial weights
    league.register(NetPlayer("past_0", frozen_net_copy(net), cfg_opp, record=False,
                              use_belief=use_belief, n_worlds=max(2, n_worlds // 2),
                              greedy=True, rng=rng), "past")

    history = []
    with LorcanaEngine(timeout=240) as engine:
        deck_ids = [d["id"] for d in engine.list_decks()]
        for it in range(iterations):
            learner.set_reference()  # KL reference = policy at the start of this round
            buffer = ReplayBuffer()
            wins = games = 0
            for g in range(games_per_iter):
                opp = league.pfsp_sample(main_id, mode="hard")
                if opp is None:
                    break
                main_seat = SEAT_ONE if g % 2 == 0 else SEAT_TWO
                opp_seat = SEAT_TWO if main_seat == SEAT_ONE else SEAT_ONE
                p1, p2 = deck_pair(rng, deck_ids)
                res = play_match(engine, {main_seat: main, opp_seat: opp},
                                 seed=f"L{it}-{g}", deck_p1=p1, deck_p2=p2)
                buffer.extend(res.samples.get(main_id, []))
                league.record_result(main_id, opp.id, res.winner_id)
                games += 1
                if res.winner_seat == main_seat:
                    wins += 1
                if verbose:
                    print(f"  it{it} g{g}: main({main_seat}) vs {opp.id} "
                          f"-> winner={res.winner_seat} (+{len(res.samples.get(main_id, []))})",
                          flush=True)

            stats = {}
            for _ in range(epochs):
                if len(buffer):
                    stats = learner.train_epoch(buffer, batch_size=64)

            if (it + 1) % freeze_every == 0:
                pid = f"past_{it + 1}"
                league.register(NetPlayer(pid, frozen_net_copy(net), cfg_opp, record=False,
                                          use_belief=use_belief, n_worlds=max(2, n_worlds // 2),
                                          greedy=True, rng=rng), "past")

            rec = {"iter": it, "winrate": wins / max(games, 1),
                   "samples": len(buffer), "loss": stats.get("loss"),
                   "kl": stats.get("kl"), "elo_main": round(league.elo[main_id], 1),
                   "league_size": len(league.players)}
            history.append(rec)
            if verbose:
                print(f"[league] iter {it+1}/{iterations}: winrate={rec['winrate']:.2f} "
                      f"loss={rec['loss']} kl={rec['kl']} elo_main={rec['elo_main']} "
                      f"league={rec['league_size']}", flush=True)

        # gating evaluation: main vs the scripted anchors
        anchors = [league.players[i] for i in league.players if league.kind[i] == "anchor"]
        main_eval = NetPlayer("main_eval", net, cfg, record=False, use_belief=use_belief,
                              n_worlds=n_worlds, greedy=True, rng=rng)
        gaunt = gauntlet(engine, main_eval, anchors, n_games_per=2, rng=rng,
                         seed_prefix="gate", deck_ids=deck_ids)
        exploit = None
        if run_exploit:
            exp = exploitability_proxy(engine, net, lambda: LorcanaNet(d_model=d_model, n_layers=layers),
                                       cfg, n_iters=1, games_per_iter=4, epochs=1, rng=rng,
                                       use_belief=use_belief)
            exploit = exp["exploitability"]

    import torch
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": net.state_dict(), "d_model": d_model, "layers": layers}, out)
    if verbose:
        print(f"[league] gauntlet vs anchors: {gaunt}", flush=True)
        if exploit is not None:
            print(f"[league] exploitability proxy (exploiter win-rate vs main): {exploit:.2f}",
                  flush=True)
        print(f"[league] saved -> {out}", flush=True)
    return {"history": history, "gauntlet": gaunt, "exploitability": exploit,
            "elo": dict(league.elo)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--freeze-every", type=int, default=2)
    ap.add_argument("--sims", type=int, default=12)
    ap.add_argument("--n-worlds", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--c-kl", type=float, default=0.5)
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--exploit", action="store_true")
    ap.add_argument("--out", type=str, default="lorcana-bot/checkpoints/league.pt")
    args = ap.parse_args()
    run_league(init=args.init, iterations=args.iterations, games_per_iter=args.games,
               epochs=args.epochs, freeze_every=args.freeze_every, sims=args.sims,
               n_worlds=args.n_worlds, d_model=args.d_model, layers=args.layers,
               c_kl=args.c_kl, use_belief=not args.no_belief, run_exploit=args.exploit,
               out=args.out)


if __name__ == "__main__":
    main()
