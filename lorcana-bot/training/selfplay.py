"""Phase 1D — self-play actor loop (Sebulba-style single-process version).

Plays games where both seats are driven by B-ISMCTS guided by the current
LorcanaNet; logs (information set, MCTS visit policy, eventual outcome z) at
each decision as AlphaZero-style training targets (architecture doc §5.2). The
full distributed actor/inference/learner split (§4.2) is a later optimization;
this keeps the loop correct and end-to-end on one box.
"""

from __future__ import annotations

import argparse

import numpy as np

from engine.bridge import LorcanaEngine
from engine.serialization import encode_obs, encode_belief
from search.ismcts import BISMCTS, SearchConfig
from search.evaluator import NetEvaluator
from training.learner import Sample, ReplayBuffer, Learner
from training.tier_a_guard import require_tier_a_clean_label_belief_training_ready


def play_game(engine: LorcanaEngine, net, cfg: SearchConfig, seed: str,
              max_steps: int = 400, rng: np.random.Generator | None = None,
              use_belief: bool = True, n_worlds: int = 6,
              deck_p1: str | None = None, deck_p2: str | None = None) -> list[Sample]:
    """Self-play one game and return training targets.

    Belief-guided label generation is blocked until the Tier-A release gate passes.
    Non-belief runs continue to use the Phase-1 single-world search.
    """
    require_tier_a_clean_label_belief_training_ready(
        use_belief=use_belief, context="training.selfplay.play_game")
    rng = rng or np.random.default_rng()
    evaluator = NetEvaluator(net)
    mcts = BISMCTS(engine, evaluator, cfg, rng)
    obs = engine.reset(seed, deck_p1, deck_p2)
    pending: list[tuple[dict, np.ndarray, str, dict]] = []
    steps = 0
    while not obs.get("done") and steps < max_steps:
        legal = obs.get("legal", [])
        actor = obs.get("actor")
        if not legal:
            break
        if use_belief:
            raise AssertionError("Tier-A guard must block belief-guided sample writing")
        else:
            res = mcts.run(obs)  # leaves the engine on the root decision state
        if actor is not None and len(res.pi) == len(legal):
            pending.append((encode_obs(obs), res.pi.copy(), actor, encode_belief(obs)))
        # sample an action from the (temperature-shaped) visit policy
        a = int(rng.choice(len(res.pi), p=res.pi)) if res.pi.sum() > 0 else 0
        key = legal[a]["stableKey"]
        obs = engine.step(key)["obs"]
        steps += 1

    winner = obs.get("winner")
    samples = []
    for enc, pi, actor, bel in pending:
        z = 0.0 if winner is None else (1.0 if str(winner) == str(actor) else -1.0)
        samples.append(Sample(enc=enc, pi=pi, z=z, belief=bel))
    return samples


def run_iteration(net, cfg: SearchConfig, n_games: int, seed_prefix: str,
                  learner: Learner, buffer: ReplayBuffer, epochs: int = 1,
                  batch_size: int = 64, verbose: bool = True,
                  use_belief: bool = True) -> dict:
    require_tier_a_clean_label_belief_training_ready(
        use_belief=use_belief, context="training.selfplay.run_iteration")
    rng = np.random.default_rng()
    import random
    pair_rng = random.Random(seed_prefix)
    with LorcanaEngine() as engine:
        deck_ids = [d["id"] for d in engine.list_decks()]
        for g in range(n_games):
            p1 = p2 = None
            if len(deck_ids) >= 2:
                p1, p2 = pair_rng.sample(deck_ids, 2)
            samples = play_game(engine, net, cfg, f"{seed_prefix}-{g}", rng=rng,
                                deck_p1=p1, deck_p2=p2, use_belief=use_belief)
            buffer.extend(samples)
            if verbose:
                tag = f" [{p1} vs {p2}]" if p1 else ""
                print(f"  selfplay game {g+1}/{n_games}{tag}: +{len(samples)} "
                      f"(buffer={len(buffer)})", flush=True)
    last = {}
    for _ in range(epochs):
        last = learner.train_epoch(buffer, batch_size=batch_size)
    return last


def main() -> None:
    import os
    import torch
    from network.model import LorcanaNet

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="", help="bootstrap checkpoint to warm-start")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--out", type=str, default="lorcana-bot/checkpoints/selfplay.pt")
    args = ap.parse_args()
    use_belief = not args.no_belief
    require_tier_a_clean_label_belief_training_ready(
        use_belief=use_belief, context="training.selfplay.main")

    d_model, layers = args.d_model, args.layers
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location="cpu")
        d_model, layers = ckpt.get("d_model", d_model), ckpt.get("layers", layers)
        net = LorcanaNet(d_model=d_model, n_layers=layers)
        net.load_state_dict(ckpt["model"])
        print(f"[selfplay] warm-started from {args.init}", flush=True)
    else:
        net = LorcanaNet(d_model=d_model, n_layers=layers)

    learner = Learner(net)
    buffer = ReplayBuffer()
    cfg = SearchConfig(simulations=args.sims, depth_limit=args.depth,
                       dirichlet_eps=0.25, temperature=1.0)
    for it in range(args.iterations):
        stats = run_iteration(net, cfg, args.games, f"sp{it}", learner, buffer,
                              use_belief=use_belief)
        print(f"[selfplay] iter {it+1}/{args.iterations}: "
              f"loss={stats.get('loss',0):.4f} policy={stats.get('policy',0):.4f} "
              f"value={stats.get('value',0):.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": net.state_dict(), "d_model": d_model, "layers": layers}, args.out)
    print(f"[selfplay] saved checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
