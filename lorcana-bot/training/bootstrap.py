"""Phase 1D — behaviour cloning from the engine's scripted automation.

Generates games where `bestDeckAwareOracleLoreRaceAutomatedActionStrategy`
plays both seats, recording at each decision the information set and the action
the strategy chose (one-hot policy target) plus the eventual game outcome z
(from the acting player's perspective). Supervised-training LorcanaNet on these
gives a competent prior before any self-play (architecture doc §5.1 Stage 0).
"""

from __future__ import annotations

import argparse

import numpy as np

from engine.bridge import LorcanaEngine
from engine.serialization import encode_obs, encode_belief
from training.learner import Sample, ReplayBuffer, Learner


def _legal_index(legal: list[dict], stable_key: str | None) -> int | None:
    if stable_key is None:
        return None
    for i, a in enumerate(legal):
        if a["stableKey"] == stable_key:
            return i
    return None


def generate_game(engine: LorcanaEngine, seed: str, strategy: str = "best",
                  max_steps: int = 400, deck_p1: str | None = None,
                  deck_p2: str | None = None) -> list[Sample]:
    """Play one fully-scripted game; return BC samples (pi target filled at the
    end once the winner is known)."""
    obs = engine.reset(seed, deck_p1, deck_p2)
    pending: list[tuple[dict, np.ndarray, str, dict]] = []  # (enc, pi, actor, belief)
    steps = 0
    while not obs.get("done") and steps < max_steps:
        legal = obs.get("legal", [])
        actor = obs.get("actor")
        enc = encode_obs(obs)
        bel = encode_belief(obs)
        r = engine.step_auto(strategy)
        if r.get("policy") == "oracle":
            raise ValueError(
                f"Refusing to behaviour-clone an ORACLE strategy ('{strategy}'): it "
                "chooses with full-deck visibility, so its actions are not learnable "
                "from the fog-of-war observation. Use a fair strategy (e.g. 'best')."
            )
        idx = _legal_index(legal, r.get("executed"))
        if idx is not None and actor is not None and legal:
            pi = np.zeros(len(legal), dtype=np.float32)
            pi[idx] = 1.0
            pending.append((enc, pi, actor, bel))
        obs = r["obs"]
        steps += 1

    winner = obs.get("winner")
    samples: list[Sample] = []
    for enc, pi, actor, bel in pending:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if str(winner) == str(actor) else -1.0
        samples.append(Sample(enc=enc, pi=pi, z=z, belief=bel))
    return samples


def generate_dataset(n_games: int, strategy: str = "best", seed_prefix: str = "bc",
                     max_steps: int = 400, verbose: bool = True,
                     sample_decks: bool = True) -> ReplayBuffer:
    """Generate scripted games across the deck metagame. With `sample_decks`,
    each game draws a distinct random real-deck pair (deck diversity, §6)."""
    import random
    buf = ReplayBuffer()
    with LorcanaEngine() as engine:
        deck_ids = [d["id"] for d in engine.list_decks()] if sample_decks else []
        rng = random.Random(seed_prefix)
        for g in range(n_games):
            p1 = p2 = None
            if len(deck_ids) >= 2:
                p1, p2 = rng.sample(deck_ids, 2)
            samples = generate_game(engine, f"{seed_prefix}-{g}", strategy, max_steps, p1, p2)
            buf.extend(samples)
            if verbose:
                tag = f" [{p1} vs {p2}]" if p1 else ""
                print(f"  game {g+1}/{n_games}{tag}: +{len(samples)} samples "
                      f"(buffer={len(buf)})", flush=True)
    return buf


def main() -> None:
    import torch
    from network.model import LorcanaNet

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="lorcana-bot/checkpoints/bc.pt")
    args = ap.parse_args()

    print(f"[bootstrap] generating {args.games} scripted games...", flush=True)
    buf = generate_dataset(args.games)
    print(f"[bootstrap] dataset: {len(buf)} samples", flush=True)

    net = LorcanaNet(d_model=args.d_model, n_layers=args.layers)
    learner = Learner(net, lr=args.lr)
    for ep in range(args.epochs):
        stats = learner.train_epoch(buf, batch_size=args.batch_size)
        belief = (f" belief={stats['belief']:.4f} sep={stats['belief_sep']:.3f} "
                  f"cnt_mae={stats['belief_count_mae']:.2f}") if "belief" in stats else ""
        print(f"[bootstrap] epoch {ep+1}/{args.epochs}: "
              f"loss={stats['loss']:.4f} policy={stats['policy']:.4f} "
              f"value={stats['value']:.4f} H={stats['entropy']:.3f}{belief}", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": net.state_dict(),
                "d_model": args.d_model, "layers": args.layers}, args.out)
    print(f"[bootstrap] saved checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
