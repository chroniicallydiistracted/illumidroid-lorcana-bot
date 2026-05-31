"""Evaluators map an observation to (priors over its legal actions, value).

`NetEvaluator` wraps a trained LorcanaNet (single-obs encode -> infer).
`UniformEvaluator` is a network-free baseline (uniform priors, zero value) used
to exercise the search machinery in tests and before any training.
"""

from __future__ import annotations

import numpy as np

from engine.serialization import encode_obs, encode_belief, collate, collate_belief


class UniformEvaluator:
    def __call__(self, obs: dict) -> tuple[np.ndarray, float]:
        n = len(obs.get("legal", []))
        if n == 0:
            return np.zeros(0, dtype=np.float32), 0.0
        return np.ones(n, dtype=np.float32) / n, 0.0

    def batch_eval(self, obs_list: list[dict]) -> list[tuple[np.ndarray, float]]:
        return [self(o) for o in obs_list]


class NetEvaluator:
    def __init__(self, net, device=None) -> None:
        import torch
        self.net = net
        self.device = device or next(net.parameters()).device

    def __call__(self, obs: dict) -> tuple[np.ndarray, float]:
        n = len(obs.get("legal", []))
        if n == 0:
            return np.zeros(0, dtype=np.float32), 0.0
        batch = collate([encode_obs(obs)])
        priors, values = self.net.infer(batch, self.device)
        return priors[0, :n].astype(np.float32), float(values[0])

    def batch_eval(self, obs_list: list[dict]) -> list[tuple[np.ndarray, float]]:
        """Evaluate many leaves in ONE forward (the batched-GPU-inference path).
        Returns per-leaf (priors over its legal actions, value)."""
        if not obs_list:
            return []
        encoded = [encode_obs(o) for o in obs_list]
        priors, values = self.net.infer(collate(encoded), self.device)
        out = []
        for i, o in enumerate(obs_list):
            n = len(o.get("legal", []))
            out.append((priors[i, :n].astype(np.float32), float(values[i])))
        return out


class BeliefEvaluator:
    """Maps an observation to (opponent pool instance-ids, P(in opp hand)).

    The probabilities align to the pool order used by `encode_belief`
    (hand cards then deck cards). Used by the determinization sampler.
    """

    def __init__(self, net, device=None) -> None:
        self.net = net
        self.device = device or next(net.parameters()).device

    def __call__(self, obs: dict) -> tuple[list[str], np.ndarray]:
        hidden = obs.get("hidden") or {}
        pool_ids = [c["id"] for c in hidden.get("hand", [])] + \
                   [c["id"] for c in hidden.get("deck", [])]
        if not pool_ids:
            return [], np.zeros(0, dtype=np.float32)
        bel = encode_belief(obs)
        batch = {**collate([encode_obs(obs)]), **collate_belief([bel])}
        probs = self.net.belief_probs(batch, self.device, normalize=True)
        return pool_ids, probs[0, :len(pool_ids)].astype(np.float32)


class UniformBeliefEvaluator:
    """Belief-free baseline: every pool card equally likely to be in hand."""

    def __call__(self, obs: dict) -> tuple[list[str], np.ndarray]:
        hidden = obs.get("hidden") or {}
        pool_ids = [c["id"] for c in hidden.get("hand", [])] + \
                   [c["id"] for c in hidden.get("deck", [])]
        n = len(pool_ids)
        hc = obs.get("players", {}).get("opp", {}).get("handCount", 0)
        p = (hc / n) if n else 0.0
        return pool_ids, np.full(n, p, dtype=np.float32)
