"""Evaluators map an observation to (priors over its legal actions, value).

`NetEvaluator` wraps a trained LorcanaNet (single-obs encode -> infer).
`UniformEvaluator` is a network-free baseline (uniform priors, zero value) used
to exercise the search machinery in tests and before any training.
"""

from __future__ import annotations

import numpy as np

from engine.serialization import encode_obs, collate


class UniformEvaluator:
    def __call__(self, obs: dict) -> tuple[np.ndarray, float]:
        n = len(obs.get("legal", []))
        if n == 0:
            return np.zeros(0, dtype=np.float32), 0.0
        return np.ones(n, dtype=np.float32) / n, 0.0


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
