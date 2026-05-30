"""Phase 1D — learner: replay buffer + gradient updates for LorcanaNet.

Loss (Phase 1 subset of architecture doc §5.3):
  L = L_policy + c_value * L_value + c_l2 * ||theta||^2 - c_entropy * H(pi)
  * L_policy : cross-entropy from the network policy to a target distribution
    over legal actions (one-hot scripted choice for BC, MCTS visit dist for
    self-play), legal-masked.
  * L_value  : cross-entropy to the two-hot projection of the outcome z.
Belief loss is deferred to Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from engine.serialization import collate
from network.heads import NEG_INF


@dataclass
class Sample:
    enc: dict            # output of encode_obs (numpy arrays)
    pi: np.ndarray       # target policy over this obs's legal actions [n]
    z: float             # outcome in [-1, 1] from the acting player's POV


@dataclass
class ReplayBuffer:
    capacity: int = 200_000
    data: list[Sample] = field(default_factory=list)

    def add(self, s: Sample) -> None:
        self.data.append(s)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]

    def extend(self, samples: list[Sample]) -> None:
        for s in samples:
            self.add(s)

    def __len__(self) -> int:
        return len(self.data)


def _build_batch(samples: list[Sample], device: torch.device):
    batch = collate([s.enc for s in samples])
    B = len(samples)
    max_a = batch["action_mask"].shape[1]
    pi_target = np.zeros((B, max_a), dtype=np.float32)
    z = np.zeros(B, dtype=np.float32)
    for i, s in enumerate(samples):
        n = len(s.pi)
        if n:
            pi_target[i, :n] = s.pi
        z[i] = s.z
    tb = {k: torch.as_tensor(v).to(device) for k, v in batch.items()}
    return tb, torch.as_tensor(pi_target).to(device), torch.as_tensor(z).to(device)


def loss_terms(net, tb, pi_target, z, c_value=1.0, c_entropy=0.0):
    out = net.forward(tb)
    logits = out["policy_logits"]                       # [B, A], illegal = -inf
    mask = tb["action_mask"]
    logp = torch.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=-1)
    logp = torch.nan_to_num(logp, neginf=0.0)           # padded cols contribute 0
    policy_loss = -(pi_target * logp).sum(dim=-1).mean()

    value_logits = out["value_logits"]
    target_dist = net.value.two_hot_target(z)
    value_logp = torch.log_softmax(value_logits, dim=-1)
    value_loss = -(target_dist * value_logp).sum(dim=-1).mean()

    # policy entropy (for monitoring / optional anti-collapse bonus)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1).mean()

    total = policy_loss + c_value * value_loss - c_entropy * entropy
    return total, {
        "loss": float(total.detach()),
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
    }


class Learner:
    def __init__(self, net, lr=1e-3, weight_decay=1e-4, device=None,
                 c_value=1.0, c_entropy=0.0):
        self.device = device or next(net.parameters()).device
        self.net = net.to(self.device)
        self.opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.c_value = c_value
        self.c_entropy = c_entropy

    def train_epoch(self, buffer: ReplayBuffer, batch_size=64, shuffle=True,
                    max_batches: int | None = None) -> dict:
        self.net.train()
        idx = np.arange(len(buffer))
        if shuffle:
            np.random.shuffle(idx)
        agg = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
        nb = 0
        for start in range(0, len(idx), batch_size):
            chunk = [buffer.data[i] for i in idx[start:start + batch_size]]
            if not chunk:
                continue
            tb, pi_t, z = _build_batch(chunk, self.device)
            total, stats = loss_terms(self.net, tb, pi_t, z, self.c_value, self.c_entropy)
            self.opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.opt.step()
            for k in agg:
                agg[k] += stats[k]
            nb += 1
            if max_batches and nb >= max_batches:
                break
        for k in agg:
            agg[k] /= max(nb, 1)
        agg["batches"] = nb
        return agg
