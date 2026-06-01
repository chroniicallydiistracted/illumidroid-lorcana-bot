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

from engine.serialization import collate, collate_belief
from network.heads import NEG_INF


@dataclass
class Sample:
    enc: dict            # output of encode_obs (numpy arrays)
    pi: np.ndarray       # target policy over this obs's legal actions [n]
    z: float             # outcome in [-1, 1] from the acting player's POV
    belief: dict | None = None   # output of encode_belief (Phase 2 target), optional


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
    if all(s.belief is not None for s in samples):
        bel = collate_belief([s.belief for s in samples])
        for k, v in bel.items():
            tb[k] = torch.as_tensor(v).to(device)
    return tb, torch.as_tensor(pi_target).to(device), torch.as_tensor(z).to(device)


def _masked_logp(logits, mask):
    logp = torch.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=-1)
    return torch.nan_to_num(logp, neginf=0.0)           # padded/illegal cols -> 0


def loss_terms(net, tb, pi_target, z, c_value=1.0, c_entropy=0.0,
               c_belief=1.0, c_count=0.1, ref_logp=None, c_kl=0.0):
    out = net.forward(tb)
    logits = out["policy_logits"]                       # [B, A], illegal = -inf
    mask = tb["action_mask"]
    logp = _masked_logp(logits, mask)
    policy_loss = -(pi_target * logp).sum(dim=-1).mean()

    value_logits = out["value_logits"]
    target_dist = net.value.two_hot_target(z)
    value_logp = torch.log_softmax(value_logits, dim=-1)
    value_loss = -(target_dist * value_logp).sum(dim=-1).mean()

    # policy entropy (for monitoring / optional anti-collapse bonus)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1).mean()

    total = policy_loss + c_value * value_loss - c_entropy * entropy
    stats = {
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
    }

    # Phase 3: KL trust-region / proximal term to a frozen reference policy
    # (anti-collapse, no catastrophic forgetting — architecture doc §5.1).
    # Illegal/padded columns contribute 0 since (logp - ref_logp) = 0 there.
    if ref_logp is not None and c_kl > 0.0:
        kl = (logp.exp() * (logp - ref_logp)).sum(dim=-1).mean()
        total = total + c_kl * kl
        stats["kl"] = float(kl.detach())

    # Phase 2: belief loss (BCE per candidate + count-consistency) — leak-free
    if "belief_logits" in out:
        bmask = tb["belief_mask"]
        blabel = tb["belief_label"]
        blogits = torch.nan_to_num(out["belief_logits"], neginf=0.0)
        bce = F.binary_cross_entropy_with_logits(blogits, blabel, reduction="none")
        denom = bmask.float().sum().clamp_min(1.0)
        belief_loss = (bce * bmask.float()).sum() / denom
        probs = torch.sigmoid(blogits) * bmask.float()
        pred_count = probs.sum(dim=-1)
        count_loss = F.l1_loss(pred_count, tb["belief_handcount"].float())
        total = total + c_belief * belief_loss + c_count * count_loss
        # calibration: mean predicted prob on true in-hand vs not
        with torch.no_grad():
            pos = (probs * blabel).sum() / blabel.sum().clamp_min(1.0)
            neg = (probs * (1 - blabel) * bmask.float()).sum() / \
                  ((1 - blabel) * bmask.float()).sum().clamp_min(1.0)
        stats["belief"] = float(belief_loss.detach())
        stats["belief_count_mae"] = float(count_loss.detach())
        stats["belief_sep"] = float((pos - neg).detach())  # >0 means it separates

        # inkwell belief: which cards the opponent chose to ink (hidden, strategic)
        if "belief_logits_ink" in out and "belief_label_ink" in tb:
            ilabel = tb["belief_label_ink"]
            ilogits = torch.nan_to_num(out["belief_logits_ink"], neginf=0.0)
            ibce = F.binary_cross_entropy_with_logits(ilogits, ilabel, reduction="none")
            ink_loss = (ibce * bmask.float()).sum() / denom
            iprobs = torch.sigmoid(ilogits) * bmask.float()
            ink_count_loss = F.l1_loss(iprobs.sum(dim=-1), tb["belief_inkcount"].float())
            total = total + c_belief * ink_loss + c_count * ink_count_loss
            with torch.no_grad():
                ipos = (iprobs * ilabel).sum() / ilabel.sum().clamp_min(1.0)
                ineg = (iprobs * (1 - ilabel) * bmask.float()).sum() / \
                       ((1 - ilabel) * bmask.float()).sum().clamp_min(1.0)
            stats["belief_ink"] = float(ink_loss.detach())
            stats["belief_ink_sep"] = float((ipos - ineg).detach())

    stats["loss"] = float(total.detach())
    return total, stats


class Learner:
    def __init__(self, net, lr=1e-3, weight_decay=1e-4, device=None,
                 c_value=1.0, c_entropy=0.0, c_belief=1.0, c_kl=0.0):
        self.device = device or next(net.parameters()).device
        self.net = net.to(self.device)
        self.opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.c_belief = c_belief
        self.c_kl = c_kl
        self._reference = None   # frozen reference net for the KL trust region

    def set_reference(self) -> None:
        """Freeze the current policy as the KL reference (call before a round)."""
        import copy
        if self.c_kl <= 0.0:
            return
        self._reference = copy.deepcopy(self.net).to(self.device)
        for p in self._reference.parameters():
            p.requires_grad_(False)
        self._reference.eval()

    def train_epoch(self, buffer: ReplayBuffer, batch_size=64, shuffle=True,
                    max_batches: int | None = None) -> dict:
        self.net.train()
        idx = np.arange(len(buffer))
        if shuffle:
            np.random.shuffle(idx)
        agg: dict[str, float] = {}
        nb = 0
        for start in range(0, len(idx), batch_size):
            chunk = [buffer.data[i] for i in idx[start:start + batch_size]]
            if not chunk:
                continue
            tb, pi_t, z = _build_batch(chunk, self.device)
            ref_logp = None
            if self._reference is not None and self.c_kl > 0.0:
                with torch.no_grad():
                    ref_logits = self._reference.forward(tb)["policy_logits"]
                    ref_logp = _masked_logp(ref_logits, tb["action_mask"])
            total, stats = loss_terms(self.net, tb, pi_t, z, self.c_value,
                                      self.c_entropy, self.c_belief,
                                      ref_logp=ref_logp, c_kl=self.c_kl)
            self.opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.opt.step()
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
            if max_batches and nb >= max_batches:
                break
        for k in agg:
            agg[k] /= max(nb, 1)
        agg["batches"] = nb
        return agg
