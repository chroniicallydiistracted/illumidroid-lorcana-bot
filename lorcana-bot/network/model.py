"""Phase 1B model — LorcanaNet: shared trunk + policy head + value head.

Dual-headed (policy + value) per the Phase 1 plan; the belief head is deferred
to Phase 2. `infer` is the search/actor entry point: numpy batch in, numpy
priors + scalar values out (the only tensor work the actor does per leaf).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.trunk import Trunk
from network.heads import PolicyHead, ValueHead, BeliefHead


def to_tensor_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(v)
        if t.dtype == torch.bool:
            pass
        elif t.dtype in (torch.float64, torch.float16):
            t = t.float()
        out[k] = t.to(device)
    return out


class LorcanaNet(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 n_atoms: int = 51) -> None:
        super().__init__()
        self.trunk = Trunk(d_model, n_heads, n_layers)
        self.policy = PolicyHead(d_model)
        self.value = ValueHead(d_model, n_atoms=n_atoms)
        self.belief = BeliefHead(d_model)   # Phase 2 third head (leak-free)

    def forward(self, batch: dict) -> dict:
        trunk_vec, token_reps = self.trunk(batch)
        policy_logits = self.policy(
            trunk_vec, token_reps,
            batch["action_cat"], batch["action_src"],
            batch["action_tgt"], batch["action_mask"],
        )
        value_logits = self.value(trunk_vec)
        out = {
            "policy_logits": policy_logits,     # [B, A] (illegal = -inf)
            "value_logits": value_logits,       # [B, n_atoms]
            "value": self.value.expected(value_logits),  # [B]
        }
        if "belief_ids" in batch:
            # candidate identity embeddings reuse the trunk's identity vocabulary
            cand_emb = self.trunk.card_encoder.id_embed(batch["belief_ids"])  # [B,P,D]
            out["belief_logits"] = self.belief(trunk_vec, cand_emb, batch["belief_mask"])
        return out

    @torch.no_grad()
    def infer(self, batch: dict, device: torch.device | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """numpy collated batch -> (priors [B, A], values [B]).

        priors are legal-masked probabilities (illegal entries = 0).
        """
        device = device or next(self.parameters()).device
        self.eval()
        tb = to_tensor_batch(batch, device)
        out = self.forward(tb)
        priors = F.softmax(out["policy_logits"], dim=-1)
        priors = priors * tb["action_mask"].float()
        denom = priors.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        priors = priors / denom
        return priors.cpu().numpy(), out["value"].cpu().numpy()

    @torch.no_grad()
    def belief_probs(self, batch: dict, device: torch.device | None = None,
                     normalize: bool = True) -> np.ndarray:
        """numpy belief batch -> per-candidate P(in opp hand) [B, P].

        With `normalize`, probabilities are scaled to sum to the known hand size
        (count-consistency). Requires `belief_ids`, `belief_mask`, and (for
        normalization) `belief_handcount` in the batch.
        """
        device = device or next(self.parameters()).device
        self.eval()
        tb = to_tensor_batch(batch, device)
        out = self.forward(tb)
        logits = out["belief_logits"]
        probs = torch.sigmoid(logits) * tb["belief_mask"].float()
        if normalize and "belief_handcount" in tb:
            probs = self.belief.normalize_to_count(probs, tb["belief_mask"],
                                                   tb["belief_handcount"].float())
        return probs.cpu().numpy()
