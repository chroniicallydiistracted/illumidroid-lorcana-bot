"""Phase 1B heads — factored pointer policy + distributional (C51) value.

Policy (architecture doc §2.5): the factored grammar (category x source x
target) is realized as an action-scoring head over the *enumerated legal
candidates*. Each action's score combines its category embedding with the
trunk-contextualized representations of its source and target card tokens
(gathered by card-position; position 0 is the NULL token). Scores are
softmaxed over the legal set only — legality stays authoritative in the engine.

Value (§5.4): distributional over 51 atoms on [-1, 1], trained on game outcome
z (two-hot target), not lore.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.serialization import NUM_CATEGORIES

NEG_INF = -1e9


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, cat_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, cat_dim)
        self.scorer = nn.Sequential(
            nn.Linear(cat_dim + 3 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, trunk_vec: torch.Tensor, token_reps: torch.Tensor,
                action_cat: torch.Tensor, action_src: torch.Tensor,
                action_tgt: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Returns masked policy logits [B, A] (illegal -> NEG_INF)."""
        B, A = action_cat.shape
        D = token_reps.shape[-1]

        def gather(pos: torch.Tensor) -> torch.Tensor:
            idx = pos.clamp(min=0, max=token_reps.shape[1] - 1)
            idx = idx.unsqueeze(-1).expand(B, A, D)             # [B, A, D]
            return torch.gather(token_reps, 1, idx)             # [B, A, D]

        src_rep = gather(action_src)
        tgt_rep = gather(action_tgt)
        cat_rep = self.cat_embed(action_cat)                    # [B, A, cat_dim]
        trunk_rep = trunk_vec.unsqueeze(1).expand(B, A, D)      # [B, A, D]

        feat = torch.cat([cat_rep, src_rep, tgt_rep, trunk_rep], dim=-1)
        logits = self.scorer(feat).squeeze(-1)                  # [B, A]
        logits = logits.masked_fill(~action_mask, NEG_INF)
        return logits


class ValueHead(nn.Module):
    def __init__(self, d_model: int, n_atoms: int = 51,
                 v_min: float = -1.0, v_max: float = 1.0, hidden: int = 128) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.v_min, self.v_max = v_min, v_max
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_atoms),
        )

    def forward(self, trunk_vec: torch.Tensor) -> torch.Tensor:
        """Returns value distribution logits [B, n_atoms]."""
        return self.net(trunk_vec)

    def expected(self, logits: torch.Tensor) -> torch.Tensor:
        """Scalar expected value in [v_min, v_max], shape [B]."""
        probs = F.softmax(logits, dim=-1)
        return (probs * self.atoms).sum(dim=-1)

    def two_hot_target(self, z: torch.Tensor) -> torch.Tensor:
        """Project scalar outcomes z [B] -> two-hot distributions [B, n_atoms]."""
        z = z.clamp(self.v_min, self.v_max)
        atoms = self.atoms
        n = self.n_atoms
        delta = (self.v_max - self.v_min) / (n - 1)
        pos = (z - self.v_min) / delta                  # [B] in [0, n-1]
        lower = pos.floor().clamp(0, n - 1).long()
        upper = pos.ceil().clamp(0, n - 1).long()
        w_upper = (pos - lower.float())
        w_lower = 1.0 - w_upper
        target = torch.zeros(z.shape[0], n, device=z.device, dtype=atoms.dtype)
        target.scatter_add_(1, lower.unsqueeze(1), w_lower.unsqueeze(1))
        target.scatter_add_(1, upper.unsqueeze(1), w_upper.unsqueeze(1))
        return target
