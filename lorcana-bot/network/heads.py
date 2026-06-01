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

from engine.serialization import NUM_CATEGORIES, CAND_FEAT_DIM, NUM_ROLES

NEG_INF = -1e9
_NAMED_ROLE = NUM_ROLES   # role id for the named-card token (0..NUM_ROLES-1 are board roles)


class PolicyHead(nn.Module):
    """Candidate-transformer policy (ACE). Each legal candidate is encoded as a
    small token set — a CLS token (candidate scalar features + family + trunk),
    a named-card token, and role-tagged pointers into the trunk's card tokens
    (source / targets / singers / shift / banish / discard / exert). A shared
    mini-transformer attends over that set; the CLS output scores the candidate.
    Scores are softmaxed over the legal set. This distinguishes candidates that
    share family/src/first-target but differ in singer / shift target / 2nd
    target / named card / cost cards / destination / optional / choice-index —
    which the old factored (cat,src,tgt) head collapsed to identical features."""

    def __init__(self, d_model: int, cat_dim: int = 32, n_heads: int = 4,
                 n_layers: int = 1, ff_mult: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, cat_dim)
        self.role_embed = nn.Embedding(NUM_ROLES + 1, d_model)   # 0=pad .. NUM_ROLES=named
        self.feat_proj = nn.Linear(CAND_FEAT_DIM, d_model)
        self.cls_proj = nn.Linear(cat_dim + 2 * d_model, d_model)  # cat + feat + trunk -> CLS
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, trunk_vec: torch.Tensor, token_reps: torch.Tensor,
                action_cat: torch.Tensor, action_mask: torch.Tensor,
                cand_feats: torch.Tensor, cand_tok_pos: torch.Tensor,
                cand_tok_role: torch.Tensor, cand_tok_mask: torch.Tensor,
                named_emb: torch.Tensor, cand_named_id: torch.Tensor) -> torch.Tensor:
        """Returns masked policy logits [B, A] (illegal -> NEG_INF)."""
        B, A = action_cat.shape
        D = self.d_model
        T = cand_tok_pos.shape[-1]
        L = token_reps.shape[1]

        # gather the trunk-contextualized rep for each candidate pointer -> [B,A,T,D]
        idx = cand_tok_pos.clamp(0, L - 1).unsqueeze(-1).expand(B, A, T, D)
        base = token_reps.unsqueeze(1).expand(B, A, L, D)
        tok = torch.gather(base, 2, idx) + self.role_embed(cand_tok_role)   # + role

        named_tok = (named_emb + self.role_embed.weight[_NAMED_ROLE]).unsqueeze(2)  # [B,A,1,D]

        cat_rep = self.cat_embed(action_cat)                                # [B,A,cat]
        feat_rep = self.feat_proj(cand_feats)                               # [B,A,D]
        trunk_rep = trunk_vec.unsqueeze(1).expand(B, A, D)                  # [B,A,D]
        cls = self.cls_proj(torch.cat([cat_rep, feat_rep, trunk_rep], dim=-1)).unsqueeze(2)

        seq = torch.cat([cls, named_tok, tok], dim=2)                       # [B,A,2+T,D]
        S = seq.shape[2]

        # key-padding mask (True = ignore): CLS always valid; named iff present
        pad = torch.ones(B, A, S, dtype=torch.bool, device=seq.device)
        pad[..., 0] = False
        pad[..., 1] = cand_named_id <= 0
        pad[..., 2:] = ~cand_tok_mask

        out = self.encoder(seq.reshape(B * A, S, D),
                           src_key_padding_mask=pad.reshape(B * A, S))      # [B*A,S,D]
        cls_out = out[:, 0, :].reshape(B, A, D)                             # CLS per candidate
        logits = self.scorer(cls_out).squeeze(-1)                          # [B,A]
        return logits.masked_fill(~action_mask, NEG_INF)


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


class BeliefHead(nn.Module):
    """Phase 2 — leak-free belief over the opponent's hidden hand.

    Scores each candidate pool card (its identity embedding) against the trunk
    vector (built ONLY from the acting player's filtered view). Output is a
    per-candidate logit for `P(card currently in opp hand)`. Candidate
    embeddings never enter the trunk, so the policy/value heads cannot see
    opponent identities — the belief is predicted, not observed.
    """

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        # two outputs per candidate: P(in opp hand), P(in opp inkwell)
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, trunk_vec: torch.Tensor, cand_emb: torch.Tensor,
                cand_mask: torch.Tensor) -> torch.Tensor:
        """trunk_vec [B,D], cand_emb [B,P,D], cand_mask [B,P] -> logits [B,P,2]
        (channel 0 = in hand, channel 1 = in inkwell)."""
        B, P, D = cand_emb.shape
        trunk_rep = trunk_vec.unsqueeze(1).expand(B, P, D)
        feat = torch.cat([trunk_rep, cand_emb], dim=-1)
        logits = self.scorer(feat)                        # [B, P, 2]
        return logits.masked_fill(~cand_mask.unsqueeze(-1), NEG_INF)

    @staticmethod
    def normalize_to_count(probs: torch.Tensor, mask: torch.Tensor,
                           handcount: torch.Tensor, iters: int = 3) -> torch.Tensor:
        """Project per-card probabilities so they sum to the known hand size and
        stay in [0,1] (count-consistency, architecture doc §2.4). A plain scale +
        clamp loses mass when a scaled prob exceeds 1, so we iterate
        scale-then-clamp a few times to restore the target total."""
        m = mask.float()
        probs = (probs * m).clamp(0.0, 1.0)
        target = handcount.unsqueeze(-1)
        for _ in range(iters):
            s = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            probs = (probs * (target / s)).clamp(0.0, 1.0) * m
        return probs
