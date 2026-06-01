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

from engine.serialization import NUM_CATEGORIES, CAND_FEAT_DIM, NUM_ROLES, AUX_DIM

NEG_INF = -1e9


class PolicyHead(nn.Module):
    """Candidate-ranking policy (ACE, lightweight). Each legal candidate is scored
    by an MLP over: its family embedding, its scalar descriptor (`cand_feats` —
    cost type / singers / targets / optional / choice / destinations), a masked
    MEAN-POOL of its role-tagged card-token reps (source / targets / singers /
    shift / banish-discard-exert), the named-card embedding, and the trunk vector.

    This keeps ACE's distinguishing power (candidates differing in singer / shift /
    2nd target / named card / cost / optional get different features) but replaces
    the per-candidate TRANSFORMER (attention over leaves×candidates rows — 82% of
    CPU search time, untrainable on this hardware) with a cheap pool+MLP that is
    ~50-100x faster per forward. Scores are softmaxed over the legal set."""

    def __init__(self, d_model: int, cat_dim: int = 32, hidden: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, cat_dim)
        self.role_embed = nn.Embedding(NUM_ROLES + 1, d_model)   # 0=pad .. NUM_ROLES=named
        self.feat_proj = nn.Linear(CAND_FEAT_DIM, d_model)
        # scorer input: cat + cand_feat(D) + trunk(D) + pooled MEAN(D) + pooled SUM(D) + named(D).
        # Both mean and sum: mean is order/count-invariant; sum is count/identity-sensitive
        # so candidates differing by an extra target/singer/cost-card stay distinct.
        self.scorer = nn.Sequential(
            nn.Linear(cat_dim + 5 * d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

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
        tok = torch.gather(base, 2, idx) + self.role_embed(cand_tok_role)   # + role tag
        m = cand_tok_mask.unsqueeze(-1).to(tok.dtype)                       # [B,A,T,1]
        tok_sum = (tok * m).sum(2)                                         # [B,A,D] (count/identity-sensitive)
        pooled = tok_sum / m.sum(2).clamp_min(1.0)                         # masked mean [B,A,D]

        cat_rep = self.cat_embed(action_cat)                               # [B,A,cat]
        feat_rep = self.feat_proj(cand_feats)                              # [B,A,D]
        trunk_rep = trunk_vec.unsqueeze(1).expand(B, A, D)                 # [B,A,D]
        feat = torch.cat([cat_rep, feat_rep, trunk_rep, pooled, tok_sum, named_emb], dim=-1)
        logits = self.scorer(feat).squeeze(-1)                             # [B,A]
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


class AuxHead(nn.Module):
    """Auxiliary consequence head (race / clock). Regresses trajectory-derived
    targets from the trunk vector — final lore for self & opponent and turns until
    the game ends — so the trunk learns *where the game is going* (race / lethal
    math), not only the terminal win/loss. Targets come from the realized
    self-play game (fair, like z); the head is never an input to anything."""

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, AUX_DIM),
        )

    def forward(self, trunk_vec: torch.Tensor) -> torch.Tensor:
        """Returns raw aux logits [B, AUX_DIM] (sigmoid -> [0,1] applied in loss)."""
        return self.net(trunk_vec)


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
