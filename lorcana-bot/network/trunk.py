"""Phase 1B trunk — permutation-invariant set-transformer over card tokens.

Builds one token per card (numeric features + learned identity embedding),
prepends a learned NULL token (card-position 0, the pointer target for actions
with no source/target) and appends a global CLS token built from the scalar
features. A few self-attention layers mix them; the CLS output is the trunk
vector and the (null + card) outputs are the per-token reps the pointer policy
head gathers from. See architecture doc §2.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from engine.serialization import (
    CARD_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    HIST_FEAT_DIM,
    HIST_MAX,
    VOCAB_SIZE,
    PAD_ID,
)


class CardEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(CARD_FEATURE_DIM, d_model)
        self.id_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, card_feats: torch.Tensor, card_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.feat_proj(card_feats) + self.id_embed(card_ids))


class HistoryEncoder(nn.Module):
    """Public-history encoder (§2.3). Summarizes the rolling window of public
    events into one vector fused into the trunk's global token, sharpening the
    belief (and policy/value) — e.g. cards the opponent has played left their
    hand. `id_embed` (the card-identity table) is shared with the trunk so a
    played card and its board token mean the same thing. A learned CLS token is
    always valid, so empty/early-game histories never produce an all-padded row."""

    def __init__(self, d_model: int, id_embed: nn.Embedding, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.id_embed = id_embed
        self.feat_proj = nn.Linear(HIST_FEAT_DIM, d_model)
        self.pos_embed = nn.Embedding(HIST_MAX + 1, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.fuse = nn.Linear(d_model, d_model)

    def forward(self, hist_feats: torch.Tensor, hist_ids: torch.Tensor,
                hist_mask: torch.Tensor) -> torch.Tensor:
        """[B,H,Fh],[B,H],[B,H] -> [B,D] history summary."""
        B, H, _ = hist_feats.shape
        pos = torch.arange(H, device=hist_feats.device).clamp(max=HIST_MAX)
        tok = self.feat_proj(hist_feats) + self.id_embed(hist_ids) + self.pos_embed(pos).unsqueeze(0)
        cls = self.cls.expand(B, 1, self.d_model)
        seq = torch.cat([cls, tok], dim=1)                       # [B, 1+H, D]
        valid = torch.ones(B, 1, dtype=torch.bool, device=hist_mask.device)
        key_padding = ~torch.cat([valid, hist_mask], dim=1)      # CLS always valid
        out = self.encoder(seq, src_key_padding_mask=key_padding)
        return self.fuse(out[:, 0, :])                           # CLS summary


class Trunk(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 ff: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.card_encoder = CardEncoder(d_model)
        self.history_encoder = HistoryEncoder(d_model, self.card_encoder.id_embed, n_heads)
        self.global_proj = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.null_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (trunk_vec [B, D], token_reps [B, 1+N, D]) where token_reps
        index 0 is the NULL token and 1..N are the cards (card-position aligned).
        """
        card_feats = batch["card_feats"]            # [B, N, Fc]
        card_ids = batch["card_ids"]                # [B, N]
        card_mask = batch["card_mask"]              # [B, N] bool (True = real)
        globals_ = batch["globals"]                 # [B, G]
        B, N, _ = card_feats.shape

        card_tok = self.card_encoder(card_feats, card_ids)        # [B, N, D]
        null_tok = self.null_token.expand(B, 1, self.d_model)     # [B, 1, D]
        cls_vec = self.global_proj(globals_)                      # [B, D]
        if "hist_feats" in batch:                                 # fuse public history (§2.3)
            cls_vec = cls_vec + self.history_encoder(
                batch["hist_feats"], batch["hist_ids"], batch["hist_mask"])
        cls_tok = cls_vec.unsqueeze(1)                            # [B, 1, D]

        seq = torch.cat([null_tok, card_tok, cls_tok], dim=1)     # [B, N+2, D]
        valid = torch.ones(B, 1, dtype=torch.bool, device=card_mask.device)
        seq_valid = torch.cat([valid, card_mask, valid], dim=1)   # [B, N+2]
        key_padding_mask = ~seq_valid                             # True = ignore

        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)  # [B, N+2, D]
        token_reps = out[:, : N + 1, :]   # [B, 1+N, D]  (null + cards)
        trunk_vec = out[:, -1, :]         # [B, D]       (CLS / global)
        return trunk_vec, token_reps
