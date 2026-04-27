"""
Spatial-Temporal Transformer classifier for per-link traffic trend prediction.

Architecture (fixed):
  Input: (B, T=11, 12, 12)
    · Positions 0..T-3  = T-2 temporal delta frames  (percentage-change, signed)
    · Positions T-2..T-1 = 2 Pareto summary frames   (x_m scale, α shape — unsigned stats)

  Fix 1 — separate streams:
    The Pareto frames are statistical summaries, NOT temporal observations.
    Feeding them through the temporal Transformer with positional encoding
    caused attention to treat them as "future timesteps" and lost their meaning
    (both computed from abs(Δ), so sign info was already gone).
    They are now projected through a dedicated Linear(2→d_model//2) branch
    and fused with the temporal representation after attention.

  Fix 2 — last-token pooling instead of mean pool:
    For trend prediction the most recent delta is most informative.
    Mean pooling gave equal weight to all T-2 timesteps including the oldest.
    The last encoder token (most recent delta) is used instead.

  Full pipeline:
    1. Split input → delta sequence (T-2, 12, 12) + Pareto stats (2, 12, 12)
    2. Reshape delta sequence → (B×144, T-2, 1)
    3. Input projection: Linear(1→d_model) + LayerNorm + ReLU
    4. Sinusoidal positional encoding over T-2 steps
    5. Temporal Transformer encoder (N layers) — self-attention per link
    6. Last-token pooling → (B×144, d_model)
    7. Pareto branch: reshape → (B×144, 2) → Linear(2→d_model//2) + LN + ReLU
    8. Fusion: concat → Linear(d_model + d_model//2 → d_model) + LN + ReLU
    9. Spatial Transformer encoder (1 layer) — cross-link attention
   10. Dropout → Linear(d_model→3) → (B, 3, 12, 12)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_NETFLUX_ROOT = Path(__file__).resolve().parent.parent
if str(_NETFLUX_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETFLUX_ROOT))

from config import (
    MATRIX_SIZE,
    DROPOUT,
    NUM_CLASSES,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_USE_SPATIAL_ATTN,
)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding added to the token sequence.
    Registered as a buffer (not a learnable parameter).
    """

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape (1, max_len, d_model) for broadcasting over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PerLinkTransformer(nn.Module):
    """
    Transformer classifier applied independently to each link's time series,
    with an optional cross-link spatial attention layer.

    Forward signature:
        x : (B, T, 12, 12)   — T = WINDOW_SIZE + 1 in relative mode (default 11)
        Returns: (B, num_classes, 12, 12)

    Public method `encode(x)` → (B, 144, d_model): returns per-link embeddings
    before the classifier head for use by the RL actor-critic.
    """

    def __init__(
        self,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_encoder_layers: int = TRANSFORMER_NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        matrix_size: int = MATRIX_SIZE,
        use_spatial_attn: bool = TRANSFORMER_USE_SPATIAL_ATTN,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.matrix_size = matrix_size
        self.num_links = matrix_size * matrix_size  # 144
        self.use_spatial_attn = use_spatial_attn

        # --- Stream A: temporal delta projection (1 scalar per step → d_model) ---
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # --- Stream B: Pareto summary projection (2 stats → d_model//2) ---
        # x_m (scale) and α (shape) are window-level statistics, not timesteps.
        # Separate branch preserves their meaning and keeps signed delta info intact.
        self.pareto_proj = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
        )

        # --- Fusion: last temporal token + Pareto embedding → d_model ---
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # --- Temporal encoder (per-link, applied to B×144 sequences) ---
        # enable_nested_tensor=False: norm_first=True is incompatible with nested
        # tensor optimisation in PyTorch 2.x; disable it to suppress the warning.
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable gradient flow
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # --- Spatial encoder (one layer, across 144 link embeddings) ---
        if use_spatial_attn:
            spatial_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.spatial_encoder = nn.TransformerEncoder(
                spatial_layer,
                num_layers=1,
                norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )

        # --- Classifier head ---
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input into per-link embeddings.

        Args:
            x: (B, T, H, W)  where T = WINDOW_SIZE+1 (e.g. 11)
               x[:, :T-2]  = T-2 signed percentage-change delta frames
               x[:, T-2:]  = 2 Pareto summary frames (x_m, α)
        Returns:
            emb: (B, H*W, d_model)
        """
        B, T, H, W = x.shape
        L = H * W       # 144
        n_deltas = T - 2  # 9 actual temporal steps

        # ── Fix 1a: split delta sequence from Pareto stats ──────────────────
        # delta_seq: signed temporal frames → (B*L, n_deltas, 1)
        delta_seq = (
            x[:, :n_deltas]              # (B, n_deltas, H, W)
            .permute(0, 2, 3, 1)         # (B, H, W, n_deltas)
            .reshape(B * L, n_deltas, 1)
        )
        # pareto_stats: x_m and α for each link → (B*L, 2)
        pareto_stats = (
            x[:, n_deltas:]              # (B, 2, H, W)
            .permute(0, 2, 3, 1)         # (B, H, W, 2)
            .reshape(B * L, 2)
        )

        # ── Stream A: temporal Transformer on delta sequence ────────────────
        x_proj = self.input_proj(delta_seq)    # (B*L, n_deltas, d_model)
        x_proj = self.pos_enc(x_proj)
        x_enc  = self.temporal_encoder(x_proj) # (B*L, n_deltas, d_model)

        # ── Fix 2: last-token pooling (most recent delta) ───────────────────
        x_last = x_enc[:, -1, :]               # (B*L, d_model)

        # ── Stream B: Pareto stats projection ───────────────────────────────
        pareto_emb = self.pareto_proj(pareto_stats)  # (B*L, d_model//2)

        # ── Fusion ───────────────────────────────────────────────────────────
        fused = self.fusion(
            torch.cat([x_last, pareto_emb], dim=-1)  # (B*L, d_model + d_model//2)
        )                                             # → (B*L, d_model)

        # ── Spatial attention across 144 links ──────────────────────────────
        x_spatial = fused.reshape(B, L, self.d_model)
        if self.use_spatial_attn:
            x_spatial = self.spatial_encoder(x_spatial)  # (B, 144, d_model)

        return x_spatial  # (B, 144, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W)
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, T, H, W = x.shape
        L = H * W

        emb = self.encode(x)                         # (B, 144, d_model)
        emb_flat = self.dropout(emb.reshape(B * L, self.d_model))
        logits = self.classifier(emb_flat)            # (B*L, num_classes)
        logits = (
            logits
            .reshape(B, L, self.num_classes)
            .permute(0, 2, 1)
            .reshape(B, self.num_classes, H, W)
        )
        return logits


def get_transformer_classifier(device: torch.device) -> PerLinkTransformer:
    """Build a PerLinkTransformer and move it to `device`."""
    model = PerLinkTransformer()
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = PerLinkTransformer()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PerLinkTransformer parameters: {total:,}")

    x = torch.randn(4, 11, 12, 12)   # B=4, T=11, 12×12 matrix
    out = model(x)
    assert out.shape == (4, 3, 12, 12), f"Bad shape: {out.shape}"
    print("Forward pass shape:", out.shape)

    emb = model.encode(x)
    assert emb.shape == (4, 144, model.d_model), f"Bad embed shape: {emb.shape}"
    print("Encode shape:", emb.shape)
    print("All checks passed.")
