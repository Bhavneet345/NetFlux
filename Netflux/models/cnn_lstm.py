"""
Per-link LSTM classifier for traffic matrix trend prediction.

Optional GCN front-end: aggregate each timestep's 12×12 deltas over the Abilene
router graph (12 nodes), then fuse node embeddings per OD pair for the LSTM.

Output: (B, 3, 12, 12) logits, compatible with CrossEntropyLoss and train/evaluate.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow `python models/cnn_lstm.py` from repo tree (Netflux on path)
_NETFLUX_ROOT = Path(__file__).resolve().parent.parent
if str(_NETFLUX_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETFLUX_ROOT))

from config import (
    MATRIX_SIZE,
    WINDOW_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    USE_GNN,
)

from data.abilene_topology import get_edge_index

# Keep these for backward-compat imports (train.py calls get_classifier)
CNN_OUT_CHANNELS = 32  # unused but referenced by config
CNN_KERNEL_SIZE = 3    # unused but referenced by config

GNN_OUT_DIM = 16


def _batch_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """Disjoint union of `batch_size` copies of the same graph (node ids offset by b * num_nodes)."""
    device = edge_index.device
    E = edge_index.size(1)
    offs = torch.arange(batch_size, device=device, dtype=torch.long) * num_nodes
    src = edge_index[0].unsqueeze(0).expand(batch_size, E) + offs.unsqueeze(1)
    dst = edge_index[1].unsqueeze(0).expand(batch_size, E) + offs.unsqueeze(1)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)


class PerLinkLSTM(nn.Module):
    """
    LSTM classifier applied independently to each link's sequence.

    Forward:
      x: (B, T, 12, 12)
      Returns: (B, num_classes, 12, 12)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        matrix_size: int = MATRIX_SIZE,
        use_gnn: bool | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.matrix_size = matrix_size
        self.num_nodes = matrix_size
        self.num_links = matrix_size * matrix_size
        self.use_gnn = USE_GNN if use_gnn is None else use_gnn

        if self.use_gnn:
            try:
                from torch_geometric.nn import GCNConv
            except ImportError as e:
                raise ImportError("Run: pip install torch_geometric") from e

            self.gcn = GCNConv(2, GNN_OUT_DIM)  # 2 features: outgoing avg + incoming avg
            self.link_fusion = nn.Linear(GNN_OUT_DIM * 2 + 1, GNN_OUT_DIM)
            self.fusion_norm = nn.LayerNorm(GNN_OUT_DIM)
            self.fusion_act = nn.Tanh()
            self.register_buffer("edge_index", get_edge_index(), persistent=False)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, GNN_OUT_DIM),
                nn.LayerNorm(GNN_OUT_DIM),
                nn.Tanh(),
            )

        self.lstm = nn.LSTM(
            input_size=GNN_OUT_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def _encode_timestep_gnn(self, xt: torch.Tensor) -> torch.Tensor:
        """
        xt: (B, 12, 12) — one delta matrix per batch item.
        Returns: (B, 12, 12, GNN_OUT_DIM) per-link features for this timestep.
        """
        B, H, W = xt.shape
        assert H == W == self.num_nodes

        # Two features per router: outgoing avg and incoming avg — keeps direction info
        # outgoing: row mean = avg traffic leaving this router
        # incoming: col mean = avg traffic arriving at this router
        row_m = xt.mean(dim=2)                          # (B, 12) outgoing avg
        col_m = xt.mean(dim=1)                          # (B, 12) incoming avg
        node_x = torch.stack([row_m, col_m], dim=-1)   # (B, 12, 2)

        x_flat = node_x.reshape(B * H, 2)              # (B*12, 2)
        ei = self.edge_index.to(xt.device)
        ei_batched = _batch_edge_index(ei, B, H)
        node_emb = self.gcn(x_flat, ei_batched)  # (B*12, GNN_OUT_DIM)
        emb = node_emb.view(B, H, GNN_OUT_DIM)  # (B, 12, 16)

        emb_i = emb.unsqueeze(2).expand(B, H, W, GNN_OUT_DIM)
        emb_j = emb.unsqueeze(1).expand(B, H, W, GNN_OUT_DIM)
        delta = xt.unsqueeze(-1)
        fused = torch.cat([emb_i, emb_j, delta], dim=-1)  # (B, 12, 12, 33)
        z = self.link_fusion(fused)
        z = self.fusion_norm(z)
        z = self.fusion_act(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape

        if self.use_gnn:
            feats = []
            for t in range(T):
                z_t = self._encode_timestep_gnn(x[:, t, :, :])
                feats.append(z_t.unsqueeze(1))
            x_seq = torch.cat(feats, dim=1)  # (B, T, H, W, GNN_OUT_DIM)
            x_seq = x_seq.reshape(B, T, H * W, GNN_OUT_DIM).permute(0, 2, 1, 3)
            x_seq = x_seq.reshape(B * H * W, T, GNN_OUT_DIM)
        else:
            x_seq = (
                x.reshape(B, T, H * W)
                .permute(0, 2, 1)
                .reshape(B * H * W, T, 1)
            )
            x_seq = self.input_proj(x_seq)

        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]
        last = self.dropout(last)
        logits = self.fc(last)
        logits = logits.reshape(B, H * W, self.num_classes)
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(B, self.num_classes, H, W)
        return logits


# ---------------------------------------------------------------------------
# Legacy stubs
# ---------------------------------------------------------------------------


class CNNLSTMClassifier(PerLinkLSTM):
    """Alias for backward compatibility with existing imports."""

    def __init__(self, window_size=WINDOW_SIZE, **kwargs):
        super().__init__(**kwargs)


class CNNLSTM(nn.Module):
    """Regression stub (unchanged interface, not used in classification)."""

    def __init__(self, window_size=WINDOW_SIZE, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(
            1,
            LSTM_HIDDEN_SIZE,
            LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE, MATRIX_SIZE * MATRIX_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        x = x.reshape(B, T, H * W).permute(0, 2, 1).reshape(B * H * W, T, 1)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1])
        return logits.reshape(B, H, W)


def get_model(device: torch.device) -> CNNLSTM:
    return CNNLSTM().to(device)


def get_classifier(device: torch.device) -> PerLinkLSTM:
    """Build per-link LSTM classifier and move to device."""
    model = PerLinkLSTM()
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = PerLinkLSTM()
    x = torch.randn(4, 11, 12, 12)  # 9 deltas + Pareto x_m + α (matches relative dataset)
    out = model(x)
    assert out.shape == (4, 3, 12, 12), f"Wrong shape: {out.shape}"
    print("Shape check passed:", out.shape)