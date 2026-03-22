"""
Per-link LSTM classifier for traffic matrix trend prediction.

Architecture: treat each of the 144 links as an independent time series.
Input (B, T, 12, 12) is reshaped to (B*144, T, 1) so the LSTM sees one
link at a time — eliminating the false spatial assumption of the CNN and
the catastrophic 4608→64 bottleneck of the old CNNLSTMClassifier.

Output: (B, 3, 12, 12) logits, compatible with CrossEntropyLoss and the
existing train.py / evaluate.py without any other changes.
"""

import torch
import torch.nn as nn

from config import (
    MATRIX_SIZE,
    WINDOW_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
)

# Keep these for backward-compat imports (train.py calls get_classifier)
CNN_OUT_CHANNELS = 32  # unused but referenced by config
CNN_KERNEL_SIZE = 3    # unused but referenced by config


class PerLinkLSTM(nn.Module):
    """
    LSTM classifier applied independently to each link's delta sequence.

    Forward pass:
      x: (B, T, 12, 12)  — T delta frames (window_size - 1 in relative mode)
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.matrix_size = matrix_size
        self.num_links = matrix_size * matrix_size  # 144

        # Input projection: scale the single percentage-change value to a richer embedding
        # before the LSTM sees it. This gives the LSTM more to work with per timestep.
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
        )

        # Single LSTM shared across all links (parameter efficiency)
        self.lstm = nn.LSTM(
            input_size=16,           # matches input_proj output
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, H, W)  where H=W=12, T = number of delta timesteps
        Returns: (B, num_classes, H, W)
        """
        B, T, H, W = x.shape

        # Reshape: treat each link as a separate sequence in the batch
        # (B, T, H, W) -> (B, T, H*W) -> (B, H*W, T) -> (B*H*W, T, 1)
        x = x.reshape(B, T, H * W)          # (B, T, 144)
        x = x.permute(0, 2, 1)              # (B, 144, T)
        x = x.reshape(B * H * W, T, 1)      # (B*144, T, 1)

        # Project scalar percentage-change to richer embedding
        x = self.input_proj(x)              # (B*144, T, 16)

        # LSTM: output last hidden state
        out, _ = self.lstm(x)               # (B*144, T, hidden)
        last = out[:, -1, :]                # (B*144, hidden)
        last = self.dropout(last)

        # Classify
        logits = self.fc(last)              # (B*144, num_classes)

        # Reshape back to (B, num_classes, H, W)
        logits = logits.reshape(B, H * W, self.num_classes)   # (B, 144, 3)
        logits = logits.permute(0, 2, 1)                       # (B, 3, 144)
        logits = logits.reshape(B, self.num_classes, H, W)     # (B, 3, 12, 12)
        return logits


# ---------------------------------------------------------------------------
# Legacy stubs — kept so that any code importing CNNLSTM / CNNLSTMClassifier
# still works, but both now delegate to PerLinkLSTM internally.
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
        self.lstm = nn.LSTM(1, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0)
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