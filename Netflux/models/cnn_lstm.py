"""
CNN-LSTM model for traffic matrix forecasting.
CNN per timestep -> spatial features -> LSTM over time -> FC -> 12x12.
"""

import torch
import torch.nn as nn

from config import (
    MATRIX_SIZE,
    WINDOW_SIZE,
    CNN_OUT_CHANNELS,
    CNN_KERNEL_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
)


class CNNEncoder(nn.Module):
    """Extract spatial features from a single 12x12 traffic matrix."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = CNN_OUT_CHANNELS,
        kernel_size: int = CNN_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        )
        # 12x12 -> same size; then flatten
        self.out_dim = out_channels * MATRIX_SIZE * MATRIX_SIZE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 12, 12) or (B, 12, 12) -> (B, C, 12, 12)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.conv(x)  # (B, C, 12, 12)
        return out.flatten(1)  # (B, C*12*12)


class CNNLSTM(nn.Module):
    """
    CNN applied per timestep -> LSTM over time -> FC -> 144 -> reshape 12x12.
    Input: (batch_size, k, 12, 12). Output: (batch_size, 12, 12).
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        cnn_out_channels: int = CNN_OUT_CHANNELS,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.cnn = CNNEncoder(1, cnn_out_channels, CNN_KERNEL_SIZE)
        cnn_feat_dim = self.cnn.out_dim
        self.lstm = nn.LSTM(
            cnn_feat_dim,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden_size, MATRIX_SIZE * MATRIX_SIZE)
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
        # x: (B, k, 12, 12)
        B, k, H, W = x.shape
        # Encode each timestep: (B, k, 12, 12) -> (B, k, cnn_feat_dim)
        feats = []
        for t in range(k):
            feats.append(self.cnn(x[:, t]))
        feats = torch.stack(feats, dim=1)  # (B, k, cnn_feat_dim)
        out, _ = self.lstm(feats)  # (B, k, lstm_hidden)
        last_hidden = out[:, -1]  # (B, lstm_hidden)
        logits = self.fc(last_hidden)  # (B, 144)
        return logits.view(B, MATRIX_SIZE, MATRIX_SIZE)


def get_model(device: torch.device) -> CNNLSTM:
    """Build CNN-LSTM (regression) and move to device."""
    model = CNNLSTM()
    return model.to(device)


class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM for classification: output (batch_size, 3, 12, 12) logits per link.
    0=Decreasing, 1=Stable, 2=Increasing. No softmax in model (use with CrossEntropyLoss).
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        cnn_out_channels: int = CNN_OUT_CHANNELS,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_classes = num_classes
        self.cnn = CNNEncoder(1, cnn_out_channels, CNN_KERNEL_SIZE)
        cnn_feat_dim = self.cnn.out_dim
        self.lstm = nn.LSTM(
            cnn_feat_dim,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden_size, MATRIX_SIZE * MATRIX_SIZE * num_classes)
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
        B, k, H, W = x.shape
        feats = []
        for t in range(k):
            feats.append(self.cnn(x[:, t]))
        feats = torch.stack(feats, dim=1)
        out, _ = self.lstm(feats)
        last_hidden = out[:, -1]
        logits = self.fc(last_hidden)  # (B, 144*3)
        return logits.view(B, self.num_classes, MATRIX_SIZE, MATRIX_SIZE)


def get_classifier(device: torch.device) -> CNNLSTMClassifier:
    """Build CNN-LSTM classifier and move to device."""
    model = CNNLSTMClassifier()
    return model.to(device)
