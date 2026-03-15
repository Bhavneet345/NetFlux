"""
Baseline predictors for traffic matrix forecasting.
Regression: persistence and mean_last_k return (batch_size, 12, 12) float.
Classification: persistence_labels and mean_labels return (batch_size, 12, 12) long (0/1/2).
"""

import torch
import numpy as np
from typing import Optional

from config import TOLERANCE


def persistence(seq: torch.Tensor) -> torch.Tensor:
    """Regression: predict M_t = M_{t-1}. seq: (B, k, 12, 12) -> (B, 12, 12)."""
    return seq[:, -1].clone()


def mean_last_k(seq: torch.Tensor) -> torch.Tensor:
    """Regression: predict next matrix as mean of last k. seq: (B, k, 12, 12) -> (B, 12, 12)."""
    return seq.mean(dim=1)


def _labels_from_delta(delta: torch.Tensor, tolerance: float) -> torch.Tensor:
    """delta (B, 12, 12) -> labels (B, 12, 12) long: 0=Decreasing, 1=Stable, 2=Increasing."""
    labels = torch.ones_like(delta, dtype=torch.long, device=delta.device)
    labels[delta > tolerance] = 2
    labels[delta < -tolerance] = 0
    return labels


def persistence_labels(seq: torch.Tensor, tolerance: float = TOLERANCE) -> torch.Tensor:
    """
    Classification: label from M[t-1] vs M[t] (last two in window).
    seq: (B, k, 12, 12). Returns (B, 12, 12) long.
    """
    M_prev = seq[:, -2]  # (B, 12, 12)
    M_curr = seq[:, -1]
    delta = (M_curr - M_prev).float()
    return _labels_from_delta(delta, tolerance)


def mean_labels(seq: torch.Tensor, tolerance: float = TOLERANCE) -> torch.Tensor:
    """
    Classification: label from mean(last k) vs M[t] (last in window).
    seq: (B, k, 12, 12). Returns (B, 12, 12) long.
    """
    mean_mat = seq.mean(dim=1)  # (B, 12, 12)
    M_curr = seq[:, -1]
    delta = (M_curr - mean_mat).float()
    return _labels_from_delta(delta, tolerance)


def run_baseline(
    name: str,
    seq: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    classification: bool = False,
) -> torch.Tensor:
    """Run baseline by name. classification=True -> return labels (B, 12, 12) long."""
    if classification:
        if name == "persistence":
            return persistence_labels(seq)
        if name == "mean":
            return mean_labels(seq)
    else:
        if name == "persistence":
            return persistence(seq)
        if name == "mean":
            return mean_last_k(seq)
    raise ValueError(f"Unknown baseline: {name}")
