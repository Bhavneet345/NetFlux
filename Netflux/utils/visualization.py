"""
Visualization utilities for traffic matrix forecasting.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_traffic_matrix(
    matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Traffic matrix",
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot a single 12x12 traffic matrix as heatmap."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Source")
    plt.colorbar(im, ax=ax, label="Traffic (Gbytes/s)")
    return ax


def plot_prediction_vs_actual(
    pred: np.ndarray,
    actual: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Side-by-side heatmaps: predicted vs actual traffic matrix."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if pred.ndim == 3:
        pred, actual = pred[0], actual[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax = max(np.max(pred), np.max(actual)) or 1e-9
    axes[0].imshow(pred, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    axes[0].set_title("Predicted")
    axes[0].set_xlabel("Destination")
    axes[0].set_ylabel("Source")
    im = axes[1].imshow(actual, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    axes[1].set_title("Actual")
    axes[1].set_xlabel("Destination")
    axes[1].set_ylabel("Source")
    plt.colorbar(im, ax=axes, label="Traffic (Gbytes/s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[Path] = None,
) -> None:
    """Plot train and validation loss over epochs."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses, label="Val loss", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
