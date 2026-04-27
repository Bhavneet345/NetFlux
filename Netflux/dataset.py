"""
Data loader and sliding-window dataset for traffic matrix forecasting.
Supports classification: predict per-link trend (Decreasing / Stable / Increasing).
"""

import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    MEASURED_DIR,
    PREPROCESSED_NPY,
    WINDOW_SIZE,
    MATRIX_SIZE,
    TOLERANCE,
    LABEL_MODE,
    EPSILON,
    set_seed,
)
from utils.scaler import TrafficScaler


def load_abilene_matrices(data_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load traffic matrices as (T, 12, 12).
    Uses PREPROCESSED_NPY if it exists; otherwise loads from .dat files in MEASURED_DIR.
    """
    data_dir = data_dir or MEASURED_DIR
    npy_path = PREPROCESSED_NPY

    if npy_path is not None and npy_path.exists():
        data = np.load(npy_path).astype(np.float32)
        assert data.ndim == 3 and data.shape[1] == data.shape[2] == MATRIX_SIZE
        return data

    pattern = os.path.join(data_dir, "tm.*.dat")
    dat_files = sorted(glob.glob(pattern))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found at {pattern}")

    rows = []
    for path in dat_files:
        arr = np.loadtxt(path, delimiter=",", comments="#")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.size != MATRIX_SIZE * MATRIX_SIZE:
            arr = arr.flatten()
            if arr.size != MATRIX_SIZE * MATRIX_SIZE:
                continue
        mat = arr.reshape(MATRIX_SIZE, MATRIX_SIZE)
        rows.append(mat)

    return np.stack(rows, axis=0)


def aggregate_matrices(data: np.ndarray, steps: int) -> np.ndarray:
    """
    Temporal aggregation: average every `steps` consecutive matrices.
    (T, 12, 12) -> (T // steps, 12, 12). With 5-min raw data, steps=6 -> 30-min resolution.
    """
    if steps <= 1:
        return data
    t = data.shape[0]
    t_new = t // steps
    return data[: t_new * steps].reshape(t_new, steps, *data.shape[1:]).mean(axis=1).astype(np.float32)


def compute_labels(
    M_curr: np.ndarray,
    M_next: np.ndarray,
    tolerance: float,
    relative: bool = False,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Compute per-link class labels for transition from M_curr (t) to M_next (t+1).
    0 = Decreasing, 1 = Stable, 2 = Increasing.

    When relative=True, tolerance is a fraction (e.g. 0.05 = 5%) and the
    comparison uses per-link percentage change: delta / (|M_curr| + epsilon).
    """
    delta = M_next - M_curr
    if relative:
        delta = delta / (np.abs(M_curr) + epsilon)
    labels = np.ones_like(M_curr, dtype=np.int64)  # Stable by default
    labels[delta > tolerance] = 2   # Increasing
    labels[delta < -tolerance] = 0  # Decreasing
    return labels


def get_class_distribution(
    data: np.ndarray,
    tolerance: float,
    num_classes: int = 3,
    relative: bool = False,
    epsilon: float = 1e-9,
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """
    Compute class distribution over all consecutive pairs in data.
    data: (T, 12, 12). Uses transitions t -> t+1 for t in 0..T-2.
    Returns (counts, percentages) where counts and percentages are length num_classes.
    """
    counts = [0] * num_classes
    for t in range(data.shape[0] - 1):
        labels = compute_labels(data[t], data[t + 1], tolerance, relative=relative, epsilon=epsilon)
        for c in range(num_classes):
            counts[c] += int((labels == c).sum())
    total = sum(counts)
    pcts = tuple((counts[c] / total * 100.0) if total > 0 else 0.0 for c in range(num_classes))
    return tuple(counts), pcts


class TrafficMatrixDataset(Dataset):
    """
    Sliding-window dataset.

    Classification, relative mode: input (window_size + 1, 12, 12) — (window_size - 1) signed
    delta frames plus two Pareto MLE summary frames (scale x_m, shape α); target (12, 12) int.

    Classification, absolute mode: input (window_size, 12, 12) raw matrices; same target.

    Regression: input (window_size, 12, 12), target (12, 12) float; optional scaler.
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = WINDOW_SIZE,
        tolerance: float = TOLERANCE,
        classification: bool = True,
        scaler: Optional[TrafficScaler] = None,
        relative: bool = (LABEL_MODE == "relative"),
        epsilon: float = EPSILON,
    ) -> None:
        """
        Args:
            data: (T, 12, 12) array, chronologically ordered.
            window_size: number of past matrices (k).
            tolerance: threshold for classification (fraction when relative=True, absolute otherwise).
            classification: if True, target is (12, 12) int labels; else (12, 12) float.
            scaler: optional; only for regression; transform seq and target.
            relative: if True, use per-link percentage change for labeling.
            epsilon: div-by-zero protection for relative mode.
        """
        self.data = data
        self.window_size = window_size
        self.tolerance = tolerance
        self.classification = classification
        self.scaler = scaler
        self.relative = relative
        self.epsilon = epsilon

        # Denominator floor for percentage changes: 1% of the 10th percentile of
        # positive entries (scales with data units; avoids exploding deltas on tiny loads).
        nonzero = data[data > 0]
        self.pct_floor = float(np.percentile(nonzero, 10)) * 0.01 if len(nonzero) > 0 else epsilon
        self.T = data.shape[0]
        if classification:
            # Need M[t] and M[t+1]; t from window_size to T-2
            self.num_samples = max(0, self.T - self.window_size - 1)
        else:
            self.num_samples = max(0, self.T - self.window_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.window_size + idx
        seq = self.data[t - self.window_size : t].copy()  # (k, 12, 12)

        if self.classification:
            target = compute_labels(
                self.data[t], self.data[t + 1], self.tolerance,
                relative=self.relative, epsilon=self.epsilon,
            )
            if self.relative:
                seq_raw = seq.copy()
                deltas = np.diff(seq_raw, axis=0) / (np.abs(seq_raw[:-1]) + self.pct_floor)
                np.clip(deltas, -3.0, 3.0, out=deltas)        # (k-1, 12, 12)

                # Professor Abbas's Pareto distribution approach (Javadtalab et al. 2015)
                # |Δ| per link; Pareto MLE scale and shape as two summary frames.
                # Use absolute values because Pareto is defined for positive values only.
                abs_deltas = np.abs(deltas) + 1e-8   # (k-1, 12, 12) strictly positive

                # Pareto MLE estimates (Javadtalab et al. 2015)
                # Scale parameter x_m: minimum value in window per link
                # Shape parameter alpha: MLE estimate from window observations
                n = abs_deltas.shape[0]   # number of delta frames (k-1 = 9)

                x_m = abs_deltas.min(axis=0, keepdims=True)          # (1, 12, 12) scale
                x_m = np.maximum(x_m, 1e-8)                          # floor

                log_ratios = np.log(abs_deltas / x_m + 1e-8)         # (k-1, 12, 12)
                alpha = n / (log_ratios.sum(axis=0, keepdims=True) + 1e-8)  # (1, 12, 12) shape

                # Clip alpha to reasonable range (Pareto shape typically 0.5 - 10)
                alpha = np.clip(alpha, 0.1, 10.0)

                # Append x_m (scale) and alpha (shape) as two summary frames
                # Sequence: [δ1, δ2, ..., δ(k-1), x_m_frame, alpha_frame]
                seq = np.concatenate([deltas, x_m, alpha], axis=0)  # (k+1, 12, 12)
            if self.scaler is not None:
                seq = self.scaler.transform(seq)
            return (
                torch.from_numpy(seq).float(),
                torch.from_numpy(target).long(),
            )
        else:
            target = self.data[t].copy()
            if self.scaler is not None:
                seq = self.scaler.transform(seq)
                target = self.scaler.transform(target)
            return (
                torch.from_numpy(seq).float(),
                torch.from_numpy(target).float(),
            )


def chronological_split(
    data: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data chronologically into train, val, test (no shuffling)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    T = data.shape[0]
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    return data[:t1], data[t1:t2], data[t2:]


if __name__ == "__main__":
    set_seed()
    data = load_abilene_matrices()
    print("Data shape:", data.shape)
    train, val, test = chronological_split(data)
    ds = TrafficMatrixDataset(train, window_size=WINDOW_SIZE, classification=True)
    x, y = ds[0]
    print("Classification sample seq shape:", x.shape, "target shape:", y.shape, "dtype:", y.dtype)
    print("Label counts:", np.bincount(y.numpy().ravel(), minlength=3))

    # Test Pareto MLE (x_m, alpha) summary frames
    x, y = ds[0]
    T = x.shape[0]
    print(f"Sequence length with Pareto MLE frames: {T}")
    print(f"Expected: WINDOW_SIZE-1+2 = {WINDOW_SIZE + 1}")
    assert T == WINDOW_SIZE + 1, f"Wrong T: {T}"
    print("Dataset sequence-length check passed")