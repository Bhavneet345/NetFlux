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
    (T, 12, 12) -> (T//steps, 12, 12). With 5-min raw data, steps=6 -> 30-min resolution.
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
    Classification: input (k, 12, 12), target (12, 12) of labels 0/1/2.
    Regression (classification=False): input (k, 12, 12), target (12, 12) float; optional scaler.
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

        # Data-relative denominator floor for percentage changes:
        # 1% of the 10th percentile of non-zero values in the dataset.
        if relative:
            nonzero = data[data > 0]
            if len(nonzero) > 0:
                self.pct_floor = float(np.percentile(nonzero, 10) * 0.01)
            else:
                self.pct_floor = epsilon
        else:
            self.pct_floor = epsilon
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
                # Feed percentage changes instead of raw values: (k-1, 12, 12)
                # pct_floor is data-relative: 1% of 10th-percentile non-zero traffic
                seq = np.diff(seq, axis=0) / (np.abs(seq[:-1]) + self.pct_floor)
                np.clip(seq, -3.0, 3.0, out=seq)
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