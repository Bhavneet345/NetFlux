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


def compute_labels(M_curr: np.ndarray, M_next: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Compute per-link class labels for transition from M_curr (t) to M_next (t+1).
    Labels correspond to timestamp t+1.
    0 = Decreasing, 1 = Stable, 2 = Increasing.
    M_curr, M_next: (12, 12). Returns (12, 12) int64.
    """
    delta = M_next - M_curr
    labels = np.ones_like(M_curr, dtype=np.int64)  # Stable by default
    labels[delta > tolerance] = 2   # Increasing
    labels[delta < -tolerance] = 0  # Decreasing
    return labels


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
    ) -> None:
        """
        Args:
            data: (T, 12, 12) array, chronologically ordered.
            window_size: number of past matrices (k).
            tolerance: for classification, |delta| <= tolerance -> Stable (in same units as data).
            classification: if True, target is (12, 12) int labels; else (12, 12) float.
            scaler: optional; only for regression; transform seq and target.
        """
        self.data = data
        self.window_size = window_size
        self.tolerance = tolerance
        self.classification = classification
        self.scaler = scaler
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
            # Target: labels for transition t -> t+1
            target = compute_labels(self.data[t], self.data[t + 1], self.tolerance)
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
