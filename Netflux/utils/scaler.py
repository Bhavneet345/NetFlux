"""
Standardization scaler for traffic matrices.
Per-element (12×12) μ and σ fitted on train split only; no data leakage.
"""

from pathlib import Path
from typing import Union

import numpy as np


EPSILON = 1e-8  # avoid divide-by-zero when std is 0


class TrafficScaler:
    """
    Standardization: X' = (X - μ) / σ.
    μ and σ are (12, 12) per-element statistics, fitted on training data only.
    """

    def __init__(self, epsilon: float = EPSILON) -> None:
        self.epsilon = epsilon
        self._mean: np.ndarray = None  # (12, 12)
        self._std: np.ndarray = None   # (12, 12)

    def fit(self, train_data: np.ndarray) -> "TrafficScaler":
        """
        Compute μ and σ from training data only.
        train_data: (T, 12, 12).
        """
        if train_data.ndim != 3:
            raise ValueError("train_data must be (T, 12, 12)")
        self._mean = np.mean(train_data, axis=0).astype(np.float32)  # (12, 12)
        self._std = np.std(train_data, axis=0).astype(np.float32)    # (12, 12)
        self._std = np.maximum(self._std, self.epsilon)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply standardization: (data - μ) / σ.
        data: (..., 12, 12) or (12, 12). Broadcasts μ, σ over leading dims.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Scaler not fitted. Call fit(train_data) first.")
        return ((data - self._mean) / self._std).astype(np.float32)

    def inverse_transform(self, data: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Reverse standardization: data * σ + μ.
        data: (..., 12, 12). Returns numpy in original units (Gbytes/s).
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Scaler not fitted. Call fit(train_data) first.")
        if hasattr(data, "cpu"):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        return (data * self._std + self._mean).astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Save mean and std to .npz for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self._mean, std=self._std)

    @classmethod
    def load(cls, path: Union[str, Path], epsilon: float = EPSILON) -> "TrafficScaler":
        """Load mean and std from .npz."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        data = np.load(path)
        scaler = cls(epsilon=epsilon)
        scaler._mean = data["mean"].astype(np.float32)
        scaler._std = np.maximum(data["std"].astype(np.float32), epsilon)
        return scaler
