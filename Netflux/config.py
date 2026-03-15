"""
Configuration for traffic matrix forecasting.
Paths are relative to project root (traffic_forecasting/).
"""

import os
import random
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# Abilene 2004 data: parent/Datasets/Abilene/2004
DATA_DIR = PROJECT_ROOT.parent / "Datasets" / "Abilene" / "2004"
MEASURED_DIR = DATA_DIR / "Measured"
# Preprocessed array (optional): if present, load this instead of .dat files
PREPROCESSED_NPY = PROJECT_ROOT / "data" / "abilene_2004_Tx12x12.npy"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCALER_PATH = CHECKPOINT_DIR / "scaler.npz"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
MATRIX_SIZE = 12  # 12x12 traffic matrix
WINDOW_SIZE = 10  # number of past matrices to predict next
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classification (link trend: Decreasing / Stable / Increasing)
TOLERANCE = 0.01  # Gbytes/s; |delta| <= tolerance -> Stable
NUM_CLASSES = 3   # 0=Decreasing, 1=Stable, 2=Increasing
# Class weights for CrossEntropyLoss: "balanced" = inverse frequency (recommended for imbalanced data), None = no weighting
CLASS_WEIGHTS = "balanced"

# -----------------------------------------------------------------------------
# Model (CNN-LSTM)
# -----------------------------------------------------------------------------
CNN_OUT_CHANNELS = 32
CNN_KERNEL_SIZE = 3
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
DROPOUT = 0.2

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
# Device: prefer cuda, then MPS (Apple Metal / M-series GPU), then cpu
DEVICE = "cuda"  # fallback if get_device() not used

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42


def get_device():
    """Return best available device: CUDA > MPS (Apple Metal) > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (M1/M2/M3/M4) GPU
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
