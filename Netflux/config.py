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
DATA_DIR = PROJECT_ROOT.parent / "Datasets" / "Abilene" / "2004"
MEASURED_DIR = DATA_DIR / "Measured"
PREPROCESSED_NPY = PROJECT_ROOT / "data" / "abilene_2004_Tx12x12.npy"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCALER_PATH = CHECKPOINT_DIR / "scaler.npz"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
MATRIX_SIZE = 12   # 12x12 traffic matrix
WINDOW_SIZE = 10   # 10 windows × 30min = 5 hours of history → 9 delta frames
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Temporal aggregation: average consecutive 5-min snapshots (30 min = 6 steps).
# Smoothes noise; labels and windows use this coarser series. Set to 1 to disable.
AGGREGATION_STEPS = 6

# Classification (link trend: Decreasing / Stable / Increasing)
LABEL_MODE = "relative"   # "absolute" or "relative"
TOLERANCE = 0.1         # 10% relative change
EPSILON = 1e-9            # div-by-zero protection
NUM_CLASSES = 3           # 0=Decreasing, 1=Stable, 2=Increasing
CLASS_WEIGHTS = "balanced"

# -----------------------------------------------------------------------------
# Model (Per-Link LSTM — CNN params kept for any legacy imports)
# -----------------------------------------------------------------------------
CNN_OUT_CHANNELS = 32    # unused by PerLinkLSTM, kept for import compatibility
CNN_KERNEL_SIZE = 3      # unused by PerLinkLSTM, kept for import compatibility
LSTM_HIDDEN_SIZE = 128   # increased: more context (10 windows) needs more capacity
LSTM_NUM_LAYERS = 2
DROPOUT = 0.3            # slightly higher to match increased hidden size

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
BATCH_SIZE = 128          # larger batch: stable gradients for per-link processing
EPOCHS = 100
LEARNING_RATE = 3e-4      # lower LR for deeper context window
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15  # longer patience: 10-window model takes more epochs to converge
DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
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