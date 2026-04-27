"""
Configuration for Netflux traffic-matrix classification / forecasting.
Paths are anchored at this file's package root (`Netflux/`).
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
WINDOW_SIZE = 10   # 10 raw matrices → 9 delta + 2 Pareto MLE (x_m, α) frames = 11 timesteps (see dataset.py)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Temporal aggregation: average consecutive 5-min snapshots (30 min = 6 steps).
# Smoothes noise; labels and windows use this coarser series. Set to 1 to disable.
AGGREGATION_STEPS = 6

# Classification (link trend: Decreasing / Stable / Increasing)
LABEL_MODE = "relative"   # "absolute" or "relative"
TOLERANCE = 0.15        # 15% relative change (Stable if |Δ| ≤ this)
EPSILON = 1e-9            # div-by-zero protection
NUM_CLASSES = 3           # 0=Decreasing, 1=Stable, 2=Increasing
CLASS_WEIGHTS = "balanced"

# -----------------------------------------------------------------------------
# Model (Per-Link LSTM — CNN params kept for any legacy imports)
# -----------------------------------------------------------------------------
USE_GNN = True  # True: GCN over Abilene topology before LSTM; False: original Linear(1→16) projection

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
LEARNING_RATE = 1e-4      # was 3e-4 — gentler for Transformer pretrain (shared by train.py / train_transformer.py)
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 25  # was 15 — more room to escape val-loss plateau before RL
DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Transformer (PerLinkTransformer)
# -----------------------------------------------------------------------------
TRANSFORMER_D_MODEL = 64          # embedding dimension (divisible by TRANSFORMER_NHEAD)
TRANSFORMER_NHEAD = 4             # attention heads
TRANSFORMER_NUM_LAYERS = 3        # temporal encoder depth
TRANSFORMER_USE_SPATIAL_ATTN = True  # cross-link spatial attention after temporal encoder

# -----------------------------------------------------------------------------
# RL / REINFORCE with running baseline fine-tuning
# -----------------------------------------------------------------------------
RL_LR = 1e-4                      # learning rate (backbone gets RL_LR * 0.1)
RL_ENTROPY_COEF = 0.05            # entropy bonus — higher than PPO to prevent collapse
RL_EPOCHS = 20                    # total fine-tuning epochs
RL_FREEZE_BACKBONE = False        # True → freeze encoder, only update classifier head

# Running-mean baseline: b ← β·b + (1−β)·mean(batch_rewards)
# β close to 1 = slow-moving baseline; tracks the long-run mean reward
RL_BASELINE_BETA = 0.99

# Shaped reward — softer wrong-prediction penalty avoids the all-Stable collapse
# that occurred with -1.0: minority classes were punished more harshly than they
# were rewarded, so the policy converged to always predicting Stable.
RL_REWARD_CORRECT_MINORITY = 3.0  # correct Decreasing (0) or Increasing (2)
RL_REWARD_CORRECT_STABLE = 1.0    # correct Stable (1)
RL_REWARD_WRONG = -0.5            # any wrong prediction (was -1.0 in PPO version)

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