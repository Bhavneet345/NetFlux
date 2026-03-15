"""
Single-sample inspection for the CNN-LSTM classifier.
Given one test window (k past matrices), show predicted vs actual trend labels
for the next 12x12 traffic matrix and basic metrics for that sample.
"""

import numpy as np
import torch

from config import (
    set_seed,
    SEED,
    get_device,
    WINDOW_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    TOLERANCE,
    CHECKPOINT_DIR,
)
from dataset import load_abilene_matrices, chronological_split, TrafficMatrixDataset
from models.cnn_lstm import CNNLSTMClassifier
from utils.metrics import accuracy, confusion_matrix

CLASS_NAMES = ["Decreasing", "Stable", "Increasing"]


def main() -> None:
    # Reproducibility and device
    set_seed(SEED)
    device = get_device()

    # Load full sequence and build classification test dataset
    data = load_abilene_matrices()
    _, _, test_data = chronological_split(data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    test_ds = TrafficMatrixDataset(
        test_data,
        window_size=WINDOW_SIZE,
        tolerance=TOLERANCE,
        classification=True,
    )

    # Load trained classifier
    model = CNNLSTMClassifier().to(device)
    ckpt = torch.load(CHECKPOINT_DIR / "best_cnn_lstm.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Choose one sample from the test set (index 0 by default)
    idx = 0
    seq, target = test_ds[idx]
    seq = seq.unsqueeze(0).to(device)      # (1, k, 12, 12)
    target = target.unsqueeze(0).to(device)  # (1, 12, 12)

    with torch.no_grad():
        logits = model(seq)                # (1, 3, 12, 12)
        pred = logits.argmax(dim=1)        # (1, 12, 12)

    # Metrics for this single matrix
    acc = accuracy(pred, target)
    cm = confusion_matrix(pred, target, num_classes=3)

    print("Single test sample (timestep index in test set =", idx, ")")
    print("Input: previous", WINDOW_SIZE, "matrices → Output: 12x12 trend labels")
    print("-" * 60)
    print(f"Per-link accuracy for this matrix: {acc:.4f}")
    print("Confusion matrix for this matrix (rows=true, cols=pred):")
    print(cm)
    print("-" * 60)

    pred_np = pred.cpu().numpy()[0]
    target_np = target.cpu().numpy()[0]

    print("Predicted labels (top-left 4x4):")
    print(pred_np[:4, :4])
    print("Actual labels (top-left 4x4):")
    print(target_np[:4, :4])

    # Optional: show class counts for this matrix
    unique_pred, counts_pred = np.unique(pred_np, return_counts=True)
    unique_true, counts_true = np.unique(target_np, return_counts=True)
    print("-" * 60)
    print("Predicted class distribution (this matrix):")
    for u, c in zip(unique_pred, counts_pred):
        name = CLASS_NAMES[int(u)] if int(u) < len(CLASS_NAMES) else str(u)
        print(f"  {name} ({int(u)}): {c}")
    print("True class distribution (this matrix):")
    for u, c in zip(unique_true, counts_true):
        name = CLASS_NAMES[int(u)] if int(u) < len(CLASS_NAMES) else str(u)
        print(f"  {name} ({int(u)}): {c}")


if __name__ == "__main__":
    main()
