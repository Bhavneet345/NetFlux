"""
Evaluation script: load the best Per-Link LSTM checkpoint and report test metrics.

Writes `outputs/test_metrics.txt`. For the transformer pipeline, see metrics from
`train_rl.py` (`outputs/rl_test_metrics.txt`) or run your own eval on
`checkpoints/best_transformer.pt`.
"""

from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from config import (
    set_seed,
    SEED,
    get_device,
    WINDOW_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    TOLERANCE,
    NUM_CLASSES,
    AGGREGATION_STEPS,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)
from dataset import load_abilene_matrices, aggregate_matrices, chronological_split, TrafficMatrixDataset
from models.cnn_lstm import CNNLSTMClassifier
from utils.metrics import compute_classification_metrics

CLASS_NAMES = ["Decreasing", "Stable", "Increasing"]


def evaluate_checkpoint_and_return_metrics(
    checkpoint_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """
    Load checkpoint, run on test_loader, return metrics dict (accuracy, macro_f1, per_class_f1, etc.).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = CNNLSTMClassifier().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_pred, all_target = [], []
    with torch.no_grad():
        for seq, target in test_loader:
            seq = seq.to(device)
            logits = model(seq)
            pred = logits.argmax(dim=1)
            all_pred.append(pred)
            all_target.append(target.to(device))

    pred = torch.cat(all_pred, dim=0)
    target_t = torch.cat(all_target, dim=0)
    return compute_classification_metrics(pred, target_t, num_classes=num_classes)


def main() -> None:
    set_seed(SEED)
    device = get_device()

    data = load_abilene_matrices()
    data = aggregate_matrices(data, AGGREGATION_STEPS)
    _, _, test_data = chronological_split(data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    test_ds = TrafficMatrixDataset(
        test_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    ckpt_path = CHECKPOINT_DIR / "best_cnn_lstm.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
        return
    m = evaluate_checkpoint_and_return_metrics(ckpt_path, test_loader, device, NUM_CLASSES)

    print("Per-Link LSTM (test set)")
    print("-" * 40)
    print(f"Accuracy: {m['accuracy']:.2f}")
    print()
    print("F1 per class:")
    for c, name_c in enumerate(CLASS_NAMES):
        print(f"  {name_c}: {m['per_class_f1'][c]:.2f}")
    print(f"Macro F1:   {m['macro_f1']:.2f}")
    print()
    print("Per-class precision:")
    for c, name_c in enumerate(CLASS_NAMES):
        print(f"  {name_c}: {m['per_class_precision'][c]:.2f}")
    print("Per-class recall:")
    for c, name_c in enumerate(CLASS_NAMES):
        print(f"  {name_c}: {m['per_class_recall'][c]:.2f}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(m["confusion_matrix"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_metrics.txt", "w") as f:
        f.write(f"Accuracy: {m['accuracy']:.2f}\n")
        f.write("F1 per class:\n")
        for c, name_c in enumerate(CLASS_NAMES):
            f.write(f"  {name_c}: {m['per_class_f1'][c]:.2f}\n")
        f.write(f"Macro F1:   {m['macro_f1']:.2f}\n")
        f.write("Per-class precision: " + str(m["per_class_precision"]) + "\n")
        f.write("Per-class recall: " + str(m["per_class_recall"]) + "\n")
        f.write("Confusion matrix:\n" + str(m["confusion_matrix"]) + "\n")


if __name__ == "__main__":
    main()
