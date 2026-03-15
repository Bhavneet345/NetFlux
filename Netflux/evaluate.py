"""
Evaluation script: load best classifier, report CNN-LSTM accuracy.
"""

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
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)
from dataset import load_abilene_matrices, chronological_split, TrafficMatrixDataset
from models.cnn_lstm import CNNLSTMClassifier
from utils.metrics import compute_classification_metrics

CLASS_NAMES = ["Decreasing", "Stable", "Increasing"]


def main() -> None:
    set_seed(SEED)
    device = get_device()

    data = load_abilene_matrices()
    _, _, test_data = chronological_split(data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    test_ds = TrafficMatrixDataset(
        test_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    ckpt_path = CHECKPOINT_DIR / "best_cnn_lstm.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
        return
    model = CNNLSTMClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
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

    pred_cnn = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)
    m = compute_classification_metrics(pred_cnn, target, num_classes=NUM_CLASSES)

    print("CNN-LSTM:")
    print(f"  Accuracy:    {m['accuracy']:.4f}")
    print(f"  Macro F1:   {m['macro_f1']:.4f}")
    print("  Per-class accuracy:")
    for c, name_c in enumerate(CLASS_NAMES):
        print(f"    {name_c}: {m['per_class_accuracy'][c]:.4f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print(m["confusion_matrix"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_metrics.txt", "w") as f:
        f.write("CNN-LSTM:\n")
        f.write(f"  Accuracy:    {m['accuracy']:.4f}\n")
        f.write(f"  Macro F1:   {m['macro_f1']:.4f}\n")
        f.write("  Per-class accuracy:\n")
        for c, name_c in enumerate(CLASS_NAMES):
            f.write(f"    {name_c}: {m['per_class_accuracy'][c]:.4f}\n")
        f.write("  Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(m["confusion_matrix"]) + "\n")


if __name__ == "__main__":
    main()
