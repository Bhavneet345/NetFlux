"""
Training script for per-link LSTM link-trend classification.
Predict per-link trend (Decreasing / Stable / Increasing).
CrossEntropyLoss with optional class weights, Adam + weight decay, grad clipping, early stopping.
"""

from pathlib import Path

import torch
import torch.nn as nn
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
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    TOLERANCE,
    NUM_CLASSES,
    CLASS_WEIGHTS,
    AGGREGATION_STEPS,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)
from dataset import load_abilene_matrices, aggregate_matrices, chronological_split, TrafficMatrixDataset
from models.cnn_lstm import get_classifier
from utils.visualization import plot_training_curves


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    data = load_abilene_matrices()
    data = aggregate_matrices(data, AGGREGATION_STEPS)
    print(
        f"Temporal aggregation: {AGGREGATION_STEPS}x (5 min × {AGGREGATION_STEPS} = "
        f"{5 * AGGREGATION_STEPS} min) -> T={data.shape[0]}"
    )
    train_data, val_data, test_data = chronological_split(
        data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    train_ds = TrafficMatrixDataset(
        train_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    val_ds = TrafficMatrixDataset(
        val_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    CLASS_NAMES = ["Decreasing", "Stable", "Increasing"]
    class_counts = [0] * NUM_CLASSES
    for _, target in train_ds:
        for c in range(NUM_CLASSES):
            class_counts[c] += int((target.numpy() == c).sum())
    total_samples = sum(class_counts)

    print("Class distribution:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_counts[c]}")

    if CLASS_WEIGHTS == "balanced":
        weights = [
            total_samples / (NUM_CLASSES * (class_counts[c] + 1e-6))
            for c in range(NUM_CLASSES)
        ]
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        weight_tensor = weight_tensor / weight_tensor.mean()
        print("Class weights (normalized, mean=1):")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {weight_tensor[c].item():.4f}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    print("-" * 40)
    model = get_classifier(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,   # L2 regularization
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, target)
            loss.backward()
            # Gradient clipping — prevents exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        train_loss = epoch_train_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        model.eval()
        epoch_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(device), target.to(device)
                logits = model(seq)
                loss = criterion(logits, target)
                epoch_val_loss += loss.item() * seq.size(0)
                n_val += seq.size(0)
        val_loss = epoch_val_loss / max(n_val, 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}  Train loss: {train_loss:.6f}  Val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = CHECKPOINT_DIR / "best_cnn_lstm.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "seed": SEED,
            }, ckpt_path)
            print(f"  -> Saved best model to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    plot_training_curves(
        train_losses, val_losses,
        save_path=OUTPUT_DIR / "training_curves.png",
    )
    print("Training complete. Best checkpoint:", CHECKPOINT_DIR / "best_cnn_lstm.pt")


if __name__ == "__main__":
    main()