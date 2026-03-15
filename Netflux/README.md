# NetFlux: Learning Temporal Structure in Backbone Traffic

NetFlux is a baseline **link-trend classification** suite for backbone network traffic. Given a sliding window of past 12×12 matrices, the model predicts for each link (i, j) whether traffic will be **Decreasing**, **Stable**, or **Increasing** at the next timestep.

---

## Project Overview

This repository implements:

- **Data loading** from raw Abilene 2004 `.dat` files or a preprocessed `(T, 12, 12)` NumPy array
- **Sliding-window** sequence generation; targets are **per-link class labels** (0/1/2)
- **Baseline predictors**: persistence (label from M[t-1] vs M[t]) and mean (label from mean(last k) vs M[t])
- **CNN-LSTM classifier**: spatial CNN per timestep, LSTM over time, FC to 3 classes per link
- **Training** with CrossEntropyLoss, Adam optimizer, early stopping on validation loss
- **Evaluation** with accuracy, macro F1, per-class accuracy, and confusion matrix

The code is modular and suitable for extending to PCA, RPCA, Transformer, or GNN-based models.

---

## Dataset Description

- **Source**: Abilene 2004 Internet2 traffic matrix (ingress/egress, router-level)
- **Location**: `DATA` = `../Datasets/Abilene/2004` (relative to project root)
- **Format**: Either
  - Raw: `Measured/tm.YYYY-MM-DD.HH-MM-SS.dat` (one 12×12 matrix per file, comma-separated, header lines starting with `#`), or
  - Preprocessed: NumPy array of shape `(T, 12, 12)` saved as `data/abilene_2004_Tx12x12.npy`
- **Temporal resolution**: 5-minute intervals
- **Assumptions**: Chronologically ordered; no missing timestamps

---

## Classification Formulation

- **Why classification**: We predict the **trend** of each link (Decreasing / Stable / Increasing) instead of raw values. Useful for anomaly detection and interpretable alerts; reduces sensitivity to scale.
- **Label definition**: For link (i,j), T_ij = M[t][i][j], T_next = M[t+1][i][j]. With `tolerance` (default 0.01 Gbytes/s): **Stable (1)** if |T_next - T_ij| <= tolerance; **Increasing (2)** if T_next - T_ij > tolerance; **Decreasing (0)** else. Labels correspond to timestamp t+1.
- **Threshold role**: Defines what counts as "no change." Smaller tolerance makes Stable rarer; larger merges small fluctuations into Stable.
- **Evaluation**: Accuracy, macro F1, per-class accuracy, confusion matrix (no inverse-transform).

---

## Problem Formulation

- **Input**: A sequence of k past traffic matrices (shape k x 12 x 12).
- **Output**: A 12×12 matrix of **class labels** (0, 1, or 2) for the next timestep.
- **Objective**: Minimize cross-entropy; maximize accuracy and macro F1. This is **classification**: 144 independent 3-way classifications (one per link).

---

## Sliding Window Strategy

- For each t with t+1 available: **Sequence** = M[t-k], ..., M[t-1] (k x 12 x 12); **Target** = label matrix for t to t+1 (12 x 12, values 0/1/2). Number of samples: T - k - 1. Data split **chronologically** (70% / 15% / 15%).

---

## Model Architecture (CNN-LSTM Classifier)

1. **CNN encoder** (per timestep): Two 2D conv layers to spatial feature vector.
2. **LSTM** over the sequence; use last hidden state.
3. **Fully connected**: 144*3 logits, reshape to (batch_size, 3, 12, 12). No softmax (CrossEntropyLoss expects logits). **Input**: (batch_size, k, 12, 12). **Output**: (batch_size, 3, 12, 12).

---

## Baselines

| Baseline | Description (classification) |
|----------|------------------------------|
| **Persistence** | Label from M[t-1] vs M[t] (last two in window) |
| **Mean** | Label from mean(last k) vs M[t] |

Both output (batch_size, 12, 12) of class indices 0/1/2. Same tolerance as dataset.

---

## Training Procedure

- **Loss**: CrossEntropyLoss (targets long, logits from model). **Optimizer**: Adam. **Split**: 70% / 15% / 15% chronological. **Early stopping**: Validation loss. Best model saved to `checkpoints/best_cnn_lstm.pt`. No scaling for classification; labels from raw traffic with `tolerance` in Gbytes/s.

---

## Data Scaling Strategy

- **Why scaling**: Traffic values (Gbytes/s) vary by orders of magnitude across OD pairs; standardization stabilizes training and improves convergence.
- **Train-only fitting**: μ and σ are computed **only on the training split**. Validation and test data are transformed using these same parameters. Fitting on the full dataset would leak future/test information and inflate metrics.
- **Per-element statistics**: We use per-element (12×12) mean and standard deviation, so each matrix entry has its own μ and σ. A small epsilon is added to σ to avoid divide-by-zero.
- **Inverse-transform at evaluation**: Model and baselines predict in **scaled space**. Predictions (and targets) are inverse-transformed before computing MSE, RMSE, and MAE, so **all reported metrics are in original units (Gbytes/s)** and remain interpretable.
- **Reproducibility**: The fitted scaler (μ, σ) is saved to `checkpoints/scaler.npz` and loaded by `evaluate.py`, ensuring identical scaling at inference.

---

## Evaluation Metrics

- **Accuracy**: Fraction of correct link predictions.
- **Macro F1**: Average of per-class F1 (Decreasing, Stable, Increasing).
- **Per-class accuracy**: Accuracy within each class.
- **Confusion matrix**: Rows = true, columns = predicted.

Reported for CNN-LSTM and both baselines.

---

## How to Run

**Prerequisites**: Python 3.10+, Data at `../Datasets/Abilene/2004/Measured/` (or preprocessed `data/abilene_2004_Tx12x12.npy`).

**Create virtual environment and install dependencies (recommended):**

```bash
cd traffic_forecasting

# Create virtual environment
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Install latest dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Train and evaluate:**

```bash
# Train CNN-LSTM (early stopping, best model saved to checkpoints/)
python train.py

# Evaluate best model and baselines (accuracy, macro F1, confusion matrix)
python evaluate.py
```

- Training curves are saved to `outputs/training_curves.png`.
- Test metrics are printed and written to `outputs/test_metrics.txt`.

---

## Project Structure

```
traffic_forecasting/
├── DATA (reference: ../Datasets/Abilene/2004)
├── config.py           # Paths, hyperparameters, seed
├── dataset.py          # Load (T,12,12), sliding window, chronological split
├── baselines.py        # Persistence and mean-of-last-k
├── models/
│   └── cnn_lstm.py     # CNN-LSTM regression + classifier
├── utils/
│   ├── metrics.py      # Regression (MSE/RMSE/MAE) + classification (accuracy, F1, confusion)
│   ├── scaler.py       # Standardization (fit on train, save/load)
│   └── visualization.py
├── train.py            # Training loop, early stopping
├── evaluate.py         # Load best model, run baselines, report metrics
├── checkpoints/        # Best model checkpoint
├── outputs/            # Plots and metric logs
├── data/               # Optional: preprocessed .npy
└── README.md
```

---

## Future Improvements

- **Transformer**: Replace LSTM with a temporal Transformer over CNN-derived tokens; add positional encoding.
- **GNN**: Model the Abilene topology as a graph; use GNN layers for spatial aggregation and a temporal module (LSTM/Transformer) for forecasting.
- **RPCA / PCA**: Low-rank or robust PCA on the traffic matrix time series for denoising or as a feature front-end.
- **Multi-step forecasting**: Predict several future matrices (e.g. \(M_{t+1}, \ldots, M_{t+h}\)) with a single model or autoregressive setup.
