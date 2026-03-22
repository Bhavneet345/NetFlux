# NetFlux: Learning Temporal Structure in Backbone Traffic

NetFlux is a **link-trend classification** pipeline for backbone traffic matrices. Given a sliding window of past 12×12 matrices, the model predicts for each link (i, j) whether traffic will be **Decreasing**, **Stable**, or **Increasing** at the next timestep.

---

## Project Overview

This repository (`Netflux/`) implements:

- **Data loading** from raw Abilene 2004 `.dat` files or a preprocessed `(T, 12, 12)` NumPy array
- **Temporal aggregation** (optional): average consecutive raw snapshots to 30-minute resolution (see `AGGREGATION_STEPS` below)
- **Sliding-window** sequences; targets are **per-link class labels** (0/1/2)
- **Per-Link LSTM classifier** (not a spatial CNN): each of 144 links is a 1-D time series; a shared LSTM + small input projection predicts 3 classes per link
- **Relative labeling & inputs**: percentage-change tolerance, percentage-change features with `pct_floor` and clipping (see `dataset.py`)
- **Training**: CrossEntropyLoss with balanced class weights, Adam (`weight_decay`), gradient clipping, early stopping on validation loss
- **Evaluation**: accuracy, macro F1, per-class F1, precision, recall, confusion matrix

---

## Configuration snapshot (`config.py`)

Values below are the single source of truth; update this table when you change `config.py`.

| Setting | Value | Notes |
|--------|-------|--------|
| `MATRIX_SIZE` | 12 | 12×12 traffic matrix |
| `WINDOW_SIZE` | **10** | 10 matrices of history; with 30-min aggregation ≈ **5 hours** of context |
| `AGGREGATION_STEPS` | **6** | 5 min × 6 = **30-minute** timesteps; set to `1` for raw 5-minute data |
| `LABEL_MODE` | `"relative"` | Per-link % change vs `|M_curr| + ε` |
| `TOLERANCE` | **0.1** | **10%** relative change → Stable band |
| `EPSILON` | `1e-9` | Denominator floor in relative formulas |
| `CLASS_WEIGHTS` | `"balanced"` | Inverse-frequency weights in CrossEntropyLoss |
| `LSTM_HIDDEN_SIZE` | **128** | Per-link LSTM hidden size |
| `LSTM_NUM_LAYERS` | 2 | Stacked LSTM |
| `DROPOUT` | **0.3** | On LSTM (between layers) and before classifier |
| `BATCH_SIZE` | 128 | |
| `EPOCHS` | 100 | (may stop earlier) |
| `LEARNING_RATE` | **3e-4** | Adam |
| `WEIGHT_DECAY` | 1e-4 | L2 on Adam |
| `EARLY_STOPPING_PATIENCE` | **15** | Validation loss |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | 0.70 / 0.15 / 0.15 | Chronological, no shuffle |
| `SEED` | 42 | Reproducibility |

**Relative mode inputs:** After aggregation, each sample uses `WINDOW_SIZE` raw matrices in the window; the dataset converts them to **percentage-change frames** along time (`WINDOW_SIZE - 1` frames, e.g. **9** when `WINDOW_SIZE = 10`).

---

## Dataset Description

- **Source**: Abilene 2004 Internet2 traffic matrix (ingress/egress, router-level)
- **Location**: `../Datasets/Abilene/2004` relative to `Netflux/` (see `DATA_DIR` / `MEASURED_DIR` in `config.py`)
- **Format**: Either
  - Raw: `Measured/tm.YYYY-MM-DD.HH-MM-SS.dat` (one 12×12 matrix per file, comma-separated, lines starting with `#` are comments), or
  - Preprocessed: `Netflux/data/abilene_2004_Tx12x12.npy` with shape `(T, 12, 12)`
- **Raw temporal resolution**: 5-minute intervals
- **Effective resolution for training/eval** (default): **30 minutes** after `aggregate_matrices(..., AGGREGATION_STEPS=6)`
- **Assumptions**: Chronologically ordered; no missing timestamps

---

## Classification Formulation

- **Why classification**: We predict each link’s **trend** (Decreasing / Stable / Increasing) instead of raw values — useful for alerts and interpretability.
- **Label definition (relative mode, default):** For each link, \(\Delta = (M_{next} - M_{curr}) / (|M_{curr}| + \epsilon)\). With **`TOLERANCE = 0.1` (10%)**:
  - **Stable (1)** if \(|\Delta| \le \text{TOLERANCE}\)
  - **Increasing (2)** if \(\Delta > \text{TOLERANCE}\)
  - **Decreasing (0)** if \(\Delta < -\text{TOLERANCE}\)
- **Absolute mode** (if `LABEL_MODE = "absolute"`): same structure but \(\Delta = M_{next} - M_{curr}\) and `TOLERANCE` is in the same units as traffic.
- **Evaluation**: Accuracy, macro F1, per-class F1, precision, recall, confusion matrix.

---

## Problem Formulation

- **Input**: A sequence of `WINDOW_SIZE` past matrices → in relative mode, **percentage-change** tensor of shape **`(WINDOW_SIZE - 1, 12, 12)`** per sample.
- **Output**: A 12×12 matrix of class labels (0, 1, 2) for the transition to the next timestep.
- **Objective**: Minimize weighted cross-entropy; report accuracy and macro F1. **144** independent 3-way decisions per sample (one per link).

---

## Sliding Window Strategy

- For each valid index: sequence ends at \(t\); target is the label matrix for \(t \to t+1\).
- **Number of classification samples:** \(T - \text{WINDOW\_SIZE} - 1\) (per split), where \(T\) is the length of that split **after** aggregation.
- **Split:** Chronological **70% / 15% / 15%** (train / val / test).

---

## Model Architecture (Per-Link LSTM)

Implemented in `models/cnn_lstm.py` as `PerLinkLSTM` (also exported as `CNNLSTMClassifier` for compatibility).

1. **Reshape** `(B, T, 12, 12)` → `(B×144, T, 1)` so each link is its own batch item.
2. **Input projection:** `Linear(1 → 16)`, `LayerNorm`, `tanh`.
3. **LSTM:** `LSTM_NUM_LAYERS` layers, hidden size `LSTM_HIDDEN_SIZE` (**128**), `batch_first=True`, dropout between layers when `num_layers > 1`.
4. **Last timestep** → `Dropout` → `Linear(hidden → 3)` per link.
5. **Reshape** logits to `(B, 3, 12, 12)` for `CrossEntropyLoss`.

There is **no** spatial CNN over the 12×12 grid; neighbors in the matrix are not treated as image pixels.

---

## Baselines

| Baseline | Description (classification) |
|----------|------------------------------|
| **Persistence** | Label from M[t-1] vs M[t] (last two in window) |
| **Mean** | Label from mean(last k) vs M[t] |

Same tolerance / label mode as the dataset when you run them with shared config.

---

## Training Procedure

- **Loss:** `CrossEntropyLoss` with **balanced** class weights (normalized to mean 1).
- **Optimizer:** Adam, **`LEARNING_RATE`**, **`WEIGHT_DECAY`**.
- **Gradient clipping:** max norm **1.0** (see `train.py`).
- **Early stopping:** validation loss, patience **`EARLY_STOPPING_PATIENCE`**.
- **Checkpoint:** `checkpoints/best_cnn_lstm.pt`.
- **Data pipeline:** Load matrices → **`aggregate_matrices`** if `AGGREGATION_STEPS > 1` → chronological split → `TrafficMatrixDataset`.

---

## Data Scaling (classification path)

For **classification**, the model uses **aggregated raw traffic** inside the dataset to build **relative** percentage-change inputs and labels. Standardization (`scaler.npz`) applies to **regression** workflows if used elsewhere; the default classification training path does **not** require fitting a scaler for the tensors described above.

---

## Evaluation Metrics

Run `evaluate.py` after training. Metrics are written to **`outputs/test_metrics.txt`** and printed to the console. Exact numbers depend on your run, seed, and checkpoint.

Example **historical** test-set numbers (5-minute data, earlier setup; **re-run** after changing `config.py`):

| Metric | Example |
|--------|--------|
| Accuracy | ~0.49 |
| Macro F1 | ~0.45 |

---

## How to Run

**Prerequisites:** Python 3.10+, data at `../Datasets/Abilene/2004/Measured/` or `Netflux/data/abilene_2004_Tx12x12.npy`.

```bash
cd Netflux

# Optional: virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

**Train and evaluate:**

```bash
python train.py    # saves best checkpoint to checkpoints/best_cnn_lstm.pt
python evaluate.py   # loads checkpoint, reports metrics
```

Optional: **`python experiments/threshold_sweep.py`** — class distribution vs relative thresholds (uses same aggregation as config).

- Training curves: `outputs/training_curves.png`
- Test metrics: `outputs/test_metrics.txt`

---

## Project Structure

```
Final Project/
├── Datasets/Abilene/2004/   # data (reference path in config)
└── Netflux/
    ├── config.py
    ├── dataset.py            # load, aggregate_matrices, sliding window, labels
    ├── baselines.py
    ├── train.py
    ├── evaluate.py
    ├── diagnose.py
    ├── experiments/
    │   └── threshold_sweep.py
    ├── models/
    │   └── cnn_lstm.py       # PerLinkLSTM (+ legacy names)
    ├── utils/
    ├── checkpoints/
    ├── outputs/
    ├── data/                 # optional preprocessed .npy
    └── requirements.txt
```

---

## Future Improvements

- **Transformer** over per-link or matrix-level tokens.
- **GNN** if explicit topology is available (Abilene graph).
- **RPCA / PCA** as a denoising front-end.
- **Multi-step** horizons (predict \(t \to t+h\)) via `HORIZON` in dataset/config if you add it.

---

## Evolution of the Approach

### 1. Why We Switched from Absolute to Relative Tolerance

Fixed absolute thresholds are not comparable across links with very different traffic scales. **Relative (percentage) labeling** with **`TOLERANCE = 0.1`** (10%) makes “stable” vs “up/down” comparable across the matrix.

### 2. Why Per-Link LSTM Instead of CNN-LSTM

A 12×12 matrix is not a natural image; **Per-Link LSTM** treats each OD pair as its own time series with a **shared** LSTM, avoiding false spatial convolutions.

**Input alignment:** Features are percentage changes (with **`pct_floor`** and **±3.0** clip) consistent with relative labels.

### 3. Training Stack (matches `config.py`)

- Balanced class weights, Adam with **weight decay**, **gradient clipping**, early stopping (**patience 15** in current config).
- **Dropout 0.3**, **hidden size 128** — tuned for longer windows (`WINDOW_SIZE = 10`).

### 4. Temporal Aggregation (30-Minute Windows)

With **`AGGREGATION_STEPS = 6`**, five-minute snapshots are averaged into **30-minute** steps before labeling and windows. Set **`AGGREGATION_STEPS = 1`** to train on full 5-minute resolution. **`WINDOW_SIZE = 10`** is then **10** coarse steps of history (≈ **5 hours** at 30-minute resolution).

### 5. Latest Results (Test Set)

Run **`python evaluate.py`** and read **`outputs/test_metrics.txt`** for metrics that match your current **`config.py`** and checkpoint. Older reported figures (e.g. accuracy ~0.49, macro F1 ~0.45) were from a **5-minute** pipeline; after switching to **30-minute aggregation** and the **wider window / larger LSTM**, **re-train and re-evaluate** and paste updated numbers here if you want them fixed in the README.
