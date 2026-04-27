# NETFLUX: PER-LINK TRAFFIC TREND CLASSIFICATION USING A SPATIAL-TEMPORAL TRANSFORMER AND REINFORCE FINE-TUNING

NetFlux is an end-to-end **link-trend classification** pipeline for backbone Internet traffic matrices. Given a sliding window of past 12×12 origin-destination (OD) matrices, the model independently predicts for each of the **144 links** whether traffic will be **Decreasing (0)**, **Stable (1)**, or **Increasing (2)** at the next timestep.

The project evolved from a Per-Link LSTM baseline through a Spatial-Temporal Transformer (separate delta vs Pareto streams, last-token temporal pooling), then optional REINFORCE fine-tuning to reshape errors on minority classes.

---

## Quick Results

**Headline numbers below are from one fully converged reference run.** After you train locally, treat `Netflux/outputs/test_metrics.txt` (LSTM) and `Netflux/outputs/rl_test_metrics.txt` (transformer + RL) as the source of truth—REINFORCE can hurt macro-F1 if the supervised transformer is underfit.

| Model | Accuracy | Macro F1 | Dec F1 | Stable F1 | Inc F1 |
|---|---|---|---|---|---|
| Per-Link LSTM (baseline) | 0.490 | 0.460 | 0.400 | 0.620 | 0.350 |
| ST-Transformer (supervised only) | 0.475 | 0.457 | 0.424 | 0.586 | 0.362 |
| **ST-Transformer + REINFORCE** | **0.499** | **0.463** | 0.352 | **0.628** | **0.410** |

In that reference run, REINFORCE improved Increasing-class F1 and Stable F1 over the supervised transformer alone; whether RL beats the LSTM on your machine depends on supervised pretraining quality and hyperparameters (`config.py`).

---

## Dataset

- **Source**: Abilene 2004 Internet2 backbone traffic matrix (12 nodes, router-level ingress/egress)
- **Location**: `../Datasets/Abilene/2004/Measured/` — one `tm.YYYY-MM-DD.HH-MM-SS.dat` file per 5-minute interval; or preprocessed as `Netflux/data/abilene_2004_Tx12x12.npy` with shape `(T, 12, 12)`
- **Raw resolution**: 5-minute intervals
- **Effective resolution**: **30 minutes** after 6× temporal aggregation (`AGGREGATION_STEPS = 6`)
- **Total timesteps after aggregation**: ~8,064
- **Split**: Chronological 70 / 15 / 15 (train / val / test) — no shuffle, no leakage

---

## Problem Formulation

- **Input**: `(WINDOW_SIZE + 1, 12, 12)` = `(11, 12, 12)` tensor per sample
  - Frames 0–8: 9 signed percentage-change delta frames
  - Frames 9–10: Pareto MLE summary frames (scale x_m, shape α)
- **Output**: `(12, 12)` integer label matrix — 144 independent 3-way decisions per sample
- **Label rule (relative mode)**: For each link, Δ = (M_{t+1} − M_t) / (|M_t| + ε)
  - Stable (1): |Δ| ≤ 0.15
  - Increasing (2): Δ > 0.15
  - Decreasing (0): Δ < −0.15
- **Class imbalance**: Stable is the majority class (~52%), Decreasing (~23%) and Increasing (~25%) are minorities

---

## Feature Engineering

### Sliding window
For each valid index t, the input window is matrices M[t − k], …, M[t − 1] (k = WINDOW_SIZE = 10). The label is the transition M[t] → M[t+1].

### Delta frames (frames 0–8)
```
δ_i = (M[t-k+i+1] − M[t-k+i]) / (|M[t-k+i]| + pct_floor)
```
where `pct_floor = 1% × P10(positive traffic values)` prevents division blow-up on low-traffic links. Clipped to [−3.0, +3.0]. These 9 frames are **signed** — direction is preserved.

### Pareto MLE summary frames (frames 9–10)
Computed from `abs(δ)` across the 9 delta frames, per link, following Javadtalab et al. (2015):
- **x_m** (scale): minimum of |δ| per link across the window — the minimum observed magnitude
- **α** (shape): MLE estimate `α = n / Σ log(|δ_i| / x_m)` — tail heaviness indicator

These are **unsigned window statistics**, not temporal observations. α is clipped to [0.1, 10.0].

---

## Model 1: Per-Link LSTM (Baseline)

**File**: `models/cnn_lstm.py` — `PerLinkLSTM`

### Architecture (default: `USE_GNN = True` in `config.py`)
1. For each timestep `t ∈ {0,…,10}`, take the slice `x[:, t, :, :]` of shape `(B, 12, 12)`.
2. **GCN front-end:** Build node features (row mean = outgoing avg, column mean = incoming avg), run `GCNConv(2 → 16)` on the Abilene `edge_index`, then per OD pair fuse `[emb_src, emb_dst, raw scalar]` with `Linear(33 → 16)` + `LayerNorm` + `Tanh` → tensor `(B, 11, 12, 12, 16)`.
3. Reshape to `(B×144, 11, 16)` — each link is its own 11-step sequence of 16-D vectors.
4. **LSTM:** 2 layers, hidden size 128, dropout between layers, `batch_first=True`.
5. Last hidden state → `Dropout(0.3)` → `Linear(128→3)` → reshape to `(B, 3, 12, 12)` logits.

### Without GCN (`USE_GNN = False`)
Steps 1–2 are replaced by reshaping `(B, 11, 12, 12)` → `(B×144, 11, 1)` and applying `Linear(1→16)` + `LayerNorm` + `Tanh` per timestep, then the same LSTM head as above.

### Why per-link, not CNN
A 12×12 OD matrix is not a spatial image. Matrix position (i, j) represents a source-destination router pair; spatial neighbors carry no physical meaning. A shared LSTM over each link's time series is the correct inductive bias.

---

## Model 2: Spatial-Temporal Transformer

**File**: `models/st_transformer.py` — `PerLinkTransformer` (207,139 parameters)

### Two architectural fixes over the naive design

#### Fix 1 — Separate Pareto stream
**Problem**: The original design fed all 11 frames through the temporal Transformer. Positional encoding assigned Pareto frames positions 10 and 11 in the sequence — treating unsigned statistics as temporal observations at future-like positions. Attention then cross-attended between directional deltas and unsigned stats, destroying sign semantics.

**Fix**: Split the input before any projection:
- Delta stream (frames 0–8) → temporal Transformer
- Pareto stream (frames 9–10) → dedicated `Linear(2→32)` branch
- Fused after attention via `Linear(96→64)`

#### Fix 2 — Last-token pooling
**Problem**: Mean pooling averaged all 9 temporal encoder outputs equally, giving the delta from 4.5 hours ago the same weight as the most recent delta.

**Fix**: Use `x_enc[:, -1, :]` — the encoder output for the most recent timestep — as the per-link embedding. For trend prediction, the most recent state dominates.

### Full pipeline
```
Input (B, 11, 12, 12)
  │
  ├── frames 0–8  →  reshape → (B×144, 9, 1)
  │     → Linear(1→64) + LayerNorm + ReLU          [input projection]
  │     → Sinusoidal positional encoding
  │     → TransformerEncoder(3 layers, 4 heads, pre-norm)
  │     → x_enc[:, -1, :]  → (B×144, 64)           [last-token pooling]
  │
  └── frames 9–10 → reshape → (B×144, 2)
        → Linear(2→32) + LayerNorm + ReLU           [Pareto branch]
        → (B×144, 32)

concat → (B×144, 96)
  → Linear(96→64) + LayerNorm + ReLU               [fusion]
  → reshape → (B, 144, 64)
  → TransformerEncoder(1 layer, 4 heads, pre-norm)  [spatial attention]
  → Dropout(0.3) → Linear(64→3)
  → (B, 3, 12, 12)
```

### Hyperparameters
| Parameter | Value |
|---|---|
| d_model | 64 |
| nhead | 4 |
| Temporal encoder layers | 3 |
| Spatial encoder layers | 1 |
| dim_feedforward | 256 (d_model × 4) |
| Dropout | 0.3 |
| Pre-norm (norm_first) | True |
| Total parameters | 207,139 |

---

## Model 3: REINFORCE Fine-Tuning

**File**: `train_rl.py`

### Why RL at all
Supervised cross-entropy minimizes average loss but weights all classes by the loss weight alone. A policy can be further shaped to avoid specific error patterns — here, the collapse toward always-Stable predictions.

### Why REINFORCE instead of PPO
PPO requires a value network trained from scratch simultaneously with the policy. During early RL epochs value estimates are noisy → advantages `A = R − V(s)` are unreliable → noisy gradients propagate into the backbone → **catastrophic forgetting**. Val macro-F1 dropped from 0.45 to 0.28 in the PPO attempt.

REINFORCE with an EMA scalar baseline:
```
b  ← 0.99·b + 0.01·mean(batch rewards)
A  = reward − b
A  = (A − mean(A)) / (std(A) + 1e-8)    [per-batch normalisation]
L  = −E[log π(a|s) · stop_grad(A)] − 0.05·entropy
```
The scalar baseline converges immediately, gradient scale is independent of reward magnitude, and the backbone receives stable signals from epoch 1.

### Shaped reward
| Prediction | Ground truth | Reward |
|---|---|---|
| Correct | Decreasing or Increasing | +3.0 |
| Correct | Stable | +1.0 |
| Wrong | any | −0.5 |

**Why soft penalty**: With `reward_wrong = −1.0`, "always predict Stable" had expected reward ≈ 0 (Stable is majority class). Exploring minority classes had negative expected reward. Policy collapsed to all-Stable. Softening to −0.5 and boosting minority correct predictions to +3.0 makes exploration profitable.

### Optimizer setup
- Backbone (temporal encoder, Pareto branch, fusion, spatial encoder): `lr = RL_LR × 0.1 = 1e-5`
- Classifier head: `lr = RL_LR = 1e-4`
- Gradient clipping: max norm 1.0
- RL epochs: 20, saving best by val macro-F1

---

## Training Procedure

### Supervised (LSTM or Transformer)
- Loss: `CrossEntropyLoss` with balanced class weights (normalized to mean 1)
- Optimizer: Adam, lr = 1e-4, weight decay = 1e-4
- Gradient clipping: max norm 1.0
- Early stopping: patience 25 on validation loss
- Batch size: 128

### Class weights (train split)
| Class | Count | Weight (normalized) |
|---|---|---|
| Decreasing | ~185,458 | ~1.28 |
| Stable | ~422,251 | ~0.56 |
| Increasing | ~203,443 | ~1.16 |

---

## Configuration (`config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `MATRIX_SIZE` | 12 | 12×12 OD matrix |
| `WINDOW_SIZE` | 10 | 10 raw matrices → 9 deltas + 2 Pareto frames = 11 timesteps |
| `AGGREGATION_STEPS` | 6 | 5 min × 6 = 30-min resolution |
| `LABEL_MODE` | `"relative"` | Percentage-change labels |
| `TOLERANCE` | 0.15 | 15% change threshold for Stable vs Dec/Inc |
| `EPSILON` | 1e-9 | Denominator floor |
| `NUM_CLASSES` | 3 | Decreasing / Stable / Increasing |
| `CLASS_WEIGHTS` | `"balanced"` | Inverse-frequency weighting |
| `LSTM_HIDDEN_SIZE` | 128 | LSTM baseline hidden dim |
| `LSTM_NUM_LAYERS` | 2 | Stacked LSTM |
| `TRANSFORMER_D_MODEL` | 64 | Transformer embedding dim |
| `TRANSFORMER_NHEAD` | 4 | Attention heads |
| `TRANSFORMER_NUM_LAYERS` | 3 | Temporal encoder depth |
| `TRANSFORMER_USE_SPATIAL_ATTN` | True | Cross-link spatial attention |
| `DROPOUT` | 0.3 | Applied throughout |
| `BATCH_SIZE` | 128 | |
| `EPOCHS` | 100 | Max supervised epochs |
| `LEARNING_RATE` | 1e-4 | Adam lr (supervised) |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `EARLY_STOPPING_PATIENCE` | 25 | Val loss patience |
| `RL_LR` | 1e-4 | REINFORCE head lr |
| `RL_ENTROPY_COEF` | 0.05 | Entropy bonus weight |
| `RL_EPOCHS` | 20 | REINFORCE fine-tuning epochs |
| `RL_BASELINE_BETA` | 0.99 | EMA decay for running baseline |
| `RL_REWARD_CORRECT_MINORITY` | 3.0 | Reward: correct Dec or Inc |
| `RL_REWARD_CORRECT_STABLE` | 1.0 | Reward: correct Stable |
| `RL_REWARD_WRONG` | −0.5 | Penalty: any wrong prediction |
| `SEED` | 42 | Reproducibility |
| `TRAIN_RATIO` | 0.70 | Chronological split |
| `VAL_RATIO` | 0.15 | |
| `TEST_RATIO` | 0.15 | |

---

## How to Run

**Prerequisites**: Python 3.10+, PyTorch 2.x, `torch-geometric` (for `USE_GNN=True`), data at `../Datasets/Abilene/2004/Measured/` or `Netflux/data/abilene_2004_Tx12x12.npy`. Install deps from `Netflux/requirements.txt`.

```bash
cd Netflux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**LSTM baseline:**
```bash
python train.py       # → checkpoints/best_cnn_lstm.pt
python evaluate.py    # → outputs/test_metrics.txt (Per-Link LSTM only)
```

**Spatial-Temporal Transformer + REINFORCE (run sequentially, one terminal only):**
```bash
python -W ignore train_transformer.py && python -W ignore train_rl.py
```
- Supervised checkpoint: `checkpoints/best_transformer.pt`
- RL checkpoint: `checkpoints/best_rl_transformer.pt`
- Curves: `outputs/transformer_training_curves.png`, `outputs/rl_training_curves.png`
- Test metrics: `outputs/rl_test_metrics.txt`

---

## Project Structure

```
Final Project/
├── Datasets/Abilene/2004/        # raw .dat files
│   └── Measured/tm.*.dat
└── Netflux/
    ├── config.py                 # all hyperparameters and paths
    ├── dataset.py                # load, aggregate, sliding window, Pareto features
    ├── baselines.py              # persistence and mean baselines
    ├── train.py                  # LSTM supervised training
    ├── train_transformer.py      # Transformer supervised pretraining
    ├── train_rl.py               # REINFORCE fine-tuning
    ├── evaluate.py               # test metrics for best_cnn_lstm.pt (LSTM only)
    ├── diagnose.py               # dataset diagnostics
    ├── predict_one.py            # single-sample inference
    ├── models/
    │   ├── cnn_lstm.py           # PerLinkLSTM + legacy stubs
    │   └── st_transformer.py     # PerLinkTransformer (fixed architecture)
    ├── utils/
    │   ├── metrics.py            # accuracy, F1, confusion matrix
    │   ├── visualization.py      # training curve plots
    │   └── scaler.py             # TrafficScaler (regression path)
    ├── data/
    │   ├── abilene_topology.py   # Abilene edge_index for GCN
    │   └── abilene_2004_Tx12x12.npy  # preprocessed matrices
    ├── experiments/
    │   └── threshold_sweep.py    # class distribution vs tolerance
    ├── checkpoints/              # saved model weights
    ├── outputs/                  # metrics and plots
    └── requirements.txt
```

---

## Evolution of the Approach

### Stage 1 — CNN-LSTM (abandoned)
Initial design used a spatial CNN over the 12×12 matrix followed by an LSTM. Abandoned because OD matrix positions have no spatial locality — convolving over (i, j) neighbors mixes unrelated source-destination pairs.

### Stage 2 — Per-Link LSTM with absolute labels (abandoned)
Applied a shared LSTM independently to each of the 144 link sequences. Used a fixed absolute threshold for labeling (e.g., Δ > 1 Gbps = Increasing). Problem: links with very low base traffic were labeled Increasing on tiny fluctuations; high-traffic links were labeled Stable on significant percentage moves. Labels were not comparable across links.

### Stage 3 — Per-Link LSTM with relative labels and Pareto features (baseline)
Switched to percentage-change labels (TOLERANCE = 15%). Added `pct_floor` for low-traffic links. Replaced short/long-term mean summary frames with Pareto MLE frames (x_m, α) following Javadtalab et al. (2015). Added temporal aggregation (6× → 30-min) to smooth noise. This is the reported LSTM baseline: accuracy = 0.490, macro F1 = 0.460.

### Stage 4 — Naive Spatial-Temporal Transformer (abandoned)
Early designs fed all 11 frames (9 deltas + 2 Pareto) through one temporal Transformer with positional encoding and used mean pooling over time. Pareto statistics were treated like extra timesteps, so attention mixed signed deltas with unsigned window summaries; mean pooling also diluted recency.

### Stage 5 — Spatial-Temporal Transformer (current `PerLinkTransformer`)
`models/st_transformer.py` implements:
- **Separate Pareto stream:** Delta frames `[:, :9]` go through the temporal encoder; Pareto `(x_m, α)` is projected with `Linear(2 → d_model//2)` and fused **after** temporal encoding with the **last** delta token (not the Pareto positions inside the temporal encoder).
- **Last-token pooling:** `x_enc[:, -1, :]` replaces mean-over-time for the delta stream so the most recent change dominates.

Supervised quality still depends on learning rate and early stopping (see `LEARNING_RATE`, `EARLY_STOPPING_PATIENCE` in `config.py`); a weak Stage 5 checkpoint makes Stage 7 (REINFORCE) unreliable.

### Stage 6 — PPO fine-tuning (abandoned)
Added a `TransformerActorCritic` wrapping the pretrained backbone with a separately initialized value head. PPO with clipped surrogate objective. Result: catastrophic forgetting. Val macro-F1 dropped from 0.45 to 0.28 within a few epochs as noisy value estimates produced unreliable advantages that corrupted the backbone.

### Stage 7 — REINFORCE with EMA baseline (optional fine-tuning)
Replaced PPO with REINFORCE + scalar EMA running baseline. No value network. Shaped reward (+3 minority correct, +1 Stable correct, −0.5 wrong) to prevent all-Stable collapse. Backbone updates use 10× lower learning rate than the classifier head.

**When it works well**, test metrics can exceed the LSTM baseline on macro F1 and minority classes (see Quick Results table). **When the Stage 5 backbone is underfit**, RL may reward-hack (e.g. over-predict Increasing); the saved `best_rl_transformer.pt` may revert to the supervised weights—always read `outputs/rl_test_metrics.txt` after your run.
