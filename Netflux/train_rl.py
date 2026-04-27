"""
REINFORCE with running-mean baseline fine-tuning for PerLinkTransformer.

Why REINFORCE instead of PPO
------------------------------
PPO requires a value network trained from scratch *in parallel* with the policy.
During early RL epochs the value estimates are noisy, so the computed advantages
A = R − V(s) are unreliable.  That instability is amplified by the multiple
PPO update passes per rollout and propagated into the backbone, causing
catastrophic forgetting.

REINFORCE with an exponential-moving-average (EMA) baseline sidesteps this:

  b  ← β·b + (1−β)·mean(batch rewards)     # scalar, no network needed
  A  = reward − b                           # stable advantage from step 1
  L  = −E[ log π(a|s) · A ]  +  entropy bonus

Because the baseline is just a running scalar it converges immediately and
never produces the kind of gradient noise that collapses the backbone.

Why the old reward led to all-Stable collapse
----------------------------------------------
With reward_wrong = −1.0, a policy that predicts Decreasing or Increasing
takes a −1 penalty almost every time (minority classes are rarely correct by
chance).  The expected reward of "always predict Stable" is ≈ 0, which is
better than the negative expected reward of exploring minority classes.  The
gradient therefore pushes toward all-Stable.

Fix: RL_REWARD_WRONG = −0.5 (softer penalty in config.py), so exploring
minority predictions costs less and the shaped +3 reward can overcome it.

Algorithm (one training step)
------------------------------
  1. Forward  : model(seq) → logits (B, 3, 12, 12)
  2. Sample   : Categorical(logits) → actions (B, 144), log_probs, entropy
  3. Reward   : compute_reward(actions, target) → (B, 144) float
  4. Baseline : b ← β·b + (1−β)·mean(rewards)
  5. Advantage: A = reward − b ; normalise per batch
  6. Loss     : −mean(log_probs · stop_gradient(A)) − λ·entropy
  7. Backward + clip + step

Outputs
-------
  checkpoints/best_rl_transformer.pt   — best checkpoint by val macro-F1
  outputs/rl_training_curves.png       — reward & val macro-F1 per epoch
  outputs/rl_test_metrics.txt          — final test-set metrics
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
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
    RL_LR,
    RL_ENTROPY_COEF,
    RL_EPOCHS,
    RL_FREEZE_BACKBONE,
    RL_BASELINE_BETA,
    RL_REWARD_CORRECT_MINORITY,
    RL_REWARD_CORRECT_STABLE,
    RL_REWARD_WRONG,
)
from dataset import (
    load_abilene_matrices,
    aggregate_matrices,
    chronological_split,
    TrafficMatrixDataset,
)
from models.st_transformer import PerLinkTransformer, get_transformer_classifier
from utils.metrics import compute_classification_metrics

CLASS_NAMES = ["Decreasing", "Stable", "Increasing"]


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    actions: torch.Tensor,
    target: torch.Tensor,
    reward_minority: float = RL_REWARD_CORRECT_MINORITY,
    reward_stable: float = RL_REWARD_CORRECT_STABLE,
    reward_wrong: float = RL_REWARD_WRONG,
) -> torch.Tensor:
    """
    Per-link shaped reward.

    Args:
        actions : (B, 144) — class predictions sampled from the policy
        target  : (B, H, W) — ground-truth labels (values 0/1/2)
    Returns:
        rewards : (B, 144) float32
    """
    B = actions.shape[0]
    tgt_flat = target.reshape(B, -1).to(actions.device)   # (B, 144)

    correct = (actions == tgt_flat)                        # bool (B, 144)
    is_minority = (tgt_flat != 1)                          # Dec (0) or Inc (2)

    rewards = torch.full(
        actions.shape, reward_wrong, dtype=torch.float32, device=actions.device
    )
    rewards[correct & is_minority] = reward_minority
    rewards[correct & ~is_minority] = reward_stable
    return rewards


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: PerLinkTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Greedy inference on `loader`; returns classification metrics dict."""
    model.eval()
    all_pred, all_target = [], []
    for seq, target in loader:
        logits = model(seq.to(device))            # (B, 3, 12, 12)
        pred = logits.argmax(dim=1).cpu()         # (B, 12, 12)
        all_pred.append(pred)
        all_target.append(target)
    return compute_classification_metrics(
        torch.cat(all_pred), torch.cat(all_target), num_classes=NUM_CLASSES
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ data
    data = load_abilene_matrices()
    data = aggregate_matrices(data, AGGREGATION_STEPS)
    train_data, val_data, test_data = chronological_split(
        data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    train_ds = TrafficMatrixDataset(
        train_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    val_ds = TrafficMatrixDataset(
        val_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    test_ds = TrafficMatrixDataset(
        test_data, window_size=WINDOW_SIZE, tolerance=TOLERANCE, classification=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ----------------------------------------------- load pretrained backbone
    supervised_ckpt = CHECKPOINT_DIR / "best_transformer.pt"
    if not supervised_ckpt.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {supervised_ckpt}\n"
            "Run train_transformer.py first."
        )

    model = get_transformer_classifier(device)
    ckpt = torch.load(supervised_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Loaded pretrained backbone from {supervised_ckpt}  "
        f"(val_loss={ckpt.get('val_loss', '?'):.6f})"
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PerLinkTransformer parameters: {total_params:,}")

    # ----------------------------------------------- optimizer
    if RL_FREEZE_BACKBONE:
        # Freeze encoder; only fine-tune the classifier head
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad_(False)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=RL_LR,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone frozen — training classifier head only ({trainable:,} params).")
    else:
        # Fine-tune encoder at 10× lower LR to avoid forgetting
        encoder_params, head_params = [], []
        for name, p in model.named_parameters():
            if "classifier" in name:
                head_params.append(p)
            else:
                encoder_params.append(p)
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": RL_LR * 0.1},
                {"params": head_params,    "lr": RL_LR},
            ]
        )
        print(
            f"Fine-tuning encoder (lr={RL_LR * 0.1:.1e})  "
            f"+ classifier head (lr={RL_LR:.1e})"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rl_ckpt_path = CHECKPOINT_DIR / "best_rl_transformer.pt"

    # ----------------------------------------------- supervised baseline
    print("-" * 60)
    print("Baseline — supervised transformer (no RL fine-tuning):")
    m0 = validate(model, val_loader, device)
    print(f"  Val  accuracy={m0['accuracy']:.4f}  macro-F1={m0['macro_f1']:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name}: F1={m0['per_class_f1'][c]:.4f}")
    print("-" * 60)

    best_val_f1 = m0["macro_f1"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "val_macro_f1": best_val_f1,
            "rl_epoch": 0,
        },
        rl_ckpt_path,
    )

    # Tracking for curves
    rl_epoch_rewards: list[float] = []
    rl_val_f1s: list[float] = [best_val_f1]  # index 0 = supervised baseline

    # EMA baseline — initialised to 0; warms up within the first few batches
    baseline: float = 0.0

    # ----------------------------------------------- REINFORCE loop
    for rl_epoch in range(1, RL_EPOCHS + 1):
        model.train()

        step_losses: list[float] = []
        step_rewards: list[float] = []

        for seq, target in train_loader:
            seq_d = seq.to(device)
            tgt_d = target.to(device)
            B = seq_d.size(0)
            L = model.num_links   # 144

            # ---- forward: get logits, sample actions ----
            logits = model(seq_d)   # (B, 3, 12, 12)

            # Reshape to (B, 144, 3) for Categorical
            logits_links = (
                logits
                .permute(0, 2, 3, 1)          # (B, 12, 12, 3)
                .reshape(B, L, NUM_CLASSES)   # (B, 144, 3)
            )
            dist = Categorical(logits=logits_links)
            actions = dist.sample()           # (B, 144) — stochastic, with grad
            log_probs = dist.log_prob(actions)  # (B, 144)
            entropy = dist.entropy()            # (B, 144)

            # ---- reward & advantage ----
            with torch.no_grad():
                rewards = compute_reward(actions, tgt_d)   # (B, 144)

            batch_mean = rewards.mean().item()
            # Update EMA baseline with current batch mean
            baseline = RL_BASELINE_BETA * baseline + (1.0 - RL_BASELINE_BETA) * batch_mean

            advantage = rewards - baseline            # (B, 144) — centre on baseline
            # Per-batch normalisation: makes gradient scale independent of reward magnitude
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # ---- REINFORCE loss ----
            # stop_gradient on advantage: we're updating the policy, not the baseline
            loss_reinforce = -(log_probs * advantage.detach()).mean()
            loss_entropy = -RL_ENTROPY_COEF * entropy.mean()
            total_loss = loss_reinforce + loss_entropy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step_losses.append(total_loss.item())
            step_rewards.append(batch_mean)

        avg_loss = sum(step_losses) / len(step_losses)
        avg_reward = sum(step_rewards) / len(step_rewards)
        rl_epoch_rewards.append(avg_reward)

        # ---- validate ----
        m = validate(model, val_loader, device)
        val_f1 = m["macro_f1"]
        rl_val_f1s.append(val_f1)

        print(
            f"Epoch {rl_epoch:2d}/{RL_EPOCHS}  "
            f"loss={avg_loss:.4f}  reward={avg_reward:.4f}  "
            f"baseline={baseline:.4f}  val_macro_F1={val_f1:.4f}"
        )
        for c, name in enumerate(CLASS_NAMES):
            print(
                f"  {name}: F1={m['per_class_f1'][c]:.4f}  "
                f"prec={m['per_class_precision'][c]:.4f}  "
                f"rec={m['per_class_recall'][c]:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": val_f1,
                    "rl_epoch": rl_epoch,
                },
                rl_ckpt_path,
            )
            print(f"  -> New best val macro-F1 {val_f1:.4f}  saved to {rl_ckpt_path}")
        else:
            print(f"  (no improvement; best={best_val_f1:.4f})")

    # ----------------------------------------------- save training curves
    _save_rl_curves(rl_epoch_rewards, rl_val_f1s)

    # ----------------------------------------------- final test evaluation
    print("\n" + "=" * 60)
    print(f"Loading best RL checkpoint (epoch {best_val_f1:.4f} val F1) ...")
    best_ckpt = torch.load(rl_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    m_test = validate(model, test_loader, device)
    print(f"TEST  accuracy={m_test['accuracy']:.4f}  macro-F1={m_test['macro_f1']:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        print(
            f"  {name}: F1={m_test['per_class_f1'][c]:.4f}  "
            f"prec={m_test['per_class_precision'][c]:.4f}  "
            f"rec={m_test['per_class_recall'][c]:.4f}"
        )
    print("Confusion matrix (rows=true, cols=pred):")
    print(m_test["confusion_matrix"])

    metrics_path = OUTPUT_DIR / "rl_test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {m_test['accuracy']:.4f}\n")
        f.write("F1 per class:\n")
        for c, name in enumerate(CLASS_NAMES):
            f.write(f"  {name}: {m_test['per_class_f1'][c]:.4f}\n")
        f.write(f"Macro F1: {m_test['macro_f1']:.4f}\n")
        f.write("Per-class precision: " + str(m_test["per_class_precision"]) + "\n")
        f.write("Per-class recall: " + str(m_test["per_class_recall"]) + "\n")
        f.write("Confusion matrix:\n" + str(m_test["confusion_matrix"]) + "\n")
    print(f"Test metrics written to {metrics_path}")


# ---------------------------------------------------------------------------
# RL curve plot
# ---------------------------------------------------------------------------

def _save_rl_curves(rewards: list[float], val_f1s: list[float]) -> None:
    """Two-panel figure: rollout reward and val macro-F1 per RL epoch."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping RL curve plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(range(1, len(rewards) + 1), rewards, marker="o", color="steelblue")
    ax1.set_title("Rollout Average Reward (REINFORCE)")
    ax1.set_xlabel("RL Epoch")
    ax1.set_ylabel("Mean Reward")
    ax1.grid(True, alpha=0.3)

    # val_f1s[0] is the supervised baseline (epoch 0)
    ax2.plot(range(0, len(val_f1s)), val_f1s, marker="o", color="darkorange")
    ax2.axhline(
        val_f1s[0], linestyle="--", color="grey", alpha=0.6, label="supervised baseline"
    )
    ax2.set_title("Val Macro-F1 (REINFORCE)")
    ax2.set_xlabel("RL Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "rl_training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RL training curves saved to {out_path}")


if __name__ == "__main__":
    main()
