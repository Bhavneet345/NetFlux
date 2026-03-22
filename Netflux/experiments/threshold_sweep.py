"""
Compute class distribution per relative tolerance (before training).
Use this to inspect label balance for different percentage-change thresholds.
Training is done only via train.py using the tolerance set in config.
Run from Netflux: python experiments/threshold_sweep.py
"""

import sys
from pathlib import Path

_netflux_root = Path(__file__).resolve().parent.parent
if str(_netflux_root) not in sys.path:
    sys.path.insert(0, str(_netflux_root))

from config import set_seed, SEED, NUM_CLASSES, EPSILON, AGGREGATION_STEPS
from dataset import load_abilene_matrices, aggregate_matrices, get_class_distribution

THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def main() -> None:
    set_seed(SEED)
    data = load_abilene_matrices()
    data = aggregate_matrices(data, AGGREGATION_STEPS)
    print(f"Aggregated: {AGGREGATION_STEPS}x -> T={data.shape[0]} ({5 * AGGREGATION_STEPS}-min resolution)")
    print()
    print("Class distribution per RELATIVE threshold (aggregated data)")
    print("=" * 55)
    for th in THRESHOLDS:
        counts, pcts = get_class_distribution(
            data, th, num_classes=NUM_CLASSES, relative=True, epsilon=EPSILON,
        )
        print(f"\nThreshold = {th:.0%} relative change")
        print(f"  Decreasing: {pcts[0]:.2f}%")
        print(f"  Stable:     {pcts[1]:.2f}%")
        print(f"  Increasing: {pcts[2]:.2f}%")


if __name__ == "__main__":
    main()
