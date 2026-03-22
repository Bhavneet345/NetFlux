"""
Data diagnostic for Abilene traffic matrices.
Run this BEFORE any further model changes to understand the actual data properties.
Run from Netflux root: python diagnose.py
"""

import sys
from pathlib import Path
import numpy as np

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import WINDOW_SIZE, TOLERANCE, EPSILON, LABEL_MODE
from dataset import load_abilene_matrices, chronological_split, compute_labels


def main():
    data = load_abilene_matrices()
    T = data.shape[0]
    print(f"Data shape: {data.shape}  (T={T} timesteps)")
    print()

    # ── 1. Traffic value scale ──────────────────────────────────────────────
    flat = data.flatten()
    nonzero = flat[flat > 0]
    print("=== Traffic value scale ===")
    print(f"  Min (all):      {flat.min():.6g}")
    print(f"  Max (all):      {flat.max():.6g}")
    print(f"  Mean (all):     {flat.mean():.6g}")
    print(f"  Median (all):   {np.median(flat):.6g}")
    print(f"  % zero entries: {100*(flat==0).mean():.1f}%")
    if len(nonzero):
        print(f"  Min (nonzero):  {nonzero.min():.6g}")
        print(f"  p10 (nonzero):  {np.percentile(nonzero,10):.6g}")
        print(f"  p50 (nonzero):  {np.percentile(nonzero,50):.6g}")
        print(f"  p90 (nonzero):  {np.percentile(nonzero,90):.6g}")
        print(f"  Max (nonzero):  {nonzero.max():.6g}")
    print()

    # ── 2. Per-link activity ─────────────────────────────────────────────────
    link_means = data.mean(axis=0)   # (12,12)
    link_zeros = (data == 0).mean(axis=0)
    print("=== Per-link mean traffic (12x12 matrix) ===")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(link_means)
    print()
    print("=== Per-link fraction of zero timesteps ===")
    print(link_zeros.round(3))
    print()

    n_dead   = (link_means == 0).sum()
    n_sparse = ((link_zeros > 0.5) & (link_means > 0)).sum()
    n_active = (link_zeros <= 0.5).sum()
    print(f"  Fully dead links (mean=0):       {n_dead}")
    print(f"  Sparse links (>50% zeros):        {n_sparse}")
    print(f"  Active links (≤50% zeros):        {n_active}")
    print()

    # ── 3. pct_floor that dataset.py will compute ───────────────────────────
    nonzero2 = data[data > 0]
    pct_floor = float(np.percentile(nonzero2, 10)) * 0.01 if len(nonzero2) else EPSILON
    print(f"=== dataset.py pct_floor = {pct_floor:.6g} ===")
    print(f"  (1% of p10 of nonzero traffic values)")
    print()

    # ── 4. Percentage-change distribution ───────────────────────────────────
    print("=== Percentage-change distribution (sample of 500 transitions) ===")
    sample_t = np.linspace(1, T-2, min(500, T-2), dtype=int)
    pct_changes = []
    for t in sample_t:
        denom = np.abs(data[t-1]) + pct_floor
        pct = (data[t] - data[t-1]) / denom
        pct_changes.append(pct.flatten())
    pct_all = np.concatenate(pct_changes)
    # Only active links
    nonzero_pct = pct_all[np.abs(pct_all) > 1e-9]
    print(f"  p1:  {np.percentile(pct_all,  1):.4f}")
    print(f"  p5:  {np.percentile(pct_all,  5):.4f}")
    print(f"  p25: {np.percentile(pct_all, 25):.4f}")
    print(f"  p50: {np.percentile(pct_all, 50):.4f}")
    print(f"  p75: {np.percentile(pct_all, 75):.4f}")
    print(f"  p95: {np.percentile(pct_all, 95):.4f}")
    print(f"  p99: {np.percentile(pct_all, 99):.4f}")
    print(f"  % clipped at ±3.0: {100*(np.abs(pct_all)>3.0).mean():.1f}%")
    print(f"  % clipped at ±1.0: {100*(np.abs(pct_all)>1.0).mean():.1f}%")
    print()

    # ── 5. True label distribution with current TOLERANCE ───────────────────
    print(f"=== Label distribution (TOLERANCE={TOLERANCE}, LABEL_MODE={LABEL_MODE!r}) ===")
    relative = (LABEL_MODE == "relative")
    counts = [0, 0, 0]
    for t in range(T - 1):
        lbl = compute_labels(data[t], data[t+1], TOLERANCE, relative=relative, epsilon=EPSILON)
        for c in range(3):
            counts[c] += int((lbl == c).sum())
    total = sum(counts)
    names = ["Decreasing", "Stable", "Increasing"]
    for c, n in enumerate(names):
        print(f"  {n}: {counts[c]:>10,}  ({100*counts[c]/total:.1f}%)")
    print()

    # ── 6. Label distribution on ACTIVE links only ──────────────────────────
    active_mask = link_zeros <= 0.5   # (12,12) bool
    print(f"=== Label distribution on ACTIVE links only ({active_mask.sum()} links) ===")
    counts_a = [0, 0, 0]
    for t in range(T - 1):
        lbl = compute_labels(data[t], data[t+1], TOLERANCE, relative=relative, epsilon=EPSILON)
        for c in range(3):
            counts_a[c] += int((lbl[active_mask] == c).sum())
    total_a = sum(counts_a)
    for c, n in enumerate(names):
        print(f"  {n}: {counts_a[c]:>10,}  ({100*counts_a[c]/total_a:.1f}%)")
    print()

    # ── 7. Autocorrelation of labels (is there temporal structure?) ──────────
    print("=== Label autocorrelation (lag-1) for active links ===")
    print("  (Do consecutive timesteps tend to have the same label?)")
    # Sample 5 active links
    active_idxs = list(zip(*np.where(active_mask)))[:5]
    for (i, j) in active_idxs:
        series = []
        for t in range(T - 1):
            lbl = compute_labels(data[t], data[t+1], TOLERANCE, relative=relative, epsilon=EPSILON)
            series.append(lbl[i, j])
        s = np.array(series)
        same = (s[1:] == s[:-1]).mean()
        print(f"  Link ({i},{j}): mean={link_means[i,j]:.4g}  "
              f"label_counts={np.bincount(s,minlength=3).tolist()}  "
              f"lag-1 same-label rate={same:.2f}")
    print()
    print("Done. Use these numbers to guide next steps.")


if __name__ == "__main__":
    main()