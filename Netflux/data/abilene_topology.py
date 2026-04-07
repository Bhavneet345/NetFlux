"""
Abilene backbone: 12 nodes (routers), indexed 0–11.
Undirected physical links are expanded to bidirectional edges for message passing;
self-loops are included for GCN stability.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

# Undirected physical links (single listing per edge)
_ABILENE_UNDIRECTED: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 5),
    (3, 4),
    (3, 6),
    (4, 5),
    (4, 7),
    (5, 9),
    (6, 7),
    (7, 8),
    (8, 9),
    (8, 10),
    (9, 11),
    (10, 11),
]

# All directed edges (both directions) plus self-loops (i, i)
ABILENE_EDGES: List[Tuple[int, int]] = []
for u, v in _ABILENE_UNDIRECTED:
    ABILENE_EDGES.append((u, v))
    ABILENE_EDGES.append((v, u))
for i in range(12):
    ABILENE_EDGES.append((i, i))


def get_adjacency_matrix() -> np.ndarray:
    """Return (12, 12) binary symmetric adjacency (including self-loops)."""
    A = np.zeros((12, 12), dtype=np.float32)
    for u, v in ABILENE_EDGES:
        A[u, v] = 1.0
    return A


def get_edge_index(device: torch.device | None = None) -> torch.LongTensor:
    """
    PyTorch Geometric edge_index: shape (2, num_edges), dtype long.
    Rows are [source_nodes, target_nodes].
    """
    src = [e[0] for e in ABILENE_EDGES]
    dst = [e[1] for e in ABILENE_EDGES]
    ei = torch.tensor([src, dst], dtype=torch.long)
    if device is not None:
        ei = ei.to(device)
    return ei


if __name__ == "__main__":
    A = get_adjacency_matrix()
    print("Adjacency matrix (12x12):")
    np.set_printoptions(linewidth=120, suppress=True)
    print(A.astype(int))
    ei = get_edge_index()
    print(f"\nedge_index shape: {tuple(ei.shape)}  (num_edges = {ei.shape[1]})")
    print(f"ABILENE_EDGES length (list): {len(ABILENE_EDGES)}")
    assert ei.shape[0] == 2
    assert ei.shape[1] == len(ABILENE_EDGES)
