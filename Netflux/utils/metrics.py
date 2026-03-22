"""
Evaluation metrics for traffic matrix forecasting.
"""

import torch
import numpy as np
from typing import Union


def mse(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean squared error (element-wise)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.mean((pred - target) ** 2))


def rmse(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(pred, target)))


def mae(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean absolute error (element-wise)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.mean(np.abs(pred - target)))


def compute_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
) -> dict:
    """Return dict with MSE, RMSE, MAE (regression)."""
    return {
        "MSE": mse(pred, target),
        "RMSE": rmse(pred, target),
        "MAE": mae(pred, target),
    }


def _to_numpy_flat(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred, dtype=np.int64).ravel()
    target = np.asarray(target, dtype=np.int64).ravel()
    return pred, target


def accuracy(pred, target):
    pred, target = _to_numpy_flat(pred, target)
    return float(np.mean(pred == target))


def per_class_precision_recall_f1(pred, target, num_classes=3):
    """Return (precisions, recalls, f1s) each of length num_classes."""
    pred, target = _to_numpy_flat(pred, target)
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return precisions, recalls, f1s


def macro_f1(pred, target, num_classes=3):
    _, _, f1s = per_class_precision_recall_f1(pred, target, num_classes)
    return float(np.mean(f1s))


def confusion_matrix(pred, target, num_classes=3):
    pred, target = _to_numpy_flat(pred, target)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(pred)):
        cm[target[i], pred[i]] += 1
    return cm


def per_class_accuracy(pred, target, num_classes=3):
    pred, target = _to_numpy_flat(pred, target)
    return [float(np.mean(pred[target == c] == c)) if np.any(target == c) else 0.0 for c in range(num_classes)]


def compute_classification_metrics(pred, target, num_classes=3):
    prec, rec, f1 = per_class_precision_recall_f1(pred, target, num_classes)
    return {
        "accuracy": accuracy(pred, target),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "per_class_accuracy": per_class_accuracy(pred, target, num_classes),
        "confusion_matrix": confusion_matrix(pred, target, num_classes),
    }
