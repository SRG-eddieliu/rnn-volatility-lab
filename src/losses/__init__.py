"""Loss functions for volatility forecasting evaluation."""

from __future__ import annotations

import numpy as np


def mse_loss(
    realized_variance: np.ndarray,
    predicted_variance: np.ndarray,
) -> float:
    """Compute mean squared error on variance forecasts."""
    y_true = np.asarray(realized_variance, dtype=float)
    y_pred = np.asarray(predicted_variance, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("realized_variance and predicted_variance must share the same shape.")
    return float(np.mean((y_true - y_pred) ** 2))


def qlike_loss(
    realized_variance: np.ndarray,
    predicted_variance: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute average QLIKE loss on variance forecasts."""
    y_true = np.asarray(realized_variance, dtype=float)
    y_pred = np.asarray(predicted_variance, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("realized_variance and predicted_variance must share the same shape.")

    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)
    return float(np.mean(np.log(y_pred) + (y_true / y_pred)))


__all__ = ["mse_loss", "qlike_loss"]
