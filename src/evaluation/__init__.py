"""Evaluation helpers for backtests and model comparison."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.losses import mse_loss, qlike_loss


def evaluate_forecasts(
    df: pd.DataFrame,
    y_true_col: str = "y_true_var",
    y_pred_col: str = "y_pred_var",
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute MSE and QLIKE metrics, optionally grouped by model metadata."""
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(
            f"Input dataframe must include '{y_true_col}' and '{y_pred_col}' columns."
        )

    if group_cols is None or len(group_cols) == 0:
        work = df.copy()
        return pd.DataFrame(
            [
                {
                    "n_obs": len(work),
                    "mse": mse_loss(work[y_true_col].values, work[y_pred_col].values),
                    "qlike": qlike_loss(work[y_true_col].values, work[y_pred_col].values),
                }
            ]
        )

    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing grouping columns in dataframe: {missing}")

    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {group_col: key for group_col, key in zip(group_cols, keys)}
        row["n_obs"] = len(grp)
        row["mse"] = mse_loss(grp[y_true_col].values, grp[y_pred_col].values)
        row["qlike"] = qlike_loss(grp[y_true_col].values, grp[y_pred_col].values)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


__all__ = ["evaluate_forecasts"]
