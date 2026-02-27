"""GARCH helpers for baseline and hybrid feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_garch_t(returns: pd.Series):
    """Fit a GARCH(1,1)-t model on returns."""
    from arch import arch_model

    series = pd.Series(returns).dropna().astype(float)
    if series.empty:
        raise ValueError("Input return series is empty after dropping NaNs.")

    model = arch_model(
        series * 100.0,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )
    result = model.fit(disp="off")
    return result


def rolling_garch_forecast(
    returns: pd.Series,
    min_train_size: int = 756,
    refit_every: int = 21,
) -> pd.Series:
    """
    Generate 1-step-ahead expanding-window GARCH(1,1)-t variance forecasts.

    Forecast at time t uses returns up to t-1 only (no lookahead).
    """
    if min_train_size <= 0 or refit_every <= 0:
        raise ValueError("min_train_size and refit_every must be positive integers.")

    series = pd.Series(returns).copy()
    valid = series.dropna().astype(float)
    if len(valid) <= min_train_size:
        raise ValueError("Not enough non-null returns to run rolling GARCH forecast.")

    forecasts = pd.Series(index=valid.index, dtype=float)

    for t in range(min_train_size, len(valid)):
        should_refit = (t == min_train_size) or ((t - min_train_size) % refit_every == 0)
        if should_refit:
            fit_result = fit_garch_t(valid.iloc[:t])

        pred = fit_result.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        forecasts.iloc[t] = pred / (100.0**2)

    return forecasts.reindex(series.index)


def add_garch_feature(
    df: pd.DataFrame,
    return_col: str = "log_return",
    out_col: str = "garch_cond_var",
    min_train_size: int = 756,
    refit_every: int = 21,
) -> pd.DataFrame:
    """Attach GARCH conditional variance feature for hybrid RNN models."""
    if return_col not in df.columns:
        raise ValueError(f"Column '{return_col}' not found in dataframe.")

    out = df.copy()
    out[out_col] = rolling_garch_forecast(
        out[return_col],
        min_train_size=min_train_size,
        refit_every=refit_every,
    )
    out[out_col] = out[out_col].replace([np.inf, -np.inf], np.nan)
    return out


__all__ = ["fit_garch_t", "rolling_garch_forecast", "add_garch_feature"]
