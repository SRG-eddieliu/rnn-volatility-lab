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
    diagnostics: list[dict] | None = None,
) -> pd.Series:
    """
    Generate 1-step-ahead expanding-window GARCH(1,1)-t variance forecasts.

    Forecast at time t uses returns up to t-1 only (no lookahead).
    """
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0
           for x in (min_train_size, refit_every)):
        raise ValueError("min_train_size and refit_every must be positive integers.")

    series = pd.Series(returns).copy()
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError("Return index must be unique and chronologically increasing.")
    valid = series.dropna().astype(float)
    if not np.isfinite(valid.to_numpy()).all():
        raise ValueError("Returns must be finite after dropping NaNs.")
    if len(valid) <= min_train_size:
        raise ValueError("Not enough non-null returns to run rolling GARCH forecast.")

    forecasts = pd.Series(index=valid.index, dtype=float)

    for t in range(min_train_size, len(valid)):
        should_refit = (t == min_train_size) or ((t - min_train_size) % refit_every == 0)
        if should_refit:
            fit_result = fit_garch_t(valid.iloc[:t])
            if fit_result.convergence_flag != 0:
                raise RuntimeError(f"GARCH fit failed to converge at forecast origin {valid.index[t-1]}.")
            omega = float(fit_result.params["omega"])
            alpha = float(fit_result.params["alpha[1]"])
            beta = float(fit_result.params["beta[1]"])
            pred = float(fit_result.forecast(horizon=1, reindex=False).variance.values[-1, 0])
            if diagnostics is not None:
                diagnostics.append({"forecast_date":str(valid.index[t]), "origin":str(valid.index[t-1]),
                                    "n_fit":t, "omega_pct2":omega, "alpha":alpha, "beta":beta,
                                    "nu":float(fit_result.params["nu"]), "convergence_flag":0})
        else:
            # Parameters stay fixed, but the state assimilates each newly observed return.
            pred = omega + alpha * (float(valid.iloc[t - 1]) * 100.0) ** 2 + beta * pred
        if not np.isfinite(pred) or pred <= 0:
            raise RuntimeError(f"Invalid conditional variance at {valid.index[t]}: {pred}")
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
