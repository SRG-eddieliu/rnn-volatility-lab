"""Auditable retraining of the legacy stock-level volatility-ratio model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StockConfig:
    garch_window: int = 50
    realized_window: int = 21
    lookback: int = 10
    minimum_sequences: int = 100
    retrain_interval: int = 600
    units: int = 8
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = .001
    seed: int = 20260925


def ticker_seed(ticker, fit_number, base):
    token = int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)
    return int((base + token + fit_number) % (2**31 - 1))


def ratio_sequences(ratio, lookback):
    """Each row predicts its own date from contiguous, strictly earlier ratios."""
    values = np.asarray(ratio, float)
    if values.ndim != 1 or lookback < 1 or len(values) <= lookback:
        raise ValueError("Insufficient one-dimensional ratio history.")
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback + 1)
    positions = np.arange(lookback, len(values))
    return windows[:, :-1].copy(), windows[:, -1].copy(), positions


def training_rows(X, y, positions, forecast_position):
    valid = (positions < forecast_position) & np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid], positions[valid]


def rolling_stock_garch(returns, cfg=StockConfig()):
    """Daily rolling refits, using returns strictly before each forecast date.

    Failed/nonconverged fits use past-window sample volatility, with an explicit
    flag. This differs from the old notebook's silent use of failed optimizers.
    """
    from arch import arch_model

    r = np.asarray(returns, float)
    if r.ndim != 1 or np.isinf(r).any():
        raise ValueError("Returns must be a vector without infinities.")
    sigma = np.full(len(r), np.nan)
    status = np.full(len(r), "insufficient_history", dtype=object)
    parameters = np.full((len(r), 4), np.nan)
    for t in range(cfg.garch_window, len(r)):
        history = r[t-cfg.garch_window:t]
        if not np.isfinite(history).all():
            continue
        fallback = float(history.std(ddof=1))
        if fallback <= 1e-12:
            status[t] = "degenerate_history"
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = arch_model(history*100, mean="Zero", vol="GARCH", p=1, q=1,
                                 dist="t", rescale=False).fit(
                    disp="off", show_warning=False, options={"maxiter": 500})
            estimate = float(fit.forecast(horizon=1, reindex=False).variance.values[-1, 0])
            if fit.convergence_flag != 0 or not np.isfinite(estimate) or estimate <= 0:
                sigma[t], status[t] = fallback, "past_vol_fallback"
            else:
                sigma[t], status[t] = np.sqrt(estimate)/100, "garch"
                parameters[t] = fit.params[["omega", "alpha[1]", "beta[1]", "nu"]].to_numpy()
        except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError):
            sigma[t], status[t] = fallback, "past_vol_fallback"
    return sigma, status, parameters


def train_stock_ratio(returns, sigma, dates, forecast_mask, ticker, output,
                      cfg=StockConfig()):
    """Expanding historical ratio training with frozen models between refits.

    Batched inference is safe because each row uses only its own past features;
    no inference data are used for fitting the model or either scaler.
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    dates = pd.DatetimeIndex(dates)
    r, g = np.asarray(returns, float), np.asarray(sigma, float)
    if len(r) != len(g) or len(r) != len(dates) or len(r) != len(forecast_mask):
        raise ValueError("Stock training arrays must align.")
    realized = pd.Series(r).rolling(cfg.realized_window).std(ddof=1).to_numpy()
    ratio = np.full(len(r), np.nan)
    np.divide(realized, g, out=ratio, where=np.isfinite(g) & (g > 0))
    X, y, positions = ratio_sequences(ratio, cfg.lookback)
    feature_valid = np.isfinite(X).all(axis=1)
    training_valid = feature_valid & np.isfinite(y)
    past_counts = np.r_[0, np.cumsum(training_valid)[:-1]]
    eligible = (np.asarray(forecast_mask, bool)[positions] & feature_valid
                & np.isfinite(g[positions]) & (past_counts >= cfg.minimum_sequences))
    candidate_rows = np.flatnonzero(eligible)
    predictions = np.full(len(r), np.nan)
    fit_ids = np.full(len(r), -1, dtype=int)
    records = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for fit_number, start in enumerate(range(0, len(candidate_rows), cfg.retrain_interval)):
        block = candidate_rows[start:start+cfg.retrain_interval]
        first_position = int(positions[block[0]])
        train_X, train_y, train_positions = training_rows(X, y, positions, first_position)
        if len(train_y) < cfg.minimum_sequences or train_positions.max() >= first_position:
            raise AssertionError("Training cutoff violation.")
        sx, sy = StandardScaler().fit(train_X), StandardScaler().fit(train_y[:, None])
        scaled_X = sx.transform(train_X).astype("float32")[:, :, None]
        scaled_y = sy.transform(train_y[:, None]).astype("float32")
        tf.keras.backend.clear_session()
        seed = ticker_seed(ticker, fit_number, cfg.seed)
        tf.keras.utils.set_random_seed(seed)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg.lookback, 1)),
            tf.keras.layers.LSTM(cfg.units, unroll=True), tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss="mse")
        history = model.fit(scaled_X, scaled_y, epochs=cfg.epochs,
                            batch_size=cfg.batch_size, shuffle=True, verbose=0)
        final_loss = float(model.evaluate(scaled_X, scaled_y, batch_size=256, verbose=0))
        if not np.isfinite(final_loss):
            raise ValueError(f"Nonfinite neural fit: {ticker}/{fit_number}")
        inference = sx.transform(X[block]).astype("float32")[:, :, None]
        result = model(inference, training=False).numpy()
        estimated_ratio = sy.inverse_transform(result).ravel()
        if not np.isfinite(estimated_ratio).all():
            raise ValueError(f"Nonfinite neural prediction: {ticker}/{fit_number}")
        predictions[positions[block]] = estimated_ratio * g[positions[block]]
        fit_ids[positions[block]] = fit_number
        checkpoint = output/f"fit-{fit_number:03d}.keras"
        model.save(checkpoint)
        np.savez(output/f"scalers-{fit_number:03d}.npz", x_mean=sx.mean_, x_scale=sx.scale_,
                 y_mean=sy.mean_, y_scale=sy.scale_)
        records.append({
            "fit_id": fit_number, "seed": seed, "training_sequences": len(train_y),
            "training_first_target": str(dates[train_positions[0]].date()),
            "training_last_target": str(dates[train_positions[-1]].date()),
            "first_forecast": str(dates[positions[block[0]]].date()),
            "last_forecast": str(dates[positions[block[-1]]].date()),
            "forecast_count": len(block), "final_standardized_training_mse": final_loss,
            "epoch_training_mse": [float(value) for value in history.history["loss"]],
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "scaler_sha256": hashlib.sha256((output/f"scalers-{fit_number:03d}.npz").read_bytes()).hexdigest(),
        })
    tf.keras.backend.clear_session()
    return predictions, fit_ids, records
