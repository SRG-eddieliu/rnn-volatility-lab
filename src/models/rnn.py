"""RNN training helpers for pure and hybrid volatility models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.utils import set_seed

ArchitectureName = Literal["lstm", "gru"]
TargetTransformName = Literal["none", "standardize", "log_standardize"]


@dataclass(frozen=True)
class RNNTrainingConfig:
    """Configuration for rolling RNN volatility experiments."""

    lookback: int = 21
    hidden_units: int = 8
    dropout: float = 0.10
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 40
    patience: int = 6
    seed: int = 42
    scale_features: bool = True
    scale_target: bool = True
    target_transform: TargetTransformName = "log_standardize"
    log_garch_features: bool = True
    garch_feature_prefix: str = "garch_"
    eps: float = 1e-8
    force_linear_output: bool = True


def default_feature_columns(variant: str) -> list[str]:
    """Return default feature columns for pure/hybrid RNN variants."""
    variant_norm = variant.strip().lower()
    if variant_norm == "pure":
        return ["log_return", "sq_return", "abs_return", "rv_21d"]
    if variant_norm in {"hybrid", "hybrid_residual"}:
        return ["log_return", "sq_return", "abs_return", "rv_21d", "garch_cond_var"]
    raise ValueError(
        f"Unsupported variant='{variant}'. Expected one of ['pure', 'hybrid', 'hybrid_residual']."
    )


def _build_keras_model(
    architecture: ArchitectureName,
    input_shape: tuple[int, int],
    cfg: RNNTrainingConfig,
    output_activation: str = "softplus",
):
    from tensorflow import keras

    layer_cls = keras.layers.LSTM if architecture == "lstm" else keras.layers.GRU

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            layer_cls(cfg.hidden_units, dropout=cfg.dropout, recurrent_dropout=0.0),
            keras.layers.Dense(1, activation=output_activation),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _capture_lstm_gates(X_seq: np.ndarray, layer) -> dict[str, np.ndarray]:
    """Return mean gate activations by sample and timestep for a Keras LSTM layer."""
    kernel, recurrent_kernel, bias = layer.get_weights()
    units = layer.units
    n_samples, lookback, _ = X_seq.shape

    h = np.zeros((n_samples, units), dtype=np.float32)
    c = np.zeros((n_samples, units), dtype=np.float32)

    gate_means = {
        "input_gate": np.zeros((n_samples, lookback), dtype=np.float32),
        "forget_gate": np.zeros((n_samples, lookback), dtype=np.float32),
        "output_gate": np.zeros((n_samples, lookback), dtype=np.float32),
        "candidate_gate": np.zeros((n_samples, lookback), dtype=np.float32),
    }

    for t in range(lookback):
        z = X_seq[:, t, :] @ kernel + h @ recurrent_kernel + bias
        i = _sigmoid(z[:, :units])
        f = _sigmoid(z[:, units : 2 * units])
        g = np.tanh(z[:, 2 * units : 3 * units])
        o = _sigmoid(z[:, 3 * units :])

        c = f * c + i * g
        h = o * np.tanh(c)

        gate_means["input_gate"][:, t] = i.mean(axis=1)
        gate_means["forget_gate"][:, t] = f.mean(axis=1)
        gate_means["output_gate"][:, t] = o.mean(axis=1)
        gate_means["candidate_gate"][:, t] = g.mean(axis=1)

    return gate_means


def _capture_gru_gates(X_seq: np.ndarray, layer) -> dict[str, np.ndarray]:
    """Return mean gate activations by sample and timestep for a Keras GRU layer."""
    kernel, recurrent_kernel, bias = layer.get_weights()
    units = layer.units
    n_samples, lookback, _ = X_seq.shape

    h = np.zeros((n_samples, units), dtype=np.float32)
    gate_means = {
        "update_gate": np.zeros((n_samples, lookback), dtype=np.float32),
        "reset_gate": np.zeros((n_samples, lookback), dtype=np.float32),
        "candidate_gate": np.zeros((n_samples, lookback), dtype=np.float32),
    }

    if getattr(layer, "reset_after", True):
        input_bias = bias[0]
        recurrent_bias = bias[1]
    else:
        input_bias = bias
        recurrent_bias = np.zeros_like(bias)

    for t in range(lookback):
        x_proj = X_seq[:, t, :] @ kernel + input_bias
        h_proj = h @ recurrent_kernel + recurrent_bias

        x_z = x_proj[:, :units]
        x_r = x_proj[:, units : 2 * units]
        x_h = x_proj[:, 2 * units :]

        h_z = h_proj[:, :units]
        h_r = h_proj[:, units : 2 * units]
        h_h = h_proj[:, 2 * units :]

        z = _sigmoid(x_z + h_z)
        r = _sigmoid(x_r + h_r)
        hh = np.tanh(x_h + r * h_h)
        h = z * h + (1.0 - z) * hh

        gate_means["update_gate"][:, t] = z.mean(axis=1)
        gate_means["reset_gate"][:, t] = r.mean(axis=1)
        gate_means["candidate_gate"][:, t] = hh.mean(axis=1)

    return gate_means


def _build_gate_frame(
    gate_values: dict[str, np.ndarray],
    test_dates: np.ndarray,
    split_id: int,
    variant: str,
    architecture: str,
    train_loss: str,
) -> pd.DataFrame:
    """Convert gate activation tensors into a long-form dataframe."""
    n_samples = len(test_dates)
    any_gate = next(iter(gate_values.values()))
    lookback = any_gate.shape[1]

    repeated_dates = np.repeat(pd.to_datetime(test_dates).values, lookback)
    lookback_index = np.tile(np.arange(lookback, dtype=int), n_samples)
    lag = lookback - lookback_index

    frames = []
    for gate_name, values in gate_values.items():
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(repeated_dates),
                    "split_id": split_id,
                    "variant": variant,
                    "architecture": architecture,
                    "train_loss": train_loss,
                    "lookback_index": lookback_index,
                    "lag": lag,
                    "gate_name": gate_name,
                    "gate_value_mean": values.reshape(-1),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _capture_gate_values_for_split(
    model,
    X_test: np.ndarray,
    test_dates: np.ndarray,
    split_id: int,
    variant: str,
    architecture: str,
    train_loss: str = "mse",
) -> pd.DataFrame:
    """Capture gate values for one split/model on test sequences."""
    if len(X_test) == 0:
        return pd.DataFrame()

    recurrent_layer = model.layers[0]
    if architecture == "lstm":
        gate_values = _capture_lstm_gates(X_test, recurrent_layer)
    elif architecture == "gru":
        gate_values = _capture_gru_gates(X_test, recurrent_layer)
    else:
        raise ValueError(f"Unsupported architecture='{architecture}' for gate capture.")

    return _build_gate_frame(
        gate_values=gate_values,
        test_dates=test_dates,
        split_id=split_id,
        variant=variant,
        architecture=architecture,
        train_loss=train_loss,
    )


def _normalize_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _read_if_exists(path: Path | None, parse_dates: Sequence[str] | None = None) -> pd.DataFrame:
    if path is None or (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame()
    kwargs = {}
    if parse_dates:
        kwargs["parse_dates"] = list(parse_dates)
    return pd.read_csv(path, **kwargs)


def _append_frame(df: pd.DataFrame, path: Path | None) -> None:
    if path is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != df.columns.tolist():
            raise ValueError(
                f"Existing file schema mismatch at {path}. "
                "Use resume=False or delete the existing file before rerun."
            )
    write_header = (not path.exists()) or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _safe_std(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32)
    return np.where(std < floor, 1.0, std)


def _standardize_features_from_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize feature tensors with train-only mean/std."""
    feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feature_std = _safe_std(X_train.std(axis=(0, 1), keepdims=True))

    X_train_scaled = ((X_train - feature_mean) / feature_std).astype(np.float32)
    X_val_scaled = ((X_val - feature_mean) / feature_std).astype(np.float32)
    X_test_scaled = ((X_test - feature_mean) / feature_std).astype(np.float32)
    return X_train_scaled, X_val_scaled, X_test_scaled, feature_mean.squeeze(), feature_std.squeeze()


def _standardize_target_from_train(
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Standardize target vectors with train-only mean/std."""
    target_mean = float(np.mean(y_train))
    target_std = float(_safe_std(np.array([np.std(y_train)])).reshape(-1)[0])
    y_train_scaled = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_scaled = ((y_val - target_mean) / target_std).astype(np.float32)
    return y_train_scaled, y_val_scaled, target_mean, target_std


def _transform_features_by_train_stats(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_cols: Sequence[str],
    cfg: RNNTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Apply feature transforms with train-only statistics."""
    X_train_t = X_train.astype(np.float32).copy()
    X_val_t = X_val.astype(np.float32).copy()
    X_test_t = X_test.astype(np.float32).copy()

    garch_log_indices: list[int] = []
    if cfg.log_garch_features:
        prefix = cfg.garch_feature_prefix.lower()
        garch_log_indices = [i for i, col in enumerate(feature_cols) if str(col).lower().startswith(prefix)]
        for idx in garch_log_indices:
            X_train_t[:, :, idx] = np.log(np.clip(X_train_t[:, :, idx], cfg.eps, None))
            X_val_t[:, :, idx] = np.log(np.clip(X_val_t[:, :, idx], cfg.eps, None))
            X_test_t[:, :, idx] = np.log(np.clip(X_test_t[:, :, idx], cfg.eps, None))

    if cfg.scale_features:
        X_train_t, X_val_t, X_test_t, feature_mean, feature_std = _standardize_features_from_train(
            X_train_t,
            X_val_t,
            X_test_t,
        )
    else:
        feature_mean = np.zeros(X_train_t.shape[-1], dtype=np.float32)
        feature_std = np.ones(X_train_t.shape[-1], dtype=np.float32)

    feature_meta: dict[str, object] = {
        "feature_cols": list(feature_cols),
        "garch_log_indices": garch_log_indices,
        "scale_features": cfg.scale_features,
        "feature_mean": feature_mean.copy(),
        "feature_std": feature_std.copy(),
    }
    return X_train_t, X_val_t, X_test_t, feature_meta


def _transform_target_by_train_stats(
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: RNNTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply target transform with train-only statistics."""
    y_train_t = y_train.astype(np.float32).copy()
    y_val_t = y_val.astype(np.float32).copy()

    if cfg.target_transform == "log_standardize":
        if np.any(y_train_t < 0):
            raise ValueError(
                "target_transform='log_standardize' requires non-negative training targets. "
                "Use 'standardize' for signed targets."
            )
        y_train_t = np.log(np.clip(y_train_t, cfg.eps, None))
        y_val_t = np.log(np.clip(y_val_t, cfg.eps, None))
    elif cfg.target_transform not in {"none", "standardize"}:
        raise ValueError(
            f"Unsupported target_transform='{cfg.target_transform}'. "
            "Expected one of ['none', 'standardize', 'log_standardize']."
        )

    target_mean = 0.0
    target_std = 1.0
    if cfg.scale_target or cfg.target_transform == "standardize" or cfg.target_transform == "log_standardize":
        y_train_t, y_val_t, target_mean, target_std = _standardize_target_from_train(y_train_t, y_val_t)

    target_meta: dict[str, object] = {
        "target_transform": cfg.target_transform,
        "scale_target": cfg.scale_target or cfg.target_transform in {"standardize", "log_standardize"},
        "target_mean": target_mean,
        "target_std": target_std,
        "eps": cfg.eps,
    }
    return y_train_t, y_val_t, target_meta


def _inverse_target_transform(y_pred: np.ndarray, target_meta: dict[str, object]) -> np.ndarray:
    """Inverse-transform predictions back to original target scale."""
    out = y_pred.astype(np.float32).copy()
    if bool(target_meta.get("scale_target", False)):
        out = out * float(target_meta["target_std"]) + float(target_meta["target_mean"])

    transform = str(target_meta.get("target_transform", "none"))
    eps = float(target_meta.get("eps", 1e-8))
    if transform == "log_standardize":
        out = np.exp(out) - eps

    return out


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "sq_return",
    date_col: str = "date",
    lookback: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a time-indexed dataframe into supervised sequences."""
    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")
    required_cols = set(feature_cols) | {target_col, date_col}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns for sequence generation: {missing}")

    feature_cols = list(feature_cols)
    work = df.loc[:, [date_col, *feature_cols]].copy()
    work[target_col] = df[target_col].values
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values(date_col)
    work = work.dropna().reset_index(drop=True)

    if len(work) <= lookback:
        raise ValueError("Not enough rows after dropna to construct sequence dataset.")

    x_values = work.loc[:, feature_cols].values.astype(np.float32)
    y_values = work.loc[:, target_col].to_numpy(dtype=np.float32)
    dates = work.loc[:, date_col].values

    X, y, y_dates = [], [], []
    for t in range(lookback, len(work)):
        X.append(x_values[t - lookback : t, :])
        y.append(y_values[t])
        y_dates.append(dates[t])

    return np.asarray(X), np.asarray(y), np.asarray(y_dates)


def _slice_by_date_window(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    start_date,
    end_date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    return X[mask], y[mask], dates[mask]


def run_rolling_experiment(
    df: pd.DataFrame,
    splits_df: pd.DataFrame,
    architecture: ArchitectureName,
    variant: str,
    cfg: RNNTrainingConfig | None = None,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "sq_return",
    date_col: str = "date",
    output_activation: str = "softplus",
    verbose_fit: int = 0,
    capture_gates: bool = False,
    predictions_path: str | Path | None = None,
    train_logs_path: str | Path | None = None,
    gates_path: str | Path | None = None,
    resume: bool = False,
    collect_last_history: bool = False,
):
    """
    Run rolling train/validation/test RNN experiment.

    Each split is fit only with data available before validation/test windows.
    If `capture_gates=True`, also return gate activation dataframe.
    If output paths are provided, writes are checkpointed split-by-split.
    With `resume=True`, already completed split_ids are skipped.
    With `resume=False`, output files are overwritten from scratch.
    `output_activation` controls final Dense activation (e.g., "softplus", "linear").
    If `collect_last_history=True`, return epoch-wise train/val loss and model
    coefficients for the last trained split.
    """
    if cfg is None:
        cfg = RNNTrainingConfig()
    if feature_cols is None:
        feature_cols = default_feature_columns(variant)

    required_split_cols = [
        "split_id",
        "train_start_date",
        "train_end_date",
        "val_start_date",
        "val_end_date",
        "test_start_date",
        "test_end_date",
    ]
    missing_split_cols = [col for col in required_split_cols if col not in splits_df.columns]
    if missing_split_cols:
        raise ValueError(f"Missing required split columns: {missing_split_cols}")

    set_seed(cfg.seed)
    X, y, y_dates = build_sequence_dataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        lookback=cfg.lookback,
    )

    predictions_path = _normalize_path(predictions_path)
    train_logs_path = _normalize_path(train_logs_path)
    gates_path = _normalize_path(gates_path)

    if not resume:
        for path in (predictions_path, train_logs_path, gates_path):
            if path is not None and path.exists():
                path.unlink()

    existing_predictions = _read_if_exists(predictions_path, parse_dates=["date"]) if resume else pd.DataFrame()
    existing_train_logs = _read_if_exists(train_logs_path) if resume else pd.DataFrame()
    existing_gates = _read_if_exists(gates_path, parse_dates=["date"]) if resume else pd.DataFrame()

    completed_split_ids: set[int] = set()
    if not existing_train_logs.empty and "split_id" in existing_train_logs.columns:
        completed_split_ids = set(existing_train_logs["split_id"].astype(int).tolist())
    elif not existing_predictions.empty and "split_id" in existing_predictions.columns:
        completed_split_ids = set(existing_predictions["split_id"].astype(int).tolist())

    prediction_rows = []
    train_log_rows = []
    gate_rows = []
    last_history: dict[str, object] | None = None

    for _, split in splits_df.iterrows():
        split_id = int(split["split_id"])
        if split_id in completed_split_ids:
            continue

        X_train, y_train, _ = _slice_by_date_window(
            X, y, y_dates, split["train_start_date"], split["train_end_date"]
        )
        X_val, y_val, _ = _slice_by_date_window(X, y, y_dates, split["val_start_date"], split["val_end_date"])
        X_test, y_test, test_dates = _slice_by_date_window(
            X, y, y_dates, split["test_start_date"], split["test_end_date"]
        )

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            continue

        X_train_model, X_val_model, X_test_model, feature_meta = _transform_features_by_train_stats(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            feature_cols=feature_cols,
            cfg=cfg,
        )
        y_train_model, y_val_model, target_meta = _transform_target_by_train_stats(y_train, y_val, cfg)
        effective_output_activation = "linear" if cfg.force_linear_output else output_activation

        set_seed(cfg.seed + split_id)
        model = _build_keras_model(
            architecture=architecture,
            input_shape=(X_train_model.shape[1], X_train_model.shape[2]),
            cfg=cfg,
            output_activation=effective_output_activation,
        )

        import tensorflow as tf
        tf.keras.utils.set_random_seed(cfg.seed + split_id)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            X_train_model,
            y_train_model.reshape(-1, 1),
            validation_data=(X_val_model, y_val_model.reshape(-1, 1)),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=verbose_fit,
            callbacks=callbacks,
        )
        loss_hist = history.history.get("loss", [])
        val_loss_hist = history.history.get("val_loss", [])
        if not loss_hist or not val_loss_hist:
            raise ValueError("Training history is missing 'loss' or 'val_loss'; cannot log diagnostics.")

        best_train_loss = float(np.min(loss_hist))
        best_val_loss = float(np.min(val_loss_hist))
        final_train_loss = float(loss_hist[-1])
        final_val_loss = float(val_loss_hist[-1])
        if collect_last_history:
            recurrent_layer = model.layers[0]
            dense_layer = model.layers[-1]
            recurrent_weights = recurrent_layer.get_weights()
            dense_weights = dense_layer.get_weights()

            if architecture == "lstm":
                recurrent_map = {
                    "kernel": recurrent_weights[0].copy(),
                    "recurrent_kernel": recurrent_weights[1].copy(),
                    "bias": recurrent_weights[2].copy(),
                }
            elif architecture == "gru":
                recurrent_map = {
                    "kernel": recurrent_weights[0].copy(),
                    "recurrent_kernel": recurrent_weights[1].copy(),
                    "bias": recurrent_weights[2].copy(),
                }
            else:
                recurrent_map = {}

            last_history = {
                "split_id": split_id,
                "architecture": architecture,
                "variant": variant,
                "train_loss": "mse",
                "loss": [float(x) for x in loss_hist],
                "val_loss": [float(x) for x in val_loss_hist],
                "feature_meta": feature_meta,
                "target_meta": target_meta,
                "recurrent_weights": recurrent_map,
                "dense_weights": {
                    "kernel": dense_weights[0].copy(),
                    "bias": dense_weights[1].copy(),
                },
            }

        y_pred = model.predict(X_test_model, verbose=0).reshape(-1)
        y_pred = _inverse_target_transform(y_pred, target_meta=target_meta)
        if output_activation == "softplus" or cfg.target_transform == "log_standardize":
            y_pred = np.clip(y_pred, 1e-12, None)

        pred_df = pd.DataFrame(
            {
                "date": pd.to_datetime(test_dates),
                "split_id": split_id,
                "variant": variant,
                "architecture": architecture,
                "train_loss": "mse",
                "y_true_var": y_test,
                "y_pred_var": y_pred,
            }
        )
        prediction_rows.append(pred_df)
        _append_frame(pred_df, predictions_path)

        if capture_gates:
            gate_df = _capture_gate_values_for_split(
                model=model,
                X_test=X_test_model,
                test_dates=test_dates,
                split_id=split_id,
                variant=variant,
                architecture=architecture,
                train_loss="mse",
            )
            gate_rows.append(gate_df)
            _append_frame(gate_df, gates_path)

        train_row = pd.DataFrame(
            [
                {
                    "split_id": split_id,
                    "variant": variant,
                    "architecture": architecture,
                    "train_loss": "mse",
                    "output_activation_used": effective_output_activation,
                    "target_transform": target_meta["target_transform"],
                    "target_mean": float(target_meta["target_mean"]),
                    "target_std": float(target_meta["target_std"]),
                    "target_eps": float(target_meta["eps"]),
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                    "n_test": len(X_test),
                    "epochs_ran": len(history.history["loss"]),
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "best_gap_val_minus_train": best_val_loss - best_train_loss,
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "final_gap_val_minus_train": final_val_loss - final_train_loss,
                }
            ]
        )
        train_log_rows.append(train_row)
        _append_frame(train_row, train_logs_path)

    if not prediction_rows and existing_predictions.empty:
        raise ValueError("No predictions were generated. Check split dates and feature availability.")

    new_predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    new_train_logs = pd.concat(train_log_rows, ignore_index=True) if train_log_rows else pd.DataFrame()
    if existing_predictions.empty:
        predictions = new_predictions.copy()
    elif new_predictions.empty:
        predictions = existing_predictions.copy()
    else:
        predictions = pd.concat([existing_predictions, new_predictions], ignore_index=True)
    if existing_train_logs.empty:
        train_logs = new_train_logs.copy()
    elif new_train_logs.empty:
        train_logs = existing_train_logs.copy()
    else:
        train_logs = pd.concat([existing_train_logs, new_train_logs], ignore_index=True)
    if not predictions.empty:
        predictions = predictions.sort_values(["split_id", "date"]).drop_duplicates(
            subset=["split_id", "date", "variant", "architecture", "train_loss"], keep="last"
        ).reset_index(drop=True)
    if not train_logs.empty:
        train_logs = train_logs.sort_values("split_id").drop_duplicates(
            subset=["split_id", "variant", "architecture", "train_loss"], keep="last"
        ).reset_index(drop=True)
    if not capture_gates:
        if collect_last_history:
            return predictions, train_logs, last_history
        return predictions, train_logs

    new_gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    if existing_gates.empty:
        gates = new_gates.copy()
    elif new_gates.empty:
        gates = existing_gates.copy()
    else:
        gates = pd.concat([existing_gates, new_gates], ignore_index=True)
    if not gates.empty:
        gates = gates.sort_values(["split_id", "date", "gate_name", "lag"]).drop_duplicates(
            subset=[
                "split_id",
                "date",
                "variant",
                "architecture",
                "train_loss",
                "lookback_index",
                "lag",
                "gate_name",
            ],
            keep="last",
        ).reset_index(drop=True)
    if collect_last_history:
        return predictions, train_logs, gates, last_history
    return predictions, train_logs, gates


__all__ = [
    "RNNTrainingConfig",
    "default_feature_columns",
    "build_sequence_dataset",
    "run_rolling_experiment",
]
