"""Reusable data pipeline utilities for volatility research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for downloading and preprocessing market data."""

    ticker: str
    start_date: str
    end_date: str
    raw_dir: Path
    processed_dir: Path
    raw_filename: str = "sp500_raw.csv"
    processed_filename: str = "sp500_log_returns.csv"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_columns(columns: Iterable[object]) -> list[str]:
    normalized = []
    for col in columns:
        if isinstance(col, tuple):
            col = col[0]
        name = str(col).strip().lower().replace(" ", "_")
        normalized.append(name)
    return normalized


def download_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance in a stable schema."""
    import yfinance as yf

    prices = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if prices.empty:
        raise ValueError(
            f"No data was returned for ticker='{ticker}' from {start_date} to {end_date}."
        )

    prices = prices.copy()
    prices.columns = _normalize_columns(prices.columns)
    prices = prices.rename_axis("date").reset_index()
    prices.columns = _normalize_columns(prices.columns)

    if "adj_close" not in prices.columns and "close" in prices.columns:
        prices["adj_close"] = prices["close"]

    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"Missing required columns after download: {missing}")

    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.tz_localize(None)
    prices = prices.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    prices = prices.loc[:, ["date", "open", "high", "low", "close", "adj_close", "volume"]]
    return prices


def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Compute one-step log returns from prices."""
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in dataframe.")

    out = df.copy()
    if (out[price_col] <= 0).any():
        raise ValueError("Price column contains non-positive values; cannot compute log returns.")

    out[return_col] = np.log(out[price_col]).diff()
    return out


def add_features(df: pd.DataFrame, return_col: str = "log_return") -> pd.DataFrame:
    """Add deterministic features useful for volatility modeling."""
    if return_col not in df.columns:
        raise ValueError(f"Column '{return_col}' not found in dataframe.")

    out = df.copy()
    out["sq_return"] = out[return_col] ** 2
    out["abs_return"] = out[return_col].abs()
    out["rv_21d"] = out[return_col].rolling(window=21, min_periods=21).std() * np.sqrt(252)
    return out


def generate_rolling_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    min_train_size: int = 756,
    val_size: int = 252,
    test_size: int = 21,
    step_size: int = 21,
    expanding_train: bool = True,
) -> pd.DataFrame:
    """Create rolling train/validation/test splits with strict time ordering."""
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")
    if any(x <= 0 for x in [min_train_size, val_size, test_size, step_size]):
        raise ValueError("Split sizes must be positive integers.")

    ordered = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(ordered[date_col], errors="raise")
    n_obs = len(ordered)

    first_val_end = min_train_size + val_size
    max_val_end = n_obs - test_size
    if first_val_end > max_val_end:
        raise ValueError("Not enough observations to construct a single rolling split.")

    rows = []
    split_id = 0
    for val_end_idx in range(first_val_end, max_val_end + 1, step_size):
        train_end_idx = val_end_idx - val_size
        train_start_idx = 0 if expanding_train else max(0, train_end_idx - min_train_size)
        val_start_idx = train_end_idx
        test_start_idx = val_end_idx
        test_end_idx = test_start_idx + test_size

        rows.append(
            {
                "split_id": split_id,
                "train_start_idx": train_start_idx,
                "train_end_idx_exclusive": train_end_idx,
                "val_start_idx": val_start_idx,
                "val_end_idx_exclusive": val_end_idx,
                "test_start_idx": test_start_idx,
                "test_end_idx_exclusive": test_end_idx,
                "train_start_date": dates.iloc[train_start_idx],
                "train_end_date": dates.iloc[train_end_idx - 1],
                "val_start_date": dates.iloc[val_start_idx],
                "val_end_date": dates.iloc[val_end_idx - 1],
                "test_start_date": dates.iloc[test_start_idx],
                "test_end_date": dates.iloc[test_end_idx - 1],
                "n_train": train_end_idx - train_start_idx,
                "n_val": val_end_idx - val_start_idx,
                "n_test": test_end_idx - test_start_idx,
            }
        )
        split_id += 1

    splits = pd.DataFrame(rows)
    return splits


def run_data_pipeline(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Download prices, compute features, and persist raw/processed CSV files."""
    raw_dir = ensure_directory(config.raw_dir)
    processed_dir = ensure_directory(config.processed_dir)

    raw_df = download_price_data(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    processed_df = add_features(compute_log_returns(raw_df))
    processed_df = processed_df.dropna(subset=["log_return"]).reset_index(drop=True)

    raw_path = raw_dir / config.raw_filename
    processed_path = processed_dir / config.processed_filename
    raw_df.to_csv(raw_path, index=False)
    processed_df.to_csv(processed_path, index=False)
    return raw_df, processed_df, raw_path, processed_path
