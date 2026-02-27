"""Data helpers for volatility-blackbox."""

from pathlib import Path

from .pipeline import (
    PipelineConfig,
    add_features,
    compute_log_returns,
    download_price_data,
    ensure_directory,
    generate_rolling_splits,
    run_data_pipeline,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

__all__ = [
    "DATA_DIR",
    "PipelineConfig",
    "ensure_directory",
    "download_price_data",
    "compute_log_returns",
    "add_features",
    "generate_rolling_splits",
    "run_data_pipeline",
]
