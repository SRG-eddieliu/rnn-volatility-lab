# volatility-blackbox

Research-grade volatility forecasting pipeline comparing GARCH and RNN model families on S&P 500 daily data.

## Objective

Build a reproducible, leakage-safe benchmark across:

1. GARCH(1,1)-t baseline
2. Hybrid GARCH + LSTM
3. Hybrid GARCH + GRU
4. Pure LSTM
5. Pure GRU
6. Residual Hybrid (GARCH + LSTM / GRU on residual variance)

## Data

- Source: `yfinance`
- Ticker: `^GSPC`
- Frequency: daily
- Coverage: 2003-2024
- Raw file: `data/raw/sp500_raw.csv`
- Processed file: `data/processed/sp500_log_returns.csv`

## Engineered Columns

Implemented in `src/data/pipeline.py`:

- `log_return = log(adj_close_t) - log(adj_close_{t-1})`
- `sq_return = log_return^2` (daily variance proxy)
- `abs_return = |log_return|`
- `rv_21d = rolling_std(log_return, 21) * sqrt(252)` (annualized 21-day realized volatility)
- `garch_cond_var` (added later by GARCH rolling forecast for hybrid models)

## Rolling Split Protocol

Generated in `01_data_pipeline.ipynb` via `generate_rolling_splits`.

- `min_train_size = 756`
- `val_size = 252`
- `test_size = 21`
- `step_size = 21`
- Expanding training window

This creates non-overlapping 21-day out-of-sample test blocks with strict temporal ordering.

## Model Families

### 1) Baseline: GARCH(1,1)-t

Implemented in `src/models/garch.py`.

- Mean: Zero
- Volatility: GARCH(1,1)
- Error distribution: Student-t
- Returns scaled by 100 during fitting (`arch` convention)
- Rolling 1-step-ahead variance forecast
- Refit frequency: every 21 trading days (`refit_every=21`)
- No lookahead: forecast at time `t` uses returns up to `t-1`

### 2) Pure RNN

- `03_lstm_model.ipynb` (LSTM)
- `04_gru_model.ipynb` (GRU)

### 3) Hybrid RNN (Feature Hybrid)

- `05_hybrid_models.ipynb`
- Inputs include GARCH conditional variance feature `garch_cond_var`

### 4) Residual Hybrid

- `05a_hybrid_residual_models.ipynb`
- Target is residual variance: `residual_var = sq_return - garch_cond_var`
- Final variance forecast is reconstructed from GARCH + predicted residual

## Leakage-Safe Transformation and Standardization

Implemented in `src/models/rnn.py`, split-by-split.

All transform parameters are fit on the training part of the current rolling split only, then reused unchanged for validation and test of that split.

### Config defaults (RNNTrainingConfig)

- `hidden_units=8`
- `lookback=21`
- `dropout=0.10`
- `epochs=40`
- `patience=6`
- `batch_size=64`
- `learning_rate=1e-3`
- `scale_features=True`
- `scale_target=True`
- `target_transform="log_standardize"`
- `log_garch_features=True`
- `garch_feature_prefix="garch_"`
- `eps=1e-8`
- `force_linear_output=True`

### Feature transforms

Given split tensors `X_train`, `X_val`, `X_test` and feature list:

1. For GARCH-prefixed features (default prefix `garch_`):
   - `x <- log(clip(x, eps, +inf))`
2. Standardize all features with train statistics:
   - `x_std = (x - mean_train) / std_train`
   - `mean_train` and `std_train` are computed from `X_train` only

### Target transforms

#### For positive variance targets (`sq_return`) in pure/hybrid models

- `y_log = log(clip(y, eps, +inf))`
- `y_std = (y_log - mean_y_train) / std_y_train`

#### For signed residual target in residual-hybrid models

- `target_transform="standardize"`
- `y_std = (y - mean_y_train) / std_y_train`
- No log transform is applied because residual target can be negative

### Inverse transform at inference

Model predicts on transformed scale; predictions are mapped back to original target scale:

- If standardized:
  - `y_unstd = y_hat * std_y_train + mean_y_train`
- If target transform was `log_standardize`:
  - `y_hat_original = exp(y_unstd) - eps`

Additional safety in code:

- For `log_standardize` targets, predictions are clipped to at least `1e-12` before evaluation.

### Output activation

- RNN Dense output is forced to `linear` (`force_linear_output=True`).
- Training loss remains MSE on transformed target.

## Training and Evaluation

### Training objective

- All RNN families are trained with MSE.

### Evaluation metrics

Implemented in `src/losses/__init__.py`.

- `MSE = mean((y_true_var - y_pred_var)^2)`
- `QLIKE = mean(log(y_pred_var) + y_true_var / y_pred_var)`
  - Both `y_true_var` and `y_pred_var` are clipped by epsilon in code for numerical stability.

### Evaluation notebook

- `06_evaluation_qlike.ipynb` compares all models with MSE and QLIKE
- Includes plots vs VIX and realized volatility diagnostics

## Gate Capture and Visualization

### Gate capture

Gate activations are captured during rolling inference for LSTM/GRU models and saved to:

- `reports/predictions/*_gate_values.csv`

### Gate analytics

`07_gate_visualization.ipynb` aggregates and exports:

- `gate_summary_by_lag.csv`
- `gate_summary_by_regime_lag1.csv`
- `gate_summary_lag1_20_by_date.csv`
- `gate_summary_lag_buckets_by_date.csv`
- `gate_vix_correlation_summary.csv`
- `gate_vix_correlation_by_bucket.csv`

## Reproducibility Notes

- Seeded training via `set_seed(...)` and split-dependent seed offsets.
- `run_rolling_experiment(..., resume=False)` overwrites output artifacts for clean rerun.
- `resume=True` continues from existing split logs.
- Append writes enforce schema checks to avoid mixing incompatible artifact formats.

## Run Order

1. `01_data_pipeline.ipynb`
2. `02_garch_baseline.ipynb`
3. `03_lstm_model.ipynb`
4. `04_gru_model.ipynb`
5. `05_hybrid_models.ipynb`
6. `05a_hybrid_residual_models.ipynb`
7. `06_evaluation_qlike.ipynb`
8. `07_gate_visualization.ipynb`

## Main Artifacts

- Forecasts and logs: `reports/predictions/`
- Consolidated metrics: `reports/predictions/evaluation_metrics_mse_qlike.csv`
- Research report markdown: `reports/research_report.md`
- PDF report with plots: `reports/research_report_with_plots.pdf`
- PDF generator: `reports/generate_pdf_report.py`

## Environment

- Install: `pip install -r requirements.txt`
- NumPy compatibility: TensorFlow environment currently uses `numpy<2` to avoid ABI issues with compiled modules.

## Repository Layout

```
volatility-blackbox/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_garch_baseline.ipynb
│   ├── 03_lstm_model.ipynb
│   ├── 04_gru_model.ipynb
│   ├── 05_hybrid_models.ipynb
│   ├── 05a_hybrid_residual_models.ipynb
│   ├── 06_evaluation_qlike.ipynb
│   └── 07_gate_visualization.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── evaluation/
│   └── utils/
├── reports/
│   ├── figures/
│   ├── predictions/
│   ├── research_report.md
│   ├── research_report_with_plots.pdf
│   └── generate_pdf_report.py
└── README.md
```
