# Volatility-Blackbox Research Report

## 1) Objective
This document summarizes the full research pipeline implemented in this repository and explains the latest model results produced by notebooks `03` to `07`.

Project scope:
- Compare six volatility forecasters on S&P 500 daily data (2003-2024).
- Enforce strict rolling out-of-sample evaluation with no lookahead bias.
- Evaluate using both MSE and QLIKE on daily variance forecasts.

## 2) Pipeline Overview
The workflow is organized into modular notebooks:
1. `01_data_pipeline.ipynb`: download/clean data, engineer returns features, build rolling splits.
2. `02_garch_baseline.ipynb`: GARCH(1,1)-t expanding-window benchmark.
3. `03_lstm_model.ipynb`: pure LSTM rolling training/evaluation.
4. `04_gru_model.ipynb`: pure GRU rolling training/evaluation.
5. `05_hybrid_models.ipynb`: GARCH-feature hybrid LSTM/GRU.
6. `05a_hybrid_residual_models.ipynb`: residual hybrids (RNN on `sq_return - garch_cond_var`).
7. `06_evaluation_qlike.ipynb`: combined performance/diagnostics plots.
8. `07_gate_visualization.ipynb`: LSTM/GRU gate analysis and regime diagnostics.

Core modules:
- `src/data/pipeline.py`: data download, feature engineering, rolling split generation.
- `src/models/garch.py`: rolling GARCH fit/forecast and GARCH feature injection.
- `src/models/rnn.py`: rolling RNN trainer, checkpointing, gate capture, transformations.
- `src/evaluation/__init__.py` + `src/losses/__init__.py`: MSE/QLIKE computation.

## 3) Data, Features, and Split Design
Dataset:
- Ticker: `^GSPC` (daily)
- Period: 2003-2024

Engineered columns:
- `log_return`
- `sq_return` (variance proxy target for most models)
- `abs_return`
- `rv_21d`
- `garch_cond_var` (for hybrid variants)

Rolling split protocol:
- `min_train_size = 756`
- `val_size = 252`
- `test_size = 21`
- `step_size = 21`
- Expanding training window

This yields non-overlapping 21-day test blocks and strict temporal ordering.

## 4) Transformation and Standardization (No Leakage)
Implemented in `src/models/rnn.py` per rolling split:

Feature transforms (fit on train only, reused on val/test):
- Return features: standardize directly.
- GARCH feature(s): apply `log(garch + eps)` then standardize.

Target transforms:
- Pure + standard hybrids (`03/04/05`):
  - `y_log = log(y + eps)`
  - `y_std = (y_log - mean_train) / std_train`
- Residual hybrids (`05a`, signed target):
  - `y_std = (y - mean_train) / std_train`

Model output:
- Linear activation (`identity`) is used for training output.

Inverse transform at inference:
- Log-standardized target: `y_hat = exp(y_hat_std * std_train + mean_train) - eps`
- Standardized target: `y_hat = y_hat_std * std_train + mean_train`

Numerical constant:
- `eps = 1e-8`

## 5) Latest Performance Snapshot
Source: `reports/predictions/evaluation_metrics_mse_qlike.csv`

| variant | architecture | n_obs | mse | qlike | mse_rank | qlike_rank |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_residual | lstm | 4473 | 2.860449e-07 | 8.588371e+05 | 1 | 6 |
| baseline | garch11_t | 4515 | 3.033150e-07 | -8.088939 | 2 | 1 |
| hybrid_residual | gru | 4473 | 3.082436e-07 | 4.000313e+06 | 3 | 7 |
| pure | gru | 4515 | 3.532426e-07 | -3.792360 | 4 | 4 |
| pure | lstm | 4515 | 3.537221e-07 | -5.286387 | 5 | 2 |
| hybrid | gru | 4473 | 3.572742e-07 | -2.506646 | 6 | 5 |
| hybrid | lstm | 4473 | 3.602508e-07 | -4.088043 | 7 | 3 |


Interpretation:
- By **QLIKE rank**, `baseline garch11_t` is best in current snapshot.
- By **MSE rank**, `hybrid_residual_lstm` appears best, but this is misleading given its QLIKE behavior (see caveat below).

## 6) Overfitting Diagnostics Summary
Source: `reports/predictions/*_train_logs.csv` (best-epoch train/val gap).

| model_file | n_splits | best_gap_mean | best_gap_p95 | best_val_loss_mean |
| --- | --- | --- | --- | --- |
| hybrid_gru_train_logs.csv | 213 | 0.100770 | 0.594812 | 0.921707 |
| hybrid_lstm_train_logs.csv | 213 | 0.115448 | 0.771694 | 0.933110 |
| hybrid_residual_gru_train_logs.csv | 213 | 4.891720 | 44.537449 | 5.791817 |
| hybrid_residual_lstm_train_logs.csv | 213 | 4.933682 | 44.762732 | 5.801875 |
| pure_gru_train_logs.csv | 215 | 0.062556 | 0.295246 | 0.906108 |
| pure_lstm_train_logs.csv | 215 | 0.081709 | 0.322391 | 0.923237 |


Interpretation:
- Pure/hybrid non-residual models show moderate average gap levels.
- Residual hybrids show very large positive gap tails, indicating unstable validation behavior in some splits.

## 7) Gate Diagnostics Snapshot
Source: `reports/predictions/gate_vix_correlation_summary.csv` (pure LSTM subset).

| gate_name | pearson_corr | corr_21d_ma | spearman_corr |
| --- | --- | --- | --- |
| input_gate | 0.166951 | 0.191735 | 0.074540 |
| candidate_gate | 0.018205 | 0.016694 | 0.043424 |
| output_gate | -0.145037 | -0.176732 | 2.183229e-04 |
| forget_gate | -0.596006 | -0.659186 | -0.220642 |


Interpretation (pure LSTM example):
- `forget_gate` has strong negative correlation with VIX in current run.
- `input_gate` is mildly positively correlated.
- Relationship signs/magnitudes are sensitive to training regime and preprocessing choices; treat as descriptive diagnostics, not causal claims.

## 8) Important Caveat: Residual-Hybrid QLIKE Explosion
Residual hybrids currently show extreme positive QLIKE despite good MSE rank.

Evidence (`hybrid_residual_*_predictions.csv`):
| file | n_obs | pred_le_1e8_count | pred_min | pred_p01 |
| --- | --- | --- | --- | --- |
| hybrid_residual_lstm_predictions.csv | 4473 | 99 | 1.000000e-12 | 1.000000e-12 |
| hybrid_residual_gru_predictions.csv | 4473 | 297 | 1.000000e-12 | 1.000000e-12 |


Explanation:
- QLIKE is highly sensitive to under-prediction near zero.
- A non-trivial count of clipped near-zero predictions (`<= 1e-8`) causes very large QLIKE penalties.
- Therefore, MSE-only ranking can disagree sharply with QLIKE ranking in these variants.

## 9) Reproducibility and Execution Notes
- RNN notebooks currently use `resume=False`, so reruns overwrite outputs cleanly.
- Train-log schema now includes transform metadata and train/val diagnostics.
- Keep environment consistent (`numpy<2`, TensorFlow env, `arch`, `yfinance`).

## 10) What the Code Does (Short Summary)
- Builds leakage-safe rolling datasets.
- Trains each model family split-by-split with strict temporal boundaries.
- Saves predictions, train logs, and gate values to `reports/predictions`.
- Produces consolidated comparison metrics + diagnostic visualizations.

---
Report generated from repository artifacts on current local state.
