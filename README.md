# volatility-blackbox
Research-grade pipeline to compare GARCH and RNN-based volatility forecasting.

This repository contains reproducible notebooks and utilities to evaluate:

1) GARCH(1,1)-t baseline
2) Hybrid GARCH + LSTM
3) Hybrid GARCH + GRU
4) Pure LSTM
5) Pure GRU
6) Residual Hybrid (GARCH + LSTM/GRU on residuals)

Core design goals:
- Use log returns and avoid lookahead bias
- Rolling out-of-sample evaluation with MSE and QLIKE
- Non-overlapping monthly test blocks (`test_size=21`, `step_size=21`)
- Modular notebooks for data, baselines, training, hybrid models, and evaluation
- Reproducible TensorFlow/Keras pipelines for deep models
- Use the `arch` package for the GARCH baseline

Data source
- We use `yfinance` to fetch S&P 500 daily data (ticker `^GSPC`) covering 2003–2024.

Quickstart

1. Create a Python environment (recommended: conda or venv).
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `notebooks/01_data_pipeline.ipynb` and run cells to download and preprocess data.
4. Confirm generated files:
   - `data/raw/sp500_raw.csv`
   - `data/processed/sp500_log_returns.csv`
   - `data/processed/rolling_splits.csv`
   - Split protocol: `min_train_size=756`, `val_size=252`, `test_size=21`, `step_size=21`
5. Train deep models with `MSE` objective:
   - `notebooks/03_lstm_model.ipynb` (Pure LSTM, MSE-trained)
   - `notebooks/04_gru_model.ipynb` (Pure GRU, MSE-trained)
   - `notebooks/05_hybrid_models.ipynb` (Hybrid LSTM/GRU, MSE-trained)
   - `notebooks/05a_hybrid_residual_models.ipynb` (Residual-hybrid LSTM/GRU)
6. Run `notebooks/02_garch_baseline.ipynb` for the GARCH(1,1)-t benchmark.
7. Compare all models in `notebooks/06_evaluation_qlike.ipynb`.
8. Run `notebooks/07_gate_visualization.ipynb` after 03/04/05 (and optionally 05a) to analyze LSTM/GRU gates.

Environment note
- TensorFlow wheels in some environments are not yet NumPy-2 compatible.
- This project pins `numpy<2` in `requirements.txt` to avoid ABI mismatch errors.

Long-run training note
- Notebooks 03/04/05 checkpoint outputs split-by-split and support `resume=True`.
- If a run is interrupted, re-run the same notebook and it will continue from missing splits.
- With `resume=False`, output CSVs are overwritten from scratch for a clean rerun.

RNN experiments (all four model families)
- Pure LSTM: train with `MSE`
- Pure GRU: train with `MSE`
- Hybrid GARCH + LSTM: train with `MSE`
- Hybrid GARCH + GRU: train with `MSE`
- Residual Hybrid GARCH + LSTM/GRU: train with `MSE` on residual variance
- Preprocessing is fit per rolling split using train-only statistics (no lookahead):
  - Target (`sq_return` / positive volatility proxy): `y_log = log(y + eps)`, then standardize.
  - Return features: standardized directly (no extra log transform).
  - GARCH feature(s): `log(garch_value + eps)` then standardize.
  - Residual-hybrid target (`residual_var`): standardized (no log; target is signed).
- RNN output layer uses linear activation; predictions are inverse-transformed back to original scale.

Evaluation metrics
- `MSE` (variance forecast error)
- `QLIKE` (quasi-likelihood loss on variance forecasts)

Output artifacts
- Prediction files are written to `reports/predictions/`
- Combined evaluation table is saved as:
  - `reports/predictions/evaluation_metrics_mse_qlike.csv`
- GARCH baseline files:
  - `reports/predictions/garch_baseline_predictions.csv`
  - `reports/predictions/garch_baseline_metrics_overall.csv`
- Gate capture files (from notebooks 03/04/05):
  - `reports/predictions/pure_lstm_gate_values.csv`
  - `reports/predictions/pure_gru_gate_values.csv`
  - `reports/predictions/hybrid_lstm_gate_values.csv`
  - `reports/predictions/hybrid_gru_gate_values.csv`
- Residual-hybrid files (from notebook 05a):
  - `reports/predictions/hybrid_residual_lstm_predictions.csv`
  - `reports/predictions/hybrid_residual_gru_predictions.csv`
  - `reports/predictions/hybrid_residual_lstm_gate_values.csv`
  - `reports/predictions/hybrid_residual_gru_gate_values.csv`

Project layout

volatility-blackbox/
├── data/
│   ├── raw/           # raw downloaded CSVs (not committed)
│   └── processed/     # cleaned, reproducible CSVs used by notebooks
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
│   ├── data/           # data loading / preprocessing helpers
│   ├── models/         # model wrappers and trainers
│   ├── losses/         # QLIKE and other evaluation losses
│   ├── evaluation/     # backtest/evaluation utilities
│   └── utils/          # plotting, reproducibility helpers
├── reports/
│   ├── figures/        # generated plots for papers/slides
│   └── predictions/    # model predictions and metric tables
├── requirements.txt
├── .gitignore
└── README.md
