# RNN Volatility Lab

Volatility forecasting experiments comparing GARCH, recurrent neural networks, and hybrid models on S&P 500 daily returns.

The research focuses on variance forecast quality, temporal evaluation, and diagnostics that explain why model rankings can differ across loss functions.

**Status:** research prototype. Historical results are retained, but the GARCH baseline and cross-model comparison need revision before the rankings can be treated as validated findings. See [validation status](docs/validation-status.md).

## Research Questions

- How do GARCH, LSTM, and GRU models compare under MSE and QLIKE?
- Does using GARCH as a feature or residual baseline improve variance forecasts?
- What do near-zero variance predictions and recurrent gate diagnostics reveal about model behavior?

## Data and Models

- Data source: `yfinance`, ticker `^GSPC`, daily observations over 2003-2024.
- Forecast target: squared daily log return, used as a noisy variance proxy.
- Baseline: GARCH(1,1) with Student-t innovations.
- Neural models: pure LSTM and GRU.
- Feature hybrids: LSTM/GRU with a GARCH variance feature.
- Residual hybrids: GARCH variance plus an RNN prediction of residual variance.

Raw data, processed data, prediction CSVs, and training logs are generated locally. They are not bundled with the public repository.

## Experiment Design

The rolling split configuration uses an expanding training window with a minimum of 756 observations, 252 validation observations, and non-overlapping 21-day test blocks.

Feature and target transformation parameters are estimated on each training split and reused for its validation and test sets. Positive variance targets use a log/standardize transform; signed residual targets use standardization without a log transform.

The intended baseline refits GARCH parameters every 21 trading days. The current implementation repeats the fitted endpoint forecast between refits instead of updating the conditional variance with each new return. This also affects models using that feature; it is an open correction, not a completed fix.

### Evaluation

- **MSE:** squared error on the original variance scale.
- **QLIKE:** `mean(log(predicted_variance) + realized_proxy / predicted_variance)`, with numerical clipping in the implementation.
- Training/validation loss diagnostics and recurrent gate summaries.

The historical report illustrates why MSE alone can be misleading: residual hybrids produce some near-zero variance forecasts with very large QLIKE penalties. Model comparisons must be rerun on a common set of test dates after the baseline correction.

Gate correlations are descriptive diagnostics, not causal explanations or trading signals.

## Setup and Run Order

```bash
python -m pip install -r requirements.txt
```

Use a compatible TensorFlow environment; the dependency file constrains NumPy below version 2.

Run these notebooks from [notebooks/](notebooks) in order:

1. [Data pipeline](notebooks/01_data_pipeline.ipynb)
2. [GARCH baseline](notebooks/02_garch_baseline.ipynb)
3. [LSTM](notebooks/03_lstm_model.ipynb)
4. [GRU](notebooks/04_gru_model.ipynb)
5. [Feature hybrids](notebooks/05_hybrid_models.ipynb)
6. [Residual hybrids](notebooks/05a_hybrid_residual_models.ipynb)
7. [MSE and QLIKE evaluation](notebooks/06_evaluation_qlike.ipynb)
8. [Gate visualization](notebooks/07_gate_visualization.ipynb)

Data download requires network access. Training is not a lightweight smoke test. Existing `resume=False` runs overwrite generated outputs, so preserve any experiment artifacts you need before rerunning.

The public repository does not yet include an automated regression test suite or a complete checked-in prediction bundle.

## Code and Reports

| Location | Contents |
| --- | --- |
| [src/data](src/data) | Data preparation, features, and rolling splits |
| [src/models](src/models) | GARCH helpers and rolling RNN experiments |
| [src/losses](src/losses) | Variance forecast losses |
| [src/evaluation](src/evaluation) | Forecast evaluation |
| [reports/research_report.md](reports/research_report.md) | Historical results and diagnostics, with validation caveats |
| [docs/validation-status.md](docs/validation-status.md) | Open evaluation issues and rerun requirements |

Previously generated PDFs in [reports/](reports) preserve the original research record. They predate the validation note and should be read alongside it.

## Next Steps

1. Correct daily GARCH state updates between parameter refits and add a regression test.
2. Regenerate the baseline and all dependent hybrid features and forecasts.
3. Compare models on identical dates, with explicit missing-data and variance-floor diagnostics.
4. Record data snapshots, dependency versions, seeds, and artifact provenance for the revised experiment.
