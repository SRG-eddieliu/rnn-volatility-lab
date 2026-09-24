# RNN Volatility Lab

Volatility forecasting experiments comparing GARCH, recurrent neural networks, and hybrid models on daily S&P 500 returns.

The project investigates temporal evaluation, forecast calibration, and why MSE and QLIKE can tell different stories about the same predictions.

**Status: research prototype, validation incomplete.** A September 2026 review independently checked saved predictions and identified a stale GARCH state update and an inconsistent target inverse. The reports have been revised, but the models have not been corrected or retrained. Historical results are not validated model rankings.

## Read the Research

- [Research note: design, findings, limitations (7 pages)](reports/volatility-research.pdf)
- [Diagnostic appendix: coverage, numerical checks, gate summaries (6 pages)](reports/volatility-diagnostics.pdf)
- [Combined report (13 pages)](reports/volatility-research-with-appendix.pdf)
- [Markdown summary](reports/research_report.md)
- [Validation status and required rerun](docs/validation-status.md)
- [Aggregate review evidence and source hashes](reports/review_metrics.json)

The revised reports replace the earlier PDFs in the current tree. Original files remain recoverable in Git history. The byline is Eddie Liu; a provenance note records that the original experiments arose from a collaborative project.

## Research Questions

- Can LSTM or GRU models add useful information to a GARCH variance forecast?
- How do feature hybrids and additive residual hybrids differ in calibration and numerical stability?
- What can recurrent gate diagnostics reveal, and what do they not establish?

## Data and Models

- Data pipeline: `yfinance`, ticker `^GSPC`, daily observations over 2003-2024.
- Target: squared daily log return, a noisy second-moment proxy; its conditional-variance interpretation assumes a zero conditional mean.
- Seven specifications: Student-t GARCH(1,1), pure LSTM/GRU, GARCH-feature LSTM/GRU, and residual LSTM/GRU.
- Pure features: lagged log return, squared return, absolute return, and annualized 21-day rolling volatility. Hybrids add lagged GARCH variance.
- Residual target: `squared_return - garch_variance`; a signed correction, not a standardized return innovation.

Raw data, processed data, prediction CSVs, and training logs are generated locally. The public repository includes aggregate review evidence, not a complete prediction bundle or market-data snapshot.

## Experiment Design and Review Findings

The expanding-window split configuration uses a minimum training span of 756 observations, 252 validation observations, and non-overlapping 21-day test blocks. Neural sequences for a target date end on the previous date. Training-only transformation statistics are reused on validation and test sets. Warmup and sequence construction reduce effective sample sizes.

The notebooks use a 21-observation lookback, 8 recurrent units, dropout 0.1, a linear output, Adam at 0.001, batch size 64, up to 35 epochs, and early-stopping patience 6.

The review independently recomputed MSE and QLIKE for all seven saved forecast series on 4,473 common dates, 2007-03-08 through 2024-12-11. It also reconciled native-sample losses and 21 gate/VIX correlations.

Material unresolved issues:

1. The GARCH implementation repeats the fitted endpoint forecast between 21-day parameter refits, without incorporating intervening returns. This affects the baseline and dependent hybrids.
2. The target transform uses `log(max(y, eps))`, but its inverse subtracts eps after exponentiation. A consistent inverse and an appropriate forecast objective still need to be tested.
3. Residual LSTM and GRU reach the `1e-12` variance floor on 99 and 297 common dates. Their large QLIKE penalties are retained in the revised results.
4. Gate/VIX correlations are ex-post diagnostics across rolling refits, not causal explanations or trading signals. Training-gap logs also need checkpoint alignment.

## Setup and Run Order

```bash
python -m pip install -r requirements.txt
```

Use a compatible TensorFlow environment; the dependency file constrains NumPy below version 2. Running the current notebooks reproduces the existing workflow, including its known defects. Correct and test the open issues before treating a new run as validated.

Run the notebooks from [notebooks/](notebooks) in order:

1. [Data pipeline](notebooks/01_data_pipeline.ipynb)
2. [GARCH baseline](notebooks/02_garch_baseline.ipynb)
3. [LSTM](notebooks/03_lstm_model.ipynb)
4. [GRU](notebooks/04_gru_model.ipynb)
5. [Feature hybrids](notebooks/05_hybrid_models.ipynb)
6. [Residual hybrids](notebooks/05a_hybrid_residual_models.ipynb)
7. [MSE and QLIKE evaluation](notebooks/06_evaluation_qlike.ipynb)
8. [Gate visualization](notebooks/07_gate_visualization.ipynb)

Data download requires network access. Training is not a lightweight smoke test. Existing `resume=False` runs overwrite generated outputs, so preserve historical experiment artifacts before rerunning.

## Repository Layout

| Location | Contents |
| --- | --- |
| [src/data](src/data) | Data preparation, features, and rolling splits |
| [src/models](src/models) | GARCH helpers and rolling RNN experiments |
| [src/losses](src/losses) | Variance forecast losses |
| [src/evaluation](src/evaluation) | Forecast evaluation |
| [reports](reports) | Revised PDFs, summary, and aggregate evidence |
| [docs/validation-status.md](docs/validation-status.md) | Confirmed defects, completed checks, and rerun requirements |

The legacy report-generation script reflects the historical presentation and does not generate the revised PDFs. No comprehensive automated regression suite is currently bundled. The next substantive improvement is correcting and testing the forecast pipeline, then publishing a clearly separate rerun with uncertainty estimates.
