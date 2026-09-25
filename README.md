# RNN Volatility Lab

A study of equity-index volatility forecasts, plus a separately retrained stock-level GARCH/LSTM model and dynamic portfolio-weighting experiment.

**[Research brief (2 pages)](reports/volatility-research-brief.pdf)** | **[Full report (8 pages)](reports/volatility-complete-report.pdf)** | **[Code](src)** | **[Demo](#quick-offline-demo)**

**Latest stock retraining:** all 503 stock columns processed, with 1,968 saved LSTM fits across 502 stocks. The hybrid improves individual-stock MSE for 493 stocks but fails aggregate forecast robustness because of extreme ratio predictions. Monthly net Sharpe is 1.119 for the cap base, 1.101 for matched GARCH, and 1.115 for GARCH/LSTM; paired intervals do not resolve a hybrid advantage. [Results and failure diagnosis](docs/stock-retraining-findings.md).

The unchanged index study uses 3,738 common test dates and 178 expanding-window folds. The separate stock portfolio experiment uses 2,390 return dates. The index additive-residual model and stock multiplicative-ratio model must not be interpreted as one prediction-to-trading pipeline.

[![Forecast regression tests](https://github.com/SRG-eddieliu/rnn-volatility-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/SRG-eddieliu/rnn-volatility-lab/actions/workflows/tests.yml)

![Common-date MSE and paired uncertainty: additive MSE improves but intervals include zero; positivity and QLIKE remain problematic](reports/figures/additive-summary.png)

## Quick Offline Demo

```bash
python -m pip install -r requirements-test.txt
python scripts/smoke_demo.py
python -m unittest discover -s tests -v
```

No account, API key, market download, or TensorFlow is needed. The demo generates
synthetic returns, fits a small rolling GARCH baseline, checks forecast timing,
and evaluates GARCH/EWMA on common dates. **It does not reproduce the historical
results above or train an LSTM.** Full research reproduction is documented below.
CI runs the lightweight suite on pushes; manual workflow runs also test TensorFlow.
The figure is rebuilt with `python reports/build_readme_figure.py` from committed
aggregate evidence, without rerunning or modifying the historical experiment.

## Research and Results

Start with the [two-page research brief](reports/volatility-research-brief.pdf): page one covers index forecasting; page two covers the newly retrained stock portfolio experiment. The [eight-page report](reports/volatility-complete-report.pdf) includes both studies. The [six-page forecast-only report](reports/volatility-final-report.pdf) is preserved unchanged. This section describes the index experiment only.

The forecast study compares nine benchmark/log-target specifications plus one additive-residual LSTM with raw and floored outputs on **3,738 identical dates**, February 5, 2010 to December 11, 2024.

Results are reported whether or not a neural model improves on a simple baseline. The forecast analysis distinguishes implementation correctness, positivity, calibration, and loss-based performance. Its MSE results do not establish portfolio P&L or Sharpe improvement.

**Main finding:** the additive-residual LSTM lowers observed MSE by **5.48%** versus GARCH, from `1.600623e-7` to `1.512933e-7` with a fixed `1e-12` floor. The unbounded output improves MSE by **5.45%**. However, there are **60 nonpositive raw forecasts**, floored QLIKE is approximately **457,801** versus GARCH's **-8.4613**, and the primary 95% paired temporal interval for the MSE difference includes zero. This is an observed MSE improvement, not demonstrated robust or deployable variance forecasting.

GARCH beats all six log-target/log-ratio neural candidates on MSE and QLIKE. The additive specification was tested in a separate run after inspecting the historical sample, using the same 178 folds, benchmark predictions, architecture, seed schedule, and training budget. Neither experiment is an untouched holdout.

## Portfolio Construction and Economic Value

The stock model uses **GARCH volatility multiplied by an LSTM ratio correction**, not the index additive-residual model above. Both GARCH and the LSTM were freshly fitted from returns; old cached hybrid predictions were not reused. Each neural fit uses strictly prior training labels/scalers, fixed seeds and saved checkpoints. One stock has insufficient history for a neural fit.

Target weights multiply a base allocation by `clip(trailing_vol / forecast, 0.5, 1.5)` and renormalize. The engine uses next-close execution, simple stock returns, drifting weights, actual traded dollars, self-financing costs, and real trading-session rebalance dates. Matched GARCH uses the same finite raw model-coverage dates as the hybrid; unavailable/nonpositive forecasts receive neutral tilts.

**Primary comparison: monthly, 10 bps per dollar bought/sold, 2015-07-06 to 2024-12-31.**

| Same-universe diagnostic | Net Sharpe | Maximum drawdown |
| --- | ---: | ---: |
| Snapshot-cap base | 1.119 | -33.25% |
| Snapshot-cap + matched GARCH | 1.101 | -33.15% |
| Snapshot-cap + GARCH/LSTM | 1.115 | -32.48% |
| Equal-weight base | 0.738 | -38.34% |
| Equal-weight + matched GARCH | 0.729 | -37.79% |
| Equal-weight + GARCH/LSTM | 0.729 | -38.15% |

The cap hybrid's Sharpe difference versus matched GARCH is **+0.0139**, paired 95% temporal interval **[-0.0089, +0.0335]**. Versus the untilted cap base, it is **-0.0043**, interval **[-0.0302, +0.0191]**. Both equal-weight intervals also include zero. Drawdown improves modestly versus the cap base, but there is no resolved incremental Sharpe benefit. All predefined frequency/cost sensitivities and unrestricted GARCH comparisons are retained.

**Forecast failure is material:** common-date, date-equal MSE against trailing 21-day volatility is `1.2789e-4` for GARCH, `1.9130e-1` for the raw hybrid, and `3.0968e-6` for lagged realized-volatility persistence. AMCR accounts for 99.96% of hybrid MSE because near-zero GARCH denominators create extreme ratio targets. There are 475 nonpositive hybrid forecasts. Independent checkpoint replay reproduces the inspected failures; no stocks were removed or seeds retuned. Better median-stock MSE does not offset this aggregate failure. Clipping allocation tilts does not repair the forecasting model.

**Retrospective diagnostic, not point-in-time alpha:** later market caps are repeated across history and the universe is a later survivor list. Equal weighting does not fix survivor bias. The overlapping trailing-volatility target is not purely future realized risk; the persistence control matters. Sharpe assumes constant daily risk-free return 0.0001. Retraining makes model provenance auditable but does not validate achievable returns, production robustness or a causal LSTM contribution.

- [Retraining report (2 pages)](reports/stock-retraining-report.pdf) and [failure diagnosis](docs/stock-retraining-findings.md)
- [Fixed training protocol](docs/stock-overlay-retraining.md) and [evaluation addendum](docs/stock-overlay-evaluation.md)
- [Aggregate results, all sensitivities and source hashes](reports/stock_retraining_results.json)
- [Independent model replay and six daily portfolio reconciliations](reports/stock_retraining_verification.json)
- [Portfolio engine](src/evaluation/portfolio.py) and [regression tests](tests/test_portfolio.py)

## Forecast Implementation

1. **GARCH state updating:** daily conditional-variance recursion between 21-observation parameter refits.
2. **Chronological transforms:** training-only scaling, lagged inputs, and matching neural train/validation/test dates after GARCH warmup.
3. **Distinct hybrid objectives:** positive log-ratio reconstruction `h = g * exp(predicted_log_ratio)` and additive residual reconstruction `h = g + predicted_residual`, with raw and floored outputs retained separately.
4. **Reproducibility:** seeds set before model construction, restored-checkpoint loss diagnostics, source signatures, and completion checks.

Log-MSE and arithmetic conditional-mean variance are different objectives. The report examines that distinction alongside the additive model's positivity limitation.

## Data and Model Design

The cached `^GSPC` daily-price snapshot covers 2003-2024. The target is squared daily log return, a noisy second-moment proxy; its interpretation as conditional variance assumes a zero conditional mean.

Pure neural inputs are 21 lagged rows of log return, squared return, absolute return, and annualized 21-day rolling volatility. Hybrids add lagged GARCH variance. Forecast targets and losses use daily variance, not annualized volatility.

After common feature availability, the initial 777-row training span yields 756 sequences. Validation has 252 observations; test blocks have 21 and do not overlap. Training expands over time. An incomplete trailing test block is excluded.

Neural settings: 8 units, dropout 0.1, Adam 0.001, batch 64, maximum 35 epochs, patience 6, primary seed 42 plus split ID. CPU execution uses unrolled recurrence for the fixed short sequence.

## Reproduce

Use an isolated environment. The tested research configuration is Python 3.10.16 on macOS arm64; pinned package versions are in [requirements-research.txt](requirements-research.txt). Bitwise equivalence on other platforms is not guaranteed.

```bash
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
RUN_TF_TESTS=1 python -m unittest discover -s tests -v

python scripts/run_corrected_experiment.py \
  --raw data/raw/sp500_raw.csv --out runs/corrected-v1 --prepare
python scripts/run_all_models.py --out runs/corrected-v1 --workers 3
python scripts/evaluate_corrected_experiment.py \
  --out runs/corrected-v1 \
  --vix data/processed/vix_daily.csv \
  --destination reports/corrected_results.json

# Additive specification: reuse the benchmark artifacts without overwriting them.
python scripts/run_additive_residual.py \
  --source runs/corrected-v1 --out runs/additive-residual-v1
python scripts/evaluate_additive_residual.py \
  --source runs/corrected-v1 --out runs/additive-residual-v1 \
  --destination reports/additive_results.json

python -m pip install -r requirements-report.txt
python reports/build_readme_figure.py
python reports/build_final_report.py
python reports/build_portfolio_report.py
python reports/build_research_brief.py
python reports/build_retrained_report.py
```

The raw price snapshot, VIX snapshot, complete predictions, gates, and training logs are **not bundled**. Supply appropriately sourced input snapshots; the aggregate artifact records the exact hashes used for this run. Providing different input bytes creates a different experiment. VIX is used only for ex-post gate diagnostics, not as a training feature.

Full training is not a lightweight test. The runner uses separate model processes, bounded worker counts, checkpointed outputs, and run signatures. Reuse an output directory only with matching inputs/code/configuration. Use a separate directory for smoke tests or alternative seeds.

### Stock Retraining and Portfolio Evaluation

The supplied return and cap snapshots are described in the [training protocol](docs/stock-overlay-retraining.md). No vendor data or saved per-stock weights/forecasts are distributed. The completed run took approximately 104 minutes with eight single-thread CPU workers in the recorded environment; other machines may differ. Review the documented retrospective source limitations before running.

```bash
python scripts/retrain_stock_overlay.py \
  --source /path/to/legacy-portfolio-inputs \
  --out runs/stock-retrain-v1 --prepare
python scripts/retrain_stock_overlay.py --out runs/stock-retrain-v1 --workers 8
python scripts/evaluate_stock_retraining.py \
  --source /path/to/legacy-portfolio-inputs --run runs/stock-retrain-v1
python scripts/verify_stock_retraining.py \
  --source /path/to/legacy-portfolio-inputs --run runs/stock-retrain-v1
python reports/build_retrained_report.py
```

For an interrupted run, omit `--prepare` and keep the same signed code/configuration/inputs. Use a new output directory for a different experiment. The report builder verifies source-code hashes against the committed aggregate evidence and preserves the index report. Run it **last** after other report builders: the earlier builders otherwise restore the cached-forecast portfolio presentation.

### Earlier Cached-Forecast Diagnostic

The portfolio runner requires the three local CSV inputs described in the protocol; it neither downloads data nor trains a stock model. Review the source limitations before acknowledging them. Keep daily outputs in the ignored `runs/` directory, not in published reports.

```bash
python scripts/evaluate_portfolio_overlay.py \
  --input-dir /path/to/legacy-portfolio-inputs \
  --daily-output runs/portfolio-diagnostic \
  --acknowledge-retrospective-data
python reports/build_portfolio_report.py
python reports/build_research_brief.py
python reports/build_retrained_report.py
```

The [earlier aggregate results](reports/portfolio_overlay_results.json) and [accounting protocol](docs/portfolio-overlay-protocol.md) are retained for provenance, not as the latest model scores. They include an explicitly labeled replay of old notebook scores with old accounting defects. Historical vendor inputs and per-stock forecasts are not distributed.

## Evaluation

The following loss/target details apply to the index experiment. The stock experiment's different trailing-volatility target, positive-forecast subset, common coverage, persistence control and portfolio intervals are specified in its [evaluation addendum](docs/stock-overlay-evaluation.md).

MSE and QLIKE are computed from identical daily targets. QLIKE is `mean(log(h) + y/h)`, with a fixed `1e-12` numerical floor; lower is better, and values can be negative.

Paired stationary-bootstrap intervals quantify temporal uncertainty in candidate-minus-GARCH loss differences: mean block length 21, 2,000 replications, 95% pointwise percentile intervals, with block-length sensitivity. These are not multiple-comparison-adjusted intervals or measures of training-seed uncertainty.

The additive control is trained on affine-standardized signed residuals, not log ratios. Its raw QLIKE is undefined because of nonpositive forecasts; raw MSE remains reportable. The primary floored output uses a fixed `1e-12` floor, with floor sensitivities disclosed rather than selected for favorable test performance.

Calendar diagnostics, level calibration, positive-forecast checks, and gate/VIX summaries accompany the aggregate scores. Gate correlations remain descriptive, not causal explanations or signals.

## Code Map

| Location | Purpose |
| --- | --- |
| [src/data](src/data) | Return features and chronological splits |
| [src/models](src/models) | GARCH recursion, neural fitting, positive reconstruction |
| [src/models/stock_overlay.py](src/models/stock_overlay.py) | Separate stock rolling GARCH and expanding-history ratio LSTM |
| [src/losses](src/losses) | Original-scale forecast losses |
| [src/evaluation/portfolio.py](src/evaluation/portfolio.py) | Self-financing long-only overlay accounting, timing, drift, and metrics |
| [scripts](scripts) | Preparation, training, and evaluation workflow |
| [tests](tests) | Regression, bootstrap, and optional CPU integration checks |
| [reports/build_final_report.py](reports/build_final_report.py) | Standalone research report from the public aggregate evidence |
| [reports/build_portfolio_report.py](reports/build_portfolio_report.py) | Portfolio diagnostic and combined report; preserves the forecast-only PDF |
| [reports/build_research_brief.py](reports/build_research_brief.py) | Two-page brief with separate forecast and portfolio panels |
| [reports/build_retrained_report.py](reports/build_retrained_report.py) | Latest stock report, complete report and brief; run last |
| [docs/validation-status.md](docs/validation-status.md) | Completed checks and unresolved research limits |

### Supporting Evidence

- [Benchmark scores, intervals, and source hashes](reports/corrected_results.json)
- [Additive scores, floor sensitivity, and source hashes](reports/additive_results.json)
- [Benchmark experiment specification](docs/corrected-experiment-plan.md) and [additive specification](docs/additive-residual-plan.md)

The `src` modules and reproduction commands above are the canonical implementation. Supporting notebooks and earlier report artifacts are not required reading for the current study.

Based on a collaborative research project; portfolio report prepared by Eddie Liu.
