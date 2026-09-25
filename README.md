# RNN Volatility Lab

A controlled study of one-day S&P 500 variance forecasts using GARCH, LSTM, GRU, and residual-learning models.

**[Read the research report](reports/volatility-final-report.pdf)** | **[Explore the code](src)** | **[Run the demo](#quick-offline-demo)**

3,738 common test dates, 178 expanding-window folds, and explicit checks of error, positivity, calibration, and temporal uncertainty. This is a single-seed historical study, not a trading-alpha claim or production model.

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

The [six-page report](reports/volatility-final-report.pdf) presents the complete methods, results, uncertainty, and reproducibility details in one document. The study compares nine benchmark/log-target specifications plus one additive-residual LSTM with raw and floored outputs on **3,738 identical dates**, February 5, 2010 to December 11, 2024.

Results are reported whether or not a neural model improves on a simple baseline. The analysis distinguishes implementation correctness, positivity, calibration, and loss-based performance. No strategy P&L or Sharpe-ratio improvement is claimed.

**Main finding:** the additive-residual LSTM lowers observed MSE by **5.48%** versus GARCH, from `1.600623e-7` to `1.512933e-7` with a fixed `1e-12` floor. The unbounded output improves MSE by **5.45%**. However, there are **60 nonpositive raw forecasts**, floored QLIKE is approximately **457,801** versus GARCH's **-8.4613**, and the primary 95% paired temporal interval for the MSE difference includes zero. This is an observed MSE improvement, not demonstrated robust or deployable variance forecasting.

GARCH beats all six log-target/log-ratio neural candidates on MSE and QLIKE. The additive specification was tested in a separate run after inspecting the historical sample, using the same 178 folds, benchmark predictions, architecture, seed schedule, and training budget. Neither experiment is an untouched holdout.

## Implementation

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
```

The raw price snapshot, VIX snapshot, complete predictions, gates, and training logs are **not bundled**. Supply appropriately sourced input snapshots; the aggregate artifact records the exact hashes used for this run. Providing different input bytes creates a different experiment. VIX is used only for ex-post gate diagnostics, not as a training feature.

Full training is not a lightweight test. The runner uses separate model processes, bounded worker counts, checkpointed outputs, and run signatures. Reuse an output directory only with matching inputs/code/configuration. Use a separate directory for smoke tests or alternative seeds.

## Evaluation

MSE and QLIKE are computed from identical daily targets. QLIKE is `mean(log(h) + y/h)`, with a fixed `1e-12` numerical floor; lower is better, and values can be negative.

Paired stationary-bootstrap intervals quantify temporal uncertainty in candidate-minus-GARCH loss differences: mean block length 21, 2,000 replications, 95% pointwise percentile intervals, with block-length sensitivity. These are not multiple-comparison-adjusted intervals or measures of training-seed uncertainty.

The additive control is trained on affine-standardized signed residuals, not log ratios. Its raw QLIKE is undefined because of nonpositive forecasts; raw MSE remains reportable. The primary floored output uses a fixed `1e-12` floor, with floor sensitivities disclosed rather than selected for favorable test performance.

Calendar diagnostics, level calibration, positive-forecast checks, and gate/VIX summaries accompany the aggregate scores. Gate correlations remain descriptive, not causal explanations or signals.

## Code Map

| Location | Purpose |
| --- | --- |
| [src/data](src/data) | Return features and chronological splits |
| [src/models](src/models) | GARCH recursion, neural fitting, positive reconstruction |
| [src/losses](src/losses) | Original-scale forecast losses |
| [scripts](scripts) | Preparation, training, and evaluation workflow |
| [tests](tests) | Regression, bootstrap, and optional CPU integration checks |
| [reports/build_final_report.py](reports/build_final_report.py) | Standalone research report from the public aggregate evidence |
| [docs/validation-status.md](docs/validation-status.md) | Completed checks and unresolved research limits |

### Supporting Evidence

- [Benchmark scores, intervals, and source hashes](reports/corrected_results.json)
- [Additive scores, floor sensitivity, and source hashes](reports/additive_results.json)
- [Benchmark experiment specification](docs/corrected-experiment-plan.md) and [additive specification](docs/additive-residual-plan.md)

The `src` modules and reproduction commands above are the canonical implementation. Supporting notebooks and earlier report artifacts are not required reading for the current study.

Based on a collaborative research project; portfolio report prepared by Eddie Liu.
