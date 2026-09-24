# Volatility Forecasting: Corrected Walk-Forward Study

**Eddie Liu** | Corrected-v1 | September 24, 2026

This is the preserved nine-specification experiment. The later [additive-residual follow-up](additive_residual_report.md) restores the original signed-residual MSE objective on identical dates; it must be read separately from the log-ratio results below.

This is a fresh historical rerun after implementation repairs and a revised common-sample design. It is not an untouched prospective holdout, a multiple-seed study, or evidence of trading alpha.

## Reports and Evidence

- [Research study (8 pages)](volatility-research.pdf)
- [Technical appendix (5 pages)](volatility-diagnostics.pdf)
- [Combined report with the additive follow-up (17 pages)](volatility-research-with-appendix.pdf)
- [Aggregate results, intervals, and source hashes](corrected_results.json)
- [Experiment plan](../docs/corrected-experiment-plan.md)
- [Validation scope and limitations](../docs/validation-status.md)

## Question and Method

The study asks whether recurrent models improve one-day S&P 500 variance forecasts over parsimonious baselines. The cached daily-price snapshot spans 2003-2024. The observed target is squared daily log return, a noisy second-moment proxy; its interpretation as conditional variance assumes a zero conditional mean.

Three baselines (Student-t GARCH(1,1), lagged RV21, EWMA 0.94) are compared with pure, GARCH-feature, and positive log-ratio LSTM/GRU models. The six neural specifications each completed 178 expanding-window folds. All nine forecasts share **3,738 test dates, 2010-02-05 through 2024-12-11**.

Every neural model has the same sample dates and at least 756 initial training sequences after a 21-row lookback, with 252 validation observations and 21-observation test blocks. GARCH retains its own expanding return-estimation history. Neural features end at t-1; transformations are fitted on training data. The incomplete trailing test block is excluded.

The common period starts later than the old 2007 comparison because a full neural training history is now required after GARCH warmup. The new log-ratio hybrids replace the old additive residual models. Consequently, new and old scores are not a one-change before/after comparison.

## Same-Date Results

| Forecast | MSE | QLIKE | Mean forecast / mean proxy | Floor count |
| --- | ---: | ---: | ---: | ---: |
| GARCH-t | 1.600622585e-07 | -8.461252 | 1.0379 | 0 |
| Lagged RV21 | 1.872075038e-07 | -8.352889 | 1.0045 | 0 |
| EWMA 0.94 | 1.748182700e-07 | -8.419993 | 1.0044 | 0 |
| Pure LSTM | 1.938755211e-07 | -6.448296 | 0.2579 | 0 |
| Pure GRU | 1.919253966e-07 | -6.430305 | 0.2595 | 0 |
| Feature LSTM | 1.964149243e-07 | -6.606568 | 0.2618 | 0 |
| Feature GRU | 1.970106747e-07 | -6.514910 | 0.2560 | 0 |
| Ratio LSTM | 1.863164022e-07 | -6.659491 | 0.2985 | 0 |
| Ratio GRU | 1.851276707e-07 | -6.571780 | 0.2826 | 0 |

MSE is `mean((y-h)**2)`. QLIKE is `mean(log(h)+y/h)`, with y and h clipped at a fixed `1e-12` for numerical evaluation. Lower is better; QLIKE may be negative under this convention. Floor count is the number of stored forecasts at or below `1e-12`. Mean forecast / mean proxy is a coarse level-calibration diagnostic, not a full conditional calibration test.

**Point estimates:** GARCH-t has the lowest QLIKE (-8.461252); GARCH-t has the lowest MSE (1.600622585e-07). The lowest neural QLIKE is Ratio LSTM (-6.659491). These are observed point-estimate orderings, not a selection-adjusted claim of model superiority.

## Paired Uncertainty

For Ratio LSTM, mean QLIKE minus GARCH QLIKE is **+1.801761**, with pointwise 95% interval **[+1.567056, +2.056143]**. Negative differences favor the candidate.

Intervals use paired stationary bootstrap, average block length 21, 2,000 replications, seed 20260924. The aggregate artifact includes all candidate MSE/QLIKE differences and block-length sensitivity at 5 and 63. These intervals are conditional on the trained forecasts, not adjusted for multiple comparisons, and do not represent training-seed or model-selection uncertainty. The highlighted neural model was selected by its observed QLIKE point estimate.

## Calendar Diagnostics

| Forecast | 2010-2016 | 2017-2019 | 2020-2021 | 2022-2024 |
| --- | ---: | ---: | ---: | ---: |
| GARCH-t | -8.4901 | -8.9501 | -7.9983 | -8.2113 |
| Lagged RV21 | -8.3647 | -8.8155 | -7.8724 | -8.1816 |
| EWMA 0.94 | -8.4576 | -8.8811 | -7.9056 | -8.2128 |
| Pure LSTM | -6.8869 | -7.6318 | -4.0539 | -5.8456 |
| Pure GRU | -6.7932 | -7.6644 | -4.3032 | -5.7717 |
| Feature LSTM | -6.9629 | -7.5703 | -4.8374 | -5.9947 |
| Feature GRU | -6.8420 | -7.5936 | -4.8942 | -5.7533 |
| Ratio LSTM | -7.0743 | -7.3071 | -5.2702 | -5.9730 |
| Ratio GRU | -6.9949 | -7.2841 | -5.1179 | -5.8438 |

Values are mean QLIKE, using identical model dates within each calendar slice. Counts: 2010-2016: 1739, 2017-2019: 754, 2020-2021: 505, 2022-2024: 740. These periods were fixed before inspecting new test losses; they are descriptive slices, not a learned regime model.

## What the Revision Establishes

- The GARCH recursion now assimilates daily returns between parameter refits. All 228 refits converged; exact repeated forecasts occur on 0 of 4778 transitions (0.00%). A regression test also matches the actual arch fixed-parameter reference.
- The target inverse is consistent with its clipping/log transform. Below-floor clipping remains intentionally non-invertible.
- Positive log-ratio hybrids avoid the unconstrained additive reconstruction. Positivity does not guarantee calibration or better forecast losses.
- Training and validation diagnostics use the same restored best-validation checkpoint in inference mode. Different transformed target families still should not be compared as though their training losses shared a common estimand.
- Complete forecast coverage, identical targets, positivity, artifact hashes, and metric arithmetic are checked before the report is produced.

## What Remains Open

The primary run has one seed schedule. The data are one index and a previously studied cached snapshot, not an untouched prospective holdout. Log-target and log-ratio MSE do not generally target arithmetic conditional-mean variance after exponentiation; this is a modeling limitation distinct from the repaired inverse bug. An original-scale objective is a follow-up, not a completed result here.

Gate/VIX correlations use same-date VIX as ex-post context and pool separately trained models. They do not identify a memory horizon, causality, or alpha. No economic strategy, transaction-cost analysis, portfolio P&L, or production validation was performed.

## Reproduction and Provenance

The public repository contains the corrected scripts, tests, pinned research dependencies, this summary renderer, a PDF builder, and aggregate evidence. Raw price/VIX snapshots and complete predictions remain local; hashes identify the exact inputs and outputs used. See the [README](../README.md) for commands.

The project originated in collaborative research. This portfolio report is presented by Eddie Liu; contributor names and course identifiers are omitted. The byline does not imply sole authorship of the original experiments. Historical reports remain in Git history, and [review_metrics.json](review_metrics.json) remains a clearly separate pre-rerun audit.

## References

- [Bollerslev (1986), GARCH](https://doi.org/10.1016/0304-4076(86)90063-1)
- [Patton (2011), volatility forecast comparison with imperfect proxies](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
- [arch forecasting documentation](https://arch.readthedocs.io/en/latest/univariate/forecasting.html)
- [arch stationary bootstrap](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html)
