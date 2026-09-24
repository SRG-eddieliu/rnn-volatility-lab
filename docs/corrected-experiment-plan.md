# Corrected Experiment Plan

Specified before inspecting the new test results on September 24, 2026. This is a repair-and-research iteration on an already studied historical dataset, not a preregistered untouched holdout study.

## Fixed Design

- Input: existing raw `^GSPC` daily snapshot. Recompute returns and deterministic features from adjusted close; hash the input. No fresh data download or retrospective price replacement.
- GARCH: zero-mean GARCH(1,1), Student-t innovations, 756-return warmup, parameter refit every 21 observations. Update the conditional variance after every newly observed return. Reject failed fits and nonpositive/non-finite forecasts.
- Common availability: begin the modeling frame only when all GARCH and recurrent features exist. Use a 21-observation lookback and a 777-row initial training span to obtain 756 usable sequences, then 252 validation observations and non-overlapping 21-observation test blocks. All neural models use identical training, validation, and test dates.
- Six neural specifications: pure LSTM/GRU, lagged-GARCH-feature LSTM/GRU, and multiplicative log-ratio LSTM/GRU. The latter replace the historical additive residual models and are explicitly new specifications.
- Positive reconstruction: learn `log(max(squared_return, 1e-8) / garch_variance)` with a signed standardized target; output `garch_variance * exp(predicted_log_ratio)`. No prediction floor is selected against test losses. Reject numeric overflow/underflow rather than silently hiding it.
- Other neural settings: 8 units, dropout 0.1, Adam 0.001, batch 64, at most 35 epochs, early-stopping patience 6, seed 42 plus split index. Set the seed before constructing the model. CPU execution uses unrolled recurrent operations for the fixed short sequence; this changes execution strategy, not the model equations.
- Evaluation: original daily-variance scale; MSE and QLIKE, identical dates, coverage, floor incidence, calibration and temporal slices. Include lagged 21-day mean squared return and EWMA(lambda=0.94) as simple baselines. No claim of a tradable strategy.
- Primary run: one seed for all six neural specifications. Any subsequent seed sensitivity is separately labeled, with its exact coverage. Do not represent temporal loss uncertainty as neural training-seed uncertainty.
- Calendar diagnostics, fixed before the new test losses are inspected: 2010-2016, 2017-2019, 2020-2021, and 2022-2024. These are descriptive slices, not a regime model selected using the realized outcomes.
- Temporal uncertainty: paired stationary-block bootstrap of the loss differences versus GARCH, 2,000 replications, mean block length 21, seed 20260924, 95% pointwise percentile intervals. Show block-length sensitivity at 5 and 63 with 1,000 replications. Intervals are conditional on the trained forecasts, assume suitable time-series regularity, and are not adjusted for multiple model comparisons.
- Logs: evaluate train and validation losses in inference mode at the same restored best-validation checkpoint. Keep independent minima only as legacy diagnostics, not as a generalization gap.

## Interpretation Boundaries

The common sample begins later than the old 2007 comparison because a full training history is required after GARCH warmup. Report the actual start and end dates and do not compare the new metrics directly with the earlier unequal-history design.

The corrected inverse resolves an implementation mismatch. It does not make log-MSE an arithmetic conditional-mean estimator. The multiplicative specification addresses positivity, not all forecast calibration or objective mismatch. Report unsuccessful results as clearly as improvements.

The original additive notebooks remain historical examples. The canonical corrected workflow is `scripts/run_corrected_experiment.py`, with separately named outputs and hashed run signatures. Old outputs must not be resumed into a corrected run.
