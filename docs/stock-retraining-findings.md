# Stock Retraining v1: Findings and Failure Diagnosis

All 503 supplied columns were processed. The run saved 1,968 LSTM fits across
502 stocks; one stock had insufficient history for a neural fit. All stock
receipts, checkpoints, scalers and strict prior-date training cutoffs were checked.
No old hybrid forecasts were reused. Core settings were fixed before evaluation;
no assets, seeds or forecasts were removed in response to these results.

## Forecast Quality Is Not Robust

Date-equal mean squared error against the actual legacy target, trailing 21-day
sample volatility, is 0.0001278946 for GARCH, 0.1913049 for the raw hybrid and
0.0000030968 for lagged trailing-volatility persistence. The hybrid's conditional
MSE difference interval versus GARCH is entirely above zero, indicating worse
loss in this experiment, not a demonstrated forecasting improvement.

493 of 502 individual stocks have lower hybrid MSE, and the median stock-level
hybrid/GARCH MSE ratio is 0.4961. Those statistics do not excuse the overall loss:
AMCR contributes 99.9643% of date-equal hybrid MSE. Its training correction ratios
reach tens of thousands when the GARCH denominator is near zero. The first
model's historical target mean/standard deviation are approximately 4,645/5,451,
far from an order-one volatility correction. An optimizer convergence flag alone
therefore did not establish forecast plausibility.

Across the forecast panel there are 475 nonpositive neural outputs: TTWO 463,
AMCR 6 and TYL 6. There are also 1,311 positive hybrid forecasts above one in
daily fractional-volatility units. The largest absolute hybrid sigma is 124.13.
The unconstrained ratio output is unsuitable for a robust volatility estimator.

Independent NumPy replay of saved LSTM weights and historical scaler statistics
reproduces the inspected extrema, including all four AMCR and TTWO model blocks.
This supports a model/target failure rather than an aggregation or neural-inference
implementation mismatch. It does not authenticate the vendor's historical
security mapping or prove why the source price history has its observed pattern.

Squared-return MSE and QLIKE are separately reported on the same positive-forecast
subset. GARCH's extremely small forecasts also damage QLIKE. The persistence
control outperforms both on those mean scores; neither sophisticated model earns
a blanket robustness claim. The variance floor is 1e-12 and invalid-output
coverage is disclosed rather than silently converted into valid forecasts.

## Economic Value Is Unresolved

Monthly rebalancing, 10 bps per dollar traded, next-close execution:

| Base | No tilt Sharpe | Matched GARCH Sharpe | Hybrid Sharpe |
| --- | ---: | ---: | ---: |
| Snapshot cap | 1.1189 | 1.1007 | 1.1146 |
| Equal weight | 0.7375 | 0.7290 | 0.7286 |

The cap hybrid's observed Sharpe advantage over matched GARCH is +0.0139, with
95% paired temporal interval [-0.0089, +0.0335]. Versus no tilt, the difference is
-0.0043, interval [-0.0302, +0.0191]. Both equal-weight intervals also include zero.
Maximum drawdown for the cap base/hybrid is -33.25%/-32.48% in this sample.

The hybrid's mean target-weight L1 distance from the cap base is 0.1129, versus
0.1524 for GARCH. Thus a lower-error forecast on many stocks does not simply map
to stronger tilts. Portfolio clipping can mask extreme forecasts without fixing
their underlying numerical behavior. This is descriptive, not causal attribution.

All six primary daily portfolio ledgers are independently reconciled using dollar
holdings and a bisection solution for transaction costs. Untilted base portfolios
also reproduce the preceding accounting experiment. There is no resolved
incremental Sharpe benefit, and the surviving-stock universe and later cap snapshot
still preclude a point-in-time alpha claim.

## Next Experiment, Not a Retroactive Repair

A follow-on study would need a separately fixed protocol for historical data
quality, training-only scale-aware volatility safeguards, a positive/robust target
parameterization, and a forecasting target aligned with the actual trading horizon.
Use multiple seeds and a genuinely new holdout. Do not remove AMCR, floor bad
forecasts after observing losses, or rerun seeds until this version looks favorable.

Evidence: [aggregate results](../reports/stock_retraining_results.json),
[independent verification](../reports/stock_retraining_verification.json),
[training protocol](stock-overlay-retraining.md), and
[evaluation addendum](stock-overlay-evaluation.md).
