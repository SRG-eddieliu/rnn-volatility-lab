# Validation Status

Review date: September 24, 2026.

**Documentation and historical arithmetic reviewed; models not corrected or retrained.** The revised [research note](../reports/volatility-research.pdf) and [diagnostic appendix](../reports/volatility-diagnostics.pdf) replace the legacy PDFs in the current repository tree. The originals remain in Git history and the local source archive. Do not interpret the new publication date as a new experiment.

## Confirmed Implementation Issues

### GARCH State Updates

In [rolling_garch_forecast](../src/models/garch.py), the fitted result is reused between parameter refits. Repeated calls to its one-step forecast do not incorporate intervening returns. The intended parameter-refit cadence of 21 trading days is distinct from the missing daily state recursion.

The retained baseline repeats its previous prediction on 4,300 of 4,514 transitions (95.26%). Its median and maximum constant run are 21 observations. A focused test with a mocked fit confirms that the existing control flow does not pass new returns to the reused result. This is not a test of corrected GARCH estimation.

The baseline, GARCH-feature hybrids, residual targets, and residual reconstruction all require regeneration after a code correction.

### Target Transform and Inverse

In [rnn.py](../src/models/rnn.py), the log-target forward transform uses `log(max(y, eps))`, while the inverse subtracts eps after exponentiation. With `eps=1e-8`, a round trip maps `1e-6` to approximately `9.90e-7`. Clipping cannot be inverted below the floor, but this discrepancy above the floor is avoidable.

Even after fixing the inverse, exponentiating a conditional log-target prediction does not generally recover conditional mean variance. The training objective and forecast estimand need to be treated separately.

### Residual Positivity

The residual target is the signed difference `squared_return - garch_variance`, not a standardized return innovation. Reconstruction can produce nonpositive forecasts that are clipped at `1e-12`. The saved residual LSTM and GRU hit this floor on 99 and 297 of 4,473 dates (2.21% and 6.64%). The large QLIKE penalties are retained, not removed as inconvenient observations.

Prediction-floor sensitivity is included as a diagnostic, with the observed-target floor held fixed at `1e-12`. It is not a validated floor choice or a substitute for a positive forecast construction.

## Checks Completed on Saved Outputs

- All seven prediction series have unique dates, finite values, nonnegative targets, and positive stored forecasts.
- Native-sample MSE and QLIKE reconcile to the stored metric file to numerical rounding.
- Cross-model losses were independently recalculated on the common 4,473 dates from 2007-03-08 to 2024-12-11. Three series originally have 4,515 observations; four have 4,473.
- Common-date targets agree within `rtol=1e-6, atol=1e-12`; the maximum absolute stored difference is approximately `5.20e-10`.
- All 21 architecture/variant/activation Pearson correlations with VIX reconcile to the saved summary. Spearman correlations were recalculated from ranks.
- A focused GARCH control-flow check and a target-transform round-trip check were run against the reviewed code.

These checks validate arithmetic and identify failures. They do not establish model validity or prove that every historical data-processing step is leakage-free. [Aggregate evidence and source hashes](../reports/review_metrics.json) identify the reviewed artifacts; raw prediction CSVs are retained locally, not included in the public repository.

## Interpretation Corrections

Gate/VIX correlations use same-date VIX close as ex-post context. They do not show causality, identify a memory horizon, or establish a trading signal. Pooled gate series combine separately trained models across rolling splits.

The original training log's minimum training loss and minimum validation loss may come from different epochs. Their difference is not a paired checkpoint generalization gap. Transformed losses across log-target and signed-residual models are not on comparable scales.

Both MSE and QLIKE can be robust to an unbiased noisy volatility proxy under suitable assumptions. Their disagreement is not proof that one loss is categorically invalid. No confidence intervals, repeated-seed experiment, formal predictive-accuracy test, or economic backtest have been completed as part of this review.

## Required Rerun

1. Correct the daily GARCH state update; add a regression test that responds to a new return with parameters fixed.
2. Correct the target inverse, test boundary behavior, and evaluate positive forecast parameterizations.
3. Audit forecast timestamps and information availability; regenerate all affected artifacts.
4. Rerun all specifications on identical dates and report coverage, floor incidence, calibration, and regime diagnostics.
5. Add simple persistence/EWMA comparisons, multiple seeds, and paired loss-difference inference appropriate for time series.
6. Record a data snapshot, dependency versions, configurations, and artifact provenance. Evaluate train/validation losses at the same selected checkpoint.

The legacy report-generation script still produces the historical presentation; it is not the source of these revised PDFs. The public repository does not yet include a comprehensive automated regression suite or complete prediction bundle. The current reports are suitable for discussing research design and failure analysis, not validated model superiority or production deployment.
