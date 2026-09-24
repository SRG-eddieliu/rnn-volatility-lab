# Volatility Forecasting: Revised Research Note

**Eddie Liu** | Revised September 24, 2026

> Historical forecasts, not a corrected model run. The baseline state update and target inverse still require correction. No models were retrained for this review.

## Reports

- [Research note (7 pages)](volatility-research.pdf)
- [Diagnostic appendix (6 pages)](volatility-diagnostics.pdf)
- [Combined report (13 pages)](volatility-research-with-appendix.pdf)
- [Aggregate review evidence and source hashes](review_metrics.json)

## Research Design

Seven specifications compare a zero-mean Student-t GARCH(1,1), pure LSTM/GRU, GARCH-feature LSTM/GRU, and additive residual LSTM/GRU. The data pipeline uses daily `^GSPC` observations over 2003-2024. Squared log return is a noisy second-moment proxy; its variance interpretation assumes a zero conditional mean.

Pure inputs are lagged log return, squared return, absolute return, and annualized 21-day rolling volatility. Hybrids also use lagged GARCH variance. Neural sequences for date t end at t-1. The expanding-window configuration uses a minimum training span of 756 observations, 252 validation observations, and non-overlapping 21-day test blocks. Warmup and sequence construction reduce effective sample sizes.

The residual target is `squared_return - garch_variance`, not a standardized return innovation. The correction is added to the saved GARCH forecast and clipped at `1e-12`.

## Common-Date Historical Results

All rows use **4,473 dates, 2007-03-08 through 2024-12-11**. This reconciles coverage, not the underlying model defects. No ranking is asserted as validated.

| Saved forecast | MSE | QLIKE | Forecasts at 1e-12 |
| --- | ---: | ---: | ---: |
| GARCH-t | 3.058102231e-07 | -8.085829 | 0 |
| Pure LSTM | 3.566812058e-07 | -5.272290 | 0 |
| Pure GRU | 3.561952277e-07 | -3.776601 | 0 |
| Feature LSTM | 3.602507815e-07 | -4.088043 | 0 |
| Feature GRU | 3.572741835e-07 | -2.506646 | 0 |
| Residual LSTM | 2.860448899e-07 | 858837.115884 | 99 |
| Residual GRU | 3.082435871e-07 | 4000312.632582 | 297 |

MSE is `mean((y-h)**2)`. QLIKE is `mean(log(h)+y/h)`, clipping y and h at `1e-12`. Lower is better; QLIKE may be negative under this convention. Both MSE and QLIKE can be robust to an unbiased noisy proxy under suitable conditions; their error weighting and finite-sample behavior differ. See [Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf).

## Material Findings

1. **GARCH state is stale between refits.** 4,300 of 4,514 adjacent predictions are identical. The median and maximum constant run are 21 observations. Source inspection and a mocked-fit control-flow check confirm that new returns are not incorporated between parameter refits. Baseline and dependent hybrids require regeneration.
2. **Target transform and inverse disagree.** The forward map uses `log(max(y, eps))`, while the inverse subtracts eps after exponentiation. Even after fixing the inverse, a log-MSE objective does not generally target conditional mean variance after naive exponentiation.
3. **Residual positivity fails.** Residual LSTM and GRU hit the floor on 99 (2.21%) and 297 (6.64%) common dates. A prediction-floor sensitivity check changes the penalty substantially, but is not a validated floor selection or a model fix.
4. **Gate correlations do not identify mechanism or alpha.** Pure LSTM forget-gate summaries have Pearson -0.596 and Spearman -0.221 versus same-date VIX. This is an ex-post pooled diagnostic across separately trained folds, not a causal or predictive result.
5. **Training gaps need checkpoint alignment.** Separate minima of train and validation loss can come from different epochs. Their difference is not a paired checkpoint generalization gap; different target transforms also preclude direct cross-family gap comparisons.

## Verification Scope

All seven prediction CSVs passed duplicate-date and finite-value checks. Stored native-sample metrics reconcile to rounding. Common-date targets agree within a relative tolerance of 1e-6 and absolute tolerance of 1e-12. All 21 gate/VIX Pearson correlations reconcile to the saved summary. Source hashes and focused test outputs are in [review_metrics.json](review_metrics.json). Raw prediction files are retained locally and not included in the public repository.

This is suitable as a portfolio discussion of research design and failure analysis, not yet a validated model ranking or production model. No trading strategy, economic backtest, multi-seed uncertainty estimate, or fresh training run is represented. See [validation status](../docs/validation-status.md).

## Research Provenance

The underlying experiments originated in a collaborative research project. This revised portfolio report is presented by Eddie Liu. Contributor names and course identifiers have been omitted; the byline does not imply sole authorship of the original experiments.
