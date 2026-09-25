# Validation Status

## Completed Stock Retraining - September 25, 2026

All 503 stock columns were processed without unhandled worker failures. The run
saved 1,968 LSTM fits across 502 stocks; one has insufficient neural training
history. Training-only scaler statistics, prior-label cutoffs, input/code hashes
and all 1,968 checkpoint files were verified. GARCH was freshly refitted from
returns, and old cached hybrid forecasts were not reused.

All **53 tests pass including TensorFlow**, with no skipped tests. New coverage
checks stock timing, chronological scaling, forecast coverage, future-data
invariance, evaluation loss alignment and independent portfolio accounting.
The older TensorFlow test setup now sets thread environment defaults before
initialization so both integration suites can run together; the index training
implementation, aggregate scores and six-page report are unchanged.

Independent NumPy recurrence replay checked 69 predictions across 14 fit blocks,
including AMCR/TTWO failure extrema, with scalers reconstructed from prior labels.
Fifteen fresh GARCH fits were checked. Six primary portfolio return paths were
recomputed with dollar holdings and bisection transaction-cost accounting over
2,390 dates each; maximum daily-return disagreement is below 6e-16. These are
implementation checks, not independent certification of the source data.

**The retrained model is not robust:** 475 raw outputs are nonpositive, while
AMCR contributes 99.96% of aggregate hybrid MSE through extreme ratio targets
associated with near-zero GARCH denominators. Although 493/502 stock-level MSEs
improve, aggregate MSE is worse and the persistence control outperforms both.
Monthly cap-base/GARCH/hybrid Sharpes are 1.1189/1.1007/1.1146; primary paired
hybrid-minus-comparator intervals include zero. Later survivor stocks and cap
snapshots remain a source of bias, so no point-in-time alpha claim is justified.

See [findings](stock-retraining-findings.md),
[aggregate evidence](../reports/stock_retraining_results.json), and
[independent checks](../reports/stock_retraining_verification.json).
The latest brief/report use this completed run; the notes below retain the
chronology of earlier work and must not be read as the latest stock-model status.

## Earlier Cached-Forecast Portfolio Extension - September 25, 2026

The separate stock portfolio accounting diagnostic is now implemented and rerun.
Fourteen new unit tests cover weighted simple returns, weight drift, execution
lags, holiday schedules, self-financing transaction costs, missing held returns,
future-data invariance, neutral forecast fallback, and performance definitions.
The full lightweight suite passes 38 tests; four unchanged TensorFlow integration
tests are optional and were not rerun for this extension.

A separate dollar-position calculation, using bisection instead of the engine's
fixed-point cost solution, reconciles all 2,390 primary daily returns, NAVs, costs
and traded notionals for the cap base and overlay. The largest daily-return
disagreement is below 6e-16. The saved original notebook Sharpes are also reproduced
from the same inputs, but those legacy values retain the old accounting defects.

Monthly net Sharpe is 1.119 for the matched cap base and 1.115 for the tilt;
the paired primary interval includes zero. The source is **not point-in-time**:
one later cap snapshot and a survivor stock list are reused across history.
Stock forecasts are a different multiplicative model, were not retrained, and
lack complete independently verifiable training logs. The diagnostic does not
validate alpha or show that the index forecasting MSE change improves trading.
See [protocol](portfolio-overlay-protocol.md) and
[aggregate evidence](../reports/portfolio_overlay_results.json).

The forecast-only validation notes below describe Part A, not the portfolio
extension. The six-page forecast PDF and all forecast experiment scores remain
unchanged; the eight-page report appends the distinct portfolio diagnostic.

Corrected historical experiment: September 24, 2026.

**Ready within the tested implementation scope; share research findings with the limitations below.** The previously documented GARCH-state and target-inverse defects have been repaired. Six neural specifications were freshly trained across 178 folds on a common sample. This does not amount to independent production validation or a prospective alpha study.

## Repairs and Tests

- Daily GARCH state recursion assimilates each new return between parameter refits. A test compares the result with the actual arch library under fixed parameters; shock and timing tests ensure an already-issued forecast is not changed by a later return.
- The log-target inverse no longer subtracts epsilon. Boundary and round-trip tests cover values above and below the target clipping floor.
- Validation data do not determine training transform statistics. Duplicate dates and non-finite sequence values are rejected; sequence tests confirm the target-date row is excluded.
- Positive log-ratio reconstruction is a new specification replacing the old additive residual design. It uses a positive baseline and an exponential correction, with explicit numerical overflow/underflow rejection.
- Neural seeds are set before model construction. CPU integration tests cover LSTM and GRU, deterministic initialization, best-checkpoint train/validation evaluation, and retraining an incomplete checkpoint.
- Bootstrap tests cover constant paired differences, seed repeatability, expected shape, and invalid inputs. The training supervisor and evaluator check current source hashes against preparation; a regression test rejects changed source before continuing.

The complete test suite contains 27 tests, including four TensorFlow integration tests enabled with `RUN_TF_TESTS=1`. All were run successfully in the recorded environment. Additive-specific checks cover signed targets through training, zero-correction fallback, the residual-MSE identity, explicit clipping, undefined raw QLIKE, and float32 CSV round-trip reconstruction.

## Additive Follow-up

The original signed additive-residual LSTM has now been retrained on the same 178 folds, using the unchanged corrected-v1 feature frame and benchmark forecasts. The code, configuration, input hashes, exact date/target coverage, and training counts are checked. See the [follow-up plan](additive-residual-plan.md), [results](../reports/additive_residual_report.md), and [aggregate evidence](../reports/additive_results.json).

Observed MSE falls by 5.45% before clipping and 5.48% with the original fixed `1e-12` floor. However, 60 raw forecasts are nonpositive; clipped QLIKE is approximately 457,801 versus -8.4613 for GARCH. The paired 95% temporal interval for clipped MSE minus GARCH MSE is approximately `[-2.47306e-8, +3.38485e-11]`, which includes zero. Share the observed improvement with these caveats, not as a robust superiority or production-readiness claim.

TensorFlow's residual predictions are float32. The evaluator restores that dtype when reading their short CSV representation and verifies the float64 reconstructed forecasts exactly to numerical tolerance; the source files themselves are unchanged. This is a serialization audit, not a new model choice or a rescore with tuned predictions.

## Corrected Experimental Design

The original early hybrid folds had much less usable training history than pure RNNs. The corrected frame starts after common GARCH-feature availability, then allocates 756 neural training sequences, 252 validation observations, and 21-observation tests. The resulting common test sample has 3,738 observations from 2010-02-05 to 2024-12-11.

All neural specifications share sample dates. GARCH still has its own expanding return-estimation history; equal neural sample availability does not imply identical effective memory for every estimator. The incomplete trailing block is excluded.

The 2008 crisis falls in training history, not the corrected test period. The new scores are not a controlled one-bug difference from the old 2007-start results because sample availability and the residual specification changed.

## Output Verification

The evaluation script requires all six model completion records and their matching forecast hashes. It checks 178 distinct split records per neural model, expected training and validation availability, positive finite forecasts, identical test dates, target agreement, and complete gate/VIX matches.

Independent array calculations reconcile with the repository loss functions. Aggregate results include level-calibration diagnostics, calendar slices, paired temporal intervals, GARCH convergence records, environment versions, and input/code/prediction hashes. See [corrected_results.json](../reports/corrected_results.json).

Raw snapshots and prediction CSVs are retained locally and not distributed in this repository. Hashes identify the reviewed artifacts but do not replace access to them for an independent full rerun.

## Remaining Research Limitations

1. **Objective alignment and positivity:** log-MSE and log-ratio MSE do not generally target arithmetic conditional-mean variance after exponentiation. The additive follow-up restores an original-scale MSE-aligned objective but does not guarantee positive variance. A positive-output model trained directly under an original-scale objective remains untested.
2. **Seed and model-selection uncertainty:** the primary run uses one training seed schedule. Temporal loss bootstrap intervals are conditional on those trained forecasts and are not multiple-comparison adjusted.
3. **Historical reuse:** the cached dataset was inspected in previous research. This is chronological walk-forward testing, not an untouched prospective final holdout.
4. **Data and proxy scope:** one equity index and a noisy squared-return proxy. The snapshot was not independently reconciled against another vendor, and corporate data revisions were not reconstructed point in time.
5. **Gate interpretation:** same-date VIX is ex-post context; pooled gate averages across separate model fits do not establish causality, memory horizon, or alpha.
6. **Economic and operational validation of the index model:** no portfolio P&L or execution model is established for the index forecasts. The separate stock model now has cost-aware retrospective accounting, but its biased sources do not validate achievable returns. Neither study has production monitoring or independent model-governance sign-off.

## Historical Artifacts

The old PDFs are recoverable in Git history. [review_metrics.json](../reports/review_metrics.json) contains the historical pre-rerun audit, including the stale baseline and clipped additive residual outputs. It is retained for provenance, not as current performance evidence.

The original notebooks and `reports/generate_pdf_report.py` retain historical workflows and may contain superseded interpretation. Use the corrected scripts and report builder documented in the [README](../README.md).
