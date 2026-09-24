# Additive Residual LSTM: Controlled Follow-up

Fixed before inspecting follow-up test losses, September 24, 2026.

## Question

Does the original additive residual LSTM improve overall variance MSE against the repaired GARCH baseline? The earlier corrected-v1 log-ratio model is a different specification and cannot answer this question on its own.

## Fixed Design

- Read the existing corrected-v1 feature frame, chronological splits, and benchmark forecasts without modifying them. Require their source-code signature to match the repaired implementation.
- Train one LSTM on `residual_var = sq_return - garch_cond_var`, using training-only affine target standardization, a linear output, and MSE. Do not log-transform the signed residual or constrain its sign.
- Reuse the exact 178 splits, 3,738 test dates, 21-row lookback, eight units, dropout 0.1, Adam 0.001, batch 64, 35-epoch maximum, patience six, and seed schedule 42 plus split ID. Capture gates as before; no new gate/VIX interpretation is required.
- Reconstruct the unbounded forecast as `raw = garch + predicted_residual`. Retain it for the exact identity `MSE(y, raw) = MSE(y-garch, predicted_residual)` and invalid-variance diagnostics.
- The primary reported forecast follows the original notebook: `clipped = max(raw, 1e-12)`. This floor is fixed before observing results. Report raw and clipped MSE separately; never present clipping as learned calibration.
- Compare with corrected GARCH, pure LSTM, GARCH-feature LSTM, and log-ratio LSTM on identical dates and targets. Previous GRU results remain available but no new GRU is trained.

## Evaluation and Decision Rules

Report MSE, relative MSE change versus GARCH, QLIKE for valid positive forecasts, mean calibration, raw nonpositive count, floor hits, and minimum prediction. Raw QLIKE is undefined if any raw forecast is nonpositive; do not silently floor it. Evaluate the clipped forecast under the existing fixed-floor QLIKE convention and disclose all floor hits.

Use paired stationary-bootstrap loss differences versus GARCH: mean block length 21, 2,000 replications, seed 20260924. Retain block-length sensitivities 5 and 63 with 1,000 replications. Use the same pre-existing calendar periods: 2010-2016, 2017-2019, 2020-2021, 2022-2024. Inspect concentration of the aggregate MSE difference and report the share of folds with lower MSE; do not use either to select or remove test dates.

As a numerical sensitivity diagnostic only, also score reconstructed forecasts with fixed floors 1e-10, 1e-8, and 1e-6. None may replace the primary 1e-12 result based on its test performance. No shrinkage, hyperparameter search, alternate seed, or sample change is part of this run.

Lower observed MSE is a point-estimate improvement. A pointwise temporal interval excluding zero is conditional evidence, not multi-seed robustness or a selection-adjusted discovery. If MSE improves but positivity or QLIKE fails, report both findings. The historical sample has already been examined and is not an untouched holdout.

## Preservation

Use a separate output directory and input/code/output hashes. Keep corrected-v1 and its reports intact as the previous experiment. Publish a clearly named additive follow-up and update the repository overview; do not silently relabel log-ratio results as additive residual results. Raw data, predictions, and private interview notes stay local.
