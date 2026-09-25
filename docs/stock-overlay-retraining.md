# Stock Overlay Retraining v1

Fixed September 25, 2026 before scoring the newly trained models. This sample
was already inspected in earlier work; no untouched holdout is claimed.

## Scope

Retrain all 503 supplied stock columns from the local returns, not the cached
hybrid forecasts. Preserve the older cached-result experiment. This is the
stock-level multiplicative volatility-ratio LSTM, not the separate index-level
additive variance-residual model.

Keep the legacy core configuration: daily rolling GARCH(1,1)-t fits with zero
mean on 50 prior returns, realized volatility from 21 returns, 10-ratio LSTM
inputs, 8 units, linear output, Adam 0.001, 10 epochs, batch 8, and refitting after
each 600 eligible stock forecast dates. Training expands across available past
data starting in 2000; at least 100 valid supervised sequences are required.
The forecast period is 2015-07-01 through 2024-12-31, 2,392 trading dates.

Target at t: trailing sample volatility through t / GARCH sigma predicted for t.
Input at t: ratios t-10 through t-1. All training labels and scaler observations
are strictly earlier than the first prediction date of each refit block. Later
rows can be predicted in a batch with a frozen model, but each row only uses its
own earlier features. No validation-driven tuning or best-seed selection.

## Explicit Repairs Versus the Notebook

- Compute each stock's GARCH panel once instead of twice.
- Preserve trading-date gaps; never compress missing observations into a
  seemingly contiguous sequence.
- Reject nonconverged/invalid GARCH fits, replacing them with the same 50-prior-
  return sample volatility. Record every substitution. This comparator is
  GARCH with an explicit fallback, not a claim that every fit succeeded.
- Stable ticker/refit seeds, single-thread CPU kernels and unrolled recurrence;
  persist checkpoints, scalers, training cutoffs, fit diagnostics and hashes.
- Do not clip negative neural volatility to a misleading near-zero risk estimate.
  Retain raw predictions; portfolio invalid forecasts receive a neutral tilt.
- Models unavailable during short histories also receive a neutral portfolio
  tilt. Evaluate forecast errors only on common eligible finite predictions and
  report coverage/invalid-output counts separately.

These repairs and deterministic execution mean this is a new experiment, not
bit-for-bit reproduction of the unseeded original notebook.

## Comparisons Fixed Before Evaluation

Forecasts: GARCH baseline versus retrained hybrid, on identical stock/date
observations. Compare squared error against trailing 21-day realized volatility
(the actual legacy target); separately assess squared-return variance MSE and
QLIKE where predictions are valid. Predicting a trailing window is not equivalent
to forecasting fresh future realized volatility. Report negative raw outputs.

Portfolio: unchanged tested accounting, monthly rebalancing, 10 bps per dollar
bought/sold, next-close execution, constant assumed daily risk-free 0.0001. Run
snapshot-cap base, GARCH tilt, and hybrid tilt; repeat for equal-weight bases.
Use the same eligible stocks within each base, common dates, and identical costs.
Report hybrid-minus-GARCH and hybrid-minus-untilted-base Sharpe differences with
paired stationary bootstrap (mean block 21, 2,000 draws, seed 20260925).
Show all daily/weekly/monthly/yearly 10 bps and monthly 0/10/25 bps sensitivities,
not just the best-performing configuration.

## Completion and Limits

Full completion requires all 503 stock receipts, immutable input/code signatures,
checkpoint hashes, no unhandled worker failures, and verified temporal cutoffs.
Pilot subsets are only engineering checks, never full-universe performance claims.
Keep raw returns, model weights, forecasts and per-stock fit artifacts private in
ignored runs/. Publish authored implementation and aggregate results only.

Retraining repairs missing model provenance, not historical source bias: later
surviving constituents and a repeated later market-cap snapshot remain. Equal
weighting does not remove survivorship. No contemporaneous universe/caps,
delisting data, capacity study or untouched holdout is supplied. Do not call these
results validated alpha or achievable returns.
