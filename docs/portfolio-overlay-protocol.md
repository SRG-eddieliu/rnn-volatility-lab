# Portfolio Overlay: Fixed Diagnostic Protocol

Specified before scoring the repaired accounting implementation, September 25,
2026. The old sample and notebook performance had already been inspected; this
is not preregistration on an untouched holdout.

## Question and Separate Model Lineage

Does the legacy volatility-ratio tilt add economic value relative to the same
stock pool and base weights, after correcting portfolio accounting?

The legacy model is per-stock GARCH volatility multiplied by an LSTM correction
ratio. Its saved forecasts are not the additive variance-residual LSTM from the
index forecasting experiment. Do not transfer the index MSE result to this
portfolio or claim a causal LSTM contribution without a matched GARCH-only panel.

## Fixed Primary Comparison

- Inputs: saved `hybrid_vol_df.csv`, `mcap_weight_df.csv`, `log_ret_bt.csv`.
- Signal at close d: base weight times
  `clip(trailing_21_day_vol[d] / saved_forecast[d], 0.5, 1.5)`, normalized.
- The saved forecast dated d was constructed using returns before d in the
  notebook. This preserves the legacy as-of-close risk-ratio signal; it is not
  relabeled as a fresh d+2 volatility forecast. Per-fit logs and source signatures
  for that historical training are unavailable, so forecast lineage is not
  independently reproduced here.
- Eligibility: a finite trailing 21-return history and positive base allocation,
  assessed at the signal date, never using next-period return availability.
- Missing/nonpositive forecasts receive a neutral multiplier of one. Count and
  disclose these substitutions; do not treat negative volatility as low risk.
- Execute at the next observed trading close. Earn returns from that close to
  the following close: signal d, execute d+1, return end d+2.
- Primary: monthly rebalance at the first observed trading close of the month;
  initialize at the first available execution close. Daily, weekly, yearly and
  zero-lag same-close execution are sensitivities, not model-selection candidates.
- Primary cost: 10 bps per dollar bought or sold (5 bps fee + 5 bps slippage).
  Solve costs and post-cost target allocations jointly. Include initial entry;
  final NAV is marked to market with no terminal liquidation fee.
- Between rebalances hold shares fixed, allowing weights to drift. Aggregate
  simple stock returns, not weighted log returns. Missing held-asset returns
  are fatal rather than silently treated as zero. Uninvested cash earns zero.
- Matched controls: identical eligible stock pool, base allocations, dates,
  execution convention, rebalance frequency and cost model, without the tilt.
- Run both legacy snapshot-cap and equal-weight bases. Equal weighting removes
  the cap snapshot itself, not survivorship bias in the constituent list.

## Metrics and Fixed Sensitivities

Report net CAGR, annualized arithmetic mean, annualized volatility, excess Sharpe,
maximum drawdown including initial wealth, traded notional and costs. Use 252
trading periods/year, sample standard deviation, and the old constant daily
risk-free assumption 0.0001; this is an assumption, not a historical rate series.
Show zero-risk-free Sharpe as a sensitivity. Use paired stationary-bootstrap
Sharpe-difference intervals (mean block 21, 2,000 resamples, fixed seed 20260925;
block lengths 5/63 at 1,000 resamples). These intervals are conditional on cached
forecasts and the biased sample; they cannot repair source or selection bias.

Fixed sensitivities: daily/weekly/monthly/yearly at 10 bps; monthly at 0/10/25 bps;
monthly idealized same-close execution versus the one-close-lag primary.
Report all of them, not just the best outcome.

## Hard Data Limitations

The supplied cap-weight panel repeats one later market-cap snapshot across
history, and the stock list consists of later surviving constituents. Historical
membership, delisting outcomes and contemporaneous market caps are not supplied.
Stocks use adjusted-close returns, while ^GSPC is a price-index reference, so
outperformance versus that index is not a clean total-return alpha comparison.
The index is contextual only; the matched portfolio is the primary comparator.

The deliverable is a reproducible **retrospective accounting diagnostic** and
tested portfolio engine. It is not a survivorship-free, point-in-time alpha study,
production strategy or claim that LSTM forecasts improve portfolio Sharpe.
Do not publish raw stock returns, cap snapshots, forecasts or original notebooks.
Publish only authored code, tests, aggregate diagnostic results and clearly
qualified figures. Preserve the six-page forecast-only report unchanged.
