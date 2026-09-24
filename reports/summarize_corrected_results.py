"""Render the public Markdown summary from the same aggregate evidence as the PDFs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'corrected_results.json').read_text())
rows=d['metrics'];bykey={r['model']:r for r in rows};best=min(rows,key=lambda r:r['qlike'])
best_nn=min(rows[3:],key=lambda r:r['qlike']);best_mse=min(rows,key=lambda r:r['mse'])
ci=next(r for r in d['paired_loss_intervals'] if r['model']==best_nn['model'] and r['loss']=='qlike')
metric_lines=['| Forecast | MSE | QLIKE | Mean forecast / mean proxy | Floor count |',
              '| --- | ---: | ---: | ---: | ---: |']
for r in rows:metric_lines.append(f"| {r['label']} | {r['mse']:.9e} | {r['qlike']:.6f} | {r['forecast_to_proxy_mean_ratio']:.4f} | {r['floor_count_1e12']} |")
periods=['2010-2016','2017-2019','2020-2021','2022-2024']
calendar={(r['model'],r['period']):r for r in d['calendar_periods']}
period_lines=['| Forecast | '+' | '.join(periods)+' |','| --- | ---: | ---: | ---: | ---: |']
for r in rows:period_lines.append('| '+r['label']+' | '+' | '.join(f"{calendar[r['model'],period]['qlike']:.4f}" for period in periods)+' |')
garch=d['garch_diagnostics']
text=f"""# Volatility Forecasting: Corrected Walk-Forward Study

**Eddie Liu** | Corrected-v1 | September 24, 2026

This is a fresh historical rerun after implementation repairs and a revised common-sample design. It is not an untouched prospective holdout, a multiple-seed study, or evidence of trading alpha.

## Reports and Evidence

- [Research study (8 pages)](volatility-research.pdf)
- [Technical appendix (5 pages)](volatility-diagnostics.pdf)
- [Combined report (13 pages)](volatility-research-with-appendix.pdf)
- [Aggregate results, intervals, and source hashes](corrected_results.json)
- [Experiment plan](../docs/corrected-experiment-plan.md)
- [Validation scope and limitations](../docs/validation-status.md)

## Question and Method

The study asks whether recurrent models improve one-day S&P 500 variance forecasts over parsimonious baselines. The cached daily-price snapshot spans 2003-2024. The observed target is squared daily log return, a noisy second-moment proxy; its interpretation as conditional variance assumes a zero conditional mean.

Three baselines (Student-t GARCH(1,1), lagged RV21, EWMA 0.94) are compared with pure, GARCH-feature, and positive log-ratio LSTM/GRU models. The six neural specifications each completed 178 expanding-window folds. All nine forecasts share **3,738 test dates, 2010-02-05 through 2024-12-11**.

Every neural model has the same sample dates and at least 756 initial training sequences after a 21-row lookback, with 252 validation observations and 21-observation test blocks. GARCH retains its own expanding return-estimation history. Neural features end at t-1; transformations are fitted on training data. The incomplete trailing test block is excluded.

The common period starts later than the old 2007 comparison because a full neural training history is now required after GARCH warmup. The new log-ratio hybrids replace the old additive residual models. Consequently, new and old scores are not a one-change before/after comparison.

## Same-Date Results

{chr(10).join(metric_lines)}

MSE is `mean((y-h)**2)`. QLIKE is `mean(log(h)+y/h)`, with y and h clipped at a fixed `1e-12` for numerical evaluation. Lower is better; QLIKE may be negative under this convention. Floor count is the number of stored forecasts at or below `1e-12`. Mean forecast / mean proxy is a coarse level-calibration diagnostic, not a full conditional calibration test.

**Point estimates:** {best['label']} has the lowest QLIKE ({best['qlike']:.6f}); {best_mse['label']} has the lowest MSE ({best_mse['mse']:.9e}). The lowest neural QLIKE is {best_nn['label']} ({best_nn['qlike']:.6f}). These are observed point-estimate orderings, not a selection-adjusted claim of model superiority.

## Paired Uncertainty

For {best_nn['label']}, mean QLIKE minus GARCH QLIKE is **{ci['difference_vs_garch']:+.6f}**, with pointwise 95% interval **[{ci['ci95_lower']:+.6f}, {ci['ci95_upper']:+.6f}]**. Negative differences favor the candidate.

Intervals use paired stationary bootstrap, average block length 21, 2,000 replications, seed 20260924. The aggregate artifact includes all candidate MSE/QLIKE differences and block-length sensitivity at 5 and 63. These intervals are conditional on the trained forecasts, not adjusted for multiple comparisons, and do not represent training-seed or model-selection uncertainty. The highlighted neural model was selected by its observed QLIKE point estimate.

## Calendar Diagnostics

{chr(10).join(period_lines)}

Values are mean QLIKE, using identical model dates within each calendar slice. Counts: {', '.join(f"{p}: {calendar['garch_t',p]['n']}" for p in periods)}. These periods were fixed before inspecting new test losses; they are descriptive slices, not a learned regime model.

## What the Revision Establishes

- The GARCH recursion now assimilates daily returns between parameter refits. All {garch['refits']} refits converged; exact repeated forecasts occur on {garch['same_as_previous']} of {garch['transitions']} transitions ({garch['share_repeated']:.2%}). A regression test also matches the actual arch fixed-parameter reference.
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
"""
(ROOT/'research_report.md').write_text(text)
print('Updated research_report.md from corrected_results.json')
