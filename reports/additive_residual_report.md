# Additive Residual LSTM: Controlled Follow-up

**Eddie Liu** | additive-residual-v1 | September 24, 2026

## Answer

The original additive-residual LSTM, with its fixed output floor, has 5.48% lower observed MSE than corrected GARCH: 1.512933e-07 versus 1.600623e-07.

The 95% paired temporal interval for the MSE difference includes zero; the observed ordering is not a statistically resolved improvement in this sample.

The unbounded model produces 60 nonpositive forecasts. The primary reconstruction clips 60 forecasts to 1e-12; its QLIKE is 4.5780e+05, versus -8.46125 for GARCH. The MSE ranking alone does not establish valid or well-calibrated variance forecasts.

The previous corrected-v1 run replaced additive residuals with log ratios. This follow-up restores the original signed additive target; the two specifications must not be conflated. The current evidence is a single-seed historical rerun, not an untouched holdout or an alpha claim.

## Identical-Date Comparison

| Specification | MSE | Change vs GARCH | QLIKE | Nonpositive / floor |
| --- | ---: | ---: | ---: | ---: |
| GARCH-t | 1.600622585e-07 | +0.00% | -8.46125 | 0 / 0 |
| Pure LSTM | 1.938755211e-07 | +21.13% | -6.44830 | 0 / 0 |
| Feature LSTM | 1.964149243e-07 | +22.71% | -6.60657 | 0 / 0 |
| Ratio LSTM | 1.863164022e-07 | +16.40% | -6.65949 | 0 / 0 |
| Additive LSTM: raw | 1.513428189e-07 | -5.45% | Undefined | 60 / 60 |
| Additive LSTM: clipped | 1.512933361e-07 | -5.48% | 4.5780e+05 | 0 / 60 |

All rows use the same 3,738 daily squared-return targets, 2010-02-05 through 2024-12-11. Negative percentage changes mean lower MSE. The raw and clipped rows come from one newly trained LSTM, not two separate fits. The four benchmark files are byte-identical to corrected-v1. Raw QLIKE is undefined if any raw forecast is nonpositive; the clipped series uses the existing fixed-floor convention.

## Method

Target: `e = y - g`, where y is squared daily log return and g is the corrected one-day GARCH variance forecast. Train on an affine-standardized signed residual, with a linear output and MSE. Restore residual units, then compute `raw = g + predicted_e` and `clipped = max(raw, 1e-12)`.

Before clipping, `(y - raw)^2 = (e - predicted_e)^2`. The standardization is a constant affine transform fitted on each fold's training data, so the within-fold training objective is proportional to original-scale residual MSE. It is not log-target or log-ratio MSE.

The same 178 expanding-window folds, 21-row inputs ending at t-1, feature set, eight units, dropout 0.1, Adam 0.001, batch 64, maximum 35 epochs, patience six, and seed schedule 42 plus split ID were used. Epoch caps and stopping rules match; actual stopping epochs may differ with the objective. Mean epochs in this run: 19.43.

## Uncertainty

| Output and loss | Mean difference vs GARCH | 95% lower | 95% upper |
| --- | ---: | ---: | ---: |
| additive_raw, mse | -8.719440e-09 | -2.467417e-08 | +1.192146e-10 |
| additive_clipped, mse | -8.768922e-09 | -2.473057e-08 | +3.384849e-11 |
| additive_clipped, qlike | +4.578095e+05 | +2.395828e+05 | +7.545538e+05 |

Negative differences favor the additive model. Paired stationary-bootstrap percentile intervals use average block length 21, 2,000 replications, and seed 20260924. These are pointwise temporal intervals conditional on the trained forecasts, not multiple-seed or model-selection uncertainty. The aggregate file retains block-length sensitivities 5 and 63.

## Stability and Numerical Limits

The clipped additive model has lower MSE in 98 of 178 test blocks. This is descriptive, not an independent win-rate test. The ten largest absolute daily MSE differences account for 42.36% of the total absolute difference. No dates were removed from the primary comparison.

Calendar-period results and fixed-floor sensitivity at 1e-12, 1e-10, 1e-8, and 1e-6 are reported in the PDF and aggregate JSON. The latter three floors are diagnostics only; none was selected to improve the reported test outcome.

## Evidence and Reproduction

- [Four-page follow-up](volatility-additive-residual.pdf)
- [Combined follow-up, original corrected study, and appendix](volatility-research-with-appendix.pdf)
- [Verified aggregate results and hashes](additive_results.json)
- [Plan fixed before follow-up scoring](../docs/additive-residual-plan.md)
- [Runner](../scripts/run_additive_residual.py) and [evaluator](../scripts/evaluate_additive_residual.py)

```bash
python scripts/run_additive_residual.py --source runs/corrected-v1 --out runs/additive-residual-v1
python scripts/evaluate_additive_residual.py --source runs/corrected-v1 --out runs/additive-residual-v1 --destination reports/additive_results.json
python reports/build_additive_report.py
```

Generate corrected-v1 first using the repository README. Input snapshots and full forecasts are not bundled. No existing corrected-v1 predictions are changed by this follow-up. Future work should not tune against this repeatedly examined test period. No trading P&L, execution, or production validation is represented.

The underlying project originated in collaborative research. This revised portfolio report is presented by Eddie Liu; contributor names and course identifiers are omitted without implying sole original authorship.
