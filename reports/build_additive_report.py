"""Render a four-page additive follow-up and prepend it to the preserved corrected study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.platypus import PageBreak, Spacer

from build_corrected_report import fonts, p, table, pdf, WIDTH


def number(value):
    if value is None:
        return 'Undefined'
    return f'{value:.5f}' if abs(value) < 100 else f'{value:.4e}'


def findings(data):
    rows = {r['model']: r for r in data['metrics']}
    baseline, primary, raw = rows['garch_t'], rows['additive_clipped'], rows['additive_raw']
    interval = next(r for r in data['paired_intervals'] if r['model'] == 'additive_clipped' and r['loss'] == 'mse')
    direction = 'lower' if primary['mse'] < baseline['mse'] else 'higher'
    mse_sentence = (f"The original additive-residual LSTM, with its fixed output floor, has {abs(primary['mse_change_pct_vs_garch']):.2f}% "
                    f"{direction} observed MSE than corrected GARCH: {primary['mse']:.6e} versus {baseline['mse']:.6e}.")
    if interval['ci95_lower'] <= 0 <= interval['ci95_upper']:
        uncertainty = 'The 95% paired temporal interval for the MSE difference includes zero; the observed ordering is not a statistically resolved improvement in this sample.'
    elif interval['ci95_upper'] < 0:
        uncertainty = 'The 95% paired temporal interval is below zero, providing conditional evidence of lower MSE. It does not establish seed robustness or selection-adjusted superiority.'
    else:
        uncertainty = 'The 95% paired temporal interval is above zero, providing conditional evidence of higher MSE in this comparison.'
    if data['raw_nonpositive_count']:
        validity = (f"The unbounded model produces {data['raw_nonpositive_count']:,} nonpositive forecasts. "
                    f"The primary reconstruction clips {data['primary_clipped_count']:,} forecasts to 1e-12; "
                    f"its QLIKE is {number(primary['qlike'])}, versus {number(baseline['qlike'])} for GARCH. "
                    'The MSE ranking alone does not establish valid or well-calibrated variance forecasts.')
    else:
        validity = (f"All unbounded forecasts are positive; {data['primary_clipped_count']:,} predictions require the fixed floor. "
                    f"The primary QLIKE is {number(primary['qlike'])}, versus {number(baseline['qlike'])} for GARCH. "
                    'Positivity alone does not establish conditional calibration.')
    return mse_sentence, uncertainty, validity


def write_markdown(data, destination):
    mse_sentence, uncertainty, validity = findings(data)
    rows = ['| Specification | MSE | Change vs GARCH | QLIKE | Nonpositive / floor |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in data['metrics']:
        rows.append(f"| {r['label']} | {r['mse']:.9e} | {r['mse_change_pct_vs_garch']:+.2f}% | {number(r['qlike'])} | {r['nonpositive_count']} / {r['floor_count']} |")
    intervals = ['| Output and loss | Mean difference vs GARCH | 95% lower | 95% upper |',
                 '| --- | ---: | ---: | ---: |']
    for r in data['paired_intervals']:
        intervals.append(f"| {r['model']}, {r['loss']} | {r['difference_vs_garch']:+.6e} | {r['ci95_lower']:+.6e} | {r['ci95_upper']:+.6e} |")
    destination.write_text(f"""# Additive Residual LSTM: Controlled Follow-up

**Eddie Liu** | additive-residual-v1 | September 24, 2026

## Answer

{mse_sentence}

{uncertainty}

{validity}

The previous corrected-v1 run replaced additive residuals with log ratios. This follow-up restores the original signed additive target; the two specifications must not be conflated. The current evidence is a single-seed historical rerun, not an untouched holdout or an alpha claim.

## Identical-Date Comparison

{chr(10).join(rows)}

All rows use the same {data['n']:,} daily squared-return targets, {data['test_start']} through {data['test_end']}. Negative percentage changes mean lower MSE. The raw and clipped rows come from one newly trained LSTM, not two separate fits. The four benchmark files are byte-identical to corrected-v1. Raw QLIKE is undefined if any raw forecast is nonpositive; the clipped series uses the existing fixed-floor convention.

## Method

Target: `e = y - g`, where y is squared daily log return and g is the corrected one-day GARCH variance forecast. Train on an affine-standardized signed residual, with a linear output and MSE. Restore residual units, then compute `raw = g + predicted_e` and `clipped = max(raw, 1e-12)`.

Before clipping, `(y - raw)^2 = (e - predicted_e)^2`. The standardization is a constant affine transform fitted on each fold's training data, so the within-fold training objective is proportional to original-scale residual MSE. It is not log-target or log-ratio MSE.

The same 178 expanding-window folds, 21-row inputs ending at t-1, feature set, eight units, dropout 0.1, Adam 0.001, batch 64, maximum 35 epochs, patience six, and seed schedule 42 plus split ID were used. Epoch caps and stopping rules match; actual stopping epochs may differ with the objective. Mean epochs in this run: {data['training']['mean_epochs']:.2f}.

## Uncertainty

{chr(10).join(intervals)}

Negative differences favor the additive model. Paired stationary-bootstrap percentile intervals use average block length 21, 2,000 replications, and seed 20260924. These are pointwise temporal intervals conditional on the trained forecasts, not multiple-seed or model-selection uncertainty. The aggregate file retains block-length sensitivities 5 and 63.

## Stability and Numerical Limits

The clipped additive model has lower MSE in {data['concentration']['folds_with_lower_mse']} of {data['concentration']['folds']} test blocks. This is descriptive, not an independent win-rate test. The ten largest absolute daily MSE differences account for {100*data['concentration']['top10_share_absolute_difference']:.2f}% of the total absolute difference. No dates were removed from the primary comparison.

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
""")


def build(results, out, source_reports):
    fonts()
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(results.read_text())
    bykey = {r['model']: r for r in data['metrics']}
    mse_sentence, uncertainty, validity = findings(data)
    result_rows = [['Forecast', 'MSE / 10^-7', 'MSE change', 'QLIKE', 'Floor hits']]
    for row in data['metrics']:
        result_rows.append([row['label'], f"{row['mse']/1e-7:.5f}", f"{row['mse_change_pct_vs_garch']:+.2f}%",
                            number(row['qlike']), str(row['floor_count'])])
    story = [p('ADDITIVE RESIDUAL FOLLOW-UP / SEPTEMBER 24, 2026', 'kicker'),
             p('Can Residual Learning<br/>Improve GARCH?', 'title'), p('<b>Eddie Liu</b>'),
             p('A controlled test of the original additive LSTM, distinct from the log-ratio alternative.', 'subtitle'),
             p('Executive Summary', 'h2'), p(mse_sentence), p(uncertainty), p(validity),
             p('Same-date comparison', 'h2'), table(result_rows, [149, 90, 82, 105, 78], compact=True),
             p('Table 1. Lower MSE and QLIKE are better. MSE change is relative to GARCH; negative means improvement. '
               'The raw and clipped rows represent one model. Raw floor hits count values at or below 1e-12, '
               'not an applied raw-output floor. Undefined QLIKE is not replaced with a favorable score.', 'small'),
             p(f"Source: additive-residual-v1 and unchanged corrected-v1 forecasts; {data['n']:,} common dates, "
               f"{data['test_start']} to {data['test_end']}. One primary seed schedule; previously examined historical data.", 'small'),
             PageBreak()]

    story += [p('01 / Restore the Original Question', 'h1'),
              p('The previous corrected run changed the additive residual model into a multiplicative log-ratio '
                'model to guarantee positive forecasts. That change did not test whether the original additive '
                'residual objective improves overall variance MSE. This follow-up supplies that missing control.'),
              p('Target and reconstruction', 'h2'),
              p('e<sub>t</sub> = y<sub>t</sub> - g<sub>t</sub><br/>'
                'h<sub>t</sub><super>raw</super> = g<sub>t</sub> + predicted e<sub>t</sub><br/>'
                'h<sub>t</sub><super>clipped</super> = max(h<sub>t</sub><super>raw</super>, 10<super>-12</super>)', 'eq'),
              p('Here y is squared daily log return and g is the corrected GARCH forecast, not a return residual '
                'or standardized GARCH innovation. Signed residuals receive training-only affine standardization; '
                'they are not logged and their sign is not constrained.'),
              table([['Control', 'Specification'],
                     ['Information and sample', '178 expanding-window folds; 3,738 common test dates. Sequences end at t-1. Initial training has 756 sequences; validation has 252 observations; test blocks have 21.'],
                     ['Network and inputs', 'LSTM, eight units, 21-row lookback, dropout 0.1. The same lagged returns, squared/absolute returns, rolling volatility, and GARCH-feature transformation as the prior hybrid models.'],
                     ['Optimization', 'Linear output; standardized signed-residual MSE; Adam 0.001; batch 64; maximum 35 epochs; patience six; restored best-validation checkpoint.'],
                     ['Randomness and budget', f"Seed 42 plus split ID; deterministic CPU operations. Same epoch cap and stopping rule, not identical realized epoch counts. Mean epochs: {data['training']['mean_epochs']:.2f}."],
                     ['Reused evidence', 'Corrected GARCH and all benchmark prediction files remain byte-identical. No test-selected shrinkage, alternative seed, or floor is introduced.']], [117, 387], compact=True),
              p('Why residual MSE is the relevant objective', 'h2'),
              p('(y - h<super>raw</super>)<super>2</super> = (e - predicted e)<super>2</super>', 'eq'),
              p('The identity is verified numerically. Within each fold, affine target standardization only '
                'rescales the squared-loss objective by a fixed positive factor. It does not imply that '
                'a finite-sample neural fit will improve out-of-sample error. Clipping changes the reconstruction '
                'and is reported separately.'), PageBreak()]

    interval_rows = [['Additive minus GARCH', 'Difference', '95% lower', '95% upper']]
    for row in data['paired_intervals']:
        scale = 1e-7 if row['loss'] == 'mse' else 1
        label = ('Raw' if row['model'] == 'additive_raw' else 'Clipped') + (' MSE / 10^-7' if row['loss'] == 'mse' else ' QLIKE')
        interval_rows.append([label, *[number(row[key]/scale) for key in ['difference_vs_garch', 'ci95_lower', 'ci95_upper']]])
    calendar = {(r['period'], r['model']): r for r in data['calendar_periods']}
    period_rows = [['Calendar period', 'n', 'GARCH MSE', 'Raw MSE', 'Clipped MSE']]
    for period in ['2010-2016', '2017-2019', '2020-2021', '2022-2024']:
        period_rows.append([period, str(calendar[period, 'garch_t']['n']),
                            *[f"{calendar[period, key]['mse']/1e-7:.5f}" for key in ['garch_t', 'additive_raw', 'additive_clipped']]])
    concentration = data['concentration']
    story += [p('02 / Uncertainty and Stability', 'h1'),
              table(interval_rows, [177, 109, 109, 109]),
              p('Table 2. Negative differences favor the additive model. MSE differences are scaled by 10^-7; '
                'QLIKE differences are unscaled. Paired stationary-bootstrap percentile intervals: average '
                'block length 21, 2,000 replications, seed 20260924.', 'small'),
              p(uncertainty),
              p('These are pointwise temporal intervals conditional on the trained forecasts. They do not '
                'include variation across training seeds, repeated research choices, or future regimes. '
                'Block lengths 5 and 63 are sensitivity checks retained in the aggregate evidence.'),
              p('Calendar-period MSE', 'h2'), table(period_rows, [128, 55, 107, 107, 107]),
              p('Table 3. All three MSE columns are divided by 10^-7. These calendar periods were retained '
                'from the previous plan. No dates are excluded from the headline comparison.', 'small'),
              p('How concentrated is the observed difference?', 'h2'),
              p(f"The clipped model has lower MSE in <b>{concentration['folds_with_lower_mse']} of "
                f"{concentration['folds']} test blocks</b>. The ten largest absolute daily MSE differences "
                f"account for <b>{100*concentration['top10_share_absolute_difference']:.2f}%</b> of total absolute "
                'daily differences. Both are descriptive diagnostics, not independent hypothesis tests.'),
              p(f"Those ten dates contribute {concentration['top10_contribution_to_mean_difference']:.6e} "
                'to the full-sample mean difference. Selecting or deleting influential dates after seeing '
                'the losses would answer a different question; the reported ranking uses every date.'),
              PageBreak()]

    floor_rows = [['Output floor', 'MSE / 10^-7', 'QLIKE', 'Mean h / mean y']]
    for row in data['floor_sensitivity']:
        floor_rows.append([f"{row['floor']:.0e}", f"{row['mse']/1e-7:.5f}", number(row['qlike']),
                           f"{row['mean_forecast_to_mean_proxy']:.3f}"])
    story += [p('03 / Validity and Research Judgment', 'h1'),
              p(validity),
              table(floor_rows, [104, 130, 130, 140]),
              p('Table 4. Only 1e-12 is the primary, original-notebook floor. Other fixed floors are '
                'numerical sensitivity diagnostics, not candidate models selected by test performance. '
                'QLIKE uses log(h) + y/h, with the established numerical floor for zero targets.', 'small'),
              p('MSE improvement is not enough for deployment', 'h2'),
              p('The additive objective directly addresses the original question, unlike log-ratio MSE. '
                'However, an unconstrained correction may create invalid variance forecasts. A fixed floor '
                'can prevent a numerical domain error without resolving the statistical problem. '
                'Any claim must distinguish point-estimate MSE, uncertainty, positivity, and QLIKE.'),
              p('Reproducibility', 'h2'),
              p('The public repository includes the pre-scoring plan, additive runner, evaluator, tests, '
                'PDF builder, and aggregate hashes. Source inputs and all prior benchmark forecasts are '
                'verified unchanged. Raw and clipped outputs are retained separately; the residual-MSE '
                'identity, checkpoint diagnostics, and complete dates are checked.'),
              p('Run scripts/run_additive_residual.py against the existing corrected-v1 directory, then '
                'scripts/evaluate_additive_residual.py. Build this report from reports/additive_results.json '
                'using reports/build_additive_report.py. Raw market data and complete predictions are local '
                'and not bundled; input hashes identify, but do not replace, those files.', 'small'),
              p('Limits and provenance', 'h2'),
              p('One equity index, one seed schedule, a noisy squared-return proxy, and repeatedly examined '
                'historical data. No return-alpha strategy, portfolio P&amp;L, or production validation is claimed. '
                'The previous nine-specification study follows this note in the combined PDF and remains '
                'a distinct experiment.', 'small'),
              p('The underlying project originated in collaborative research. This revised portfolio report '
                'is presented by Eddie Liu; contributor names and course identifiers are omitted without '
                'implying sole original authorship.', 'small'),
              p('<link href="https://github.com/SRG-eddieliu/rnn-volatility-lab">github.com/SRG-eddieliu/rnn-volatility-lab</link>', 'small')]
    followup = out/'volatility-additive-residual.pdf'
    pdf(followup, 'Volatility Forecasting - Additive Follow-up', story, 4)
    merged = PdfWriter()
    merged.append(followup, outline_item='Latest: Additive Residual Follow-up')
    merged.append(source_reports/'volatility-research.pdf', outline_item='Previous: Nine-Specification Corrected Study')
    merged.append(source_reports/'volatility-diagnostics.pdf', outline_item='Previous: Technical Appendix')
    merged.add_metadata({'/Title': 'Volatility Forecasting - Additive Follow-up and Corrected Study', '/Author': 'Eddie Liu'})
    combined = out/'volatility-research-with-appendix.pdf'
    with combined.open('wb') as stream:
        merged.write(stream)
    if len(PdfReader(combined).pages) != 17:
        raise ValueError('Expected four new pages plus the preserved thirteen-page study.')
    write_markdown(data, results.with_name('additive_residual_report.md'))
    print(json.dumps({'followup_pages': 4, 'combined_pages': 17, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('additive_results.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--source-reports', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    build(args.results, args.output_dir, args.source_reports)
