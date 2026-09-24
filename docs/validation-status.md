# Validation Status

Review date: 2026-09-24.

This note qualifies the historical Markdown and PDF reports. Documentation has been updated; the underlying models have not been corrected or retrained as part of this cleanup.

## GARCH State Updates

In [rolling_garch_forecast](../src/models/garch.py), a fitted result is reused between parameter refits. Calling its one-step forecast repeatedly does not incorporate the new returns that arrive in that interval.

Consequently, the current implementation repeats the fitted endpoint forecast between refits rather than producing a daily state-updated forecast. The baseline and any hybrid features derived from it need to be regenerated after correction.

## Comparable Evaluation Dates

The historical report includes different observation counts across model families. Cross-model loss comparisons need an explicit common-date intersection, with the same target definition and missing-data policy.

The published MSE/QLIKE ordering is a historical output, not a validated ranking of model families.

## Residual Variance and QLIKE

Some residual-hybrid forecasts are clipped close to zero. The historical report records large QLIKE penalties for those predictions despite relatively low MSE.

The revised experiment should report how often forecasts are clipped and check sensitivity to the variance floor. A favorable MSE alone does not establish a useful variance forecast.

## Required Rerun

1. Add a regression test showing that new returns update the next conditional variance between parameter refits.
2. Check forecast timestamps and information availability.
3. Regenerate baseline, hybrid features, and all dependent predictions.
4. Evaluate on a common test-date set; retain coverage and clipping diagnostics.
5. Record dependency versions, data snapshot details, configurations, seeds, and artifact locations.
6. Publish the revised comparison separately from the historical report.

The current repository does not include a comprehensive automated regression test suite. The report's referenced CSV predictions and logs are generated artifacts, not files bundled in the public prediction directory.

## Scope of Existing Material

The notebooks and reports remain useful for inspecting model design, transformations, and diagnostic questions. Existing PDF reports are preserved as historical research records; no new performance claims or revised metrics are asserted here.
