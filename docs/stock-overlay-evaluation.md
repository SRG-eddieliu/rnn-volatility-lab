# Retraining Evaluation Addendum

Fixed before evaluating the new stock-model performance. The original retraining
configuration and its signed protocol are unchanged.

- In portfolio comparisons, the primary GARCH tilt is masked to the dates/assets
  where a neural model issued a finite raw forecast. Both models therefore have
  the same model-coverage opportunity set; all other eligible stocks retain the
  neutral base tilt. Negative neural outputs are still counted and receive neutral
  hybrid tilts, not positive or floored forecasts. Label this GARCH comparator
  "matched coverage" rather than implying it uses every available GARCH forecast.
- Also disclose the unrestricted GARCH tilt as a monthly 10 bps sensitivity.
- Forecast comparisons use common finite GARCH/hybrid observations. The primary
  volatility MSE compares forecasts with trailing 21-day realized volatility.
  Include lagged trailing-volatility persistence as a transparent control for the
  target's mechanical overlap. This is not a new candidate selected on performance.
- Daily loss differences first average across the same available stocks at each
  date, then average dates equally. Bootstrap entire daily cross-sectional means
  with paired stationary blocks, not individual stock/date observations as IID.
- Squared-return variance MSE and QLIKE use the same strictly positive forecast
  subset, with a disclosed numerical variance floor of 1e-12. Report invalid and
  missing counts separately; this conditional comparison does not erase failures.
- Record all public evaluator/portfolio-engine hashes separately from the frozen
  training signature. No training code or signed configuration changes during the
  run. Do not select model hyperparameters or seeds from any evaluated scores.
