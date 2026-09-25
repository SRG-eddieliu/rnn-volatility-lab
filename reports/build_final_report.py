"""Build the standalone portfolio report from committed, unchanged evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from reportlab.platypus import Image, PageBreak

from build_corrected_report import WIDTH, fonts, p, pdf, table


def load_evidence(directory):
    parent_path = directory / "corrected_results.json"
    parent = json.loads(parent_path.read_text())
    additive = json.loads((directory / "additive_results.json").read_text())
    expected = additive["evidence"]["parent_results_sha256"]
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != expected:
        raise ValueError("The additive evidence does not identify this benchmark artifact.")
    for source_key, target_key in [("test_observations", "n"),
                                   ("test_start", "test_start"), ("test_end", "test_end")]:
        if parent["prepared"][source_key] != additive[target_key]:
            raise ValueError("The experiments do not describe the same evaluation sample.")
    if not all(additive["verification"].values()):
        raise ValueError("The committed additive verification is incomplete.")
    base_rows = {row["model"]: row for row in parent["metrics"]}
    for row in additive["metrics"]:
        if row["model"] in base_rows:
            for metric in ("mse", "qlike"):
                if not math.isclose(row[metric], base_rows[row["model"]][metric], rel_tol=1e-12):
                    raise ValueError("Shared benchmark scores disagree.")
    return parent, additive


def score(value):
    return "Undefined" if value is None else f"{value:,.4f}"


def build(directory, destination):
    fonts()
    parent, additive = load_evidence(directory)
    rows = {row["model"]: row for row in additive["metrics"]}
    baseline, raw, floored = [rows[key] for key in ("garch_t", "additive_raw", "additive_clipped")]
    interval = next(row for row in additive["paired_intervals"]
                    if row["model"] == "additive_clipped" and row["loss"] == "mse")
    if not interval["ci95_lower"] <= 0 <= interval["ci95_upper"]:
        raise ValueError("Reassess the narrative: the reported uncertainty has changed.")
    if len(parent["metrics"]) != 9 or additive["raw_nonpositive_count"] != 60:
        raise ValueError("Reassess the narrative against the changed evidence.")

    story = [
        p("RESEARCH REPORT / SEPTEMBER 2026", "kicker"),
        p("Volatility Forecasting:<br/>GARCH and Recurrent Models", "title"),
        p("Error reduction, forecast validity, and temporal uncertainty", "subtitle"),
        p("<b>Eddie Liu</b>"),
        p("Executive Summary", "h2"),
        p(f"An additive-residual LSTM achieves <b>{abs(floored['mse_change_pct_vs_garch']):.2f}% "
          f"lower observed MSE</b> than GARCH-t with a fixed output floor, and "
          f"{abs(raw['mse_change_pct_vs_garch']):.2f}% lower without it. The comparison uses "
          f"{additive['n']:,} identical daily targets across 178 walk-forward folds."),
        p("The result does not establish robust superiority: the primary 95% paired temporal "
          "interval includes zero. The raw model produces 60 nonpositive forecasts, and flooring "
          f"them yields QLIKE of {floored['qlike']:,.0f}, versus {baseline['qlike']:.4f} for GARCH."),
        Image(str(directory / "figures" / "additive-summary.png"), width=WIDTH, height=226.8),
        p("Finding: aligning the training objective with variance MSE helps this point estimate, "
          "but does not guarantee positive, calibrated variance forecasts. This is a single-seed "
          "historical study, not evidence of trading alpha or a deployment-ready model.", "small"),
        p(f"Evaluation: {additive['test_start']} to {additive['test_end']}. "
          "Source: committed benchmark and additive aggregate artifacts; no new training or "
          "market-data download was performed to assemble this report.", "small"),
        PageBreak(),
        p("01 / Experiment Design", "h1"),
        table([
            ["Component", "Specification"],
            ["Data and target", "Cached S&amp;P 500 index (^GSPC) daily-price snapshot, 2003-2024. "
             "Target y is squared daily log return, a noisy second-moment proxy. Its interpretation "
             "as conditional variance assumes a zero conditional mean."],
            ["Benchmarks", "Student-t GARCH(1,1), a lagged 21-day mean of squared returns, and "
             "EWMA(0.94). GARCH parameters are refitted on expanding returns every 21 observations; "
             "the conditional-variance state updates each day."],
            ["Neural candidates", "Pure and GARCH-feature LSTM/GRU models; multiplicative "
             "log-ratio LSTM/GRU models; and an additive-residual LSTM. Raw and floored additive "
             "forecasts are two outputs of the same fitted model."],
            ["Input availability", "21 lagged rows ending at t-1: log return, squared return, "
             "absolute return, and annualized 21-day rolling volatility. Feature and residual "
             "hybrids add lagged GARCH variance. Targets remain in daily variance units."],
            ["Walk-forward splits", "After a 756-return GARCH warmup, initial neural training "
             "contains 756 sequences from 777 rows. Validation has 252 observations; test blocks "
             "have 21. Training expands across 178 non-overlapping test blocks. The incomplete "
             "trailing block is excluded."],
            ["Training", "Eight recurrent units; dropout 0.1; Adam 0.001; batch 64; at most "
             "35 epochs; patience six; seed 42 plus split ID; best-validation checkpoint restored. "
             "CPU recurrence is unrolled for the fixed sequence length."],
        ], [112, 392], compact=True),
        p("Information discipline", "h2"),
        p("A forecast for t uses information through t-1. Feature and target transforms are "
          "fitted on each fold's training sample only. Validation chooses early stopping. "
          "A past test observation may enter a later fold's training history once it is observable."),
        p("Experiment scope", "h2"),
        p("The additive specification was evaluated in a separate run after inspection of the "
          "historical sample. It uses the same dates, benchmark predictions, architecture, "
          "seed schedule, and training budget. The report combines the evidence, not the fits. "
          "Neither experiment is an untouched prospective holdout."),
        PageBreak(),
        p("02 / Complete Same-Date Results", "h1"),
    ]

    metric_rows = [["Forecast", "MSE / 10^-7", "QLIKE", "Mean h / mean y"]]
    for row in parent["metrics"]:
        metric_rows.append([row["label"], f"{row['mse']/1e-7:.5f}", score(row["qlike"]),
                            f"{row['forecast_to_proxy_mean_ratio']:.3f}"])
    for row in (raw, floored):
        label = "Additive LSTM: raw" if row is raw else "Additive LSTM: floored"
        metric_rows.append([label, f"{row['mse']/1e-7:.5f}", score(row["qlike"]),
                            f"{row['mean_forecast_to_mean_proxy']:.3f}"])
    story += [
        table(metric_rows, [164, 101, 130, 109], compact=True),
        p("Table 1. All rows use the same 3,738 squared-return targets. Lower MSE and QLIKE "
          "are better. The final two rows share one trained LSTM; the floor is 1e-12. Undefined "
          "raw QLIKE is not replaced by a favorable score. Mean h / mean y is a level-calibration "
          "diagnostic, not a test of conditional calibration.", "small"),
        p("How to interpret the ranking", "h2"),
        p("GARCH-t beats all six log-target/log-ratio recurrent candidates on both MSE and "
          "QLIKE. The additive model has a lower MSE point estimate, but its raw outputs can "
          "be invalid variances. A favorable ordering on one loss is not an overall model verdict."),
        p("Scoring conventions", "h2"),
        p("MSE = mean[(y - h)<super>2</super>]<br/>"
          "QLIKE = mean[log(h) + y / h]", "eq"),
        p("Reported QLIKE uses a 1e-12 numerical floor for positive-output evaluation, including "
          "zero targets. It can be negative. Raw additive QLIKE remains undefined because some "
          "forecasts are nonpositive. Calendar and uncertainty diagnostics use matched dates.", "small"),
        PageBreak(),
        p("03 / Uncertainty and Stability", "h1"),
    ]

    interval_rows = [["Additive minus GARCH", "Difference", "95% lower", "95% upper"]]
    for row in additive["paired_intervals"]:
        scale = 1e-7 if row["loss"] == "mse" else 1
        label = "Raw" if row["model"] == "additive_raw" else "Floored"
        label += " MSE / 10^-7" if row["loss"] == "mse" else " QLIKE"
        interval_rows.append([label, *[f"{row[key]/scale:+.6g}" for key in
                                       ("difference_vs_garch", "ci95_lower", "ci95_upper")]])
    calendar = {(row["period"], row["model"]): row for row in additive["calendar_periods"]}
    period_rows = [["Period", "n", "GARCH MSE", "Raw MSE", "Floored MSE"]]
    for period in ("2010-2016", "2017-2019", "2020-2021", "2022-2024"):
        period_rows.append([period, calendar[period, "garch_t"]["n"], *[
            f"{calendar[period, key]['mse']/1e-7:.5f}" for key in
            ("garch_t", "additive_raw", "additive_clipped")]])
    concentration = additive["concentration"]
    story += [
        table(interval_rows, [165, 113, 113, 113]),
        p("Table 2. Negative differences favor the additive model. Paired stationary-bootstrap "
          "95% percentile intervals use mean block length 21 and 2,000 replications. Both MSE "
          "intervals include zero; these are pointwise temporal intervals conditional on the "
          "trained forecasts, not seed- or selection-adjusted uncertainty.", "small"),
        p("Block-length sensitivity", "h2"),
        p("The stored sensitivity checks use 1,000 replications at mean block lengths 5 and 63. "
          "The floored-MSE interval just excludes zero at length 5 but includes it at 21 and 63. "
          "The significance conclusion is therefore not stable across these dependence assumptions."),
        p("Calendar-period MSE", "h2"),
        table(period_rows, [116, 58, 110, 110, 110]),
        p("Table 3. MSE columns are divided by 10^-7. Calendar slices are descriptive; all dates "
          "remain in the headline comparison. The 2008 crisis is in training history, not the test period.", "small"),
        p("Concentration of the difference", "h2"),
        p(f"The floored model has lower MSE in {concentration['folds_with_lower_mse']} of "
          f"{concentration['folds']} test blocks. The ten largest absolute daily MSE differences "
          f"account for {100*concentration['top10_share_absolute_difference']:.2f}% of total "
          "absolute difference. These diagnostics are not independent win-rate tests; no "
          "influential dates were removed."),
        PageBreak(),
        p("04 / Objective and Forecast Validity", "h1"),
        p("What the additive model learns", "h2"),
        p("e<sub>t</sub> = y<sub>t</sub> - g<sub>t</sub><br/>"
          "h<sub>t</sub><super>raw</super> = g<sub>t</sub> + predicted e<sub>t</sub><br/>"
          "(y - h<super>raw</super>)<super>2</super> = (e - predicted e)<super>2</super>", "eq"),
        p("The target is a variance-forecast error, not a return residual or standardized GARCH "
          "innovation. Training-only affine standardization preserves signed residuals; a linear "
          "output minimizes a loss proportional to residual MSE within a fold. This identity "
          "aligns the objective with variance MSE but does not ensure generalization or positivity."),
        p("Why positive alternatives answer a different question", "h2"),
        p("Direct log-target models exponentiate a log prediction; log-ratio hybrids reconstruct "
          "h = g * exp(predicted log ratio). These transformations ensure positive finite outputs "
          "when numerical checks pass, but log-MSE does not generally estimate the arithmetic "
          "conditional mean required to minimize variance MSE. Positivity and objective alignment "
          "are separate properties."),
        p("Fixed-floor sensitivity", "h2"),
    ]
    floor_rows = [["Floor", "MSE / 10^-7", "QLIKE", "Mean h / mean y"]]
    for row in additive["floor_sensitivity"]:
        floor_rows.append([f"{row['floor']:.0e}", f"{row['mse']/1e-7:.5f}", score(row["qlike"]),
                           f"{row['mean_forecast_to_mean_proxy']:.3f}"])
    story += [
        table(floor_rows, [88, 122, 154, 140]),
        p("Table 4. The primary floor is 1e-12. The other floors are diagnostics only, not "
          "test-selected production settings. Flooring changes the forecast and can strongly "
          "affect QLIKE; it does not supply a statistical explanation for near-zero variance.", "small"),
        p("Research implication", "h2"),
        p("A useful next experiment would combine an explicitly positive variance parameterization "
          "with a mean-targeted objective, tested under a fixed multi-seed protocol and new data. "
          "That experiment has not been run. The present evidence supports the objective/validity "
          "tradeoff, not a deployable improvement."),
        PageBreak(),
        p("05 / Code, Evidence, and Scope", "h1"),
        p("Implementation and reproducibility", "h2"),
        table([
            ["Repository path", "Role"],
            ["src/data; src/models; src/losses", "Chronological features and splits; daily GARCH "
             "recursion; recurrent training/reconstruction; loss definitions."],
            ["scripts/run_corrected_experiment.py; scripts/run_all_models.py", "Prepare the "
             "benchmark experiment and train the six log-target/log-ratio neural candidates."],
            ["scripts/run_additive_residual.py; scripts/evaluate_additive_residual.py", "Train "
             "the signed-residual LSTM and score raw/floored outputs on matching dates."],
            ["reports/corrected_results.json; reports/additive_results.json", "Scores, temporal "
             "intervals, diagnostics, input/code/prediction hashes, and environment records."],
            ["tests; scripts/smoke_demo.py", "Timing, reconstruction, state-recursion, bootstrap, "
             "and optional TensorFlow checks; a synthetic offline workflow demo."],
        ], [226, 278], compact=True),
        p("The README provides the full command sequence. Build this PDF with "
          "reports/build_final_report.py. Its source checks reconcile the shared benchmark "
          "scores and parent-artifact hash before combining the evidence. It does not retrain "
          "models or change the stored evaluation.", "small"),
        p("Data access and interpretation limits", "h2"),
        p("Raw market snapshots, full predictions, and training logs are not bundled. Their hashes "
          "identify the reviewed artifacts but are not substitutes for access in an independent "
          "rerun. Supplying different input bytes creates a different experiment. The offline "
          "demo uses synthetic returns; it does not reproduce these historical scores."),
        p("The study covers one equity index and one training-seed schedule. Squared returns are "
          "noisy variance proxies; repeated inspection creates research-selection risk. Temporal "
          "intervals omit seed uncertainty and multiple-comparison adjustment. VIX/gate "
          "diagnostics in the supporting artifacts are descriptive, not causal evidence. "
          "No trading P&amp;L, execution benefit, multi-asset robustness, or production validation is claimed."),
        p("Project provenance", "h2"),
        p("Based on a collaborative research project; this portfolio report is prepared by Eddie Liu.", "small"),
        p('<link href="https://github.com/SRG-eddieliu/rnn-volatility-lab">'
          "github.com/SRG-eddieliu/rnn-volatility-lab</link>", "small"),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf(destination, "Volatility Forecasting", story, 6)
    print(f"Built {destination} (6 pages; unchanged source evidence)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("volatility-final-report.pdf"))
    args = parser.parse_args()
    build(args.evidence_dir, args.output)
