"""Render completed retraining evidence; preserve the original forecast report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
from reportlab.platypus import Image, PageBreak

from build_corrected_report import WIDTH, TEAL, GRAY, fonts, p, pdf, table

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/SRG-eddieliu/rnn-volatility-lab"
LABELS = {"cap_base": "Snapshot-cap base", "cap_garch": "Snapshot-cap + GARCH",
          "cap_hybrid": "Snapshot-cap + GARCH/LSTM", "equal_base": "Equal-weight base",
          "equal_garch": "Equal-weight + GARCH", "equal_hybrid": "Equal-weight + GARCH/LSTM"}


def load(directory):
    data = json.loads((directory/"stock_retraining_results.json").read_text())
    if data["training_audit"]["stocks"] != 503:
        raise ValueError("Report requires the completed full-universe experiment.")
    signatures = [data["training_signature"]["code_sha256"], data["evaluation_code_sha256"]]
    for signature in signatures:
        for name, expected in signature.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evidence/code mismatch: {name}")
    metrics = {row["id"]: row for row in data["portfolio_metrics"]}
    primary = {name: metrics[f"{name}_monthly_10bps"] for name in LABELS}
    intervals = {row["base"]: row for row in data["paired_sharpe_intervals"] if row["block_length"] == 21}
    return data, metrics, primary, intervals


def describe(interval):
    if interval["ci95_lower"] > 0:
        return "The conditional interval is above zero."
    if interval["ci95_upper"] < 0:
        return "The conditional interval is below zero."
    return "The conditional interval includes zero."


def plot_intervals(intervals, destination):
    keys = ["cap_base", "cap_garch", "equal_base", "equal_garch"]
    fig, ax = plt.subplots(figsize=(8.4, 2.7))
    fig.subplots_adjust(left=.24, right=.97, top=.79, bottom=.26)
    lower, upper = 0., 0.
    for i, key in enumerate(keys):
        row = intervals[key]
        point, low, high = [row[k] for k in ("difference", "ci95_lower", "ci95_upper")]
        ax.hlines(i, low, high, color=TEAL)
        ax.vlines([low, high], i-.09, i+.09, color=TEAL)
        ax.plot(point, i, "o", color=TEAL)
        lower, upper = min(lower, low), max(upper, high)
    padding = max((upper-lower)*.12, .003)
    ax.set_xlim(lower-padding, upper+padding)
    ax.set_yticks(range(4), ["Cap: no tilt", "Cap: GARCH", "Equal: no tilt", "Equal: GARCH"])
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_title("Hybrid minus comparator: net Sharpe", loc="left", fontsize=12, pad=12)
    ax.set_xlabel("95% paired temporal intervals; positive favors hybrid", fontsize=9)
    ax.tick_params(labelsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, facecolor="white")
    plt.close(fig)


def build(directory):
    fonts()
    data, metrics, primary, intervals = load(directory)
    audit, fm = data["training_audit"], data["forecast_metrics"]
    source_report = directory/"volatility-final-report.pdf"
    old_cached = directory/"volatility-portfolio-overlay.pdf"
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source_report, old_cached)}
    brief = directory/"volatility-research-brief.pdf"
    brief_source = PdfReader(brief)
    if len(brief_source.pages) != 2:
        raise ValueError("Expected an existing two-page index/stock brief.")
    chart = directory/"figures/stock-retrained-sharpe.png"
    plot_intervals(intervals, chart)
    volatility = fm["volatility_mse"]
    loss_interval = fm["paired_loss_intervals"]["hybrid_vs_garch"]
    mse_change = (volatility["hybrid_raw"]["date_equal_mse"]/volatility["garch"]["date_equal_mse"]-1)*100
    comparison = intervals["cap_garch"]
    rows = [["Forecast / trailing 21-day target", "Date-equal MSE", "MSE / GARCH"]]
    for key, label in (("garch", "GARCH with recorded fit fallback"),
                       ("hybrid_raw", "GARCH x LSTM ratio (raw)"), ("persistence", "Previous trailing volatility")):
        value = volatility[key]["date_equal_mse"]
        rows.append([label, f"{value:.4e}", f"{value/volatility['garch']['date_equal_mse']:.3f}x"])
    story = [
        p("PART B / RETRAINED STOCK OVERLAY", "kicker"),
        p("Stock-Level GARCH + LSTM:<br/>Retraining and Economic Value", "title"),
        p("<b>Eddie Liu</b> | Reproducible retrospective experiment | September 2026", "small"),
        p("Completed run; forecast robustness failed", "h2"),
        p(f"Processed <b>{audit['stocks']} stocks</b> and saved <b>{audit['lstm_fits']:,} LSTM fits</b> "
          "for forecasts dated 2015-07-01 through 2024-12-31. GARCH was also refitted from "
          "the raw return history; the earlier cached hybrid forecasts were not reused."),
        p("This is the separate stock-level multiplicative volatility model, not the index "
          "additive variance-residual LSTM in Part A. The studies are not one shared model.", "small"),
        p("What was trained", "h2"),
        p("Daily GARCH(1,1)-t on 50 prior returns. An 8-unit LSTM predicts the ratio of "
          "trailing 21-day realized volatility to GARCH sigma using 10 past ratios. "
          "Expanding-history fitting uses 10 epochs, batch 8, and refits every 600 eligible "
          "forecast dates. Scaling and training labels stop strictly before each fit block."),
        p("Forecast quality on common observations", "h2"),
        table(rows, [290, 115, 99], compact=True),
        p(f"{fm['common_asset_dates']:,} common stock/date observations across "
          f"{fm['common_dates']:,} dates and {fm['assets_with_common_forecasts']} stocks. "
          "Daily losses average stocks first, then dates equally; lower MSE is better.", "small"),
        p(f"Hybrid MSE change versus GARCH: <b>{mse_change:+.2f}%</b> (worse). "
          f"Hybrid-minus-GARCH loss interval: [{loss_interval['ci95_lower']:.3e}, "
          f"{loss_interval['ci95_upper']:.3e}]. {describe(loss_interval)}"),
        p(f"<b>Failure concentration:</b> {fm['assets_with_lower_hybrid_mse']}/"
          f"{fm['assets_with_common_forecasts']} stocks improve individually, yet "
          f"{fm['failure_diagnostics']['dominant_error_stock']} contributes "
          f"{100*fm['failure_diagnostics']['dominant_share_of_date_equal_hybrid_mse']:.2f}% "
          "of hybrid MSE. Near-zero GARCH denominators create extreme ratio targets. "
          "Independent checkpoint replay reproduces the failure; no stocks were removed.", "small"),
        p("<b>Target caveat:</b> consecutive 21-day volatility windows overlap. The persistence "
          "control tests this mechanical predictability; lower trailing-volatility error "
          "is not automatically better fresh future-risk prediction or trading performance.", "small"),
        p("Audit trail and limitations", "h2"),
        p(f"Saved checkpoints, scalers, seeds, fit cutoffs and source hashes. "
          f"{fm['nonpositive_hybrid_asset_dates']:,} nonpositive raw hybrid outputs; "
          f"{audit['stocks_without_trained_model']} stock(s) have insufficient history for a neural fit. "
          f"GARCH uses recorded past-volatility fallback on "
          f"{audit['garch_evaluation_status_counts'].get('past_vol_fallback', 0):,} evaluation asset-days.", "small"),
        p("Later constituents/cap weights remain biased; one seed per fit and no untouched "
          "holdout. Reproducible model provenance does not establish robust forecasts or alpha.", "small"),
        PageBreak(),
        p("PART B / STOCK PORTFOLIO RESULTS", "kicker"),
        p("Does the LSTM Add Trading Value?", "h1"),
        p("Weight = base weight x clip(trailing volatility / forecast, 0.5, 1.5), normalized. "
          "Compare the hybrid with both an untilted allocation and a GARCH-only tilt."),
    ]
    portfolio_rows = [["Monthly, 10 bps", "Net Sharpe", "Max drawdown"]]
    for key, row in primary.items():
        portfolio_rows.append([LABELS[key], f"{row['sharpe']:.3f}", f"{row['max_drawdown']*100:.2f}%"])
    story += [
        table(portfolio_rows, [290, 107, 107], compact=True),
        p("2,390 common return dates, 2015-07-06 to 2024-12-31. Next-close execution; "
          "10 bps per dollar bought/sold; drifting holdings and self-financing costs. "
          "Sharpe assumes daily risk-free return 0.0001.", "small"),
        Image(str(chart), width=WIDTH, height=144),
        p(f"Snapshot-cap hybrid minus matched GARCH Sharpe: {comparison['difference']:+.4f}; "
          f"95% interval [{comparison['ci95_lower']:+.4f}, {comparison['ci95_upper']:+.4f}]. "
          f"{describe(comparison)} Mean bootstrap block 21 days, 2,000 paired resamples.", "small"),
        p("Matched controls and implementation", "h2"),
        p("Both tilts use the same stock pool and model-availability dates. Unavailable or "
          "nonpositive forecasts receive neutral tilts. Unrestricted GARCH and all predefined "
          "frequency/cost sensitivities remain in the aggregate results.", "small"),
        p("What this does not establish", "h2"),
        p(f"<b>Forecast robustness failed:</b> {fm['nonpositive_hybrid_asset_dates']} nonpositive "
          "outputs and AMCR-driven extreme errors. Capping portfolio tilts does not repair "
          "the underlying forecast model.", "small"),
        p("Later constituents and repeated later market caps create survivorship and look-ahead "
          "bias. Equal weights do not fix the universe. These conditional comparisons are "
          "not validated alpha, achievable-return estimates or a causal proof of LSTM value.", "small"),
        p(f'<b><link href="{URL}/blob/main/reports/volatility-complete-report.pdf">Full report (8 pages)</link>'
          f' | <link href="{URL}">Code, all sensitivities and source hashes</link></b>', "small"),
        p("Source: reports/stock_retraining_results.json. Raw vendor data and per-stock model "
          "artifacts remain local. Based on collaborative research; report prepared by Eddie Liu.", "small"),
    ]
    stock_report = directory/"stock-retraining-report.pdf"
    pdf(stock_report, "Stock Retraining", story, 2)
    combined = PdfWriter()
    combined.append(source_report, outline_item="Part A: Index Forecasting")
    combined.append(stock_report, outline_item="Part B: Retrained Stock Overlay")
    combined.add_metadata({"/Title": "Volatility Research: Index Forecasts and Retrained Stock Overlays", "/Author": "Eddie Liu"})
    with (directory/"volatility-complete-report.pdf").open("wb") as stream:
        combined.write(stream)
    short = PdfWriter()
    short.add_page(brief_source.pages[0])
    short.add_page(PdfReader(stock_report).pages[1])
    short.add_metadata({"/Title": "Volatility Research Brief: Index Forecasts and Retrained Stock Portfolios", "/Author": "Eddie Liu"})
    with brief.open("wb") as stream:
        short.write(stream)
    for path, expected in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Historical standalone report changed unexpectedly.")
    if len(PdfReader(directory/"volatility-complete-report.pdf").pages) != 8:
        raise ValueError("Expected eight-page complete report.")
    print("Built two-page retraining report, eight-page complete report, and two-page research brief.")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
