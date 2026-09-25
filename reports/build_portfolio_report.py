"""Render the portfolio diagnostic and append it to the unchanged forecast study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
from reportlab.platypus import Image, PageBreak

from build_corrected_report import WIDTH, TEAL, GRAY, fonts, p, pdf, table

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"snapshot_cap": "Snapshot-cap base", "hybrid_cap": "Snapshot-cap + overlay",
          "equal_weight": "Equal-weight base", "hybrid_equal": "Equal-weight + overlay"}


def load_portfolio(directory):
    data = json.loads((directory / "portfolio_overlay_results.json").read_text())
    for path, expected in data["code_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Portfolio code/protocol changed after scoring: {path}")
    metrics = {row["id"]: row for row in data["metrics"]}
    primary = {name: metrics[f"{name}_monthly_10bps_lag1"] for name in LABELS}
    interval = next(row for row in data["paired_sharpe_intervals"]
                    if row["baseline"] == "snapshot_cap" and row["block_length"] == 21)
    if not interval["ci95_lower"] <= 0 <= interval["ci95_upper"]:
        raise ValueError("Reassess the summary against the changed evidence.")
    return data, metrics, primary, interval


def build_portfolio_chart(data, destination):
    rows = [row for row in data["paired_sharpe_intervals"] if row["block_length"] == 21]
    fig, ax = plt.subplots(figsize=(8.4, 2.5))
    fig.subplots_adjust(left=.25, right=.93, top=.76, bottom=.29)
    for i, row in enumerate(rows):
        point, lower, upper = [row[key] for key in ("difference", "ci95_lower", "ci95_upper")]
        ax.errorbar(point, i, xerr=[[point-lower], [upper-point]], fmt="o", color=TEAL,
                    markersize=6, capsize=5)
    ax.set_yticks([0, 1], ["Snapshot-cap base", "Equal-weight base"])
    ax.set_ylim(1.6, -.6)
    ax.set_xlim(-.036, .026)
    ax.axvline(0, linestyle="--", color=GRAY, linewidth=1)
    ax.set_xlabel("Overlay minus matched-base net Sharpe (positive favors overlay)", fontsize=9)
    ax.set_title("No resolved incremental Sharpe improvement", fontsize=13, loc="left", pad=15)
    ax.tick_params(labelsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, facecolor="white")
    plt.close(fig)


def primary_table(primary, compact=False):
    if compact:
        rows = [["Monthly, 10 bps", "Net Sharpe", "Max drawdown"]]
        for name, row in primary.items():
            rows.append([LABELS[name], f"{row['sharpe']:.3f}", f"{100*row['max_drawdown']:.2f}%"])
        return table(rows, [270, 117, 117], compact=True)
    rows = [["Monthly, 10 bps", "Net CAGR", "Net Sharpe", "Max DD"]]
    for name, row in primary.items():
        rows.append([LABELS[name], f"{100*row['cagr']:.2f}%", f"{row['sharpe']:.3f}",
                     f"{100*row['max_drawdown']:.2f}%"])
    return table(rows, [237, 89, 89, 89], compact=True)


def build(directory):
    fonts()
    data, metrics, primary, interval = load_portfolio(directory)
    base, overlay = primary["snapshot_cap"], primary["hybrid_cap"]
    chart = directory/"figures/portfolio-sharpe-difference.png"
    build_portfolio_chart(data, chart)
    story = [
        p("PART B / PORTFOLIO ACCOUNTING DIAGNOSTIC", "kicker"),
        p("Dynamic Volatility Weighting:<br/>Economic Value and Controls", "title"),
        p("<b>Eddie Liu</b> | Separate legacy stock-model experiment", "small"),
        p("Research finding", "h2"),
        p(f"The monthly overlay has net Sharpe <b>{overlay['sharpe']:.3f}</b> versus "
          f"<b>{base['sharpe']:.3f}</b> for the same snapshot-cap base without the tilt. "
          f"The difference is {interval['difference']:+.4f}, with a paired 95% temporal interval "
          f"[{interval['ci95_lower']:+.4f}, {interval['ci95_upper']:+.4f}]. "
          "This diagnostic does not establish incremental Sharpe improvement."),
        p("<b>Important:</b> later surviving constituents and a later cap snapshot are used "
          "throughout history. These are conditional accounting results, not point-in-time alpha "
          "or achievable-return estimates. Cached forecast training was not rerun.", "small"),
        primary_table(primary),
        p(f"Table 1. {data['data']['observations']:,} common return dates, "
          f"{data['data']['start']} to {data['data']['end']}. Next-close execution, monthly "
          "rebalance, 10 bps per dollar bought/sold. CAGR is geometric; Sharpe subtracts "
          "a constant assumed daily risk-free return of 0.0001. Final positions are not liquidated.", "small"),
        Image(str(chart), width=WIDTH, height=150),
        p("Paired stationary-bootstrap intervals: mean block 21, 2,000 resamples. Both include "
          "zero. They condition on the cached forecasts and biased sample and do not repair source bias.", "small"),
        p("What is being tested", "h2"),
        p("The legacy stock model multiplies GARCH volatility by an LSTM ratio correction. "
          "Weights are proportional to base weights times clipped trailing-volatility / saved-forecast "
          "ratios (0.5-1.5). It is not the index additive-residual LSTM in Part A; the two studies "
          "must not be interpreted as one prediction-to-P&amp;L experiment."),
        PageBreak(),
        p("Portfolio Mechanics and Sensitivities", "h1"),
        p("Accounting and information timing", "h2"),
        p("Signals formed after close d execute at close d+1, earning the return ending d+2. "
          "Weights drift between trades. The engine compounds weighted simple stock returns, "
          "charges actual buy/sell notional, and solves post-cost allocations self-consistently. "
          "Rebalances use actual observed trading sessions, not calendar-day filters. Missing "
          "held-asset returns fail rather than silently become zero."),
        p("Net Sharpe by rebalance frequency", "h2"),
    ]
    rows = [["10 bps, one-close lag", "Matched cap base", "Cap overlay", "Difference"]]
    for frequency in ("daily", "weekly", "monthly", "yearly"):
        b, h = [metrics[f"{name}_{frequency}_10bps_lag1"] for name in ("snapshot_cap", "hybrid_cap")]
        rows.append([frequency.capitalize(), f"{b['sharpe']:.3f}", f"{h['sharpe']:.3f}",
                     f"{h['sharpe']-b['sharpe']:+.4f}"])
    story += [table(rows, [165, 113, 113, 113], compact=True),
              p("Monthly was fixed as primary. The small positive yearly point estimate is not "
                "selected as the headline outcome; frequency sensitivity is not a fresh holdout test.", "small"),
              p("Net Sharpe by transaction-cost assumption", "h2")]
    cost_rows = [["Monthly cost / traded $", "Matched cap base", "Cap overlay", "Difference"]]
    for cost in (0, 10, 25):
        b, h = [metrics[f"{name}_monthly_{cost}bps_lag1"] for name in ("snapshot_cap", "hybrid_cap")]
        cost_rows.append([f"{cost} bps", f"{b['sharpe']:.3f}", f"{h['sharpe']:.3f}",
                          f"{h['sharpe']-b['sharpe']:+.4f}"])
    story += [
        table(cost_rows, [165, 113, 113, 113], compact=True),
        p(f"Annualized traded notional is {base['annualized_traded_notional']:.2f}x wealth "
          f"for the base and {overlay['annualized_traded_notional']:.2f}x for the overlay. "
          "The small gross Sharpe advantage disappears under the primary cost assumption. "
          "The idealized same-close sensitivity also does not improve monthly net Sharpe.", "small"),
        p("Data quality, scope, and reproducibility", "h2"),
        p(f"The panel has {data['data']['assets']} stock columns and one repeated cap-weight row. "
          f"{data['data']['neutral_fallback_asset_days']:,} eligible asset-days receive neutral "
          f"tilts, including {data['data']['nonpositive_forecast_asset_days']} nonpositive "
          "forecasts. Historical membership, delistings, contemporaneous caps, and independent "
          "per-fit forecast records remain unavailable. Equal weighting does not remove survivor bias.", "small"),
        p("The ^GSPC price index is context only: it is not a matched total-return comparator for "
          "adjusted-close stock portfolios. A same-universe GARCH-only panel is also missing, so "
          "these comparisons cannot isolate an LSTM contribution. No execution-capacity test is claimed.", "small"),
        p("Code: src/evaluation/portfolio.py; runner: scripts/evaluate_portfolio_overlay.py; "
          "tests: tests/test_portfolio.py; protocol: docs/portfolio-overlay-protocol.md. "
          "reports/portfolio_overlay_results.json stores scores, sensitivities, hashes and monthly "
          "portfolio aggregates. Raw inputs and daily position-level data are not distributed.", "small"),
        p("Based on a collaborative research project; portfolio report prepared by Eddie Liu.", "small"),
    ]
    appendix = directory/"volatility-portfolio-overlay.pdf"
    pdf(appendix, "Portfolio Overlay Diagnostic", story, 2)
    full = directory/"volatility-final-report.pdf"
    before = hashlib.sha256(full.read_bytes()).hexdigest()
    writer = PdfWriter()
    writer.append(full, outline_item="Part A: Index Forecasting Study")
    writer.append(appendix, outline_item="Part B: Separate Portfolio Diagnostic")
    writer.add_metadata({"/Title": "Volatility Research: Forecasting and Portfolio Overlays", "/Author": "Eddie Liu"})
    combined = directory/"volatility-complete-report.pdf"
    with combined.open("wb") as stream:
        writer.write(stream)
    if len(PdfReader(combined).pages) != 8 or hashlib.sha256(full.read_bytes()).hexdigest() != before:
        raise ValueError("Combined report or original report preservation failed.")
    print(f"Created {appendix} (2 pages) and {combined} (8 pages); six-page original unchanged.")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
