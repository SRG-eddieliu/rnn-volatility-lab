"""Build a two-page research brief without changing the full report or experiment."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from build_corrected_report import INK, TEAL, GRAY, WIDTH, fonts, footer, p, table
from build_final_report import load_evidence
from build_portfolio_report import load_portfolio, build_portfolio_chart, primary_table


ROOT_URL = "https://github.com/SRG-eddieliu/rnn-volatility-lab"


def build_chart(additive, destination):
    rows = {row["model"]: row for row in additive["metrics"]}
    fig, (mse, ci) = plt.subplots(1, 2, figsize=(8.4, 2.8),
                                  gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.16, right=.96, top=.81, bottom=.25, wspace=.65)
    keys = ["garch_t", "additive_raw", "additive_clipped"]
    values = [rows[key]["mse"] / 1e-7 for key in keys]
    mse.barh(["GARCH-t", "LSTM: raw", "LSTM: floored"], values,
             color=["#707A80", TEAL, TEAL], height=.5)
    mse.invert_yaxis()
    mse.set_xlim(0, 1.98)
    mse.set_title("Observed error", loc="left", fontsize=11, fontweight="bold", pad=14)
    mse.set_xlabel("MSE / 10^-7", fontsize=9)
    mse.tick_params(labelsize=9)
    for i, value in enumerate(values):
        mse.text(value + .035, i, f"{value:.4f}", va="center", fontsize=9)
    intervals = [row for row in additive["paired_intervals"] if row["loss"] == "mse"]
    for i, row in enumerate(intervals):
        point, lo, hi = [row[key] / 1e-8 for key in
                         ("difference_vs_garch", "ci95_lower", "ci95_upper")]
        ci.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", color=TEAL,
                    capsize=4, markersize=5)
    ci.set_yticks([0, 1], ["Raw", "Floored"], fontsize=9)
    ci.set_ylim(1.6, -.6)
    ci.set_xlim(-2.8, .3)
    ci.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ci.set_title("95% temporal intervals", loc="left", fontsize=11, fontweight="bold", pad=14)
    ci.set_xlabel("MSE difference vs GARCH / 10^-8", fontsize=9)
    ci.tick_params(axis="x", labelsize=9)
    ci.text(.01, -.36, "Negative favors LSTM; both intervals include zero.",
            transform=ci.transAxes, fontsize=7.7, color=GRAY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, facecolor="white")
    plt.close(fig)


def build(directory, destination):
    protected = [directory/name for name in ("volatility-final-report.pdf",
                 "volatility-portfolio-overlay.pdf", "volatility-complete-report.pdf")]
    if destination.resolve() in {path.resolve() for path in protected}:
        raise ValueError("The brief must not overwrite a full or companion report.")
    protected_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in protected if path.exists()}
    fonts()
    parent, additive = load_evidence(directory)
    portfolio, _, portfolio_primary, portfolio_interval = load_portfolio(directory)
    portfolio_chart = directory / "figures/portfolio-sharpe-difference.png"
    build_portfolio_chart(portfolio, portfolio_chart)
    rows = {row["model"]: row for row in additive["metrics"]}
    benchmark = {row["model"]: row for row in parent["metrics"]}
    raw, floored, garch = [rows[key] for key in ("additive_raw", "additive_clipped", "garch_t")]
    interval = next(row for row in additive["paired_intervals"]
                    if row["model"] == "additive_clipped" and row["loss"] == "mse")
    if not interval["ci95_lower"] <= 0 <= interval["ci95_upper"]:
        raise ValueError("Reassess the summary: the uncertainty conclusion has changed.")
    neural = parent["metrics"][3:]
    if len(neural) != 6 or any(row["mse"] <= garch["mse"] or row["qlike"] <= garch["qlike"]
                               for row in neural):
        raise ValueError("Reassess the benchmark comparison against the changed evidence.")
    full_report = directory / "volatility-final-report.pdf"
    full_hash = hashlib.sha256(full_report.read_bytes()).hexdigest()
    chart = directory / "figures" / "research-brief-results.png"
    build_chart(additive, chart)

    title = ParagraphStyle("brief-title", fontName="BodyBold", fontSize=29, leading=34,
                           textColor=colors.HexColor(INK), spaceAfter=10)
    takeaway = ParagraphStyle("brief-takeaway", fontName="Body", fontSize=12, leading=17,
                              textColor=colors.HexColor(INK), spaceAfter=12)
    number = ParagraphStyle("brief-number", fontName="BodyBold", fontSize=24, leading=29,
                            textColor=colors.HexColor(TEAL))
    label = ParagraphStyle("brief-label", fontName="Body", fontSize=9, leading=13,
                           textColor=colors.HexColor(GRAY))
    metric_band = Table([
        [Paragraph(f"{additive['n']:,}", number), Paragraph("178", number),
         Paragraph("2010-2024", number)],
        [Paragraph("common test dates", label), Paragraph("walk-forward folds", label),
         Paragraph("historical evaluation period", label)],
    ], colWidths=[168, 168, 168])
    metric_band.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("LINEBELOW", (0, 1), (-1, 1), .6, colors.HexColor("#C9D5D7")),
    ]))

    story = [
        p("RESEARCH BRIEF / PART A: INDEX FORECASTING", "kicker"),
        Paragraph("Volatility Forecasting<br/>with Residual Learning", title),
        p("<b>Eddie Liu</b> | GARCH, LSTM, and empirical model evaluation", "small"),
        Paragraph(f"An additive LSTM reduces observed forecast MSE by <b>"
                  f"{abs(floored['mse_change_pct_vs_garch']):.2f}%</b> versus GARCH with a fixed floor. "
                  "But the uncertainty interval includes zero, and invalid raw variance forecasts "
                  "remain a material limitation.", takeaway),
        metric_band,
        p("Can an LSTM learn GARCH's forecast errors?", "h2"),
        p("The target is one-day S&amp;P 500 variance, measured with squared daily log returns."),
        table([
            ["01  Forecast the baseline", "02  Learn its errors", "03  Evaluate both outputs"],
            ["GARCH-t estimates next-day variance g using past returns.",
             "LSTM predicts the signed error e = y - g from lagged features.",
             "Raw h = g + predicted e.<br/>Floored h = max(raw h, 1e-12)."],
        ], [168, 168, 168]),
        p("The evidence at a glance", "h2"),
        Image(str(chart), width=WIDTH, height=168),
        p("Paired stationary bootstrap: mean block length 21; 2,000 resamples. Intervals "
          "condition on the trained forecasts, not training-seed or model-selection uncertainty.", "small"),
        p(f"Raw MSE improves {abs(raw['mse_change_pct_vs_garch']):.2f}%, versus "
          f"{abs(floored['mse_change_pct_vs_garch']):.2f}% when floored: the gain is not primarily a clipping effect.", "small"),
        p(f"<b>Validity:</b> {additive['raw_nonpositive_count']} raw forecasts are nonpositive. "
          f"Floored QLIKE is {floored['qlike']:,.0f}, versus {garch['qlike']:.4f} for GARCH; "
          "lower is better. Lower MSE alone is not a deployable-model result."),
        PageBreak(),
        p("PART B / SEPARATE STOCK PORTFOLIO DIAGNOSTIC", "kicker"),
        p("Dynamic Portfolio Weighting", "h1"),
        p("The legacy stock model applies an LSTM ratio correction to GARCH volatility. "
          "It is a separate model and sample from Part A, not a trading application of its "
          "additive-residual predictions."),
        p("The allocation rule", "h2"),
        p("Weight = base weight x clip(trailing vol / saved forecast, 0.5, 1.5), then normalize. "
          "Compare the tilt with the same eligible stocks and base weights without the tilt."),
        primary_table(portfolio_primary, compact=True),
        p(f"{portfolio['data']['observations']:,} return dates: {portfolio['data']['start']} to "
          f"{portfolio['data']['end']}. Monthly rebalance; 10 bps per dollar bought/sold; "
          "next-close execution. Sharpe uses a constant assumed daily risk-free return of 0.0001.", "small"),
        Image(str(portfolio_chart), width=WIDTH, height=150),
        p(f"Primary snapshot-cap Sharpe difference: {portfolio_interval['difference']:+.4f}; "
          f"95% paired temporal interval [{portfolio_interval['ci95_lower']:+.4f}, "
          f"{portfolio_interval['ci95_upper']:+.4f}]. Mean block 21, 2,000 resamples. "
          "The small gross benefit disappears under the primary cost assumption.", "small"),
        p("What the implementation adds", "h2"),
        p("Aligned holding periods, drifting weights, self-financing costs, trading-session "
          "rebalancing and explicit invalid-forecast fallback. An independent dollar-position "
          "calculation reconciles the primary portfolio results."),
        p("What the results do not establish", "h2"),
        p("The later constituent list and repeated later cap snapshot create survivorship and "
          "look-ahead bias. Equal weighting does not fix that universe. Cached stock forecasts "
          "were not retrained. These are retrospective accounting diagnostics, not validated "
          "alpha; Part A's MSE change cannot explain Part B's Sharpe.", "small"),
        p(f'<b><link href="{ROOT_URL}/blob/main/reports/volatility-complete-report.pdf">Full research report (8 pages)</link>'
          f' | <link href="{ROOT_URL}">Code and evidence</link></b>', "small"),
        p("Sources: reports/additive_results.json and reports/portfolio_overlay_results.json; "
          "full raw data are not bundled. Based on collaborative research; brief prepared by Eddie Liu.", "small"),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination), pagesize=(612, 792), leftMargin=54, rightMargin=54,
        topMargin=40, bottomMargin=59, title="Volatility Research - Forecasts and Portfolio Overlays",
        author="Eddie Liu", subject="Two separate experiments: index forecasts and stock portfolio diagnostics",
        creator="Eddie Liu - research portfolio",
    )
    document.build(story, onFirstPage=footer("Research Brief"), onLaterPages=footer("Research Brief"))
    count = len(PdfReader(destination).pages)
    if count != 2:
        raise ValueError(f"Expected two pages; got {count}.")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, digest in protected_hashes.items()):
        raise ValueError("A full or companion report changed unexpectedly.")
    print(f"Built {destination} (2 pages); full report unchanged: {full_hash}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("volatility-research-brief.pdf"))
    args = parser.parse_args()
    build(args.evidence_dir, args.output)
