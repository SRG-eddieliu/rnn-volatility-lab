"""Render the README PNG directly from committed aggregate evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "additive_results.json").read_text())
    metrics = {row["model"]: row for row in data["metrics"]}
    keys = ["garch_t", "additive_raw", "additive_clipped"]
    labels = ["GARCH-t", "Additive / raw", "Additive / floored"]
    colors = ["#60666d", "#247b91", "#247b91"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, (ax, ci) = plt.subplots(1,2, figsize=(12,5.4), gridspec_kw={"width_ratios":[1,1.1]})
    fig.subplots_adjust(left=.15, right=.96, top=.73, bottom=.31, wspace=.6)
    fig.suptitle("Additive-residual LSTM: error and uncertainty", x=.04, y=.96,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(.04,.875, f"{data['n']:,} common test dates | {data['test_start']} to {data['test_end']} | one training-seed schedule", color="#555555")
    values = [metrics[key]["mse"]/1e-7 for key in keys]
    ax.barh(labels, values, color=colors, height=.55)
    ax.invert_yaxis()
    ax.set_xlim(0,1.95)
    ax.set_xlabel("MSE / 1e-7 (lower is better)")
    for i, value in enumerate(values): ax.text(value+.025, i, f"{value:.4f}", va="center")
    rows = [row for row in data["paired_intervals"] if row["loss"]=="mse"]
    for i,row in enumerate(rows):
        point = row["difference_vs_garch"] / 1e-8
        lo, hi = row["ci95_lower"]/1e-8, row["ci95_upper"]/1e-8
        ci.errorbar(point, i, xerr=[[point-lo],[hi-point]], fmt="o", capsize=5,
                    color="#247b91", markersize=7)
    ci.set_yticks([0,1], ["Raw", "Floored"])
    ci.set_ylim(1.6,-.6)
    ci.axvline(0,color="#666666",linestyle="--",linewidth=1)
    ci.set_xlim(-2.8,.3)
    ci.set_xlabel("MSE difference vs GARCH / 1e-8")
    ci.set_title("Paired temporal 95% intervals", fontsize=12)
    floored=metrics["additive_clipped"]
    fig.text(.04,.18, f"Observed MSE change: {floored['mse_change_pct_vs_garch']:.2f}% | both intervals include zero", fontsize=12, fontweight="bold")
    fig.text(.04,.12, f"{data['raw_nonpositive_count']} nonpositive raw forecasts; floored QLIKE {floored['qlike']:,.0f} vs GARCH {metrics['garch_t']['qlike']:.4f}.", color="#555555")
    fig.text(.04,.065, "Historical follow-up, not an untouched holdout. Intervals condition on trained forecasts; no seed or selection adjustment.", fontsize=9, color="#555555")
    fig.text(.04,.025, "Source: reports/additive_results.json | paired stationary bootstrap, mean block 21, 2,000 resamples", fontsize=9, color="#555555")
    destination = root / "figures" / "additive-summary.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)
    return destination


if __name__ == "__main__":
    print(build())
