from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "reports" / "predictions"
OUT_PDF = ROOT / "reports" / "research_report_with_plots.pdf"
PAGE_SIZE = (13.5, 8.0)
EPS = 1e-12


def pretty_model_name(variant: str, architecture: str) -> str:
    if variant == "baseline":
        return "Baseline GARCH-t"
    clean_variant = str(variant).replace("_", " ").title()
    return f"{clean_variant} {str(architecture).upper()}"


def model_label(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [pretty_model_name(v, a) for v, a in zip(df["variant"], df["architecture"])],
        index=df.index,
    )


def safe_read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def page_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, fontsize=17, y=0.985, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10, color="#333333")


def save_page(pdf: PdfPages, fig: plt.Figure, tight: bool = True) -> None:
    if tight:
        fig.tight_layout(rect=[0.03, 0.05, 0.98, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, lines: list[str], subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    page_title(fig, title, subtitle)
    y = 0.88
    for line in lines:
        fig.text(0.055, y, line, fontsize=11, va="top")
        y -= 0.045
    save_page(pdf, fig, tight=False)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, subtitle: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    page_title(fig, title, subtitle)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    save_page(pdf, fig)


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_model_daily(all_pred: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    model_daily = (
        all_pred.groupby(["date", "model"], as_index=False)
        .agg(y_true_var=("y_true_var", "mean"), y_pred_var=("y_pred_var", "mean"))
        .sort_values(["model", "date"])
        .reset_index(drop=True)
    )

    model_daily["sq_error"] = (model_daily["y_true_var"] - model_daily["y_pred_var"]) ** 2
    model_daily["rolling_mse_63d"] = model_daily.groupby("model")["sq_error"].transform(
        lambda s: s.rolling(63, min_periods=63).mean()
    )

    y_true = model_daily["y_true_var"].clip(lower=EPS)
    y_pred = model_daily["y_pred_var"].clip(lower=EPS)
    ratio = y_true / y_pred
    model_daily["qlike_daily"] = ratio - np.log(ratio) - 1.0
    model_daily["rolling_qlike_63d"] = model_daily.groupby("model")["qlike_daily"].transform(
        lambda s: s.rolling(63, min_periods=63).mean()
    )

    model_daily["realized_vol_21d"] = np.sqrt(
        model_daily.groupby("model")["y_true_var"].transform(lambda s: s.rolling(21, min_periods=21).mean())
        * 252.0
    )
    model_daily["forecast_vol_21d"] = np.sqrt(
        model_daily.groupby("model")["y_pred_var"].transform(lambda s: s.rolling(21, min_periods=21).mean())
        * 252.0
    )

    if not vix_df.empty:
        out = model_daily.merge(vix_df[["date", "vix_close"]], on="date", how="left")
        out["vix_implied_var"] = (out["vix_close"] / 100.0) ** 2 / 252.0
        return out
    return model_daily


def collect_train_logs() -> pd.DataFrame:
    log_rows = []
    required = {
        "split_id",
        "variant",
        "architecture",
        "best_train_loss",
        "best_val_loss",
        "best_gap_val_minus_train",
    }
    for path in sorted(PRED_DIR.glob("*_train_logs.csv")):
        try:
            d = safe_read(path)
        except Exception:
            continue
        if not required.issubset(set(d.columns)):
            continue
        d["model"] = model_label(d[["variant", "architecture"]])
        d["source_file"] = path.name
        log_rows.append(d)
    if not log_rows:
        return pd.DataFrame()
    return pd.concat(log_rows, ignore_index=True)


def collect_gate_raw() -> pd.DataFrame:
    req = {"date", "variant", "architecture", "lag", "gate_name", "gate_value_mean"}
    rows = []
    for path in sorted(PRED_DIR.glob("*_gate_values.csv")):
        try:
            d = safe_read(path, parse_dates=["date"])
        except Exception:
            continue
        if not req.issubset(set(d.columns)):
            continue
        rows.append(d[list(req)].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def add_metric_pages(pdf: PdfPages, eval_df: pd.DataFrame) -> None:
    d = eval_df.copy().sort_values("mse")

    fig, axes = plt.subplots(1, 2, figsize=PAGE_SIZE)
    page_title(fig, "Model Performance: MSE and QLIKE", "Lower is better for both metrics")

    sns.barplot(data=d, x="mse", y="model", ax=axes[0], color="#4C78A8")
    axes[0].set_title("MSE by Model")
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("")

    sns.barplot(data=d, x="qlike", y="model", ax=axes[1], color="#F58518")
    axes[1].set_title("QLIKE by Model")
    axes[1].set_xlabel("QLIKE")
    axes[1].set_ylabel("")

    save_page(pdf, fig)

    no_resid = d[d["variant"] != "hybrid_residual"].copy()
    if not no_resid.empty:
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        page_title(fig, "QLIKE by Model (Residual-Hybrid Excluded)")
        sns.barplot(data=no_resid.sort_values("qlike"), x="qlike", y="model", ax=ax, color="#2A9D8F")
        ax.set_xlabel("QLIKE")
        ax.set_ylabel("")
        save_page(pdf, fig)


def add_forecast_pages(pdf: PdfPages, model_daily: pd.DataFrame, models: list[str]) -> None:
    for model_name in models:
        d = model_daily[model_daily["model"] == model_name].sort_values("date")
        if d.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=PAGE_SIZE, sharex=True)
        page_title(fig, f"{model_name}: Forecast vs Realized")

        axes[0].plot(d["date"], d["y_true_var"], color="#666666", lw=1.0, label="Realized Var")
        axes[0].plot(d["date"], d["y_pred_var"], color="#1f77b4", lw=1.0, label="Forecast Var")
        axes[0].set_ylabel("Variance")
        axes[0].legend(loc="upper right", frameon=False)
        axes[0].grid(alpha=0.2)

        dv = d.dropna(subset=["realized_vol_21d", "forecast_vol_21d"])
        axes[1].plot(dv["date"], dv["realized_vol_21d"], color="#666666", lw=1.0, label="Realized Vol 21d")
        axes[1].plot(dv["date"], dv["forecast_vol_21d"], color="#1f77b4", lw=1.0, label="Forecast Vol 21d")
        axes[1].set_ylabel("Annualized Vol")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="upper right", frameon=False)
        axes[1].grid(alpha=0.2)

        save_page(pdf, fig)


def add_vix_diagnostic_pages(pdf: PdfPages, model_daily: pd.DataFrame, models: list[str]) -> None:
    if "vix_close" not in model_daily.columns:
        return

    for model_name in models:
        d = model_daily[model_daily["model"] == model_name].sort_values("date")
        if d.empty or d["vix_close"].notna().sum() == 0:
            continue

        fig, axes = plt.subplots(3, 1, figsize=PAGE_SIZE, sharex=True)
        page_title(fig, f"{model_name}: VIX and Forecast/Error Diagnostics")

        axes[0].plot(d["date"], d["y_pred_var"], color="#1f77b4", lw=1.0, label="Forecast Var")
        ax0b = axes[0].twinx()
        ax0b.plot(d["date"], d["vix_close"], color="#E76F51", lw=0.9, alpha=0.5, label="VIX")
        axes[0].set_ylabel("Forecast Var")
        ax0b.set_ylabel("VIX")
        axes[0].grid(alpha=0.2)

        dm = d.dropna(subset=["rolling_mse_63d"])
        axes[1].plot(dm["date"], dm["rolling_mse_63d"], color="#2A9D8F", lw=1.0, label="Rolling MSE (63d)")
        ax1b = axes[1].twinx()
        ax1b.plot(dm["date"], dm["vix_close"], color="#E76F51", lw=0.9, alpha=0.5, label="VIX")
        axes[1].set_ylabel("Rolling MSE")
        ax1b.set_ylabel("VIX")
        axes[1].grid(alpha=0.2)

        dq = d.dropna(subset=["rolling_qlike_63d"])
        axes[2].plot(dq["date"], dq["rolling_qlike_63d"], color="#264653", lw=1.0, label="Rolling QLIKE (63d)")
        ax2b = axes[2].twinx()
        ax2b.plot(dq["date"], dq["vix_close"], color="#E76F51", lw=0.9, alpha=0.5, label="VIX")
        axes[2].set_ylabel("Rolling QLIKE")
        ax2b.set_ylabel("VIX")
        axes[2].set_xlabel("Date")
        axes[2].grid(alpha=0.2)

        save_page(pdf, fig)


def add_calibration_pages(pdf: PdfPages, model_daily: pd.DataFrame, models: list[str]) -> None:
    if "vix_implied_var" not in model_daily.columns:
        return

    model_to_cal = {}
    for model_name in models:
        d = model_daily[model_daily["model"] == model_name].dropna(subset=["vix_implied_var"]).copy()
        if len(d) < 40:
            continue
        try:
            d["rv_bin"] = pd.qcut(d["y_true_var"], q=10, duplicates="drop")
        except ValueError:
            continue
        cal = (
            d.groupby("rv_bin", observed=True)
            .agg(
                mean_true=("y_true_var", "mean"),
                mean_model=("y_pred_var", "mean"),
                mean_vix=("vix_implied_var", "mean"),
            )
            .dropna()
            .reset_index(drop=True)
        )
        if len(cal) >= 3:
            model_to_cal[model_name] = cal

    if not model_to_cal:
        return

    names = list(model_to_cal.keys())
    for block in chunked(names, 4):
        fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE, sharex=False, sharey=False)
        page_title(fig, "Calibration vs Realized Variance", "Model forecasts and VIX-implied variance by realized-variance decile")
        axes = np.array(axes).reshape(-1)

        for ax, model_name in zip(axes, block):
            cal = model_to_cal[model_name]
            ax.plot(cal["mean_true"], cal["mean_model"], marker="o", lw=1.4, label="Model")
            ax.plot(cal["mean_true"], cal["mean_vix"], marker="s", lw=1.2, label="VIX Implied")
            low = min(cal["mean_true"].min(), cal["mean_model"].min(), cal["mean_vix"].min())
            high = max(cal["mean_true"].max(), cal["mean_model"].max(), cal["mean_vix"].max())
            ax.plot([low, high], [low, high], ls="--", color="gray", lw=1.0, label="Ideal")
            ax.set_title(model_name, fontsize=10)
            ax.set_xlabel("Mean Realized Var")
            ax.set_ylabel("Mean Forecast Var")
            ax.grid(alpha=0.2)

        for ax in axes[len(block) :]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        save_page(pdf, fig)


def add_rolling_mse_vs_realized_pages(pdf: PdfPages, model_daily: pd.DataFrame, models: list[str]) -> None:
    for block in chunked(models, 4):
        fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE, sharex=True)
        page_title(fig, "Rolling 63D MSE vs Realized Volatility")
        axes = np.array(axes).reshape(-1)

        for ax, model_name in zip(axes, block):
            d = model_daily[model_daily["model"] == model_name].dropna(subset=["rolling_mse_63d", "realized_vol_21d"]) 
            if d.empty:
                ax.axis("off")
                continue
            ax.plot(d["date"], d["rolling_mse_63d"], color="#2A9D8F", lw=1.0)
            ax.set_title(model_name, fontsize=10)
            ax.set_ylabel("Rolling MSE")
            ax.grid(alpha=0.2)
            ax2 = ax.twinx()
            ax2.plot(d["date"], d["realized_vol_21d"], color="#6D6875", lw=0.9, alpha=0.7)
            ax2.set_ylabel("Realized Vol 21d")

        for ax in axes[len(block) :]:
            ax.axis("off")

        save_page(pdf, fig)


def add_overfit_pages(pdf: PdfPages, logs: pd.DataFrame) -> None:
    if logs.empty:
        return

    summary = (
        logs.groupby("model", as_index=False)
        .agg(
            best_gap_mean=("best_gap_val_minus_train", "mean"),
            best_gap_p95=("best_gap_val_minus_train", lambda s: s.quantile(0.95)),
            best_val_mean=("best_val_loss", "mean"),
        )
        .sort_values("best_gap_mean")
    )

    fig, axes = plt.subplots(1, 2, figsize=PAGE_SIZE)
    page_title(fig, "Overfit Diagnostics (Best Epoch Across Rolling Splits)")

    sns.barplot(data=summary, x="best_gap_mean", y="model", ax=axes[0], color="#54A24B")
    axes[0].axvline(0.0, color="gray", ls="--", lw=1)
    axes[0].set_title("Mean Gap (Val - Train)")
    axes[0].set_xlabel("Gap")
    axes[0].set_ylabel("")

    sns.barplot(data=summary, x="best_val_mean", y="model", ax=axes[1], color="#B279A2")
    axes[1].set_title("Mean Best Validation Loss")
    axes[1].set_xlabel("Best Val Loss")
    axes[1].set_ylabel("")

    save_page(pdf, fig)

    keep = logs[["model", "split_id", "best_train_loss", "best_val_loss"]].dropna().copy()
    keep["split_id"] = keep["split_id"].astype(int)
    model_names = sorted(keep["model"].unique())

    for block in chunked(model_names, 4):
        fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE, sharex=True)
        page_title(fig, "Best Loss by Rolling Split")
        axes = np.array(axes).reshape(-1)

        for ax, model_name in zip(axes, block):
            d = keep[keep["model"] == model_name].sort_values("split_id")
            if d.empty:
                ax.axis("off")
                continue
            ax.plot(d["split_id"], d["best_train_loss"], lw=1.0, color="#1f77b4", label="Best Train")
            ax.plot(d["split_id"], d["best_val_loss"], lw=1.0, color="#ff7f0e", label="Best Val")
            ax.set_title(model_name, fontsize=10)
            ax.set_xlabel("Split ID")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.2)

        for ax in axes[len(block) :]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        save_page(pdf, fig)


def add_gate_summary_pages(
    pdf: PdfPages,
    gate_lag: pd.DataFrame,
    gate_regime: pd.DataFrame,
    gate_corr: pd.DataFrame,
    gate_bucket_corr: pd.DataFrame,
) -> None:
    if not gate_lag.empty:
        for arch in sorted(gate_lag["architecture"].dropna().unique()):
            d = gate_lag[gate_lag["architecture"] == arch].copy()
            if d.empty:
                continue
            d["series"] = d["variant"].str.replace("_", " ").str.title() + " | " + d["gate_name"]
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            page_title(fig, f"{arch.upper()} Gate Profiles by Lag")
            sns.lineplot(data=d, x="lag", y="gate_value_mean", hue="series", ax=ax)
            ax.invert_xaxis()
            ax.set_xlabel("Lag (1 = most recent)")
            ax.set_ylabel("Mean Gate Activation")
            ax.grid(alpha=0.2)
            ax.legend(title="Variant | Gate", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            save_page(pdf, fig)

    if not gate_regime.empty:
        for arch in sorted(gate_regime["architecture"].dropna().unique()):
            d = gate_regime[gate_regime["architecture"] == arch].copy()
            if d.empty:
                continue
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            page_title(fig, f"{arch.upper()} Lag-1 Gate Activation by Regime")
            sns.barplot(data=d, x="gate_name", y="gate_value_mean", hue="vol_regime", ax=ax)
            ax.set_xlabel("Gate")
            ax.set_ylabel("Mean Gate Activation")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(title="Vol Regime", frameon=False)
            save_page(pdf, fig)

    if not gate_corr.empty:
        for arch in sorted(gate_corr["architecture"].dropna().unique()):
            d = gate_corr[gate_corr["architecture"] == arch].copy()
            if d.empty:
                continue
            d["model"] = d["variant"].str.replace("_", " ").str.title()
            pivot = d.pivot(index="gate_name", columns="model", values="pearson_corr")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            page_title(fig, f"{arch.upper()} Gate-VIX Correlation (Pearson)")
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
            ax.set_xlabel("Variant")
            ax.set_ylabel("Gate")
            save_page(pdf, fig)

    if not gate_bucket_corr.empty:
        d = gate_bucket_corr.dropna(subset=["pearson_corr"]).copy()
        for arch in sorted(d["architecture"].dropna().unique()):
            da = d[d["architecture"] == arch].copy()
            if da.empty:
                continue
            da["variant_bucket"] = da["variant"].str.replace("_", " ").str.title() + " | " + da["lag_bucket"].astype(str)
            pivot = da.pivot_table(index="gate_name", columns="variant_bucket", values="pearson_corr", aggfunc="mean")
            if pivot.empty:
                continue
            fig_w = max(PAGE_SIZE[0], 0.75 * len(pivot.columns) + 6)
            fig, ax = plt.subplots(figsize=(fig_w, PAGE_SIZE[1]))
            page_title(fig, f"{arch.upper()} Bucket Gate-VIX Correlation (Pearson)")
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
            ax.set_xlabel("Variant | Lag Bucket")
            ax.set_ylabel("Gate")
            save_page(pdf, fig)


def add_gate_date_pages(
    pdf: PdfPages,
    lag1_by_date: pd.DataFrame,
    lag1_20_by_date: pd.DataFrame,
    lag_bucket_by_date: pd.DataFrame,
) -> None:
    if not lag1_by_date.empty and "vix_close" in lag1_by_date.columns:
        for (arch, variant), d0 in lag1_by_date.groupby(["architecture", "variant"], sort=True):
            d0 = d0.sort_values("date")
            gates = sorted(d0["gate_name"].dropna().unique())
            if not gates:
                continue
            ncols = 2
            nrows = int(np.ceil(len(gates) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_SIZE[0], 3.0 * nrows + 1.5), sharex=True)
            page_title(fig, f"{arch.upper()} | {str(variant).replace('_', ' ').title()}: Lag-1 Gate vs VIX")
            axes = np.array(axes).reshape(-1)
            for ax, gate in zip(axes, gates):
                d = d0[d0["gate_name"] == gate]
                ax.plot(d["date"], d["gate_value_mean"], color="#1f77b4", lw=1.1)
                ax2 = ax.twinx()
                ax2.plot(d["date"], d["vix_close"], color="#E76F51", lw=0.9, alpha=0.4)
                ax.set_title(gate.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Gate")
                ax2.set_ylabel("VIX")
                ax.grid(alpha=0.2)
            for ax in axes[len(gates) :]:
                ax.axis("off")
            save_page(pdf, fig)

    if not lag1_20_by_date.empty and "vix_close" in lag1_20_by_date.columns:
        for (arch, variant), d0 in lag1_20_by_date.groupby(["architecture", "variant"], sort=True):
            d0 = d0.sort_values("date")
            gates = sorted(d0["gate_name"].dropna().unique())
            if not gates:
                continue
            ncols = 2
            nrows = int(np.ceil(len(gates) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_SIZE[0], 3.0 * nrows + 1.5), sharex=True)
            page_title(fig, f"{arch.upper()} | {str(variant).replace('_', ' ').title()}: Mean Gate (Lags 1-20) vs VIX")
            axes = np.array(axes).reshape(-1)
            for ax, gate in zip(axes, gates):
                d = d0[d0["gate_name"] == gate]
                ax.plot(d["date"], d["gate_value_lag1_20_mean"], color="#1f77b4", lw=1.1)
                ax2 = ax.twinx()
                ax2.plot(d["date"], d["vix_close"], color="#E76F51", lw=0.9, alpha=0.4)
                ax.set_title(gate.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Gate")
                ax2.set_ylabel("VIX")
                ax.grid(alpha=0.2)
            for ax in axes[len(gates) :]:
                ax.axis("off")
            save_page(pdf, fig)

    if not lag_bucket_by_date.empty and "vix_close" in lag_bucket_by_date.columns:
        for (arch, variant), d0 in lag_bucket_by_date.groupby(["architecture", "variant"], sort=True):
            d0 = d0.sort_values("date")
            gates = sorted(d0["gate_name"].dropna().unique())
            if not gates:
                continue
            ncols = 2
            nrows = int(np.ceil(len(gates) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_SIZE[0], 3.2 * nrows + 1.5), sharex=True)
            page_title(fig, f"{arch.upper()} | {str(variant).replace('_', ' ').title()}: Lag-Bucket Gates vs VIX")
            axes = np.array(axes).reshape(-1)
            for ax, gate in zip(axes, gates):
                d = d0[d0["gate_name"] == gate]
                sns.lineplot(data=d, x="date", y="gate_value_mean", hue="lag_bucket", ax=ax, lw=1.0)
                ax2 = ax.twinx()
                ax2.plot(d["date"], d["vix_close"], color="#E76F51", lw=0.9, alpha=0.35)
                ax.set_title(gate.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Gate")
                ax2.set_ylabel("VIX")
                ax.grid(alpha=0.2)
                ax.legend(title="Lag Bucket", frameon=False, loc="upper left")
            for ax in axes[len(gates) :]:
                ax.axis("off")
            save_page(pdf, fig)


def add_pure_lstm_gate_insight_pages(
    pdf: PdfPages,
    lag1_20_by_date: pd.DataFrame,
    gate_bucket_corr: pd.DataFrame,
) -> None:
    d = lag1_20_by_date.copy()
    if not d.empty and {"architecture", "variant", "gate_name", "vix_close", "gate_value_lag1_20_mean"}.issubset(d.columns):
        d = d[(d["architecture"] == "lstm") & (d["variant"] == "pure")].dropna(
            subset=["vix_close", "gate_value_lag1_20_mean"]
        )
        gates = sorted(d["gate_name"].unique())
        if gates:
            ncols = 2
            nrows = int(np.ceil(len(gates) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_SIZE[0], 3.2 * nrows + 1.5), sharex=False)
            page_title(fig, "Pure LSTM: Mean Gate (Lags 1-20) vs VIX")
            axes = np.array(axes).reshape(-1)
            for ax, gate in zip(axes, gates):
                dg = d[d["gate_name"] == gate]
                sns.regplot(data=dg, x="vix_close", y="gate_value_lag1_20_mean", scatter_kws={"s": 12, "alpha": 0.35}, line_kws={"lw": 1.2}, ax=ax)
                ax.set_title(gate.replace("_", " ").title(), fontsize=10)
                ax.set_xlabel("VIX")
                ax.set_ylabel("Gate Mean (Lags 1-20)")
                ax.grid(alpha=0.2)
            for ax in axes[len(gates) :]:
                ax.axis("off")
            save_page(pdf, fig)

    gb = gate_bucket_corr.copy()
    if not gb.empty and {"architecture", "variant", "gate_name", "lag_bucket", "pearson_corr"}.issubset(gb.columns):
        gb = gb[(gb["architecture"] == "lstm") & (gb["variant"] == "pure")].dropna(subset=["pearson_corr"])
        if not gb.empty:
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            page_title(fig, "Pure LSTM: Bucket-Level Gate-VIX Correlation (Pearson)")
            sns.barplot(
                data=gb,
                x="gate_name",
                y="pearson_corr",
                hue="lag_bucket",
                hue_order=["lag1_5", "lag6_10", "lag11_20"],
                ax=ax,
            )
            ax.axhline(0.0, color="gray", ls="--", lw=1)
            ax.set_xlabel("Gate")
            ax.set_ylabel("Correlation with VIX")
            ax.legend(title="Lag Bucket", frameon=False)
            ax.grid(axis="y", alpha=0.2)
            save_page(pdf, fig)


def main() -> None:
    eval_path = PRED_DIR / "evaluation_metrics_mse_qlike.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing: {eval_path}")

    eval_df = safe_read(eval_path)
    eval_df["model"] = model_label(eval_df)

    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))
    pred_files = [
        p
        for p in pred_files
        if "raw_predictions" not in p.name and not p.name.startswith("evaluation_metrics")
    ]

    pred_frames = []
    for p in pred_files:
        d = safe_read(p, parse_dates=["date"])
        required = {"date", "variant", "architecture", "y_true_var", "y_pred_var"}
        if required.issubset(set(d.columns)):
            pred_frames.append(d)

    if not pred_frames:
        raise ValueError("No valid prediction files found in reports/predictions.")

    all_pred = pd.concat(pred_frames, ignore_index=True)
    all_pred["model"] = model_label(all_pred)

    vix_path = ROOT / "data" / "processed" / "vix_daily.csv"
    vix_df = safe_read(vix_path, parse_dates=["date"]) if vix_path.exists() else pd.DataFrame(columns=["date", "vix_close"])

    model_daily = build_model_daily(all_pred, vix_df)
    model_order = eval_df.sort_values("mse")["model"].tolist()

    logs = collect_train_logs()

    gate_lag = safe_read(PRED_DIR / "gate_summary_by_lag.csv") if (PRED_DIR / "gate_summary_by_lag.csv").exists() else pd.DataFrame()
    gate_regime = safe_read(PRED_DIR / "gate_summary_by_regime_lag1.csv") if (PRED_DIR / "gate_summary_by_regime_lag1.csv").exists() else pd.DataFrame()
    gate_corr = safe_read(PRED_DIR / "gate_vix_correlation_summary.csv") if (PRED_DIR / "gate_vix_correlation_summary.csv").exists() else pd.DataFrame()
    gate_bucket_corr = safe_read(PRED_DIR / "gate_vix_correlation_by_bucket.csv") if (PRED_DIR / "gate_vix_correlation_by_bucket.csv").exists() else pd.DataFrame()
    lag1_20_by_date = safe_read(PRED_DIR / "gate_summary_lag1_20_by_date.csv", parse_dates=["date"]) if (PRED_DIR / "gate_summary_lag1_20_by_date.csv").exists() else pd.DataFrame()
    lag_bucket_by_date = safe_read(PRED_DIR / "gate_summary_lag_buckets_by_date.csv", parse_dates=["date"]) if (PRED_DIR / "gate_summary_lag_buckets_by_date.csv").exists() else pd.DataFrame()

    gate_raw = collect_gate_raw()
    lag1_by_date = pd.DataFrame()
    if not gate_raw.empty:
        lag1_by_date = (
            gate_raw[gate_raw["lag"] == 1]
            .groupby(["date", "architecture", "variant", "gate_name"], as_index=False)
            .agg(gate_value_mean=("gate_value_mean", "mean"))
        )
        if not vix_df.empty:
            lag1_by_date = lag1_by_date.merge(vix_df[["date", "vix_close"]], on="date", how="left")

    with PdfPages(OUT_PDF) as pdf:
        add_text_page(
            pdf,
            "Volatility-Blackbox Report",
            [
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "Data: S&P 500 (^GSPC), daily, 2003-2024.",
                "Evaluation: expanding rolling OOS with test_size=21 and step_size=21.",
                "Models: baseline GARCH-t, pure LSTM/GRU, hybrid GARCH+LSTM/GRU, residual hybrids.",
                "Metrics: MSE and QLIKE on daily variance forecasts.",
                "RNN preprocessing: train-only transforms (log+standardize target for variance targets;",
                "standardize target for residual targets; log+standardize GARCH features).",
                "Output layer for RNNs is linear with inverse transform applied at inference.",
            ],
        )

        eval_show = eval_df[["model", "n_obs", "mse", "qlike"]].copy().sort_values("mse")
        eval_show["mse"] = eval_show["mse"].map(lambda x: f"{x:.6e}")
        eval_show["qlike"] = eval_show["qlike"].map(lambda x: f"{x:.6e}")
        add_table_page(pdf, "Performance Snapshot", eval_show)

        add_metric_pages(pdf, eval_df)
        add_forecast_pages(pdf, model_daily, model_order)
        add_vix_diagnostic_pages(pdf, model_daily, model_order)
        add_calibration_pages(pdf, model_daily, model_order)
        add_rolling_mse_vs_realized_pages(pdf, model_daily, model_order)
        add_overfit_pages(pdf, logs)
        add_gate_summary_pages(pdf, gate_lag, gate_regime, gate_corr, gate_bucket_corr)
        add_gate_date_pages(pdf, lag1_by_date, lag1_20_by_date, lag_bucket_by_date)
        add_pure_lstm_gate_insight_pages(pdf, lag1_20_by_date, gate_bucket_corr)

        add_text_page(
            pdf,
            "Coverage Notes",
            [
                "This PDF auto-generates all available report plots from CSV artifacts in reports/predictions.",
                "If a section is missing, the corresponding source CSV was absent or empty.",
                "Title formatting is standardized at the page level to avoid clipped or overlapping titles.",
            ],
        )

    print(f"Saved PDF report: {OUT_PDF}")


if __name__ == "__main__":
    main()
