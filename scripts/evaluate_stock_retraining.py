"""Verify a completed stock retrain and score forecasts plus matched portfolios."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.portfolio import overlay_targets, simulate_portfolio, performance_stats
from retrain_stock_overlay import verify_signature, sha, write_json
from evaluate_portfolio_overlay import paired_sharpe_interval


def daily_average(values, mask):
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.).sum(axis=1)
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def mean_difference_interval(daily, block=21, reps=2000):
    values = np.asarray(daily, float)
    values = values[np.isfinite(values)]
    if len(values) < 50:
        raise ValueError("Too few common loss dates.")
    bootstrap = StationaryBootstrap(block, values, seed=20260925)
    draws = bootstrap.apply(lambda x: np.array([x.mean()]), reps=reps)[:, 0]
    low, high = np.quantile(draws, [.025, .975])
    return {"difference": float(values.mean()), "ci95_lower": float(low),
            "ci95_upper": float(high), "dates": len(values), "mean_block": block, "resamples": reps}


def forecast_metrics(returns, garch, hybrid):
    realized = returns.rolling(21).std(ddof=1).reindex(garch.index)
    persistence = returns.rolling(21).std(ddof=1).shift(1).reindex(garch.index)
    target = realized.to_numpy()
    g, h, p = garch.to_numpy(), hybrid.to_numpy(), persistence.to_numpy()
    common = np.isfinite(g) & np.isfinite(h) & np.isfinite(p) & np.isfinite(target)
    if (common.sum(axis=1) == 0).any():
        raise ValueError("Empty forecast-evaluation date.")
    volatility = {}
    errors = {}
    for name, forecast in (("garch", g), ("hybrid_raw", h), ("persistence", p)):
        squared = (forecast-target)**2
        daily = daily_average(squared, common)
        errors[name] = daily
        volatility[name] = {"date_equal_mse": float(daily.mean()),
                            "pooled_asset_date_mse": float(squared[common].mean())}
    counts = common.sum(axis=0)
    asset_g = np.divide(np.where(common, (g-target)**2, 0.).sum(axis=0), counts,
                        out=np.full(len(counts), np.nan), where=counts > 0)
    asset_h = np.divide(np.where(common, (h-target)**2, 0.).sum(axis=0), counts,
                        out=np.full(len(counts), np.nan), where=counts > 0)
    comparison = {
        "hybrid_vs_garch": mean_difference_interval(errors["hybrid_raw"]-errors["garch"]),
        "hybrid_vs_persistence": mean_difference_interval(errors["hybrid_raw"]-errors["persistence"]),
    }
    contributions = np.where(common, (h-target)**2, 0.)/common.sum(axis=1)[:, None]
    contributions = contributions.mean(axis=0)
    dominant = int(np.argmax(contributions))
    stock_ratios = np.divide(asset_h, asset_g, out=np.full(len(counts), np.nan),
                             where=np.isfinite(asset_g) & (asset_g > 0))
    sq_return = returns.reindex(garch.index).to_numpy()**2
    positive = common & (g > 0) & (h > 0) & (p > 0) & np.isfinite(sq_return)
    variance = {}
    for name, forecast in (("garch", g), ("hybrid", h), ("persistence", p)):
        v = np.maximum(forecast**2, 1e-12)
        variance[name] = {
            "mse": float(np.nanmean(daily_average((v-sq_return)**2, positive))),
            "qlike": float(np.nanmean(daily_average(np.log(v)+sq_return/v, positive))),
        }
    return {
        "target": "Trailing 21-day sample volatility through target date; not fresh future realized volatility",
        "common_asset_dates": int(common.sum()), "common_dates": int((common.sum(axis=1)>0).sum()),
        "assets_with_common_forecasts": int((counts>0).sum()),
        "assets_with_lower_hybrid_mse": int(((asset_h<asset_g)&(counts>0)).sum()),
        "missing_hybrid_asset_dates": int((~np.isfinite(h)).sum()),
        "nonpositive_hybrid_asset_dates": int((np.isfinite(h)&(h<=0)).sum()),
        "maximum_absolute_hybrid_sigma": float(np.nanmax(np.abs(h))),
        "failure_diagnostics": {
            "dominant_error_stock": str(garch.columns[dominant]),
            "dominant_share_of_date_equal_hybrid_mse": float(contributions[dominant]/contributions.sum()),
            "median_stock_hybrid_to_garch_mse_ratio": float(np.nanmedian(stock_ratios)),
            "positive_hybrid_sigma_above_one_asset_dates": int((h>1).sum()),
            "minimum_positive_garch_sigma": float(g[g>0].min()),
            "largest_nonpositive_stock_counts": [
                {"stock": str(stock), "count": int(count)} for stock, count in
                (hybrid <= 0).sum().sort_values(ascending=False).head(3).items()],
            "interpretation": "Loss concentration diagnostic only; no stocks or forecasts removed or retuned after scoring.",
        },
        "volatility_mse": volatility, "paired_loss_intervals": comparison,
        "positive_subset_asset_dates": int(positive.sum()), "variance_floor": 1e-12,
        "squared_return_metrics_positive_subset": variance,
    }


def verify_run(run, source):
    signature = verify_signature(run)
    for package, expected in signature["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Runtime package changed: {package}")
    done = json.loads((run/"complete.json").read_text())
    if done["signature_sha256"] != sha(run/"signature.json"):
        raise ValueError("Completion belongs to another training run.")
    if signature["assets"] != 503 or signature["selection"] != "full supplied stock universe":
        raise ValueError("This full-study evaluator must not publish a pilot subset.")
    for name, expected in signature["source_sha256"].items():
        if sha(source/name) != expected:
            raise ValueError(f"Original input changed: {name}")
    for name, expected in done["panel_sha256"].items():
        if sha(run/name) != expected:
            raise ValueError(f"Forecast panel changed: {name}")
    names = json.loads((run/"tickers.json").read_text())
    diagnostics = {"stocks": len(names), "lstm_fits": 0, "strictly_prior_training_cutoffs": True,
                   "garch_full_history_status_counts": {}, "garch_evaluation_status_counts": {},
                   "stocks_without_trained_model": 0, "checkpoint_files_verified": 0}
    for ticker in names:
        directory = run/"stocks"/ticker
        if sha(directory/"complete.json") != done["stock_receipts_sha256"][ticker]:
            raise ValueError(f"Stock receipt changed: {ticker}")
        receipt = json.loads((directory/"complete.json").read_text())
        if receipt["signature_sha256"] != done["signature_sha256"]:
            raise ValueError(f"Stock lineage mismatch: {ticker}")
        for name, expected in receipt["artifacts"].items():
            if sha(directory/name) != expected:
                raise ValueError(f"Stock artifact mismatch: {ticker}/{name}")
            diagnostics["checkpoint_files_verified"] += int(name.endswith(".keras"))
        logs = json.loads((directory/"training.json").read_text())
        diagnostics["lstm_fits"] += len(logs)
        diagnostics["stocks_without_trained_model"] += int(not logs)
        for fit in logs:
            if not fit["training_last_target"] < fit["first_forecast"]:
                raise ValueError(f"Training leaked future labels: {ticker}")
        for key, value in receipt["garch_status"].items():
            old = diagnostics["garch_full_history_status_counts"].get(key, 0)
            diagnostics["garch_full_history_status_counts"][key] = old + value
        frame = pd.read_csv(directory/"garch.csv", index_col=0, parse_dates=True)
        for key, value in frame.loc[signature["start"]:signature["end"], "status"].value_counts().items():
            old = diagnostics["garch_evaluation_status_counts"].get(key, 0)
            diagnostics["garch_evaluation_status_counts"][key] = old + int(value)
    return signature, done, diagnostics


def evaluate(run, source, destination):
    signature, done, diagnostics = verify_run(run, source)
    garch = pd.read_csv(run/"garch_vol_df.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    hybrid = pd.read_csv(run/"hybrid_vol_df.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    cap = pd.read_csv(source/"mcap_weight_df.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(source/"log_ret_bt.csv", index_col=0, parse_dates=True)
    if not garch.index.equals(hybrid.index) or not garch.columns.equals(hybrid.columns):
        raise ValueError("Forecast panels do not align.")
    garch_matched = garch.where(np.isfinite(hybrid))
    targets, audits = {}, {}
    for label, base in (("cap", cap), ("equal", cap*0+1)):
        neutral, gtilt, ga = overlay_targets(returns, garch_matched, base)
        other, htilt, ha = overlay_targets(returns, hybrid, base)
        pd.testing.assert_frame_equal(neutral, other)
        _, unrestricted, _ = overlay_targets(returns, garch, base)
        targets.update({f"{label}_base": neutral, f"{label}_garch": gtilt,
                        f"{label}_hybrid": htilt, f"{label}_garch_unrestricted": unrestricted})
        audits[label] = {"garch_matched": ga, "hybrid": ha,
                         "mean_target_L1_distance_from_base": {
                             "garch": float((gtilt-neutral).abs().sum(axis=1).mean()),
                             "hybrid": float((htilt-neutral).abs().sum(axis=1).mean())}}
    configurations = [(name, "monthly", 10) for name in targets]
    for name in ("cap_base", "cap_garch", "cap_hybrid"):
        configurations += [(name, freq, 10) for freq in ("daily", "weekly", "yearly")]
        configurations += [(name, "monthly", cost) for cost in (0, 25)]
    simple = np.expm1(returns.loc[garch.index, garch.columns])
    output = run/"evaluation"
    output.mkdir(exist_ok=True)
    metrics, ledgers = [], {}
    for name, frequency, cost in configurations:
        key = f"{name}_{frequency}_{cost}bps"
        result = simulate_portfolio(simple, targets[name], frequency, cost, 1)
        ledger = result.ledger
        stats = performance_stats(ledger.net_return)
        stats.update({"id": key, "model": name, "frequency": frequency, "cost_bps": cost,
                      "start": str(ledger.index[0].date()), "end": str(ledger.index[-1].date()),
                      "annualized_traded_notional": float(ledger.traded_notional.sum()*252/len(ledger)),
                      "sharpe_rf_zero": performance_stats(ledger.net_return, 0)["sharpe"]})
        metrics.append(stats)
        ledgers[key] = ledger
        ledger.to_csv(output/f"{key}.csv")
    prior = json.loads((ROOT/"reports/portfolio_overlay_results.json").read_text())
    if any(prior["source_sha256"][name] != expected
           for name, expected in signature["source_sha256"].items()):
        raise ValueError("The cached-study control uses different source inputs.")
    for current, old in (("cap_base", "snapshot_cap"), ("equal_base", "equal_weight")):
        new_row = next(row for row in metrics if row["id"] == f"{current}_monthly_10bps")
        old_row = next(row for row in prior["metrics"] if row["id"] == f"{old}_monthly_10bps_lag1")
        for metric in ("sharpe", "cagr", "max_drawdown"):
            if not np.isclose(new_row[metric], old_row[metric], rtol=0, atol=1e-12):
                raise ValueError(f"Untilted control changed unexpectedly: {current}/{metric}")
    diagnostics["untilted_baselines_match_previous_accounting_study"] = True
    intervals = []
    for label in ("cap", "equal"):
        left = ledgers[f"{label}_hybrid_monthly_10bps"].net_return
        for base in ("base", "garch"):
            right = ledgers[f"{label}_{base}_monthly_10bps"].net_return
            if not left.index.equals(right.index):
                raise ValueError("Portfolio comparison date mismatch.")
            for block, reps in ((21, 2000), (5, 1000), (63, 1000)):
                row = paired_sharpe_interval(left, right, block=block, reps=reps)
                row.update({"base": f"{label}_{base}", "candidate": f"{label}_hybrid"})
                intervals.append(row)
    result = {
        "study": "stock-overlay-retrain-v1", "status": "Completed retraining, source-biased retrospective diagnostic",
        "training_signature": signature, "completion": done, "training_audit": diagnostics,
        "forecast_metrics": forecast_metrics(returns[garch.columns], garch, hybrid),
        "portfolio_configuration": {"primary_frequency": "monthly", "cost_bps": 10,
                                    "execution_lag": 1, "rf_daily_assumption": .0001,
                                    "garch_primary_model_coverage": "masked to finite raw hybrid forecasts"},
        "portfolio_metrics": metrics, "paired_sharpe_intervals": intervals, "eligibility": audits,
        "evaluation_code_sha256": {name: sha(ROOT/name) for name in (
            "scripts/evaluate_stock_retraining.py", "src/evaluation/portfolio.py",
            "scripts/evaluate_portfolio_overlay.py", "docs/stock-overlay-evaluation.md")},
        "private_ledger_sha256": {path.name: sha(path) for path in sorted(output.glob("*.csv"))},
        "limits": ["Later survivor universe; current cap snapshot repeated through history.",
                   "Not an untouched holdout; only one seed per fit, no seed uncertainty study.",
                   "Trailing realized-volatility target has mechanical overlap, not a pure future-risk target.",
                   "Independent GARCH comparator uses explicit recorded failed-fit fallback.",
                   "Trading-cost/next-close fills and fixed risk-free rate are assumptions, not execution evidence.",
                   "No causal LSTM alpha or achievable return claim; data bias survives retraining."],
    }
    write_json(destination, result)
    print(json.dumps({"audit": diagnostics, "forecast_metrics": result["forecast_metrics"],
                      "primary": [row for row in metrics if row["frequency"]=="monthly" and row["cost_bps"]==10],
                      "intervals": [row for row in intervals if row["block_length"]==21]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT/"reports/stock_retraining_results.json")
    args = parser.parse_args()
    evaluate(args.run, args.source, args.destination)
