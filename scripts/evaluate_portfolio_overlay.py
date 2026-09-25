"""Re-evaluate legacy portfolio accounting; no point-in-time alpha claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.portfolio import overlay_targets, simulate_portfolio, performance_stats, validate_frame


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_sharpe_interval(left, right, block=21, reps=2000, rf=.0001):
    values = np.column_stack([left, right])
    def difference(x):
        sd = x.std(axis=0, ddof=1)
        if (sd <= 1e-15).any():
            raise ValueError("Degenerate Sharpe resample.")
        scores = (x.mean(axis=0)-rf) / sd * np.sqrt(252)
        return np.array([scores[0]-scores[1]])
    draws = StationaryBootstrap(block, values, seed=20260925).apply(difference, reps=reps)
    lower, upper = np.quantile(draws[:, 0], [.025, .975])
    return {"difference": float(difference(values)[0]), "ci95_lower": float(lower),
            "ci95_upper": float(upper), "block_length": block, "resamples": reps}


def legacy_replay(log_returns, forecasts, cap, frequency):
    """Faithful score replay only, deliberately retaining the old accounting defects."""
    dates, columns = forecasts.index, forecasts.columns
    realized = log_returns[columns].rolling(21).std().reindex(dates)
    weight = cap * (realized / forecasts).clip(.5, 1.5)
    weight = weight.div(weight.sum(axis=1), axis=0)
    if frequency == "monthly":
        trade_dates = dates[dates.is_month_start]
    elif frequency == "yearly":
        trade_dates = dates[np.r_[True, dates.year[1:] != dates.year[:-1]]]
    else:
        raise ValueError("Replay supports the two displayed original cases only.")
    weight.loc[~dates.isin(trade_dates)] = np.nan
    weight = weight.ffill()
    wrong_log_returns = (weight * log_returns[columns].shift(-1).reindex(dates)).sum(axis=1)
    costs = (weight-weight.shift(1).fillna(0)).abs().sum(axis=1)*.001
    simple = np.expm1(wrong_log_returns-costs)
    result = performance_stats(simple)
    # The old drawdown starts at the first returned wealth, not initial wealth.
    wealth = (1+simple).cumprod()
    result["max_drawdown"] = float((wealth/wealth.cummax()-1).min())
    return result


def evaluate(source, output, destination, acknowledge):
    if not acknowledge:
        raise ValueError("Pass --acknowledge-retrospective-data after reading the protocol's data limits.")
    names = ["hybrid_vol_df.csv", "mcap_weight_df.csv", "log_ret_bt.csv"]
    paths = {name: source/name for name in names}
    frames = {name: pd.read_csv(path, index_col=0, parse_dates=True) for name, path in paths.items()}
    forecast, cap, log_returns = [frames[name] for name in names]
    for name, frame in frames.items():
        validate_frame(frame, name)
    if len(cap.drop_duplicates()) != 1:
        raise ValueError("This protocol expects the legacy constant cap snapshot; review changed data separately.")
    if "SP500" not in log_returns or log_returns.loc[forecast.index, "SP500"].isna().any():
        raise ValueError("Missing contextual index data.")
    dates = forecast.index
    if not log_returns.loc[dates[0]:dates[-1]].index.equals(dates):
        raise ValueError("Forecast dates have gaps against the supplied trading sessions.")
    returns = np.expm1(log_returns.loc[dates, forecast.columns])
    neutral, tilt, audit = overlay_targets(log_returns, forecast, cap)
    equal_base = pd.DataFrame(1., index=dates, columns=forecast.columns)
    equal, equal_tilt, equal_audit = overlay_targets(log_returns, forecast, equal_base)
    targets = {"snapshot_cap": neutral, "hybrid_cap": tilt,
               "equal_weight": equal, "hybrid_equal": equal_tilt}
    output.mkdir(parents=True, exist_ok=True)
    metrics, saved = [], {}
    specifications = [(name, "monthly", 10, 1) for name in targets]
    specifications += [(name, frequency, 10, 1) for name in ("snapshot_cap", "hybrid_cap")
                       for frequency in ("daily", "weekly", "yearly")]
    specifications += [(name, "monthly", cost, 1) for name in ("snapshot_cap", "hybrid_cap") for cost in (0, 25)]
    specifications += [(name, "monthly", 10, 0) for name in ("snapshot_cap", "hybrid_cap")]
    common_dates = None
    for name, frequency, cost, lag in specifications:
        key = f"{name}_{frequency}_{cost}bps_lag{lag}"
        # Align initial entry and earned returns for both latency assumptions.
        shift = 1-lag
        result = simulate_portfolio(returns.iloc[shift:], targets[name].iloc[shift:], frequency, cost, lag)
        ledger = result.ledger
        if common_dates is None:
            common_dates = ledger.index
        if not ledger.index.equals(common_dates):
            raise ValueError("Scenario does not cover the identical evaluation period.")
        view = ledger
        stats = performance_stats(view.net_return)
        stats.update({"id": key, "model": name, "frequency": frequency, "cost_bps": cost,
                      "execution_lag": lag, "start": str(common_dates[0].date()),
                      "end": str(common_dates[-1].date()),
                      "sharpe_rf_zero": performance_stats(view.net_return, 0)["sharpe"],
                      "annualized_traded_notional": float(view.traded_notional.sum()*252/len(view)),
                      "sum_daily_cost_fractions": float(view.cost_fraction.sum()),
                      "rebalances": int(view.rebalanced.sum())})
        metrics.append(stats)
        saved[key] = view
        ledger.to_csv(output/f"{key}.csv")
        print(json.dumps({"id": key, "sharpe": stats["sharpe"], "cagr": stats["cagr"]}), flush=True)
    spx = np.expm1(log_returns.loc[common_dates, "SP500"])
    index_stats = performance_stats(spx)
    index_stats.update({"id": "sp500_price_reference", "role": "context only, not a matched total-return comparator"})

    intervals = []
    for base_name, overlay_name in (("snapshot_cap", "hybrid_cap"), ("equal_weight", "hybrid_equal")):
        base = saved[f"{base_name}_monthly_10bps_lag1"].net_return
        overlay = saved[f"{overlay_name}_monthly_10bps_lag1"].net_return
        for block, reps in ((21, 2000), (5, 1000), (63, 1000)):
            interval = paired_sharpe_interval(overlay, base, block, reps)
            interval.update({"baseline": base_name, "overlay": overlay_name})
            intervals.append(interval)
    # Store only monthly portfolio aggregates, never underlying stock data.
    aggregate = pd.DataFrame({name: np.cumprod(1.+saved[f"{name}_monthly_10bps_lag1"].net_return)
                              for name in targets}, index=common_dates)
    aggregate["sp500_price_reference"] = np.cumprod(1.+spx)
    monthly = aggregate.resample("ME").last()
    curves = [{"date": str(date.date()), **{key: float(value) for key, value in row.items()}}
              for date, row in monthly.iterrows()]
    yearly = []
    for name in targets:
        s = saved[f"{name}_monthly_10bps_lag1"].net_return
        for year, values in s.groupby(s.index.year):
            yearly.append({"model": name, "year": int(year), "n": len(values),
                           "return": float(np.prod(1.+values)-1),
                           "sharpe": performance_stats(values)["sharpe"]})

    legacy = {freq: legacy_replay(log_returns, forecast, cap, freq) for freq in ("yearly", "monthly")}
    expected_sharpes = {"yearly": .959989, "monthly": .954744}
    for key, expected in expected_sharpes.items():
        if not np.isclose(legacy[key]["sharpe"], expected, atol=5e-7, rtol=0):
            raise ValueError("Saved inputs do not reproduce the inspected notebook's legacy Sharpe.")
    source_hashes = {name: sha(path) for name, path in paths.items()}
    for name in ("cleaned_code.ipynb", "Untitled2.ipynb", "data_load.ipynb"):
        if (source/name).exists():
            source_hashes[name] = sha(source/name)
    result = {
        "study": "legacy-equity-volatility-overlay-accounting-diagnostic",
        "status": "Retrospective diagnostic; not point-in-time alpha evidence",
        "data": {"forecast_rows": len(forecast), "assets": len(forecast.columns),
                 "start": str(common_dates[0].date()), "end": str(common_dates[-1].date()),
                 "observations": len(common_dates), "cap_snapshot_unique_rows": len(cap.drop_duplicates()),
                 "forecast_missing": int(forecast.isna().sum().sum()),
                 "forecast_nonpositive": int((forecast <= 0).sum().sum()),
                 "forecast_max": float(forecast.max().max()),
                 "historical_return_infinities": int(np.isinf(log_returns[forecast.columns].to_numpy()).sum()),
                 "evaluation_return_infinities": int(np.isinf(returns.to_numpy()).sum()),
                 **audit, "equal_weight_eligibility": equal_audit},
        "configuration": {"primary_frequency": "monthly", "primary_cost_bps": 10,
                          "primary_execution_lag": 1, "trailing_vol_window": 21,
                          "tilt_bounds": [.5, 1.5], "rf_daily_assumption": .0001,
                          "cash_daily_return": 0., "annualization": 252,
                          "terminal_liquidation": False, "forecast_retraining_performed": False},
        "metrics": metrics, "contextual_index": index_stats, "paired_sharpe_intervals": intervals,
        "monthly_nav": curves, "calendar_returns": yearly, "legacy_replay": legacy,
        "source_sha256": source_hashes,
        "code_sha256": {path: sha(ROOT/path) for path in
                        ("src/evaluation/portfolio.py", "scripts/evaluate_portfolio_overlay.py",
                         "docs/portfolio-overlay-protocol.md", "tests/test_portfolio.py")},
        "private_daily_output_sha256": {path.name: sha(path) for path in sorted(output.glob("*.csv"))},
        "limits": [
            "Later constituent list and constant later market-cap weights create survivorship and look-ahead bias.",
            "Equal-weight sensitivity removes the cap snapshot, not the survivor-universe selection.",
            "Legacy stock model is a multiplicative volatility-ratio LSTM, not the index additive-residual model.",
            "Cached forecast training is not rerun or independently authenticated; extreme/invalid forecasts remain a model-quality warning.",
            "The close-d risk-ratio signal uses forecast dated d and trailing vol through d; it is not a fresh d+2 forecast.",
            "Next-close fill and proportional costs are assumptions; no spreads, volumes, capacity or realized execution data.",
            "^GSPC price index versus adjusted-close stocks is not a matched total-return alpha comparison.",
            "Sharpe uses a fixed daily risk-free assumption, not historical rates.",
            "Bootstrap intervals condition on this inspected, biased sample and cannot cure those biases.",
            "No matched GARCH-only forecast panel; no causal attribution of portfolio improvement to LSTM.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(destination), "data": result["data"], "intervals": intervals}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--daily-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT/"reports/portfolio_overlay_results.json")
    parser.add_argument("--acknowledge-retrospective-data", action="store_true")
    args = parser.parse_args()
    evaluate(args.input_dir, args.daily_output, args.destination, args.acknowledge_retrospective_data)
