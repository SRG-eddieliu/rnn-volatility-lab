"""Close-to-close, self-financing diagnostics for long-only portfolio overlays."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    ledger: pd.DataFrame
    end_weights: pd.DataFrame


def validate_frame(frame, name):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError(f"{name} requires a nonempty DatetimeIndex.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and increasing.")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} has duplicate assets.")


def overlay_targets(log_returns, forecasts, base_weights, window=21, bounds=(.5, 1.5)):
    """As-of-close targets; unavailable/nonpositive forecasts leave the base tilt at one.

    Eligibility uses only past return coverage, never next-period returns. The caller
    owns the historical validity of base_weights and the provenance of forecasts.
    """
    for name, frame in (("returns", log_returns), ("forecasts", forecasts), ("base", base_weights)):
        validate_frame(frame, name)
    if window < 2 or not 0 < bounds[0] <= 1 <= bounds[1]:
        raise ValueError("Invalid history window or tilt bounds.")
    if not forecasts.index.equals(base_weights.index) or not forecasts.columns.equals(base_weights.columns):
        raise ValueError("Forecast and base labels must match exactly.")
    if not forecasts.columns.isin(log_returns.columns).all() or not forecasts.index.isin(log_returns.index).all():
        raise ValueError("Return coverage must contain every forecast label.")
    base = base_weights.to_numpy(dtype=float)
    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Base weights must be finite and nonnegative.")
    history = log_returns.loc[:, forecasts.columns].replace([np.inf, -np.inf], np.nan)
    realized = history.rolling(window, min_periods=window).std(ddof=1).reindex(forecasts.index)
    eligible = realized.notna().to_numpy() & (base > 0)
    level = forecasts.to_numpy(dtype=float)
    valid = np.isfinite(level) & (level > 0) & eligible
    factor = np.ones_like(level)
    np.divide(realized.to_numpy(), level, out=factor, where=valid)
    factor = np.clip(factor, *bounds)
    clean_base = np.where(eligible, base, 0.)
    tilted = clean_base * factor
    def normalize(values):
        sums = values.sum(axis=1, keepdims=True)
        return np.divide(values, sums, out=np.zeros_like(values), where=sums > 0)
    neutral = pd.DataFrame(normalize(clean_base), index=forecasts.index, columns=forecasts.columns)
    overlay = pd.DataFrame(normalize(tilted), index=forecasts.index, columns=forecasts.columns)
    diagnostics = {
        "eligible_asset_days": int(eligible.sum()),
        "neutral_fallback_asset_days": int((eligible & ~valid).sum()),
        "nonpositive_forecast_asset_days": int((eligible & np.isfinite(level) & (level <= 0)).sum()),
        "missing_forecast_asset_days": int((eligible & ~np.isfinite(level)).sum()),
        "lower_bound_asset_days": int((valid & (factor == bounds[0])).sum()),
        "upper_bound_asset_days": int((valid & (factor == bounds[1])).sum()),
        "min_eligible_assets": int(eligible.sum(axis=1).min()),
        "max_eligible_assets": int(eligible.sum(axis=1).max()),
    }
    return neutral, overlay, diagnostics


def rebalance_mask(dates, frequency):
    """First observed trading close of each calendar period, including holiday weeks."""
    if frequency == "daily":
        return np.ones(len(dates), dtype=bool)
    periods = {"weekly": "W-SUN", "monthly": "M", "yearly": "Y"}
    if frequency not in periods:
        raise ValueError("Unsupported rebalance frequency.")
    keys = dates.to_period(periods[frequency])
    return np.r_[True, keys[1:] != keys[:-1]]


def rebalance_after_cost(current, target, rate):
    """Solve post-cost NAV + rate * actual risky-asset dollars traded = pre-trade NAV.

    All quantities are fractions of pre-trade wealth; buys and sells both count.
    The target is measured against post-cost wealth, so no unfunded cash is created.
    """
    if not np.isfinite(rate) or not 0 <= rate < .1:
        raise ValueError("Per-dollar cost rate must be in [0, .1).")
    current, target = np.asarray(current, float), np.asarray(target, float)
    if current.shape != target.shape or current.ndim != 1:
        raise ValueError("Weight shapes disagree.")
    for weights in (current, target):
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() > 1 + 1e-10:
            raise ValueError("Long-only weights must be finite and sum to at most one.")
    if rate == 0:
        return 1., float(np.abs(target-current).sum())
    # This fixed-point map is a contraction with constant at most rate.
    post = 1.
    for _ in range(100):
        turnover = np.abs(post * target-current).sum()
        updated = 1. - rate * turnover
        if abs(updated-post) < 1e-14:
            post = updated
            break
        post = updated
    turnover = float(np.abs(post * target-current).sum())
    if not np.isclose(post + rate * turnover, 1., atol=1e-12, rtol=0):
        raise ArithmeticError("Rebalance is not self-financing.")
    return float(post), turnover


def simulate_portfolio(simple_returns, targets, frequency="monthly", cost_bps=10., execution_lag=1):
    """Targets computed after close d execute at close d+execution_lag.

    Default lag=1 means next-close execution; the first earned return ends at d+2.
    Lag=0 is explicitly an idealized same-close fill. Between rebalances, shares
    are held constant and weights drift with returns. Uninvested cash earns zero.
    """
    validate_frame(simple_returns, "returns")
    validate_frame(targets, "targets")
    if not simple_returns.index.equals(targets.index) or not simple_returns.columns.equals(targets.columns):
        raise ValueError("Return and target labels must match exactly.")
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, int) or execution_lag < 0:
        raise ValueError("Execution lag must be a nonnegative integer.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 1000:
        raise ValueError("Invalid per-dollar traded cost.")
    r, a = simple_returns.to_numpy(dtype=float), targets.to_numpy(dtype=float)
    if np.isinf(r).any() or (r[np.isfinite(r)] <= -1).any():
        raise ValueError("Returns must exceed -100%; explicit liquidation data are required otherwise.")
    if not np.isfinite(a).all() or (a < 0).any() or (a.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Targets must be finite, long-only, and not leveraged.")
    start = execution_lag + 1
    if len(r) <= start:
        raise ValueError("Insufficient returns for the execution lag.")
    schedule = rebalance_mask(simple_returns.index, frequency)
    current = np.zeros(r.shape[1])
    nav = 1.
    records, end_weights = [], []
    for i in range(start, len(r)):
        execution = i-1
        decision = execution-execution_lag
        rebalance = i == start or schedule[execution]
        if rebalance:
            weight = a[decision].copy()
            post, turnover = rebalance_after_cost(current, weight, cost_bps / 10000.)
        else:
            weight, post, turnover = current.copy(), 1., 0.
        held = weight > 0
        if not np.isfinite(r[i, held]).all():
            bad = simple_returns.columns[held & ~np.isfinite(r[i])].tolist()
            raise ValueError(f"Missing held-asset return at {simple_returns.index[i]}: {bad}")
        gross = float(np.dot(weight[held], r[i, held]))
        growth = 1. + gross
        if not np.isfinite(growth) or growth <= 0:
            raise ValueError("Portfolio wealth is nonpositive or nonfinite.")
        net = post * growth - 1.
        nav *= 1. + net
        current = np.zeros_like(weight)
        current[held] = weight[held] * (1. + r[i, held]) / growth
        if current.sum() > 1 + 1e-10:
            raise ArithmeticError("Post-return asset weights exceed wealth.")
        records.append({
            "return_date": simple_returns.index[i], "execution_date": simple_returns.index[execution],
            "signal_date": simple_returns.index[decision] if rebalance else pd.NaT,
            "rebalanced": bool(rebalance), "gross_return": gross, "net_return": net,
            "cost_fraction": 1.-post, "traded_notional": turnover,
            "nav": nav, "cash_weight": max(0., 1.-current.sum()),
        })
        end_weights.append(current.copy())
    ledger = pd.DataFrame(records).set_index("return_date")
    return PortfolioResult(ledger, pd.DataFrame(end_weights, index=ledger.index, columns=targets.columns))


def performance_stats(returns, rf_daily=.0001):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("A finite non-bankrupt return series of length >=2 is required.")
    if not np.isfinite(rf_daily):
        raise ValueError("Risk-free assumption must be finite.")
    wealth = np.r_[1., np.cumprod(1.+r)]
    if not np.isfinite(wealth).all() or (wealth <= 0).any():
        raise ValueError("Invalid compounded wealth.")
    sd = float(r.std(ddof=1))
    return {
        "n": len(r), "cagr": float(np.expm1(np.log1p(r).sum() * 252 / len(r))),
        "annualized_arithmetic_return": float(r.mean() * 252),
        "annualized_volatility": sd * np.sqrt(252),
        "sharpe": float((r.mean()-rf_daily) / sd * np.sqrt(252)) if sd > 1e-15 else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth)-1).min()),
        "terminal_wealth": float(wealth[-1]),
    }
