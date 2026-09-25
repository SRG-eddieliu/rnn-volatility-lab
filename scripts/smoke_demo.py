"""Offline synthetic GARCH/timing smoke test, not a replication of market results."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.garch import rolling_garch_forecast
from src.models.rnn import build_sequence_dataset
from src.losses import mse_loss, qlike_loss


def run_demo():
    rng = np.random.default_rng(19)
    returns = np.zeros(420)
    variance = .0001
    for t in range(len(returns)):
        returns[t] = np.sqrt(variance)*rng.normal()
        variance = .000005 + .08*returns[t]**2 + .87*variance
    series = pd.Series(returns, index=pd.bdate_range("2020-01-01", periods=len(returns)))
    garch = rolling_garch_forecast(series, min_train_size=300, refit_every=50)
    ewma = series.pow(2).ewm(alpha=.06, adjust=False).mean().shift(1)
    y = series.pow(2)
    mask = garch.notna() & ewma.notna()
    changed = series.iloc[:361].copy()
    changed.iloc[360] += .2
    before_shock = rolling_garch_forecast(changed, min_train_size=300, refit_every=50)
    np.testing.assert_allclose(garch.iloc[:361], before_shock, equal_nan=True)

    features = pd.DataFrame({"date":series.index, "return":series.values, "y":y.values})
    x, targets, dates = build_sequence_dataset(features, ["return"], "y", lookback=21)
    np.testing.assert_allclose(x[0,:,0], series.iloc[:21], rtol=1e-6)
    assert pd.Timestamp(dates[0]) == series.index[21]
    assert np.isfinite(garch[mask]).all() and (garch[mask] > 0).all()
    return {
        "data": "synthetic; no market evidence, no neural training",
        "seed": 19, "n_common_forecasts": int(mask.sum()),
        "sequence_count": len(targets), "future_return_invariance": True,
        "metrics": {
            name: {"mse": mse_loss(y[mask], h[mask]), "qlike": qlike_loss(y[mask], h[mask])}
            for name, h in [("garch",garch), ("ewma",ewma)]
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
