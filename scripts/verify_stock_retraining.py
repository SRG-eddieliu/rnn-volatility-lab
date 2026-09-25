"""Independent checkpoint spot checks and dollar-position portfolio reconciliation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def lstm_numpy(X, weights):
    kernel, recurrent, bias, dense, offset = weights
    h = np.zeros((len(X), recurrent.shape[0]))
    c = h.copy()
    for t in range(X.shape[1]):
        z = X[:, t, :] @ kernel + h @ recurrent + bias
        i, f, candidate, o = np.split(z, 4, axis=1)
        sigmoid = lambda a: 1/(1+np.exp(-np.clip(a, -700, 700)))
        c = sigmoid(f)*c + sigmoid(i)*np.tanh(candidate)
        h = sigmoid(o)*np.tanh(c)
    return (h@dense + offset).ravel()


def check_models(run, source):
    import tensorflow as tf
    from arch import arch_model
    tf.config.set_visible_devices([], "GPU")
    returns = pd.read_csv(source/"log_ret_bt.csv", index_col=0, parse_dates=True)
    checks = []
    for ticker in ("AAPL", "ABBV", "ABNB", "AMCR", "TTWO"):
        folder = run/"stocks"/ticker
        garch = pd.read_csv(folder/"garch.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        values = pd.read_csv(folder/"predictions.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        logs = json.loads((folder/"training.json").read_text())
        r = returns[ticker].reindex(garch.index)
        ratio = r.rolling(21).std()/garch.sigma
        selected = logs if ticker in ("AMCR", "TTWO") else (logs[0], logs[-1])
        for record in selected:
            number = record["fit_id"]
            first = pd.Timestamp(record["first_forecast"])
            before = ratio.loc[ratio.index < first]
            sequences, targets = [], []
            raw = before.to_numpy()
            for i in range(10, len(raw)):
                block = raw[i-10:i+1]
                if np.isfinite(block).all():
                    sequences.append(block[:-1])
                    targets.append(block[-1])
            sequences, targets = np.array(sequences), np.array(targets)
            scalers = np.load(folder/"models"/f"scalers-{number:03d}.npz")
            assert len(targets) == record["training_sequences"]
            np.testing.assert_allclose(sequences.mean(axis=0), scalers["x_mean"], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(sequences.std(axis=0), scalers["x_scale"], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(targets.mean(), scalers["y_mean"][0], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(targets.std(), scalers["y_scale"][0], rtol=1e-9, atol=1e-10)
            dates = values.index[values.fit_id == number]
            series = values.loc[dates, "hybrid_sigma_raw"]
            chosen = pd.DatetimeIndex(list(dict.fromkeys([
                dates[0], dates[len(dates)//2], dates[-1], series.abs().idxmax(), series.idxmin()])))
            X = np.array([ratio.loc[ratio.index < date].iloc[-10:].to_numpy() for date in chosen])
            X = ((X-scalers["x_mean"])/scalers["x_scale"]).astype("float32")[:, :, None]
            model = tf.keras.models.load_model(folder/"models"/f"fit-{number:03d}.keras")
            predicted = lstm_numpy(X, model.get_weights())
            reconstructed = (predicted*scalers["y_scale"][0]+scalers["y_mean"][0])*garch.loc[chosen, "sigma"].to_numpy()
            saved = values.loc[chosen, "hybrid_sigma_raw"].to_numpy()
            np.testing.assert_allclose(reconstructed, saved, atol=2e-7, rtol=2e-5)
            checks.append({"ticker": ticker, "fit_id": number, "predictions_checked": len(chosen),
                           "max_absolute_sigma_error": float(np.max(np.abs(reconstructed-saved))),
                           "scalers_recomputed_from_strictly_prior_labels": True})
        good_dates = garch.index[(garch.status == "garch") & (garch.index >= values.index[0])]
        for date in good_dates[[0, len(good_dates)//2, -1]]:
            history = r.loc[r.index < date].iloc[-50:].to_numpy()*100
            fit = arch_model(history, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
                disp="off", show_warning=False, options={"maxiter": 500})
            sigma = np.sqrt(fit.forecast(horizon=1, reindex=False).variance.values[-1, 0])/100
            np.testing.assert_allclose(sigma, garch.loc[date, "sigma"], rtol=1e-8, atol=1e-10)
        tf.keras.backend.clear_session()
    return {"scope": "3 fixed stocks first/last blocks plus all blocks for 2 observed failure cases; NumPy LSTM/scaler replay including extrema",
            "garch_refits_checked": 15, "neural_prediction_spot_checks": checks}


def dollar_ledger(returns, target):
    holdings, cash = np.zeros(len(returns.columns)), 1.
    rows = []
    for i in range(2, len(returns)):
        before = holdings.sum()+cash
        execution = returns.index[i-1]
        rebalance = i == 2 or execution.to_period("M") != returns.index[i-2].to_period("M")
        cost, traded = 0., 0.
        if rebalance:
            allocation = target.iloc[i-2].to_numpy()
            low, high = 0., before
            for _ in range(64):
                after = (low+high)/2
                fee = .001*np.abs(after*allocation-holdings).sum()
                if after+fee > before:
                    high = after
                else:
                    low = after
            new = ((low+high)/2)*allocation
            traded = np.abs(new-holdings).sum()
            cost = .001*traded
            cash = before-new.sum()-cost
            assert cash >= -1e-12
            holdings = new
        today = returns.iloc[i].to_numpy()
        assert np.isfinite(today[holdings > 0]).all()
        holdings *= 1+np.where(holdings > 0, today, 0.)
        after = holdings.sum()+cash
        rows.append([after/before-1, after, traded/before, cost/before])
    return pd.DataFrame(rows, index=returns.index[2:],
                        columns=["net_return", "nav", "traded_notional", "cost_fraction"])


def check_portfolios(run, source):
    garch = pd.read_csv(run/"garch_vol_df.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    hybrid = pd.read_csv(run/"hybrid_vol_df.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    cap = pd.read_csv(source/"mcap_weight_df.csv", index_col=0, parse_dates=True)
    log_returns = pd.read_csv(source/"log_ret_bt.csv", index_col=0, parse_dates=True)[garch.columns]
    realized = log_returns.rolling(21).std().loc[garch.index]
    simple = np.expm1(log_returns.loc[garch.index])
    checks = []
    for label, weights in (("cap", cap), ("equal", cap*0+1)):
        base = weights.where(realized.notna(), 0.)
        for model, forecast in (("base", None), ("garch", garch.where(np.isfinite(hybrid))), ("hybrid", hybrid)):
            if forecast is None:
                raw = base
            else:
                tilt = (realized/forecast.where(forecast > 0)).replace([np.inf, -np.inf], np.nan).fillna(1.).clip(.5, 1.5)
                raw = base*tilt
            desired = raw.div(raw.sum(axis=1), axis=0).fillna(0.)
            independent = dollar_ledger(simple, desired)
            saved = pd.read_csv(run/"evaluation"/f"{label}_{model}_monthly_10bps.csv", index_col=0, parse_dates=True)
            assert independent.index.equals(saved.index)
            for key in independent:
                np.testing.assert_allclose(independent[key], saved[key], rtol=1e-10, atol=1e-12)
            checks.append({"model": f"{label}_{model}", "dates": len(saved),
                           "max_daily_return_disagreement": float(abs(independent.net_return-saved.net_return).max())})
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    result = {"model_replay": check_models(args.run, args.source)}
    if not args.models_only:
        result["independent_portfolio_accounting"] = check_portfolios(args.run, args.source)
    destination = args.run/"independent-verification.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
