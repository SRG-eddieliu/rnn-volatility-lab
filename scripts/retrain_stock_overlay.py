"""Bounded, resumable stock GARCH/LSTM retraining; all raw outputs stay local."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.stock_overlay import StockConfig, rolling_stock_garch, train_stock_ratio

CODE = ("src/models/stock_overlay.py", "scripts/retrain_stock_overlay.py",
        "docs/stock-overlay-retraining.md", "tests/test_stock_overlay.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def verify_signature(out):
    sig = json.loads((out/"signature.json").read_text())
    for name, expected in sig["code_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Training code changed: {name}; use a new run directory.")
    for name, expected in sig["prepared_sha256"].items():
        if sha(out/name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    return sig


def prepare(source, out, tickers=None, start="2015-07-01", end="2024-12-31"):
    if (out/"signature.json").exists():
        raise ValueError("Run already prepared; omit --prepare to resume.")
    out.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(source/"log_ret_bt.csv", index_col=0, parse_dates=True)
    cap = pd.read_csv(source/"mcap_weight_df.csv", index_col=0, parse_dates=True)
    names = list(cap.columns) if tickers is None else tickers
    if len(set(names)) != len(names) or not set(names) <= set(cap.columns):
        raise ValueError("Unknown or duplicate stock selection.")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and ordered.")
    returns = returns.loc[:end, names]
    if np.isinf(returns.to_numpy()).any() or len(returns) < 182:
        raise ValueError("Invalid stock returns or insufficient history.")
    forecast_mask = (returns.index >= start) & (returns.index <= end)
    if not returns.index[forecast_mask].equals(cap.loc[start:end].index):
        raise ValueError("Forecast and portfolio date panels differ.")
    np.save(out/"returns.npy", returns.to_numpy(dtype=float))
    np.save(out/"dates.npy", returns.index.to_numpy())
    np.save(out/"forecast_mask.npy", forecast_mask)
    write_json(out/"tickers.json", names)
    signature = {
        "study": "stock-overlay-retrain-v1", "config": asdict(StockConfig()),
        "assets": len(names), "return_rows": len(returns), "forecast_rows": int(forecast_mask.sum()),
        "start": start, "end": end, "selection": "full supplied stock universe" if tickers is None else "explicit pilot subset",
        "source_sha256": {name: sha(source/name) for name in ("log_ret_bt.csv", "mcap_weight_df.csv")},
        "code_sha256": {name: sha(ROOT/name) for name in CODE},
        "prepared_sha256": {name: sha(out/name) for name in ("returns.npy", "dates.npy", "forecast_mask.npy", "tickers.json")},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "arch", "tensorflow", "keras", "scikit-learn")},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "machine": platform.machine(), "tensorflow_gpu": False},
        "data_limits": "Later survivor list and repeated later cap snapshot; retraining does not remove source bias.",
    }
    write_json(out/"signature.json", signature)
    print(json.dumps({"prepared": str(out), "assets": len(names), "forecast_rows": int(forecast_mask.sum())}), flush=True)


def worker(out, ticker):
    sig = verify_signature(out)
    names = json.loads((out/"tickers.json").read_text())
    column = names.index(ticker)
    folder = out/"stocks"/ticker
    folder.mkdir(parents=True, exist_ok=True)
    run_hash = sha(out/"signature.json")
    if (folder/"complete.json").exists():
        done = json.loads((folder/"complete.json").read_text())
        if done["signature_sha256"] != run_hash:
            raise ValueError("Stock checkpoint belongs to another experiment.")
        for name, expected in done["artifacts"].items():
            if sha(folder/name) != expected:
                raise ValueError(f"Stock output changed: {ticker}/{name}")
        return
    started = time.monotonic()
    returns = np.load(out/"returns.npy", mmap_mode="r")[:, column].copy()
    dates = pd.DatetimeIndex(np.load(out/"dates.npy"))
    forecast_mask = np.load(out/"forecast_mask.npy")
    cfg = StockConfig(**sig["config"])
    garch_file, garch_done = folder/"garch.csv", folder/"garch-complete.json"
    if garch_done.exists():
        checkpoint = json.loads(garch_done.read_text())
        if checkpoint["signature_sha256"] != run_hash or checkpoint["sha256"] != sha(garch_file):
            raise ValueError("GARCH checkpoint mismatch.")
        garch = pd.read_csv(garch_file, index_col=0, parse_dates=True, float_precision="round_trip")
        if not garch.index.equals(dates):
            raise ValueError("GARCH dates mismatch.")
        sigma = garch.sigma.to_numpy()
    else:
        sigma, flags, parameters = rolling_stock_garch(returns, cfg)
        garch = pd.DataFrame(parameters, index=dates, columns=["omega_pct2", "alpha", "beta", "nu"])
        garch["sigma"], garch["status"] = sigma, flags
        garch.to_csv(garch_file)
        write_json(garch_done, {"sha256": sha(garch_file), "signature_sha256": run_hash,
                               "seconds": time.monotonic()-started})
    predictions, fit_ids, records = train_stock_ratio(returns, sigma, dates, forecast_mask,
                                                     ticker, folder/"models", cfg)
    forecast = pd.DataFrame({"garch_sigma": sigma, "hybrid_sigma_raw": predictions,
                             "fit_id": fit_ids}, index=dates).loc[forecast_mask]
    forecast.to_csv(folder/"predictions.csv")
    write_json(folder/"training.json", records)
    artifacts = {str(path.relative_to(folder)): sha(path) for path in
                 [garch_file, folder/"predictions.csv", folder/"training.json", *sorted((folder/"models").glob("*"))]}
    write_json(folder/"complete.json", {
        "ticker": ticker, "signature_sha256": run_hash, "seconds": time.monotonic()-started,
        "fits": len(records), "forecast_count": int(np.isfinite(forecast.hybrid_sigma_raw).sum()),
        "nonpositive_forecasts": int((forecast.hybrid_sigma_raw <= 0).sum()),
        "garch_status": garch.status.value_counts().to_dict(), "artifacts": artifacts,
    })


def run(out, workers):
    sig = verify_signature(out)
    names = json.loads((out/"tickers.json").read_text())
    logs = out/"process_logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    def launch(ticker):
        with (logs/f"{ticker}.log").open("a") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(out),
                                     "--ticker", ticker], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            return {"ticker": ticker, "status": "failed", "returncode": result.returncode}
        return {"ticker": ticker, "status": "complete"}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, ticker) for ticker in names}
        completed = 0
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                completed += 1
                if row["status"] != "complete":
                    errors.append(row)
                print(json.dumps(row), flush=True)
            print(json.dumps({"completed": completed, "total": len(names), "errors": len(errors),
                              "elapsed_seconds": round(time.monotonic()-started)}), flush=True)
    if errors:
        write_json(out/"failures.json", errors)
        raise RuntimeError(f"{len(errors)} workers failed; inspect process logs, then resume.")
    panel_garch, panel_hybrid, audit = {}, {}, []
    for ticker in names:
        folder = out/"stocks"/ticker
        values = pd.read_csv(folder/"predictions.csv", index_col=0, parse_dates=True)
        panel_garch[ticker], panel_hybrid[ticker] = values.garch_sigma, values.hybrid_sigma_raw
        audit.append(json.loads((folder/"complete.json").read_text()))
    pd.DataFrame(panel_garch).to_csv(out/"garch_vol_df.csv")
    pd.DataFrame(panel_hybrid).to_csv(out/"hybrid_vol_df.csv")
    write_json(out/"complete.json", {"assets": len(names), "forecast_rows": sig["forecast_rows"],
                "signature_sha256": sha(out/"signature.json"), "seconds": time.monotonic()-started,
                "lstm_fits": sum(row["fits"] for row in audit),
                "panel_sha256": {name: sha(out/name) for name in ("garch_vol_df.csv", "hybrid_vol_df.csv")},
                "stock_receipts_sha256": {ticker: sha(out/"stocks"/ticker/"complete.json") for ticker in names}})
    print("ALL STOCKS COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--ticker")
    parser.add_argument("--start", default="2015-07-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use 1-8 bounded CPU workers.")
    out = args.out.resolve()
    if args.prepare:
        if args.source is None:
            parser.error("--prepare requires --source")
        prepare(args.source, out, args.tickers, args.start, args.end)
    elif args.ticker:
        worker(out, args.ticker)
    else:
        run(out, args.workers)
