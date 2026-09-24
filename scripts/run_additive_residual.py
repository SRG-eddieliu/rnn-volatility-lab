"""Run the original additive LSTM against immutable corrected-v1 inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')

import numpy as np
import pandas as pd

from scripts.run_all_models import verify_code_signature
from scripts.run_corrected_experiment import CFG, save_json, sha
from src.models.rnn import RNNTrainingConfig, run_rolling_experiment

FLOOR = 1e-12
SOURCE_FILES = ['features.csv', 'splits.csv', 'prepared.json', 'input_signature.json',
                'garch_t_predictions.csv', 'pure_lstm_predictions.csv',
                'hybrid_lstm_predictions.csv', 'hybrid_log_ratio_lstm_predictions.csv',
                'pure_lstm/train_logs.csv']


def reconstruct_additive(baseline, correction, floor=FLOOR):
    base = np.asarray(baseline, dtype=float)
    delta = np.asarray(correction, dtype=float)
    if base.shape != delta.shape or base.ndim != 1:
        raise ValueError('Expected matching one-dimensional arrays.')
    if not np.isfinite(base).all() or not np.isfinite(delta).all() or (base <= 0).any():
        raise ValueError('Baseline must be finite and positive; correction must be finite.')
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('Floor must be finite and positive.')
    with np.errstate(over='ignore', invalid='ignore'):
        raw = base + delta
    if not np.isfinite(raw).all():
        raise ValueError('Additive reconstruction overflowed.')
    return raw, np.maximum(raw, floor)


def make_signature(source, cfg, split_ids):
    verify_code_signature(source)
    parent = json.loads((source / 'input_signature.json').read_text())
    return {
        'experiment': 'additive-residual-v1',
        'parent_input_signature': parent,
        'source_sha256': {name: sha(source / name) for name in SOURCE_FILES},
        'code_sha256': {name: sha(ROOT / name) for name in [
            *parent['code_sha256'], 'scripts/run_all_models.py',
            'scripts/run_additive_residual.py']},
        'configuration': asdict(cfg), 'split_ids': split_ids,
        'target': 'sq_return - garch_cond_var',
        'reconstruction': 'garch_cond_var + predicted_residual',
        'primary_floor': FLOOR,
    }


def run(source: Path, out: Path, max_splits: int | None = None):
    if source.resolve() == out.resolve():
        raise ValueError('Use a separate directory; corrected-v1 must remain unchanged.')
    if max_splits is not None and max_splits < 1:
        raise ValueError('max_splits must be positive.')
    frame = pd.read_csv(source / 'features.csv', parse_dates=['date'])
    split_cols = pd.read_csv(source / 'splits.csv', nrows=0).columns
    splits = pd.read_csv(source / 'splits.csv', parse_dates=[c for c in split_cols if c.endswith('_date')])
    if max_splits is not None:
        splits = splits.iloc[:max_splits].copy()
    cfg = RNNTrainingConfig(**{**asdict(CFG), 'target_transform': 'standardize'})
    signature = make_signature(source, cfg, splits.split_id.tolist())
    out.mkdir(parents=True, exist_ok=True)
    signature_path = out / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Inputs or code changed. Do not mix experiments; choose a new output directory.')
    if not signature_path.exists() and any(out.iterdir()):
        raise ValueError('Refusing to adopt nonempty output without an experiment signature.')
    save_json(signature_path, signature)
    complete_path = out / 'complete.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        for name, digest in complete['output_sha256'].items():
            if sha(out / name) != digest:
                raise ValueError(f'Completed artifact changed: {name}')
        print('Already complete and hashes verified.', flush=True)
        return

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ['TF_NUM_INTRAOP_THREADS']))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ['TF_NUM_INTEROP_THREADS']))
    tf.config.experimental.enable_op_determinism()
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    frame['residual_var'] = frame.sq_return - frame.garch_cond_var
    started = time.monotonic()
    pred, logs, gates = run_rolling_experiment(
        frame, splits, 'lstm', 'hybrid_residual', cfg, target_col='residual_var',
        output_activation='linear', capture_gates=True, resume=True,
        predictions_path=out / 'target_predictions.csv', train_logs_path=out / 'train_logs.csv',
        gates_path=out / 'gates.csv')
    expected_dates = []
    for split in splits.itertuples():
        expected_dates.extend(frame.loc[frame.date.between(split.test_start_date, split.test_end_date), 'date'])
    if list(pd.to_datetime(pred.date)) != list(pd.to_datetime(expected_dates)):
        raise ValueError('Forecast dates differ from the fixed test schedule.')
    if len(logs) != len(splits) or pred.date.duplicated().any():
        raise ValueError('Incomplete folds or duplicated predictions.')
    master = frame.set_index('date').loc[pd.to_datetime(pred.date)]
    np.testing.assert_allclose(pred.y_true_var, master.residual_var, rtol=1e-6, atol=1e-12)
    raw, clipped = reconstruct_additive(master.garch_cond_var, pred.y_pred_var)
    result = pd.DataFrame({
        'date': pd.to_datetime(pred.date), 'split_id': pred.split_id,
        'y_true_var': master.sq_return.to_numpy(),
        'garch_cond_var': master.garch_cond_var.to_numpy(),
        'y_true_residual': master.residual_var.to_numpy(),
        'y_pred_residual': pred.y_pred_var.to_numpy(),
        'y_pred_raw': raw, 'y_pred_var': clipped,
        'was_clipped': raw < FLOOR,
    })
    result.to_csv(out / 'predictions.csv', index=False)
    gates[gates.lag <= 20].groupby(['date', 'gate_name'], as_index=False).gate_value_mean.mean().to_csv(
        out / 'gate_summary.csv', index=False)
    complete = {
        'experiment': 'additive-residual-v1', 'splits': len(logs), 'n': len(pred),
        'seconds_this_invocation': time.monotonic() - started,
        'tensorflow': tf.__version__, 'signature_sha256': sha(signature_path),
        'output_sha256': {name: sha(out / name) for name in [
            'predictions.csv', 'target_predictions.csv', 'train_logs.csv', 'gates.csv', 'gate_summary.csv']},
    }
    save_json(complete_path, complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-splits', type=int, help='Smoke test only; use a separate directory.')
    args = parser.parse_args()
    run(args.source, args.out, args.max_splits)
