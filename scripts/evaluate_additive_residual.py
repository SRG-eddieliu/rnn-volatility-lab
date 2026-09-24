"""Verify the additive follow-up and compare it with unchanged corrected-v1 forecasts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.evaluate_corrected_experiment import PERIODS, bootstrap_mean_intervals
from scripts.run_additive_residual import FLOOR, reconstruct_additive
from scripts.run_all_models import verify_code_signature
from scripts.run_corrected_experiment import CFG, sha, save_json
from src.losses import mse_loss, qlike_loss


def loss_summary(y, h):
    y, h = np.asarray(y, dtype=float), np.asarray(h, dtype=float)
    if y.shape != h.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError('Expected equally sized, nonempty one-dimensional arrays.')
    if not np.isfinite(y).all() or not np.isfinite(h).all() or (y < 0).any():
        raise ValueError('Invalid observed proxy or prediction.')
    mse = float(np.mean((y-h)**2))
    np.testing.assert_allclose(mse, mse_loss(y, h), rtol=1e-12)
    qlike = None
    if (h > 0).all():
        qlike = float(np.mean(np.log(np.maximum(h, FLOOR)) + np.maximum(y, FLOOR)/np.maximum(h, FLOOR)))
        np.testing.assert_allclose(qlike, qlike_loss(y, h), rtol=1e-12)
    return {'n': len(y), 'mse': mse, 'qlike': qlike,
            'mean_forecast_to_mean_proxy': float(h.mean()/y.mean()),
            'nonpositive_count': int((h <= 0).sum()), 'floor_count': int((h <= FLOOR).sum()),
            'min_forecast': float(h.min()), 'median_forecast': float(np.median(h)),
            'max_forecast': float(h.max())}


def evaluate(source: Path, out: Path, destination: Path):
    verify_code_signature(source)
    signature = json.loads((out/'signature.json').read_text())
    complete = json.loads((out/'complete.json').read_text())
    prepared = json.loads((source/'prepared.json').read_text())
    if complete['n'] != prepared['test_observations'] or complete['splits'] != prepared['splits']:
        raise ValueError('Only a complete, full-sample additive run may be reported.')
    if complete['signature_sha256'] != sha(out/'signature.json'):
        raise ValueError('Run signature changed after completion.')
    for name, digest in signature['source_sha256'].items():
        if sha(source/name) != digest:
            raise ValueError(f'Parent input changed: {name}')
    for name, digest in signature['code_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Training source changed: {name}')
    for name, digest in complete['output_sha256'].items():
        if sha(out/name) != digest:
            raise ValueError(f'Completed output changed: {name}')
    expected_cfg = vars(CFG).copy()
    expected_cfg['target_transform'] = 'standardize'
    if signature['configuration'] != expected_cfg or signature['primary_floor'] != FLOOR:
        raise ValueError('Experiment deviates from the fixed additive configuration.')
    expected_splits = pd.read_csv(source/'splits.csv')
    if signature['split_ids'] != expected_splits.split_id.tolist():
        raise ValueError('Unexpected split schedule.')

    # CSV emits short float32 decimals; restore the model output dtype before auditing sums.
    pred = pd.read_csv(out/'predictions.csv', parse_dates=['date'], float_precision='round_trip',
                       dtype={'y_pred_residual': np.float32}).set_index('date')
    if pred.index.has_duplicates or not pred.index.is_monotonic_increasing:
        raise ValueError('Prediction dates must be unique and increasing.')
    y = pred.y_true_var.to_numpy()
    raw, clipped = reconstruct_additive(pred.garch_cond_var, pred.y_pred_residual)
    np.testing.assert_allclose(raw, pred.y_pred_raw, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(clipped, pred.y_pred_var, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(pred.was_clipped, raw < FLOOR)
    residual_errors = (y-pred.garch_cond_var.to_numpy())-pred.y_pred_residual.to_numpy(dtype=float)
    np.testing.assert_allclose(residual_errors, y-raw, rtol=1e-9, atol=1e-17)
    np.testing.assert_allclose(pred.y_true_residual, y-pred.garch_cond_var.to_numpy(), rtol=1e-8, atol=1e-16)

    previous = json.loads((ROOT/'reports/corrected_results.json').read_text())
    series = {}
    rows = []
    for key, label in [('garch_t', 'GARCH-t'), ('pure_lstm', 'Pure LSTM'),
                       ('hybrid_lstm', 'Feature LSTM'), ('hybrid_log_ratio_lstm', 'Ratio LSTM')]:
        filename = f'{key}_predictions.csv'
        if sha(source/filename) != previous['forecast_sha256'][filename]:
            raise ValueError(f'Benchmark no longer matches published corrected-v1: {key}')
        benchmark = pd.read_csv(source/filename, parse_dates=['date']).set_index('date')
        if not benchmark.index.equals(pred.index):
            raise ValueError(f'Benchmark dates differ: {key}')
        np.testing.assert_allclose(benchmark.y_true_var, y, rtol=1e-12, atol=1e-15)
        if key == 'garch_t':
            np.testing.assert_allclose(benchmark.y_pred_var, pred.garch_cond_var, rtol=1e-12, atol=1e-15)
        series[key] = benchmark.y_pred_var.to_numpy()
        row = {'model': key, 'label': label, **loss_summary(y, series[key])}
        old = next(r for r in previous['metrics'] if r['model'] == key)
        np.testing.assert_allclose([row['mse'], row['qlike']], [old['mse'], old['qlike']], rtol=1e-12)
        rows.append(row)
    series['additive_raw'] = raw
    series['additive_clipped'] = clipped
    for key, label in [('additive_raw', 'Additive LSTM: raw'), ('additive_clipped', 'Additive LSTM: clipped')]:
        rows.append({'model': key, 'label': label, **loss_summary(y, series[key])})
    base_mse = rows[0]['mse']
    for row in rows:
        row['mse_change_pct_vs_garch'] = float(100*(row['mse']/base_mse-1))

    logs = pd.read_csv(out/'train_logs.csv').sort_values('split_id')
    baseline_logs = pd.read_csv(source/'pure_lstm/train_logs.csv').sort_values('split_id')
    for col in ['split_id', 'n_train', 'n_val', 'n_test']:
        np.testing.assert_array_equal(logs[col], baseline_logs[col])
    if not logs.target_transform.eq('standardize').all() or not logs.output_activation_used.eq('linear').all():
        raise ValueError('The additive model must use affine target standardization and linear output.')
    if not logs.epochs_ran.between(1, 35).all() or not logs.best_epoch.between(1, 35).all():
        raise ValueError('Invalid stopping metadata.')
    diagnostic_cols = ['checkpoint_train_loss', 'checkpoint_val_loss', 'checkpoint_gap_val_minus_train',
                       'target_mean', 'target_std']
    if not np.isfinite(logs[diagnostic_cols]).all().all() or (logs.target_std <= 0).any():
        raise ValueError('Invalid training diagnostics.')
    np.testing.assert_allclose(logs.checkpoint_val_loss, logs.best_val_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(logs.checkpoint_gap_val_minus_train,
                               logs.checkpoint_val_loss-logs.checkpoint_train_loss, rtol=1e-10, atol=1e-12)

    base = series['garch_t']
    base_mse_daily = (y-base)**2
    base_qlike_daily = np.log(base)+np.maximum(y, FLOOR)/base
    comparisons = [('additive_raw', 'mse'), ('additive_clipped', 'mse'), ('additive_clipped', 'qlike')]
    differences = np.column_stack([
        (y-series[key])**2-base_mse_daily if loss == 'mse' else
        np.log(series[key])+np.maximum(y, FLOOR)/series[key]-base_qlike_daily
        for key, loss in comparisons])
    bounds = bootstrap_mean_intervals(differences)
    intervals = [{'model': key, 'loss': loss, 'difference_vs_garch': float(differences[:, i].mean()),
                  'ci95_lower': float(bounds[0, i]), 'ci95_upper': float(bounds[1, i])}
                 for i, (key, loss) in enumerate(comparisons)]
    sensitivities = {}
    for block in [5, 63]:
        bound = bootstrap_mean_intervals(differences, block=block, reps=1000)
        sensitivities[str(block)] = {'ci95_lower': bound[0].tolist(), 'ci95_upper': bound[1].tolist()}

    periods = []
    for period, start, end in PERIODS:
        mask = pred.index.to_series().between(start, end).to_numpy()
        for key, values in series.items():
            periods.append({'period': period, 'model': key, **loss_summary(y[mask], values[mask])})
    floor_sensitivity = [{'floor': floor, **loss_summary(y, np.maximum(raw, floor))}
                         for floor in [FLOOR, 1e-10, 1e-8, 1e-6]]
    daily = pd.DataFrame({'split_id': pred.split_id.to_numpy(), 'garch_mse': base_mse_daily,
                          'additive_mse': (y-clipped)**2, 'difference': differences[:, 1]}, index=pred.index)
    folds = daily.groupby('split_id')[['garch_mse', 'additive_mse', 'difference']].mean()
    influential = np.argsort(np.abs(differences[:, 1]))[-10:][::-1]
    absolute_total = float(np.abs(differences[:, 1]).sum())
    concentration = {
        'definition': 'Ten largest absolute daily additive-clipped minus GARCH squared-loss differences; diagnostic selection only.',
        'top10_share_absolute_difference': float(np.abs(differences[influential, 1]).sum()/absolute_total) if absolute_total else 0.,
        'top10_contribution_to_mean_difference': float(differences[influential, 1].sum()/len(y)),
        'remaining_dates_mean_difference': float(np.delete(differences[:, 1], influential).mean()),
        'top_dates': [{'date': str(pred.index[i].date()), 'mse_difference': float(differences[i, 1])} for i in influential],
        'folds_with_lower_mse': int((folds.difference < 0).sum()), 'folds': len(folds),
    }
    training = {'folds': len(logs), 'min_sequences': int(logs.n_train.min()),
                'validation_observations': int(logs.n_val.iloc[0]), 'test_block': int(logs.n_test.iloc[0]),
                'mean_epochs': float(logs.epochs_ran.mean()),
                'mean_checkpoint_train_loss': float(logs.checkpoint_train_loss.mean()),
                'mean_checkpoint_val_loss': float(logs.checkpoint_val_loss.mean()),
                'mean_checkpoint_gap': float(logs.checkpoint_gap_val_minus_train.mean())}
    result = {
        'run': 'additive-residual-v1', 'parent_run': 'corrected-v1',
        'test_start': str(pred.index.min().date()), 'test_end': str(pred.index.max().date()), 'n': len(y),
        'metrics': rows, 'paired_intervals': intervals, 'calendar_periods': periods,
        'floor_sensitivity': floor_sensitivity, 'concentration': concentration,
        'training': training, 'signature': signature, 'output_sha256': complete['output_sha256'],
        'raw_residual_mse': float(np.mean(residual_errors**2)),
        'raw_nonpositive_count': int((raw <= 0).sum()), 'primary_clipped_count': int((raw < FLOOR).sum()),
        'bootstrap': {'method': 'paired stationary percentile', 'block': 21, 'repetitions': 2000,
                      'seed': 20260924, 'confidence': .95,
                      'comparison_order': [f'{key}:{loss}' for key, loss in comparisons],
                      'sensitivity_repetitions': 1000, 'block_sensitivity': sensitivities,
                      'scope': 'Pointwise, conditional on trained forecasts; not seed or selection-adjusted uncertainty.'},
        'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in
                         ['numpy', 'pandas', 'arch', 'tensorflow', 'keras']}},
        'evidence': {'parent_results_sha256': sha(ROOT/'reports/corrected_results.json'),
                     'plan_sha256': sha(ROOT/'docs/additive-residual-plan.md'),
                     'evaluation_code_sha256': sha(Path(__file__))},
        'verification': {'identical_dates_targets': True, 'parent_predictions_unchanged': True,
                         'training_dates_and_budget_match': True, 'residual_mse_identity': True,
                         'raw_and_clipped_outputs_retained': True, 'hashes_verified': True},
        'limits': ['Single-seed historical follow-up after prior inspection; not an untouched holdout.',
                   'Training MSE is affine-standardized residual MSE; raw reconstruction has no positivity guarantee.',
                   'A fixed numerical floor is not a statistical model; its QLIKE effects are disclosed.',
                   'No alpha, portfolio P&L, production validation, or multi-asset robustness is claimed.'],
    }
    save_json(destination, result)
    print(pd.DataFrame(rows)[['label', 'mse', 'qlike', 'mse_change_pct_vs_garch', 'nonpositive_count', 'floor_count']].to_string(index=False))
    print(json.dumps({'intervals': intervals, 'concentration': concentration}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.source, args.out, args.destination)
