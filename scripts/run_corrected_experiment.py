"""Reproducible CPU rerun on a supplied, immutable daily-price snapshot.

Run from the repository root. Generated market data and predictions stay local.
The published report uses aggregate results, never a silently shortened run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_DETERMINISTIC_OPS','1')

import numpy as np
import pandas as pd

from src.data.pipeline import add_features, compute_log_returns, generate_rolling_splits
from src.models.garch import rolling_garch_forecast
from src.models.rnn import RNNTrainingConfig, run_rolling_experiment, reconstruct_log_ratio_variance

MODELS = [('pure','lstm'),('pure','gru'),('hybrid','lstm'),('hybrid','gru'),
          ('hybrid_log_ratio','lstm'),('hybrid_log_ratio','gru')]
CFG = RNNTrainingConfig(epochs=35,unroll=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, content):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def prepare(raw_path: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    signature = {'raw_sha256':sha(raw_path),'configuration':asdict(CFG),
                 'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/models/garch.py',ROOT/'src/models/rnn.py',ROOT/'src/data/pipeline.py',Path(__file__)]}}
    existing = out/'input_signature.json'
    if existing.exists():
        if json.loads(existing.read_text()) != signature:
            raise ValueError('Inputs/configuration/code changed. Choose a new output directory; do not mix runs.')
        if (out/'prepared.json').exists(): return
    save_json(existing,signature)
    raw = pd.read_csv(raw_path,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if raw.date.duplicated().any(): raise ValueError('Duplicate source dates.')
    if not np.isfinite(raw.adj_close).all() or (raw.adj_close<=0).any(): raise ValueError('Invalid source prices.')
    frame=add_features(compute_log_returns(raw)).dropna(subset=['log_return']).reset_index(drop=True)
    diagnostics=[]
    frame['garch_cond_var']=rolling_garch_forecast(frame.set_index('date').log_return,
                                                 min_train_size=756,refit_every=21,diagnostics=diagnostics).to_numpy()
    frame['rv21_forecast']=frame.sq_return.rolling(21,min_periods=21).mean().shift(1)
    frame['ewma_forecast']=frame.sq_return.ewm(alpha=.06,adjust=False).mean().shift(1)
    frame['log_variance_ratio']=np.log(np.maximum(frame.sq_return,CFG.eps)/frame.garch_cond_var)
    common=frame.dropna(subset=['log_return','sq_return','abs_return','rv_21d','garch_cond_var',
                                'log_variance_ratio','rv21_forecast','ewma_forecast']).reset_index(drop=True)
    # Every architecture receives identical dates and at least 756 training sequences.
    splits=generate_rolling_splits(common,min_train_size=756+CFG.lookback,val_size=252,test_size=21,step_size=21)
    common.to_csv(out/'features.csv',index=False)
    splits.to_csv(out/'splits.csv',index=False)
    frame[['date','garch_cond_var']].dropna().to_csv(out/'garch_full_history.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'garch_fits.csv',index=False)
    test_dates=[]
    for row in splits.itertuples():
        test_dates.extend(common.loc[(common.date>=row.test_start_date)&(common.date<=row.test_end_date),'date'])
    if len(set(test_dates))!=len(test_dates): raise ValueError('Overlapping test dates.')
    test=common.set_index('date').loc[test_dates]
    for name,col in [('garch_t','garch_cond_var'),('rv21','rv21_forecast'),('ewma94','ewma_forecast')]:
        pd.DataFrame({'date':test.index,'y_true_var':test.sq_return.to_numpy(),'y_pred_var':test[col].to_numpy()}).to_csv(out/f'{name}_predictions.csv',index=False)
    info={'source_first_date':str(raw.date.min().date()),'source_last_date':str(raw.date.max().date()),
          'raw_rows':len(raw),'complete_feature_rows':len(common),'splits':len(splits),
          'test_observations':len(test),'test_start':str(test.index.min().date()),'test_end':str(test.index.max().date()),
          'garch_refits':len(diagnostics),'all_garch_fits_converged':True,
          'training_minimum_sequences':756,'validation_observations':252,'test_block':21,
          'garch_warmup':756,'ewma_lambda':.94,'base_seed':CFG.seed,'unroll':CFG.unroll,
          'method_change':'Equal availability after GARCH warmup; log-ratio hybrids replace additive residual hybrids.',
          'known_limits':['Single primary seed; no untouched prospective holdout.',
                          'Log-target MSE does not generally estimate arithmetic conditional mean variance.',
                          'Cached historical source snapshot; no new data download.']}
    save_json(out/'prepared.json',info)
    print(json.dumps(info),flush=True)


def run_model(out: Path, variant: str, architecture: str, max_splits: int|None=None, seed: int=42):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ['TF_NUM_INTRAOP_THREADS']))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ['TF_NUM_INTEROP_THREADS']))
    tf.config.experimental.enable_op_determinism()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    tf.get_logger().setLevel('ERROR')
    name=f'{variant}_{architecture}'
    frame=pd.read_csv(out/'features.csv',parse_dates=['date'])
    splits=pd.read_csv(out/'splits.csv',parse_dates=[c for c in pd.read_csv(out/'splits.csv',nrows=0).columns if c.endswith('_date')])
    if max_splits is not None: splits=splits.iloc[:max_splits].copy()
    model_out=out/name
    model_out.mkdir(exist_ok=True)
    target='log_variance_ratio' if variant=='hybrid_log_ratio' else 'sq_return'
    cfg=RNNTrainingConfig(**{**asdict(CFG),'seed':seed,'target_transform':'standardize' if target=='log_variance_ratio' else 'log_standardize'})
    signature={'configuration':asdict(cfg),'split_ids':splits.split_id.tolist(),'input_signature':json.loads((out/'input_signature.json').read_text())}
    signature_path=model_out/'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
        raise ValueError(f'Existing {name} output has a different configuration or split selection.')
    save_json(signature_path,signature)
    if (model_out/'complete.json').exists():
        print(f'Already complete: {name}',flush=True)
        return
    t=time.monotonic()
    pred, logs, gates=run_rolling_experiment(frame,splits,architecture,variant,cfg,target_col=target,
                                            output_activation='linear',capture_gates=True,
                                            predictions_path=model_out/'target_predictions.csv',
                                            train_logs_path=model_out/'train_logs.csv',gates_path=model_out/'gates.csv',resume=True)
    expected_dates=[]
    for row in splits.itertuples():
        expected_dates.extend(frame.loc[(frame.date>=row.test_start_date)&(frame.date<=row.test_end_date),'date'])
    if list(pd.to_datetime(pred.date))!=list(pd.to_datetime(expected_dates)):
        raise ValueError(f'Forecast coverage mismatch: {name}')
    master=frame.set_index('date').loc[pd.to_datetime(pred.date)]
    if variant=='hybrid_log_ratio':
        pred['predicted_log_ratio']=pred.y_pred_var
        pred['y_pred_var']=reconstruct_log_ratio_variance(pred.y_pred_var.to_numpy(),master.garch_cond_var.to_numpy())
    pred['y_true_var']=master.sq_return.to_numpy()
    if not np.isfinite(pred[['y_true_var','y_pred_var']]).all().all() or (pred.y_pred_var<=0).any():
        raise ValueError(f'Invalid reconstructed forecasts: {name}')
    pred.to_csv(out/f'{name}_predictions.csv',index=False)
    gate_summary=gates[gates.lag<=20].groupby(['date','gate_name'],as_index=False).gate_value_mean.mean()
    gate_summary.to_csv(model_out/'gate_summary.csv',index=False)
    info={'model':name,'n':len(pred),'splits':len(logs),'seconds_this_invocation':time.monotonic()-t,
          'tensorflow':tf.__version__,'numpy':np.__version__,'python':platform.python_version(),
          'checkpoint_gap_definition':'Inference-mode train and validation MSE at restored best-validation weights.',
          'forecast_sha256':sha(out/f'{name}_predictions.csv')}
    save_json(model_out/'complete.json',info)
    print(json.dumps(info),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--model',choices=[f'{v}:{a}' for v,a in MODELS])
    parser.add_argument('--max-splits',type=int,help='Smoke test only; use a separate output directory.')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.prepare:
        if args.raw is None: parser.error('--prepare requires --raw')
        prepare(args.raw,args.out)
    if args.model:
        variant,architecture=args.model.split(':')
        run_model(args.out,variant,architecture,args.max_splits,args.seed)


if __name__=='__main__': main()
