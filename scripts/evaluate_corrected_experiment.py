"""Verify complete rerun artifacts and publish aggregate results, not raw market data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.losses import mse_loss, qlike_loss
from scripts.run_all_models import verify_code_signature

LABELS={
    'garch_t':'GARCH-t','rv21':'Lagged RV21','ewma94':'EWMA 0.94',
    'pure_lstm':'Pure LSTM','pure_gru':'Pure GRU','hybrid_lstm':'Feature LSTM','hybrid_gru':'Feature GRU',
    'hybrid_log_ratio_lstm':'Ratio LSTM','hybrid_log_ratio_gru':'Ratio GRU',
}
PERIODS=[('2010-2016','2010-01-01','2016-12-31'),('2017-2019','2017-01-01','2019-12-31'),
         ('2020-2021','2020-01-01','2021-12-31'),('2022-2024','2022-01-01','2024-12-31')]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bootstrap_mean_intervals(values, block=21, reps=2000, seed=20260924):
    """Pointwise percentile CIs, paired across columns and stationary-block resampled across dates."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite observations-by-series matrix.')
    if block<=0 or reps<100: raise ValueError('Invalid bootstrap settings.')
    bootstrap=StationaryBootstrap(block,values,seed=seed)
    samples=bootstrap.apply(lambda x:x.mean(axis=0),reps=reps)
    return np.quantile(samples,[.025,.975],axis=0)


def evaluate(out: Path, destination: Path, vix_path: Path|None=None):
    verify_code_signature(out)
    prepared=json.loads((out/'prepared.json').read_text())
    inputs=json.loads((out/'input_signature.json').read_text())
    expected_n=prepared['test_observations']
    frames={}; hashes={}; rows=[]; years=[]; logs={}
    y=None; dates=None
    for key,label in LABELS.items():
        path=out/f'{key}_predictions.csv'
        if key not in ['garch_t','rv21','ewma94']:
            complete=json.loads((out/key/'complete.json').read_text())
            if complete['splits']!=prepared['splits'] or complete['n']!=expected_n:
                raise ValueError(f'Incomplete run: {key}')
            if complete['forecast_sha256']!=sha(path): raise ValueError(f'Changed forecast file: {key}')
            train=pd.read_csv(out/key/'train_logs.csv')
            if len(train)!=prepared['splits'] or train.split_id.duplicated().any(): raise ValueError(f'Invalid logs: {key}')
            loss_columns=['best_train_loss','best_val_loss','checkpoint_train_loss','checkpoint_val_loss',
                          'checkpoint_gap_val_minus_train','final_train_loss','final_val_loss']
            if not np.isfinite(train[loss_columns].to_numpy()).all(): raise ValueError(f'Non-finite training diagnostics: {key}')
            if not train.epochs_ran.between(1,35).all() or not (train.best_epoch<=train.epochs_ran).all():
                raise ValueError(f'Invalid stopping epoch: {key}')
            np.testing.assert_allclose(train.checkpoint_val_loss,train.best_val_loss,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(train.checkpoint_gap_val_minus_train,
                                       train.checkpoint_val_loss-train.checkpoint_train_loss,rtol=1e-10,atol=1e-12)
            if train.n_train.min()!=756 or train.n_val.nunique()!=1 or train.n_val.iloc[0]!=252:
                raise ValueError(f'Unequal training/validation availability: {key}')
            if train.n_test.nunique()!=1 or train.n_test.iloc[0]!=21: raise ValueError('Unexpected test block size.')
            logs[key]={'min_train':int(train.n_train.min()),'max_train':int(train.n_train.max()),
                       'median_epochs':float(train.epochs_ran.median()),'mean_epochs':float(train.epochs_ran.mean()),
                       'mean_checkpoint_train_loss':float(train.checkpoint_train_loss.mean()),
                       'mean_checkpoint_val_loss':float(train.checkpoint_val_loss.mean()),
                       'mean_checkpoint_gap':float(train.checkpoint_gap_val_minus_train.mean()),
                       'max_best_checkpoint_val_difference':float(abs(train.checkpoint_val_loss-train.best_val_loss).max())}
        frame=pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index()
        if len(frame)!=expected_n or frame.index.has_duplicates: raise ValueError(f'Coverage mismatch: {key}')
        if dates is None: dates=frame.index; y=frame.y_true_var.to_numpy()
        if not frame.index.equals(dates): raise ValueError(f'Date mismatch: {key}')
        np.testing.assert_allclose(frame.y_true_var,y,rtol=1e-12,atol=1e-15)
        h=frame.y_pred_var.to_numpy()
        if not np.isfinite(h).all() or (h<=0).any() or not np.isfinite(y).all() or (y<0).any():
            raise ValueError(f'Invalid forecast or target: {key}')
        mse=(y-h)**2
        qlike=np.log(np.maximum(h,1e-12))+np.maximum(y,1e-12)/np.maximum(h,1e-12)
        np.testing.assert_allclose(mse.mean(),mse_loss(y,h),rtol=1e-12)
        np.testing.assert_allclose(qlike.mean(),qlike_loss(y,h),rtol=1e-12)
        frame['mse_loss']=mse; frame['qlike_loss']=qlike
        frames[key]=frame; hashes[path.name]=sha(path)
        rows.append({'model':key,'label':label,'n':len(frame),'mse':float(mse.mean()),'qlike':float(qlike.mean()),
                     'mean_forecast':float(h.mean()),'mean_proxy':float(y.mean()),
                     'forecast_to_proxy_mean_ratio':float(h.mean()/y.mean()),
                     'min_forecast':float(h.min()),'median_forecast':float(np.median(h)),'max_forecast':float(h.max()),
                     'floor_count_1e12':int((h<=1e-12).sum()),'near_zero_count_1e8':int((h<=1e-8).sum())})
        for period,start,end in PERIODS:
            piece=frame.loc[start:end]
            if piece.empty: raise ValueError(f'Empty prespecified calendar period: {period}')
            years.append({'model':key,'period':period,'n':len(piece),'mse':float(piece.mse_loss.mean()),
                          'qlike':float(piece.qlike_loss.mean()),
                          'forecast_to_proxy_mean_ratio':float(piece.y_pred_var.mean()/piece.y_true_var.mean())})
    candidates=list(LABELS)[1:]
    loss_diffs=np.column_stack([frames[k][loss].to_numpy()-frames['garch_t'][loss].to_numpy()
                              for loss in ['mse_loss','qlike_loss'] for k in candidates])
    bounds=bootstrap_mean_intervals(loss_diffs)
    intervals=[]
    for j,loss in enumerate(['mse','qlike']):
        for i,key in enumerate(candidates):
            col=j*len(candidates)+i
            intervals.append({'model':key,'loss':loss,'difference_vs_garch':float(loss_diffs[:,col].mean()),
                              'ci95_lower':float(bounds[0,col]),'ci95_upper':float(bounds[1,col])})
    sensitivity={}
    for block in [5,63]:
        b=bootstrap_mean_intervals(loss_diffs,block=block,reps=1000)
        sensitivity[str(block)]={'ci95_lower':b[0].tolist(),'ci95_upper':b[1].tolist()}
    gate_stats=[]
    if vix_path is not None:
        vix=pd.read_csv(vix_path,parse_dates=['date'])
        if vix.date.duplicated().any(): raise ValueError('Duplicate VIX dates.')
        hashes['vix_daily.csv']=sha(vix_path)
        for key in list(LABELS)[3:]:
            gates=pd.read_csv(out/key/'gate_summary.csv',parse_dates=['date'])
            joined=gates.merge(vix,on='date',how='left',validate='many_to_one')
            if joined.vix_close.isna().any(): raise ValueError(f'Missing VIX matches: {key}')
            for gate,group in joined.groupby('gate_name'):
                if len(group)!=expected_n or group.date.duplicated().any(): raise ValueError('Gate coverage mismatch.')
                a=group.gate_value_mean; b=group.vix_close
                gate_stats.append({'model':key,'gate':gate,'n':len(group),
                                   'pearson':float(a.corr(b)),
                                   'spearman':float(a.rank().corr(b.rank()))})
    baseline=pd.read_csv(out/'garch_full_history.csv').garch_cond_var
    same=baseline.eq(baseline.shift())
    refits=pd.read_csv(out/'garch_fits.csv')
    environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','arch','tensorflow','keras']}
    result={'run':'corrected-v1','prepared':prepared,'input_signature':inputs,'environment':environment,
            'metrics':rows,'calendar_periods':years,'paired_loss_intervals':intervals,
            'bootstrap':{'method':'arch StationaryBootstrap; paired percentile intervals',
                         'confidence':.95,'average_block_length':21,'repetitions':2000,'seed':20260924,
                         'scope':'Pointwise temporal uncertainty conditional on trained forecasts; not multiple-testing adjusted or training-seed uncertainty.',
                         'sensitivity_repetitions':1000,'sensitivity_columns':[f'{k}:{loss}' for loss in ['mse','qlike'] for k in candidates],
                         'block_sensitivity':sensitivity},
            'garch_diagnostics':{'refits':len(refits),'all_converged':bool(refits.convergence_flag.eq(0).all()),
                                 'same_as_previous':int(same.sum()),'transitions':len(baseline)-1,
                                 'share_repeated':float(same.sum()/(len(baseline)-1)),
                                 'max_alpha_plus_beta':float((refits.alpha+refits.beta).max())},
            'training_diagnostics':logs,'gate_stats':gate_stats,'forecast_sha256':hashes,
            'verification':{'models_checked':len(frames),'unique_dates':True,'identical_dates':True,
                            'identical_targets':True,'finite_positive_forecasts':True,'metrics_reconciled':True,
                            'expected_test_count':expected_n}}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(rows)[['label','n','mse','qlike','forecast_to_proxy_mean_ratio','floor_count_1e12']].to_string(index=False))
    print(json.dumps(result['garch_diagnostics']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--vix',type=Path)
    a=parser.parse_args()
    evaluate(a.out,a.destination,a.vix)
