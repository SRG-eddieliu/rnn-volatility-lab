"""Run each corrected model in its own bounded CPU process with separate logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MODELS=['pure:lstm','pure:gru','hybrid:lstm','hybrid:gru','hybrid_log_ratio:lstm','hybrid_log_ratio:gru']

def verify_code_signature(out, root=ROOT):
    expected=json.loads((Path(out)/'input_signature.json').read_text())['code_sha256']
    if not expected:
        raise ValueError('Missing prepared source-code signature.')
    for filename, digest in expected.items():
        current=hashlib.sha256((Path(root)/filename).read_bytes()).hexdigest()
        if current!=digest:
            raise ValueError(f'Source changed since preparation: {filename}. Use a new experiment directory.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=6: parser.error('workers must be between 1 and 6')
    out=args.out.resolve()
    if not (out/'prepared.json').exists(): parser.error('Prepare the input first.')
    verify_code_signature(out)
    (out/'process_logs').mkdir(exist_ok=True)
    env=dict(os.environ,TF_CPP_MIN_LOG_LEVEL='3')
    started=time.monotonic()
    def run(model):
        log=out/'process_logs'/f'{model.replace(":","_")}.log'
        command=[sys.executable,str(ROOT/'scripts/run_corrected_experiment.py'),'--out',str(out),'--model',model]
        with log.open('a') as stream:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{model} exited with {completed.returncode}; inspect {log}')
        return model
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run,model) for model in MODELS}
        while pending:
            done,pending=wait(pending,timeout=30,return_when='FIRST_COMPLETED')
            for f in done: print('COMPLETED '+f.result(),flush=True)
            progress={}
            for model in MODELS:
                name=model.replace(':','_')
                log=out/name/'train_logs.csv'
                count=max(0,sum(1 for _ in log.open())-1) if log.exists() else 0
                progress[name]={'splits':count,'complete':(out/name/'complete.json').exists()}
            print(json.dumps({'elapsed_seconds':round(time.monotonic()-started),'progress':progress}),flush=True)
    print('ALL SIX MODELS COMPLETE',flush=True)

if __name__=='__main__': main()
