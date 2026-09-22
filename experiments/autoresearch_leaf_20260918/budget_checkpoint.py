"""Architecture-independent checkpoint ABI, shared by all generated candidates.

The Agent chooses models and writes training/prediction code. This module owns
deadlines, probability checks and atomic publication. It never opens host labels.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import numbers


def encode_parameter_fields(value, parameter_context=False):
    """Log parameter sentinels as typed tags; never sanitize scores or predictions."""
    if isinstance(value,dict):
        return {k:encode_parameter_fields(v,parameter_context or
            k in ('params','parameters') or str(k).endswith(('_params','_parameters')))
            for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [encode_parameter_fields(v,parameter_context) for v in value]
    if parameter_context and isinstance(value,numbers.Real) and not isinstance(value,bool):
        if not math.isfinite(value):
            return {'__parameter_scalar__':{'type':type(value).__module__+'.'+type(value).__qualname__,
                    'nonfinite':'nan' if math.isnan(value) else '+inf' if value>0 else '-inf'}}
        return int(value) if isinstance(value,numbers.Integral) else float(value)
    return value


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)



def task_schema():
    return json.loads((Path(os.environ.get('MLEVOLVE_PUBLIC_DATA','/data'))/'schema.json').read_text())

def classes():
    return task_schema()['classes']

def label_indices(frame):
    result=frame[task_schema()['label_column']].map({c:i for i,c in enumerate(classes())})
    if result.isna().any():raise ValueError('unknown class label')
    return result

def feature_frame(frame):
    return frame[task_schema()['input_columns']].copy()

def probabilities(value,n):
    import numpy as np
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(n,len(classes())) or not np.isfinite(value).all():raise ValueError('expected finite [N,C] probabilities')
    if (value<0).any() or (value>1).any() or not np.allclose(value.sum(1),1,atol=1e-5):raise ValueError('invalid probability simplex')
    return value


def loss(labels,probs):
    import numpy as np
    p=probabilities(probs,len(labels))
    p=np.clip(p,1e-15,1-1e-15);p/=p.sum(1,keepdims=True)
    return float(-np.log(p[np.arange(len(labels)),np.asarray(labels,dtype=int)]).mean())


class BudgetCheckpoint:
    def __init__(self):
        self.spec=json.loads(Path(os.environ['MLEVOLVE_BUDGET_SPEC']).read_text())
        self.root=Path(self.spec['artifact_root'])
        self.root.mkdir(parents=True,exist_ok=True)
        self.requested=False;self.last_publish_seconds=0
        self.best_tune=float('inf');self.published=False
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,lambda *unused:setattr(self,'requested',True))
        write_json(self.root/'STARTED.json',{'pid':os.getpid(),'node_id':self.spec['node_id'],
            'code_sha256':self.spec['code_sha256'],'started_monotonic':time.monotonic()})

    def remaining_seconds(self):
        return max(0,self.spec['execution_deadline_monotonic']-time.monotonic())

    def should_stop(self,next_step_seconds=0):
        next_step_seconds = 0 if next_step_seconds is None else float(next_step_seconds)
        reserve=max(600,2*self.last_publish_seconds)
        return (self.requested or (self.root/'STOP_TRAINING').exists()
            or time.monotonic()+max(0,next_step_seconds)>=self.spec['training_deadline_monotonic']
            or self.remaining_seconds()<=reserve+max(0,next_step_seconds))

    def record(self,**observations):
        # Append-only current-run observations, not outside memory.
        row={'elapsed_seconds':time.monotonic()-self.spec['node_started_monotonic'],**encode_parameter_fields(observations)}
        with (self.root/'training_events.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        print('TRAINING_EVENT '+json.dumps(row),flush=True)

    def publish(self,*,tune_probs,save_checkpoint,predict_checkpoint,step,metadata=None):
        """save_checkpoint(directory); predict_checkpoint(directory, feature_dataframe).

        Predictor MUST load the saved checkpoint and return columns from /data/schema.json.
        It receives only id and feature columns, never search/test labels. This works for any
        architecture, sklearn/boosting pipeline, ensemble, or custom neural net.
        """
        import numpy as np
        import pandas as pd
        data=Path(self.spec['public_data'])
        tune=pd.read_csv(data/'tune.csv')
        score=loss(label_indices(tune),tune_probs)
        self.record(step=step,tune_logloss=score,checkpoint_improved=score<self.best_tune)
        if score>=self.best_tune-1e-9:return False
        start=time.monotonic();name=f'checkpoint-{time.time_ns()}'
        tmp=self.root/(name+'.partial');tmp.mkdir()
        ckpt=tmp/'model';ckpt.mkdir()
        save_checkpoint(ckpt)
        files=[p for p in ckpt.rglob('*') if p.is_file()]
        if not files or not any(p.stat().st_size for p in files):raise ValueError('checkpoint writer saved no model')
        for p in files:
            if p.is_symlink():raise ValueError('checkpoint must be self-contained, no symlinks')
            with p.open('rb') as f:os.fsync(f.fileno())
        # Model is durable now. Predicting may still be interrupted; until the
        # whole bundle is committed, CURRENT continues pointing to the old best.
        reloaded_tune=probabilities(predict_checkpoint(ckpt,feature_frame(tune)),len(tune))
        if not np.allclose(reloaded_tune,tune_probs,rtol=1e-4,atol=1e-6):
            raise ValueError('saved checkpoint does not reproduce supplied tune probabilities')
        arrays={'tune_ids':tune.id.astype(str).to_numpy(dtype=str),'tune_probs':reloaded_tune}
        for split in ('search','test'):
            frame=pd.read_csv(data/f'{split}.csv')
            if list(frame.columns)!=task_schema()['input_columns']:raise ValueError('prediction input contains labels or extra fields')
            arrays[split+'_ids']=frame.id.astype(str).to_numpy(dtype=str)
            arrays[split+'_probs']=probabilities(predict_checkpoint(ckpt,frame.copy()),len(frame))
        np.savez_compressed(tmp/'predictions.npz',**arrays)
        pd.DataFrame({'id':arrays['test_ids'],**{c:arrays['test_probs'][:,i] for i,c in enumerate(classes())}}).to_csv(tmp/'submission.csv',index=False)
        receipt={k:self.spec[k] for k in ('node_id','code_sha256','data_manifest_sha256','contract_sha256')}
        receipt.update(step=step,tune_logloss=score,metadata=encode_parameter_fields(metadata or {}),
            files={str(p.relative_to(tmp)):sha(p) for p in tmp.rglob('*') if p.is_file()},
            publish_seconds=time.monotonic()-start)
        write_json(tmp/'receipt.json',receipt)
        for p in tmp.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
        tmp.rename(self.root/name)
        previous=json.loads((self.root/'CURRENT.json').read_text())['bundle'] if (self.root/'CURRENT.json').exists() else None
        write_json(self.root/'CURRENT.json',{'bundle':name})
        self.best_tune=score;self.published=True;self.last_publish_seconds=time.monotonic()-start
        # Migration: preserve every completed checkpoint and trial receipt.
        # CURRENT alone selects the tune incumbent; no evidence is deleted.
        return True

    def finish(self,reason='completed'):
        write_json(self.root/'TRAIN_OUTCOME.json',{'stop_reason':reason,
            'complete_checkpoint_present':(self.root/'CURRENT.json').exists(),
            'best_tune_logloss':self.best_tune if math.isfinite(self.best_tune) else None,
            'elapsed_seconds':time.monotonic()-self.spec['node_started_monotonic']})
