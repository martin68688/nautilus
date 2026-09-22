"""Minimal actual-Leaf engineering smoke, not an A/B candidate or memory result."""
import json
from pathlib import Path
import socket
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from budget_checkpoint import BudgetCheckpoint, task_schema


def predict_checkpoint(directory, frame):
    saved=joblib.load(Path(directory)/'model.joblib')
    raw=saved['model'].predict_proba(frame[saved['features']])
    order=[list(saved['model'].classes_).index(c) for c in saved['classes']]
    return raw[:,order]


def main():
    budget=BudgetCheckpoint()
    for forbidden in ('/host-labels','/workspace','/etc/mlevolve','/var/run/docker.sock'):
        assert not Path(forbidden).exists(),forbidden
    assert not any(k in os.environ for k in ('OPENAI_API_KEY','AWS_SECRET_ACCESS_KEY','AWS_ACCESS_KEY_ID'))
    first_process=Path('/proc/1/environ').read_bytes() if os.access('/proc/1/environ',os.R_OK) else b''
    assert b'OPENAI_API_KEY=' not in first_process and b'AWS_SECRET_ACCESS_KEY=' not in first_process
    try:
        Path('/helpers/unauthorized-write').write_text('negative-probe')
    except OSError:
        pass
    else:
        raise AssertionError('immutable helper filesystem is writable')
    try:
        socket.create_connection(('1.1.1.1',443),timeout=1)
    except OSError:
        pass
    else:
        raise AssertionError('network isolation failed')
    import torch
    assert torch.cuda.is_available()
    assert 'A10G' in torch.cuda.get_device_name(0)
    value=torch.randn(256,256,device='cuda')
    value=value@value
    torch.cuda.synchronize()
    budget.record(gpu_probe=True,gpu=torch.cuda.get_device_name(0),finite=bool(torch.isfinite(value).all()))
    fit=pd.read_csv('/data/fit.csv');tune=pd.read_csv('/data/tune.csv')
    schema=task_schema();features=schema['feature_columns']
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=300,random_state=0))
    model.fit(fit[features],fit.species)
    state={'model':model,'features':features,'classes':schema['classes']}
    def save(directory):joblib.dump(state,Path(directory)/'model.joblib')
    live=model.predict_proba(tune[features])
    order=[list(model.classes_).index(c) for c in schema['classes']]
    budget.record(kind='engineering_smoke_only',actual_parameters=model[-1].get_params())
    budget.publish(tune_probs=live[:,order],save_checkpoint=save,
                   predict_checkpoint=predict_checkpoint,step=1,
                   metadata={'engineering_smoke_only':True,'actual_parameters':model[-1].get_params()})
    budget.finish()


if __name__=='__main__':main()
