"""Read-only diagnostic of persisted probabilities, not a new experimental result."""
import io,json,tarfile
from pathlib import Path
import numpy as np,pandas as pd
from host import score
from core import sha,atomic_json
R=Path(__file__).resolve().parent;D=R/'diagnostic-r2-attempts1-5'
with tarfile.open(R/'input-closure.tar.gz') as t:
    schema=json.load(t.extractfile('public/schema.json'))
    frames={'search':pd.read_csv(t.extractfile('host-labels/search.csv')),
            'tune':pd.read_csv(t.extractfile('public/tune.csv'))}
classes=schema['classes'];mapping={x:i for i,x in enumerate(classes)}
rows=[]
for a in sorted(D.glob('attempt-*')):
    hp=a/'HOST.json'
    if not hp.exists():continue
    h=json.loads(hp.read_text());f=a/'artifacts'/h['bundle']/'predictions.npz'
    assert sha(f)==h['predictions_sha256']
    v=np.load(f,allow_pickle=False);stats={}
    for split,frame in frames.items():
        assert np.array_equal(v[split+'_ids'],frame.id.astype(str))
        p=v[split+'_probs'];y=frame.species.map(mapping).to_numpy()
        loss=score(frame.species,p,classes);assert abs(loss-h[split+'_logloss'])<1e-12
        stats[split]={'logloss':loss,'accuracy':float((p.argmax(1)==y).mean()),
            'mean_max_probability':float(p.max(1).mean()),'mean_true_probability':float(p[np.arange(len(y)),y].mean()),
            'geomean_true_probability':float(np.exp(-loss)),
            'entropy':float(-(p*np.log(np.maximum(p,1e-15))).sum(1).mean()),
            'true_probability_quantiles':np.quantile(p[np.arange(len(y)),y],[0,.25,.5,.75,1]).tolist()}
    rows.append({'attempt':a.name,'code_sha256':sha(a/'train.py'),'stats':stats,'host_sha256':sha(hp)})
atomic_json(D/'PROBABILITY_DIAGNOSTIC.json',{'diagnostic_only':True,'n_classes':99,
    'uniform_logloss':float(np.log(99)),'new_training':False,'rows':rows})
print(json.dumps(rows,indent=2))
