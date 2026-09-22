"""Recompute only persisted formal R2 probabilities, no source code execution."""
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np,pandas as pd
from host import score
from core import sha,atomic_json
R=Path(__file__).resolve().parent
with tarfile.open(R/'input-closure.tar.gz') as t:
 schema=json.load(t.extractfile('public/schema.json'))
 hidden=t.extractfile('host-labels/search.csv').read()
 assert hashlib.sha256(hidden).hexdigest()=='ed0a0c20a8b9817a1b771acb98a741ca29683e7f2a36af6e5ccafd4dc316870c'
 frames={'search':pd.read_csv(io.BytesIO(hidden)),'tune':pd.read_csv(t.extractfile('public/tune.csv'))}
root=R/'r2-terminal-archive-v001/restored/round2-terra-r1memory-amended-001';rows=[]
for hp in sorted(root.glob('attempt-*/HOST.json')):
 h=json.loads(hp.read_text());pred=hp.parent/'artifacts'/h['bundle']/'predictions.npz'
 assert sha(pred)==h['predictions_sha256'];values=np.load(pred,allow_pickle=False);losses={}
 for split,frame in frames.items():
  assert np.array_equal(values[split+'_ids'],frame.id.astype(str))
  loss=score(frame.species,values[split+'_probs'],schema['classes'])
  assert abs(loss-h[split+'_logloss'])<1e-12;losses[split]=loss
 rows.append({'node':hp.parent.name,'losses':losses,'host_sha256':sha(hp),'predictions_sha256':sha(pred)})
assert len(rows)==8
atomic_json(R/'R2_HOST_RECOMPUTE.json',{'passed':True,'rows':rows,'no_source_code_or_pickle_executed':True,'new_experiment':False})
print(json.dumps(rows))
