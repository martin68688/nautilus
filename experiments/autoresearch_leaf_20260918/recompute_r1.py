"""Current independent score audit from frozen arrays; never loads source models."""
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np,pandas as pd
from core import atomic_json,sha
from host import score
R=Path(__file__).resolve().parent
with tarfile.open(R/'input-closure.tar.gz') as t:
    schema=json.load(t.extractfile('public/schema.json'))
    hidden=t.extractfile('host-labels/search.csv').read()
    assert hashlib.sha256(hidden).hexdigest()=='ed0a0c20a8b9817a1b771acb98a741ca29683e7f2a36af6e5ccafd4dc316870c'
    search=pd.read_csv(io.BytesIO(hidden));tune=pd.read_csv(t.extractfile('public/tune.csv'))
rows=[]
for receipt_file in (R/'r1-original-archive').glob('runs/*/workspace/working/budget_checkpoints/*/VERIFIED_RESULT.json'):
    v=json.loads(receipt_file.read_text());b=receipt_file.parent/Path(v['bundle']).name
    pred=b/'predictions.npz';assert sha(pred)==v['receipt']['files']['predictions.npz']
    values=np.load(pred,allow_pickle=False)
    for split,frame in [('search',search),('tune',tune)]:
        assert np.array_equal(values[split+'_ids'],frame.id.astype(str))
    search_score=score(search.species,values['search_probs'],schema['classes'])
    tune_score=score(tune.species,values['tune_probs'],schema['classes'])
    assert abs(search_score-v['metric'])<1e-12
    assert abs(tune_score-v['receipt']['tune_logloss'])<1e-12
    rows.append({'node':v['receipt']['node_id'],'host_search':search_score,'tune':tune_score,
        'receipt_sha256':sha(receipt_file),'predictions_sha256':sha(pred)})
assert len(rows)==2
atomic_json(R/'r1-distillation-v001/R1_HOST_RECOMPUTE.json',{'passed':True,'model_code_executed':False,
    'checkpoint_reexecution_claimed':False,'allow_pickle':False,'rows':rows,'script_sha256':sha(Path(__file__))})
print(json.dumps(rows))
