"""Read-only container probe; no fitting, no split labels, no artifact writes."""
import json,runpy,pickle,math
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
root=Path('/readonly-attempt');current=json.loads((root/'artifacts/CURRENT.json').read_text())['bundle']
namespace=runpy.run_path(str(root/'train.py'),run_name='diagnostic_read_only')
predict=namespace['predict_checkpoint'];checkpoint=root/'artifacts'/current/'model'
frame=pd.read_csv('/data/search.csv');base=predict(checkpoint,frame)
altered_ids=frame.copy();altered_ids['id']=np.arange(len(frame))+1000000
by_id=predict(checkpoint,altered_ids)
order=np.arange(len(frame))[::-1];permuted=predict(checkpoint,frame.iloc[order].reset_index(drop=True))[::-1]
zero=frame.copy()
for c in zero.columns:
    if c!='id':zero[c]=0.0
changed=predict(checkpoint,zero)
with (checkpoint/'model.pkl').open('rb') as f:obj=pickle.load(f)
models=obj.get('models',[obj.get('model')]);models=[x for x in models if x is not None]
assert np.allclose(base,by_id) and np.allclose(base,permuted)
assert np.max(np.abs(base-changed))>1e-6
assert all(np.array_equal(m.classes_,np.arange(99)) for m in models)
assert obj['classes']==json.loads(Path('/data/schema.json').read_text())['classes']
print(json.dumps({'read_only_probe_passed':True,'id_changes_ignored':True,'row_permutation_equivariant':True,
    'features_changed_prediction_max_abs':float(np.max(np.abs(base-changed))),
    'serialized_model_types':[type(m).__module__+'.'+type(m).__name__ for m in models],
    'class_indices_exact_0_to_98':True,'class_names_equal_schema':True,'fit_only_scaler_n_samples':int(obj['scaler'].n_samples_seen_),
    'SimpleImputer_default_missing_values_is_nan':bool(math.isnan(SimpleImputer().get_params()['missing_values'])),
    'new_training':False}))
