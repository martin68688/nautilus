import importlib.util,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
mode,root,seconds=sys.argv[1:];root=Path(root)
spec=importlib.util.spec_from_file_location('candidate',root/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if mode=='train':
    params=module.train('/data',str(root/'artifacts'),float(seconds),0)
    (root/'actual_parameters.json').write_text(json.dumps(params,allow_nan=False))
else:
    arrays={}
    for split in ['tune','search','test']:
        df=pd.read_csv('/data/'+split+'.csv');p=np.asarray(module.predict(str(root/'artifacts'),df.text.fillna('').tolist()),dtype=float)
        if p.shape!=(len(df),3) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=1e-5,atol=1e-6):raise ValueError('invalid probabilities '+split)
        arrays[split]=p;arrays[split+'_ids']=df.id.to_numpy(dtype=str)
    np.savez(root/(mode+'.npz'),**arrays)
