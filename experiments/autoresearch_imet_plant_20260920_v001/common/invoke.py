import importlib.util,sys,json
from pathlib import Path
import numpy as np,pandas as pd
PLANT_CLASSES=['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']
mode,root,seconds=sys.argv[1:];root=Path(root)
spec=importlib.util.spec_from_file_location('candidate',root/'train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if mode=='train':
    params=m.train('/data',str(root/'artifacts'),float(seconds),0);(root/'actual_parameters.json').write_text(json.dumps(params,allow_nan=False))
else:
    arrays={}
    for split in ['tune','search','test']:
        df=pd.read_csv('/data/'+split+'.csv');p=np.asarray(m.predict(str(root/'artifacts'),df.path.astype(str).tolist()),dtype=float)
        expected=(len(df),12) if Path('/data/imet.flag').exists() is False else (len(df),3474)
        if expected[1]==12 and p.ndim==1 and p.shape==(len(df),):
            try:
                if np.issubdtype(p.dtype,np.number): ids=np.rint(p.astype(float)).astype(int)
                else: ids=np.array([int(float(x)) if str(x).strip().replace('.','',1).isdigit() else PLANT_CLASSES.index(str(x)) for x in p],dtype=int)
                if np.all((ids>=0)&(ids<12)):
                    q=np.zeros(expected,dtype=float); q[np.arange(len(df)),ids]=1.0; p=q
            except Exception: pass
        if p.shape!=expected or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('invalid predictions '+split)
        if expected[1]==12 and not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('not normalized')
        arrays[split]=p;arrays[split+'_ids']=df.id.to_numpy(dtype=str)
    np.savez(root/(mode+'.npz'),**arrays)
