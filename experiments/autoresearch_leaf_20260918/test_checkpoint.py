"""Real synthetic fit -> durable checkpoint -> separate-process reload -> Host.
This is NOT a Leaf score or a target-container/GPU test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
try:
    import pandas as pd
    import sklearn
    from host import verify
except ImportError:
    sklearn=None
from core import sha, atomic_json

FIXTURE='''
from pathlib import Path
import os
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from budget_checkpoint import BudgetCheckpoint

def predict_checkpoint(directory, dataframe):
    return joblib.load(Path(directory)/'model.joblib').predict_proba(dataframe[['x']])

if __name__=='__main__':
    data=Path(os.environ['MLEVOLVE_PUBLIC_DATA'])
    budget=BudgetCheckpoint()
    fit=pd.read_csv(data/'fit.csv');tune=pd.read_csv(data/'tune.csv')
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(random_state=0))
    model.fit(fit[['x']],fit.species)
    def save_checkpoint(directory):joblib.dump(model,Path(directory)/'model.joblib')
    budget.record(family='synthetic_test_logistic',C=1.0,random_state=0,imputer_params=model[0].get_params())
    budget.publish(tune_probs=model.predict_proba(tune[['x']]),save_checkpoint=save_checkpoint,
                   predict_checkpoint=predict_checkpoint,step=1,metadata={'synthetic_fixture':True,'imputer_params':model[0].get_params()})
    budget.finish()
'''


@unittest.skipIf(sklearn is None,'requires isolated local sklearn test dependency')
class Tests(unittest.TestCase):
    def test_real_synthetic_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);data=r/'public';hidden=r/'hidden';art=r/'artifacts'
            for p in (data,hidden,art):p.mkdir()
            atomic_json(data/'schema.json',{'classes':['a','b'],'label_column':'species','input_columns':['id','x']})
            pd.DataFrame({'id':['f1','f2','f3','f4'],'x':[-2,-1,1,2],
                          'species':['a','a','b','b']}).to_csv(data/'fit.csv',index=False)
            pd.DataFrame({'id':['t1','t2'],'x':[-1.5,1.5],'species':['a','b']}).to_csv(data/'tune.csv',index=False)
            pd.DataFrame({'id':['s1','s2'],'x':[-.9,.9],'species':['a','b']}).to_csv(hidden/'search.csv',index=False)
            pd.read_csv(hidden/'search.csv').drop(columns='species').to_csv(data/'search.csv',index=False)
            pd.DataFrame({'id':['u1','u2'],'x':[-.5,.5]}).to_csv(data/'test.csv',index=False)
            train=r/'train.py';train.write_text(FIXTURE)
            now=time.monotonic()
            spec={'node_id':'synthetic','code_sha256':sha(train),'data_manifest_sha256':'synthetic',
                  'contract_sha256':'test-only','public_data':str(data),'artifact_root':str(art),
                  'node_started_monotonic':now,'training_deadline_monotonic':now+1200,
                  'execution_deadline_monotonic':now+1800}
            atomic_json(r/'spec.json',spec)
            env={**os.environ,'MLEVOLVE_PUBLIC_DATA':str(data),'MLEVOLVE_BUDGET_SPEC':str(r/'spec.json'),
                 'REPLAY_CANDIDATE':str(train),'PYTHONPATH':str(Path(__file__).parent)}
            for program in (train,Path(__file__).with_name('replay.py')):
                result=subprocess.run([sys.executable,str(program)],env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
            receipt=verify(art,data,hidden,spec,sha(hidden/'search.csv'))
            self.assertTrue(receipt['verified'])
            self.assertTrue(receipt['replay']['fresh_process'])


if __name__=='__main__':unittest.main()
