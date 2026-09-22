"""Synthetic verifier regression only; never reported as Leaf training results."""
import json
import math
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import pandas as pd
    from host import score, verify
except ImportError:
    np = None
from core import sha, atomic_json


@unittest.skipIf(np is None, 'requires isolated local test dependencies')
class Tests(unittest.TestCase):
    def test_original_loss_semantics(self):
        self.assertAlmostEqual(score(['a','b'],[[.75,.25],[.2,.8]],['a','b']),
                               -(math.log(.75)+math.log(.8))/2)
        for value in ([[float('nan'),0]], [[.8,.8]], [[-1,2]]):
            with self.assertRaises(ValueError): score(['a'],value,['a','b'])

    def test_verifier_and_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);public=r/'public';hidden=r/'hidden';art=r/'artifacts';bundle=art/'checkpoint-1'
            for p in (public,hidden,bundle/'model'):p.mkdir(parents=True)
            atomic_json(public/'schema.json',{'classes':['a','b']})
            ids=['1','2'];probs=np.array([[.75,.25],[.2,.8]])
            pd.DataFrame({'id':ids,'species':['a','b']}).to_csv(public/'tune.csv',index=False)
            pd.DataFrame({'id':ids,'species':['a','b']}).to_csv(hidden/'search.csv',index=False)
            pd.DataFrame({'id':ids}).to_csv(public/'test.csv',index=False)
            (bundle/'model'/'dummy.txt').write_text('synthetic fixture, NOT a trained model')
            arrays={}
            for split in ('tune','search','test'):
                arrays[split+'_ids']=np.array(ids);arrays[split+'_probs']=probs.copy()
            np.savez_compressed(bundle/'predictions.npz',**arrays)
            pd.DataFrame({'id':ids,'a':probs[:,0],'b':probs[:,1]}).to_csv(bundle/'submission.csv',index=False)
            spec={'node_id':'test','code_sha256':'code','data_manifest_sha256':'data','contract_sha256':'contract'}
            receipt={**spec,'tune_logloss':score(['a','b'],probs,['a','b']),
                     'files':{str(f.relative_to(bundle)):sha(f) for f in bundle.rglob('*') if f.is_file()}}
            atomic_json(bundle/'receipt.json',receipt)
            atomic_json(art/'CURRENT.json',{'bundle':'checkpoint-1'})
            atomic_json(art/'REPLAY.json',{'passed':True,'bundle':'checkpoint-1'})
            result=verify(art,public,hidden,spec,sha(hidden/'search.csv'))
            self.assertTrue(result['verified'])
            with self.assertRaises(ValueError):verify(art,public,hidden,spec,'wrong')
            (bundle/'model'/'dummy.txt').write_text('tampered')
            with self.assertRaises(ValueError):verify(art,public,hidden,spec,sha(hidden/'search.csv'))


if __name__=='__main__':unittest.main()
