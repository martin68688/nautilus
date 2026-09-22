import io,json,math,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core import RunState,sha,atomic_json
from provider import complete,BASE_URL
from budget_checkpoint import encode_parameter_fields,probabilities
from memory_bridge import MemoryBridge
from agent import propose

class Tests(unittest.TestCase):
    def test_raw_inner_and_outer_parse_failures_retained(self):
        for body in (b'{"x":1} extra',json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"action":"query"} {"action":"read"}'}}]}).encode()):
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'raw.bin'
                with patch.dict(os.environ,{'OPENAI_API_KEY':'fixture-not-a-key','OPENAI_BASE_URL':BASE_URL}),patch('urllib.request.urlopen',return_value=io.BytesIO(body)):
                    with self.assertRaises(json.JSONDecodeError):complete([],time.time()+30,raw_response_path=path)
                self.assertEqual(path.read_bytes(),body)
                self.assertEqual(path.stat().st_mode&0o777,0o600)
    def test_nonfinite_parameter_tags_do_not_relax_scores(self):
        from sklearn.impute import SimpleImputer
        params=SimpleImputer().get_params()
        encoded=encode_parameter_fields({'imputer_params':params})
        self.assertEqual(encoded['imputer_params']['missing_values']['__parameter_scalar__']['nonfinite'],'nan')
        self.assertTrue(math.isnan(params['missing_values']))
        json.dumps(encoded,allow_nan=False)
        with self.assertRaises(ValueError):json.dumps(encode_parameter_fields({'tune_logloss':float('nan')}),allow_nan=False)
        with patch('budget_checkpoint.classes',return_value=['a','b']):
            for x in (float('nan'),float('inf')):
                with self.assertRaises(ValueError):probabilities(np.array([[x,.5]]),1)
    def test_memory_read_required_rejection_permitted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'pack.json'
            atomic_json(p,{'schema':'mlevolve_search_experience_v1','data_manifest_sha256':'data','cards':[{'id':'a','lesson':'alpha prior','granularity':'architecture'}]})
            b=MemoryBridge({'search':{'path':str(p),'sha256':sha(p)}},'leaf-classification','data')
            replies=iter([{'action':'submit','code':'pass','change':'skip','decisions':[]},
                {'action':'query','terms':['alpha']},{'action':'read','keys':['search:a']},
                {'action':'submit','code':'pass','change':'independent hypothesis','decisions':[{'key':'search:a','action':'reject','reason':'different representation hypothesis','code_evidence':''}]}])
            code,_=propose(lambda m:(next(replies),{}),root,'contract','',[],b)
            self.assertEqual(code,'pass')
            self.assertTrue((root/'submission-rejected-00.json').exists())
            self.assertEqual(json.loads((root/'MEMORY_DECISIONS.json').read_text())['read_ids'],['search:a'])
    def test_amendment_carries_history_without_reset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);clock={'started':time.time()-100,'deadline':time.time()+100}
            old={'revision':'old'};new={'revision':'new'}
            s=RunState(root,old,200,clock=clock)
            f=s.new_attempt();(f/'train.py').write_text('pass')
            s.finish(f,{'returncode':0,'timeout':False},{'schema':'leaf_independent_host_v1','verified':True,'search_logloss':.4})
            original=(root/'LAUNCH.json').read_bytes()
            a={'source_launch_sha256':sha(root/'LAUNCH.json'),'from_contract':old,'to_contract':new,
               'original_started':s.launch['started'],'original_deadline':s.launch['deadline']}
            atomic_json(root/'AMENDMENT.json',a)
            resumed=RunState(root,new,200,clock={'started':s.launch['started'],'deadline':s.launch['deadline']})
            self.assertEqual(resumed.incumbent['score'],.4)
            self.assertEqual((root/'LAUNCH.json').read_bytes(),original)
            with self.assertRaises(ValueError):RunState(root,new,200,clock={'started':time.time(),'deadline':time.time()+7200})
            a['source_launch_sha256']='bad';atomic_json(root/'AMENDMENT.json',a)
            with self.assertRaises(ValueError):RunState(root,new,200,clock=clock)

if __name__=='__main__':unittest.main()
