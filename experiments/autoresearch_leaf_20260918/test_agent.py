import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from agent import propose
from memory_bridge import MemoryBridge
from sandbox import command


class Tests(unittest.TestCase):
    def test_lossless_real_reader_and_decision(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); card={'id':'a','granularity':'tuning','lesson':'alpha full text',
                'initial':{'C':1},'selected':{'C':2},'trials':[{'C':1,'loss':.3},{'C':2,'loss':.2}],
                'continuation':{'depends_on':'t001'},'limits':['no causal claim']}
            data={'schema':'mlevolve_search_experience_v1','task_id':'leaf-classification',
                  'data_manifest_sha256':'data','cards':[card]}
            f=p/'pack.json';f.write_text(json.dumps(data))
            b=MemoryBridge({'search':{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}},
                           'leaf-classification','data')
            replies=iter([{'action':'query','terms':['alpha'],'granularity':'tuning'},
                {'action':'read','keys':['search:a']},
                {'action':'submit','code':'C=2\n','change':'C change',
                 'decisions':[{'key':'search:a','action':'adapt','reason':'test only','code_evidence':'C=2'}]}])
            requests=[]
            def call(messages):
                requests.append(json.loads(json.dumps(messages)))
                return next(replies),{}
            code,_=propose(call,p,'JSON contract','',[],b)
            self.assertEqual(code,'C=2\n')
            transmitted=json.loads(requests[2][-1]['content'])
            self.assertEqual(transmitted['cards'][0]['body'],card)
            self.assertFalse(json.loads((p/'MEMORY_DECISIONS.json').read_text())['execution_verified'])

    def test_no_memory_rejects_query(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                propose(lambda _:({'action':'query','terms':['alpha']},{}),Path(d),'','','',None)

    def test_same_agent_repairs_mechanical_submission(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            replies=iter([{'action':'submit','code':'bad = (','change':'broken','decisions':[]},
                          {'action':'submit','code':'good = 1','change':'syntax corrected','decisions':[]}])
            messages=[]
            def call(m):
                messages.append(json.loads(json.dumps(m)))
                return next(replies),{}
            code,_=propose(call,p,'contract','',[],None)
            self.assertEqual(code,'good = 1')
            self.assertIn('validation_error',json.loads(messages[1][-1]['content']))
            self.assertTrue((p/'submission-rejected-00.json').exists())

    def test_docker_mount_surface(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            cmd=command('image@sha256:'+'a'*64,'leaf-test',p,p,p)
            text=' '.join(cmd)
            for forbidden in ('docker.sock','/host-labels','--privileged','--pid=host','--env-file'):
                self.assertNotIn(forbidden,text)
            for required in ('--network=none','--cap-drop=ALL','--read-only','--user=65534:65534'):
                self.assertIn(required,cmd)
            with self.assertRaises(ValueError): command('image:latest','leaf-test',p,p,p)


if __name__=='__main__': unittest.main()
