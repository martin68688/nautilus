"""Sequential independent 7200s arms; launch only after online gates pass."""
import os,json,time,hashlib
from pathlib import Path
import harness as h
import aws_backend as aws
aws.install()
freeze=json.loads((h.R.parent/'FREEZE.json').read_text())
for relative,digest in freeze['files'].items():
 assert h.sha((h.R.parent/relative).read_bytes())==digest, 'frozen input mismatch '+relative
name=os.environ['AUTORESEARCH_PAIR_ID']
assert name.replace('-','').replace('_','').isalnum()
receipt=h.R/('PAIR_'+name+'.json')
if receipt.exists():raise RuntimeError('pair identity already used; explicit recovery required')
preflight=json.loads((h.R/'aws-preflight-v1/PREFLIGHT.json').read_text())
assert preflight['passed']
probe=h.R/('endpoint-probe-'+name);probe.mkdir()
reply=h.ask([{'role':'user','content':'Return JSON {"ok":true} only.'}],probe/'probe',time.monotonic()+180)
assert reply.get('ok') is True
files={p.name:h.sha(p.read_bytes()) for p in h.R.glob('*.py')}
state={'pair_id':name,'budget_per_arm':7200,'model':h.MODEL,'effort':h.EFFORT,'code_hashes':files,'data_manifest':h.DM,'resource':aws.identity(),'memory_hashes':{k:v['sha256'] for k,v in h.bridge().packs.items()},'arms':{},'terminal':False}
h.save(receipt,state)
for arm,mem in [('A',False),('B',True)]:
 state['active_arm']=arm;h.save(receipt,state)
 output=h.run_arm(name+'-'+arm,mem,7200)
 state['arms'][arm]=json.loads((output/'RUN_OUTCOME.json').read_text());h.save(receipt,state)
 if state['arms'][arm].get('provider_stopped'):break
state['terminal']=True;state['paired_arms_completed']=len(state['arms'])==2
h.save(receipt,state)
