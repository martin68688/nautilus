"""Independent AWS operational checks; no LLM calls and no formal run results."""
import os,json,time
from pathlib import Path
import harness as h
import aws_backend as aws
h.DATA=Path(os.environ['AUTORESEARCH_DATA']);h.LABELS=Path(os.environ['AUTORESEARCH_HOST_LABELS']);h.identity=aws.identity
assert h.sha((h.DATA/'manifest.json').read_bytes())==h.DM
for n,d in json.loads((h.DATA/'manifest.json').read_text())['files'].items():assert h.sha((h.DATA/n).read_bytes())==d
root=h.R/'aws-preflight-v1';root.mkdir(exist_ok=False)
results={}
for name,code,budget in [('baseline',(h.R/'baseline.py').read_text(),600),('gpu',(h.R/'gpu_smoke.py').read_text(),180),('nonzero','def train(*a):\n print("val_loss: 0.00001")\n raise RuntimeError("intentional nonzero")\ndef predict(*a): return None\n',65),('timeout','import time\ndef train(*a):\n time.sleep(120)\ndef predict(*a): return None\n',3)]:
 out=root/name;out.mkdir();result=aws.execute(code,'preflight-'+name,out,budget);results[name]=result;h.save(root/'RESULTS.json',results)
 if name in ['baseline','gpu']:assert result['status']=='valid' and result['reload_verified']
 else:assert result['status']=='failed' and result['search_loss'] is None
h.save(root/'PREFLIGHT.json',{'passed':True,'checks':results,'formal_training_started':False,'endpoint_checked':False})
