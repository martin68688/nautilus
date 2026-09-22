from pathlib import Path
import time,json,sys
import harness as h
out=h.R/(sys.argv[1] if len(sys.argv)>1 else 'agent-probe-v2');out.mkdir(exist_ok=False)
try:
 p=h.propose((h.R/'baseline.py').read_text(),[],h.bridge(),out,time.monotonic()+600)
 h.save(out/'PROBE_RESULT.json',{'actual_model':h.MODEL,'effort':h.EFFORT,'retrieval_decision_passed':True,'training_executed':False,'proposal_change':p.get('change'),'public_models':p.get('public_models',[]),'budget_seconds':p.get('budget_seconds')});print('actual retrieval and proposal passed; training not yet executed')
except Exception as e:
 h.save(out/'PROBE_RESULT.json',{'passed':False,'error_type':type(e).__name__,'error':str(e)[:500],'training_executed':False});raise
