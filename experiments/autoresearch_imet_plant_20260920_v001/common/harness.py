import os,sys,json,time,hashlib,urllib.request,urllib.error,subprocess,signal,difflib,re,csv,math
from pathlib import Path
import numpy as np
from memory_bridge import MemoryBridge
from memory_decision_validation import repair_decisions,MemoryDecisionValidationError
R=Path(__file__).resolve().parent; ROOT=R.parent
TASK=os.environ['AUTORESEARCH_TASK']; DATA=Path(os.environ['AUTORESEARCH_DATA']).resolve(); LABELS=Path(os.environ['AUTORESEARCH_HOST_LABELS']).resolve(); MEMROOT=Path(os.environ['AUTORESEARCH_MEMORY_ROOT']).resolve()
MODEL='gpt-5.6-terra'; EFFORT='low'; BUDGET=int(os.environ.get('AUTORESEARCH_BUDGET_SECONDS','7200'))
CLASSES=['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']
HOST_INSTANCE=os.environ['AUTORESEARCH_INSTANCE_ID']

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):
    p=Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def env_endpoint():
    if os.environ.get('AUTORESEARCH_API_BASE') and os.environ.get('OPENAI_API_KEY'): return os.environ['AUTORESEARCH_API_BASE'].rstrip('/'),os.environ['OPENAI_API_KEY']
    raise RuntimeError('private endpoint environment missing')
def ask(messages,out,deadline):
    remain=deadline-time.monotonic()
    if remain<=5: raise TimeoutError('deadline before Agent call')
    base,key=env_endpoint(); out=Path(out); payload={'model':MODEL,'reasoning_effort':EFFORT,'messages':messages,'response_format':{'type':'json_object'}}
    save(out.with_suffix('.request.json'),{'model':MODEL,'effort':EFFORT,'messages':messages})
    req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    try:
        with urllib.request.urlopen(req,timeout=min(240,max(10,remain))) as resp:data=json.load(resp)
    except urllib.error.HTTPError as e:
        body=e.read().decode(errors='replace');save(out.with_suffix('.error.json'),{'status':e.code,'body':body[:2000]});raise
    save(out.with_suffix('.response.json'),data);content=data['choices'][0]['message']['content'];clean=content.strip()
    if clean.startswith('```') and clean.endswith('```'):
        lines=clean.splitlines();clean='\n'.join(lines[1:-1]).strip()
    try:ans=json.loads(clean)
    except Exception as e:
        save(out.with_suffix('.parse-error.json'),{'raw_sha256':sha(content.encode()),'error':str(e)});raise
    save(out.with_suffix('.parsed.json'),ans);return ans

def task_contract():
    if TASK=='plant':
        return '''You are the single research Agent in a strict autoresearch loop for Plant Seedlings Classification (12 species). Return JSON only. Propose one complete Python candidate with train(data_dir,artifact_dir,budget_seconds,seed) and predict(artifact_dir,paths). The Host passes relative image paths; candidate loads images lazily from /data/<path>. Use only fit.csv and fit labels for fitting; tune.csv/tune labels are for internal validation and model selection; search.csv and test.csv have no labels in the candidate view. The Host scores hidden search by macro-F1 of argmax predictions. Save a self-contained model and preprocessing under artifact_dir; predict must work in a fresh process and reload identically. No network, credentials, hidden labels, historical code/weights or other experiments. Generic torchvision pretrained weights are allowed only if declared in public_models and made available offline. Do real training, report architecture, selected parameters, seed, and internal validation trajectory. Do not fabricate results.'''
    return '''You are the single research Agent in a strict autoresearch loop for iMet 2020 FGVC7. Return JSON only. Propose one complete Python candidate with train(data_dir,artifact_dir,budget_seconds,seed) and predict(artifact_dir,paths). The Host passes relative image paths; candidate loads images lazily from /data/<path>. There are 3474 independent multilabel attributes. Use only fit.csv and fit_labels.npz for fitting; tune.csv/tune_labels.npz are calibration/model-selection only; search.csv/test.csv have no labels in the candidate view. The Host scores hidden search by micro-F1 on the 3474-column indicator matrix. Use one global threshold selected from 0.02..0.50 on tune only, record it, and use argmax fallback for empty rows. Do not flatten labels. Image-only lazy decoding; no full-corpus descriptor cache. Save a self-contained model and preprocessing under artifact_dir; predict must work in a fresh process and reload identically. No network, credentials, hidden labels, historical code/weights or other experiments. Generic torchvision pretrained weights are allowed only if declared in public_models and made available offline. Do real convergence-aware training, report architecture, selected parameters, seed, threshold and internal validation trajectory. Do not fabricate results.'''

def bridge():
    binding=json.loads((MEMROOT/'BINDING.json').read_text())
    return MemoryBridge({c:{'path':str(MEMROOT/(c+'_memory.json')),'sha256':binding['sha256'][c]} for c in ['search','inner']},'autoresearch-'+TASK,os.environ['AUTORESEARCH_DATA_MANIFEST_SHA256'])

def metrics(pred,split,ids):
    import pandas as pd
    expected=pd.read_csv(DATA/f'{split}.csv')
    if not np.array_equal(ids.astype(str),expected.id.astype(str).to_numpy()):raise ValueError('row identity mismatch '+split)
    if TASK=='plant':
        if pred.shape!=(len(expected),len(CLASSES)) or not np.isfinite(pred).all() or (pred<0).any() or (pred>1).any():raise ValueError('invalid plant probabilities')
        if not np.allclose(pred.sum(1),1,atol=1e-5):raise ValueError('plant probabilities not normalized')
        from sklearn.metrics import f1_score
        y=np.array(pd.read_csv(DATA/f'{split}.csv').label if split in ['fit','tune'] else pd.read_csv(LABELS).label if split=='search' else [])
        return None if split=='test' else float(f1_score(y,pred.argmax(1),average='macro'))
    if pred.shape!=(len(expected),3474) or not np.isfinite(pred).all() or (pred<0).any() or (pred>1).any():raise ValueError('invalid iMet probabilities')
    ys=np.load(DATA/f'{split}_labels.npz')['y'] if split in ['fit','tune'] else np.load(LABELS)['y'] if split=='search' else None
    if split=='test':return None
    from sklearn.metrics import f1_score
    # frozen host threshold for comparable scoring; candidate threshold is reported but host score is fixed.
    return float(f1_score(ys,pred>=0.20,average='micro'))

def validate_predictions(a,b):
    result={}
    for split in ['tune','search','test']:
        pa,pb=a[split],b[split]
        if pa.shape!=pb.shape or not np.allclose(pa,pb,atol=1e-6,rtol=1e-5):raise ValueError('reload mismatch '+split)
        ids=a[split+'_ids'];result[split+'_score']=metrics(pa,split,ids)
    result['search_score']=result['search_score'];return result

def propose(current,history,mem,out,deadline,initial=False):
    contract=task_contract();context={'task':TASK,'current_code':current,'history':history[-12:],'remaining_seconds':max(0,deadline-time.monotonic()),'data_manifest_sha256':os.environ['AUTORESEARCH_DATA_MANIFEST_SHA256']}
    cards={};readids=[]
    if mem:
        q=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps({'context':context,'memory_catalogue':mem.catalogue(),'request':'Return {"queries":[short literal substrings]} using exact catalogue substrings, not semantic sentences.'})}],out/'01-query',deadline)
        keys=mem.grep(q.get('queries',[]));save(out/'retrieval.json',{'queries':q.get('queries',[]),'matching_keys':keys})
        catalogue=[x for x in mem.catalogue() if x['key'] in keys and x['key'].startswith('search:')]
        sel=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps({'context':context,'matching_catalogue':catalogue,'request':'Return {"read_ids":[exact allowed search: keys]}; at most 6, unique.'})}],out/'02-select',deadline)
        readids=sel.get('read_ids',[]);readids=[k for k in readids if k in [x['key'] for x in catalogue]][:6]
        save(out/'02-select-final.json',{'read_ids':readids});cards=mem.read(readids);save(out/'full_cards.json',cards)
    request={'context':context,'historical_memory':cards,'selected_memory_keys':readids,'request':'Return {"change":string,"code":complete candidate Python,"budget_seconds":number,"public_models":["org/model"],"memory_decisions":[...]}. Include one decision for every selected key, using exact key; adopt/adapt requires exact code_evidence. If no memory, decisions must be [].'}
    ans=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps(request,ensure_ascii=False)}],out/'03-edit',deadline)
    if ans.get('stop'):return ans
    if not isinstance(ans.get('code'),str):raise ValueError('Agent did not return code')
    compile(ans['code'],'candidate.py','exec')
    if mem:
        ans,_=repair_decisions(ans,readids,cards,out/'03-memory-validation',deadline,ask,save,contract);save(out/'memory_receipt.json',mem.receipt(readids,ans['memory_decisions']))
    elif ans.get('memory_decisions',[])!=[]:raise ValueError('R1 must be no-memory')
    (out/'candidate.py').write_text(ans['code']);return ans

def identity():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    if len(out.splitlines())!=1 or 'A10G' not in out:raise RuntimeError('single A10G required')
    return {'instance_id':HOST_INSTANCE,'hardware':out}

def execute(code,attempt,out,seconds):
    # aws_backend is injected by the task wrapper, then calls this helper's docker runner.
    raise RuntimeError('backend not installed')

def run_arm(name,memory,budget=7200):
    (ROOT/'runs').mkdir(exist_ok=True);out=ROOT/'runs'/name;out.mkdir(parents=False,exist_ok=False);(out/'repo').mkdir();
    identity();start=time.monotonic();deadline=start+budget
    save(out/'LAUNCH.json',{'run_id':name,'task_id':'autoresearch-'+TASK,'round':int(os.environ['AUTORESEARCH_ROUND']),'started_unix':time.time(),'budget_seconds':budget,'model':MODEL,'effort':EFFORT,'memory':memory,'data_manifest_sha256':os.environ['AUTORESEARCH_DATA_MANIFEST_SHA256'],'runtime_manifest_sha256':os.environ['AUTORESEARCH_RUNTIME_MANIFEST_SHA256'],'metric':'host_micro_f1' if TASK=='imet' else 'host_macro_f1'})
    current='';history=[];best=None;best_id=None;mem=bridge() if memory else None
    for i in range(1000):
        if deadline-time.monotonic()<120:break
        a=out/f'attempt-{i:03d}';a.mkdir();attempt=f'{name}-{i:03d}';row={'attempt':attempt,'started_elapsed':time.monotonic()-start}
        try:
            proposal=propose(current,history,mem,a,deadline,initial=(best is None))
            if proposal.get('stop'):save(a/'stop.json',proposal);break
            save(a/'public_models.json',proposal.get('public_models',[]));code=proposal['code'];(a/'candidate.py').write_text(code)
            seconds=min(float(proposal.get('budget_seconds',600)),deadline-time.monotonic()-30)
            result=execute(code,attempt,a,seconds);row.update(result,change=proposal.get('change',''))
            keep=result['status']=='valid' and (best is None or result['search_score']>best)
            row['decision']='keep' if keep else 'discard'
            if keep:best=result['search_score'];best_id=attempt;current=code;(out/'incumbent.py').write_text(code)
            (out/'repo'/'train.py').write_text(current)
        except Exception as e:
            row.update(status='failed',search_score=None,reason=type(e).__name__+': '+str(e)[:700],decision='discard')
            for f in ['train.log','predict1.log','predict2.log','execution.json']:
                if (a/f).exists():pass
        row['elapsed_seconds']=time.monotonic()-start;history.append(row);save(a/'RESULT.json',row);save(out/'STATE.json',{'best':best,'best_attempt':best_id,'history':history,'incumbent_sha256':sha(current.encode())})
        print(json.dumps(row,ensure_ascii=False),flush=True)
    save(out/'RUN_OUTCOME.json',{'terminal':True,'best_search':best,'best_attempt':best_id,'elapsed_seconds':time.monotonic()-start,'attempts':len(history),'host_verified':bool(best_id)});return out
