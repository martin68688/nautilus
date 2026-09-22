import pathlib,json,csv,math,hashlib
R=pathlib.Path(__file__).resolve().parents[2];O=pathlib.Path(__file__).resolve().parent
rows=[];excluded=[];sources={}
def load(p):
 sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def add(task,round,attempt,value,p,kind,index):
 assert math.isfinite(float(value)) and value>=0
 rows.append(dict(task=task,round=round,attempt=attempt,value=float(value),kind=kind,event_index=index,source=str(p.relative_to(R))))
for task,base in [('Spooky','experiments/autoresearch_spooky_20260918/r1-cutoff-v1/source'),('Leaf','experiments/autoresearch_leaf_20260918/r1-cutoff-adapted-v001')]:
 b=R/base;j=load(next(b.glob('runs/*/logs/journal.json')));order={n['id']:n['step'] for n in j['nodes']}
 pp=list(b.rglob('trial-*-outcome.json'));pp.sort(key=lambda p:(order[load(p)['node_id']],p.name))
 for p in pp:
  d=load(p);add(task,'R1',d['node_id'],d['tune_loss'],p,'historical_checkpoint_or_stage',d['trial_id'])
for rnd,base in [('R2','experiments/autoresearch_spooky_20260918/r2-final-audit/run'),('R3','experiments/autoresearch_spooky_20260918/lifecycle-return-v1/bundle/spooky-autoresearch-chain-r3-v1/code/runs/spooky-ar-r3-v1')]:
 for a in sorted((R/base).glob('attempt-*')):
  p=a/'actual_parameters.json'
  if not p.exists():continue
  d=load(p);tr=d.get('trajectory',d.get('inner_evaluation_trajectory',[]));n=0
  for i,t in enumerate(tr):
   stage=str(t.get('stage',t.get('setting',{}).get('stage','')))
   value=t.get('tune_logloss')
   if value is None and 'tune' in stage:value=t.get('logloss')
   if value is not None:add('Spooky',rnd,a.name,value,p,'tune_grid',i);n+=1
   else:excluded.append(dict(task='Spooky',round=rnd,attempt=a.name,source=str(p.relative_to(R)),event=t,reason='not explicit tune: fold/training score or unscored event'))
  # Add a final held-out evaluation only when no eligible inner tune is available; no duplicate final minima.
  hp=a/'HOST_RESULT.json'
  if not n and hp.exists():
   h=load(hp);add('Spooky',rnd,a.name,h['tune_loss'],hp,'final_reload_tune',0)
for rnd,base in [('R2','experiments/autoresearch_leaf_20260918/r2-terminal-archive-v001/restored/round2-terra-r1memory-amended-001'),('R3','experiments/autoresearch_leaf_20260918/lifecycle-return-v001/restored-v003/content/ebs-source/round3-terra-r1r2memory-amended-001')]:
 for a in sorted((R/base).glob('attempt-*')):
  p=a/'artifacts/training_events.jsonl'
  if not p.exists():continue
  sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
  events=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  # checkpoint publication rows are canonical; observations often repeat the same value.
  scored=[(i,t) for i,t in enumerate(events) if t.get('tune_logloss') is not None]
  for i,t in scored:add('Leaf',rnd,a.name,t['tune_logloss'],p,'checkpoint_tune',i)
  for i,t in enumerate(events):
   if t.get('tune_logloss') is None:excluded.append(dict(task='Leaf',round=rnd,attempt=a.name,event=t,reason='unscored training or duplicate observations payload'))
for task in ['Spooky','Leaf']:
 for rnd in ['R1','R2','R3']:
  sub=[x for x in rows if x['task']==task and x['round']==rnd];assert sub
  for i,x in enumerate(sub):x['ordinal']=i+1;x['progress_percent']=100*i/max(1,len(sub)-1)
  print(task,rnd,len(sub),min(x['value'] for x in sub),max(x['value'] for x in sub))
with (O/'tune_observations.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'EXCLUSIONS.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n');(O/'SOURCE_HASHES.json').write_text(json.dumps(sources,indent=2)+'\n')
