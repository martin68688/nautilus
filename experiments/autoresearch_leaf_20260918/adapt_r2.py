"""Explicit autoresearch -> unchanged trajectory-skill schema adapter.
Preserve raw archives; one recorded publication is not every unlogged grid point.
"""
import json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
SOURCE=R/'r2-terminal-archive-v001/restored'
OUT=R/'r2-adapted-v001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 assert not OUT.exists()
 old=SOURCE/'round2-terra-r1memory-001';new=SOURCE/'round2-terra-r1memory-amended-001'
 clock=read(new/'CLOCK.json');assert clock==read(old/'CLOCK.json')
 assert clock['deadline']==1789779408.6579065 and (new/'SUPERVISOR.json').exists()
 overlaps=[]
 for p in sorted(old.glob('attempt-*')):
  for f in p.rglob('*'):
   if f.is_file():
    q=new/f.relative_to(old);assert q.is_file() and sha(f)==sha(q),str(f)
    overlaps.append(str(f.relative_to(old)))
 run=OUT/'runs/r2-autoresearch-two-epochs';nodes=[];transform=[];allhash={}
 for root in [old,new]:
  for f in root.rglob('*'):
   if f.is_file():allhash[str(f.relative_to(SOURCE))]=sha(f)
 for p in sorted(new.glob('attempt-*')):
  step=int(p.name.split('-')[1]);nid='r2_'+p.name
  result=read(p/'RESULT.json') if (p/'RESULT.json').exists() else None
  host=read(p/'HOST.json') if (p/'HOST.json').exists() else None
  if host:
   assert result and result['finished']<=clock['deadline'] and host['verified'] and host['replay']['passed']
   assert result['score']==host['search_logloss'] and sha(p/'HOST.json')==result['host_sha256']
  cp=run/'workspace/working/budget_checkpoints'/nid
  code=p/'train.py'
  if code.exists():
   dest=run/'logs/candidate_checkpoints'/f'{nid}.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(code,dest)
  md=read(p/'MEMORY_DECISIONS.json') if (p/'MEMORY_DECISIONS.json').exists() else {}
  error=(result or {}).get('error')
  clean_result={k:v for k,v in (result or {}).items() if k not in ('untrusted_training_log_tail','change')}
  partial_reads=[]
  for req in p.glob('agent-request-*.json'):
   for msg in read(req):
    if msg.get('role')=='user':
     try:body=json.loads(msg['content'])
     except (ValueError,TypeError):continue
     if isinstance(body,dict):
      for card in body.get('cards',[]):
       if isinstance(card,dict) and 'body' in card and 'key' in card:partial_reads.append(card['key'])
  analysis={'result':clean_result,'generation_interrupted_for_engineering_pause':step==11 and not result,
    'generation_interrupted_at_original_deadline':step>11 and not result,
    'partial_proposal_actual_full_read_keys':sorted(set(partial_reads)),
    'epoch':'original-v006' if step<=11 else 'amended-v007',
    'memory_read_ids':md.get('read_ids',[]),'full_read_recorded':bool(md.get('read_ids')),
    'score_role':'independent_host_search_not_tune','source_directory':str(p.relative_to(SOURCE)),
    'unscored_inherited_train_py_is_not_a_new_executed_candidate':not (p/'PROPOSAL.json').exists()}
  events=[json.loads(line) for line in (p/'artifacts/training_events.jsonl').read_text().splitlines()] if (p/'artifacts/training_events.jsonl').exists() else []
  records=[e for e in events if 'actual_parameters' in e]
  trial_ids=[]
  for j,e in enumerate(records,1):
   tid=f'{nid}:observation-{j}';trial_ids.append(tid)
   write(cp/'inner_tuning'/f'trial-{j:03d}-outcome.json',{'trial_id':tid,'node_id':nid,'code_sha256':sha(code),
    'action':'recorded_fresh_fit_publication_observation','parameters':e['actual_parameters'],'status':'ok',
    'tune_loss':e['observations']['tune_logloss'],'elapsed_seconds':e['elapsed_seconds'],
    'parent_trial_id':None,'improved':True,
    'metadata':{'observation_kind':'recorded_aggregate_not_each_grid_evaluation','unlogged_grid_scores_unknown':True,
      'source_event':e,'source_event_file_sha256':sha(p/'artifacts/training_events.jsonl')}})
  params=records[-1]['actual_parameters'] if records else {}
  write(cp/'inner_tuning/SESSION.json',{'context':{'task':'leaf-classification',
    'architecture':params.get('estimator','SVC descriptor classifier' if host and step==4 else 'unscored proposal'),
    'recipe':'See actual parameter record and analyst code audit; models fresh-fit; no source weights reused.',
    'source_epoch':analysis['epoch'],'memory_read_ids':analysis['memory_read_ids'],
    'unlogged_grid_scores_unknown':True},
    'parameters':{'observed_actual_parameters':params,'unrecorded_domain_unknown':True},'initial':params,'mutable':[]})
  if host:
   bundle=p/'artifacts'/host['bundle'];receipt=read(bundle/'receipt.json')
   assert receipt['code_sha256']==sha(code)
   for name,digest in receipt['files'].items():assert sha(bundle/name)==digest
   assert records and abs(records[-1]['observations']['tune_logloss']-host['tune_logloss'])<1e-10
   receipt['metadata']['inner_trial_id']=trial_ids[-1];receipt['metadata']['parameters']=params
   v={'host_recomputed':True,'checkpoint_validated':True,'metric':host['search_logloss'],'receipt':receipt,
     'adapter_provenance':{'host_sha256':sha(p/'HOST.json'),'original_receipt_sha256':sha(bundle/'receipt.json'),
      'changes':'normalized flags/metric and trial linkage only; original files retained'}}
   write(cp/'VERIFIED_RESULT.json',v);transform.append(v['adapter_provenance'])
  nodes.append({'id':nid,'step':step,'metric':{'value':host['search_logloss'] if host else None},
    'exec_time':(result or {}).get('process',{}).get('elapsed'),
    'exc_type':error['type'] if error else ('engineering_pause' if step==11 else 'original_deadline_generation_cutoff' if not result else 'execution_failure' if not host else None),
    'analysis':json.dumps(analysis,ensure_ascii=False)})
 write(run/'logs/journal.json',{'nodes':nodes})
 launch=read(old/'LAUNCH.json');amend=read(new/'AMENDMENT.json')
 write(OUT/'control/LAUNCH.json',{'round':2,'run_id':'r2-autoresearch-two-epochs','task_id':'leaf-classification',
  'data_manifest_sha256':'7bd8ce4887d28eea1bbe26730227ac4e49160332bd087571d7331110b5580a7f',
  'runtime_manifest_sha256':sha(new/'AMENDMENT.json'),'runtime_identity_note':'composite two-epoch identity bound by AMENDMENT; not a single unchanged runtime',
  'original_launch':launch,'amendment':amend,'clock':clock,'metric':'host_search_logloss'})
 write(OUT/'ADAPTER_AUDIT.json',{'source_files':allhash,'overlap_files_hash_identical':overlaps,
  'logical_nodes':len(nodes),'host_scored':sum(n['metric']['value'] is not None for n in nodes),
  'transformations':transform,'clock':clock,'supervisor':read(new/'SUPERVISOR.json'),
  'outcome':read(new/'OUTCOME.json') if (new/'OUTCOME.json').exists() else None,
  'engineering_pause_seconds':read(new/'RESUME_LAUNCH.json')['resumed_unix']-1789773728.5101762,
  'all_pause_and_provider_failure_time_in_original_budget':True,
  'engineering_probe_scores_included':False,'adapter_sha256':sha(Path(__file__))})
 print(json.dumps({'nodes':len(nodes),'host_scored':sum(n['metric']['value'] is not None for n in nodes)}))
if __name__=='__main__':main()
