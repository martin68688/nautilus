"""Explicit old training-stage -> distillation schema adapter; never modifies archive.

Stage records are NOT Agent tuning trials. Original host receipts remain byte-exact
in the archive; normalized copies only add linkage metadata for the skill schema.
"""
import copy, datetime, hashlib, json, os, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'r1-original-archive'
OUT=ROOT/'r1-cutoff-adapted-v001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def timestamp(x):
    return datetime.datetime.fromisoformat(x).replace(tzinfo=datetime.timezone.utc).timestamp()

def main():
    assert not OUT.exists(),'immutable new output required'
    remote="import pathlib,hashlib,json; r=pathlib.Path('/export/run-2h'); print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file() and not p.is_symlink()}))"
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
    hashes=json.loads(subprocess.check_output(['kubectl','exec','-n','ecepxie','leaf-r1-cutoff-export-20260918','--','python','-c',remote],env=env))
    for name,digest in hashes.items():assert sha(SOURCE/name)==digest,name
    launch=read(SOURCE/'control/LAUNCH.json');start=launch['started_unix'];cutoff=start+7200
    assert launch['memory_enabled'] is False and launch['memory_sources']==[]
    assert launch['search_memory_mode']=='control' and launch['model']=='gpt-5.6-luna'
    run=next((SOURCE/'runs').iterdir());journal=read(run/'logs/journal.json')
    state=read(run/'logs/linear_search_state.json');iterations={x['node_id']:x for x in state['iterations']}
    requests=list((run/'logs/search_memory').glob('*.request*.json'))
    prompt_proofs=[]
    for p in requests:
        d=read(p);u=json.loads(d['prompt']['user'])
        assert 'NO historical task memory' in u['task']
        # Any supplied history is this run's own feedback; external cards must be empty.
        for key in ('memory_cards','retrieved_cards','cards'):
            if key in u:assert not u[key],(p,key)
        prompt_proofs.append({'path':str(p.relative_to(SOURCE)),'sha256':sha(p),'keys':list(u)})
    accepted=[];excluded=[];transforms=[];scores=[]
    for n in journal['nodes']:
        if n['step']==0:continue
        if not n.get('finish_time') or timestamp(n['finish_time'])>cutoff:
            excluded.append(n['id']);continue
        assert start<=n['ctime']<=timestamp(n['finish_time'])<=cutoff
        accepted.append(n)
        nid=n['id'];it=iterations[nid]
        code=run/'logs/candidate_checkpoints'/f'{nid}.py'
        assert sha(code)==it['code_sha256']
        dest=OUT/'runs'/run.name
        target=dest/'logs/candidate_checkpoints'/code.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(code,target)
        cp=run/'workspace/working/budget_checkpoints'/nid
        dst=dest/'workspace/working/budget_checkpoints'/nid
        vp=cp/'VERIFIED_RESULT.json'
        events=[json.loads(x[len('TRAINING_EVENT '):]) for x in '\n'.join(n.get('_term_out') or []).splitlines() if x.startswith('TRAINING_EVENT ')]
        stages={x['stage']:x for x in events if 'stage' in x}
        outcomes=[x for x in events if 'checkpoint_improved' in x]
        if vp.exists():
            v=read(vp);assert v['host_recomputed'] and v['checkpoint_validated'] and v['returncode']==0
            assert v['metric']==n['metric']['value']==it['metric']
            bundle=cp/Path(v['bundle']).name
            for name,digest in v['receipt']['files'].items():assert sha(bundle/name)==digest
            assert int(bundle.name.split('-')[1])/1e9<=cutoff
            scores.append({'node':nid,'host_search':v['metric'],'finished_unix':timestamp(n['finish_time']), 'elapsed_seconds':timestamp(n['finish_time'])-start})
            params={}
            for e in outcomes:
                s=stages[e['step']];tid=f'{nid}:stage-{e["step"]}'
                par={k:s[k] for k in ('iterations','depth','learning_rate','l2_leaf_reg')}
                params[e['step']]=par
                write(dst/'inner_tuning'/f'trial-{e["step"]:03d}-outcome.json',{
                    'trial_id':tid,'node_id':nid,'code_sha256':sha(code),'action':'fresh_fit_training_stage',
                    'parameters':par,'status':'ok','tune_loss':e['tune_logloss'],'improved':e['checkpoint_improved'],
                    'elapsed_seconds':e['elapsed_seconds'],'parent_trial_id':None,
                    'metadata':{'observation_kind':'training_stage_not_agent_tuning','fresh_fit':True,
                                'source_event':e,'source_parameter_event':s,'source_journal_sha256':sha(run/'logs/journal.json')}})
            write(dst/'inner_tuning/SESSION.json',{'context':{'task':'leaf-classification','architecture':'CPU CatBoost MultiClass',
                'recipe':'fit-only median imputation, 192 numeric features, fresh CatBoost each iteration cap',
                'observation_kind':'training_stage_not_agent_tuning','source_model_seed':42,'data_split_seed':0},
                'parameters':{'observed_iteration_caps':[x['iterations'] for x in stages.values()],
                              'domain_not_recorded':True},'initial':params[1],'mutable':[]})
            normalized=copy.deepcopy(v);sel=v['receipt']['step']
            normalized['receipt']['metadata']['inner_trial_id']=f'{nid}:stage-{sel}'
            normalized['receipt']['metadata']['parameters']=params[sel]
            normalized['adapter_provenance']={'original_path':str(vp.relative_to(SOURCE)), 'original_sha256':sha(vp),
                'changes':'Only metadata inner_trial_id and parameters added from exact stage event; no score changed'}
            write(dst/'VERIFIED_RESULT.json',normalized)
            transforms.append(normalized['adapter_provenance'])
        else:assert n['metric']['value'] is None
    launch2=copy.deepcopy(launch);launch2.update(round=1,run_id=run.name,task_id='leaf-classification')
    write(OUT/'control/LAUNCH.json',launch2)
    write(OUT/'runs'/run.name/'logs/journal.json',{'nodes':accepted})
    outcome=read(SOURCE/'control/POD_PROCESS_OUTCOME.json')
    assert outcome['finished_unix']<=cutoff
    report={'source_root':str(SOURCE),'source_files':hashes,'copy_hashes_verified':True,
        'copied_symlinks':False,'symlink_note':'Only input /data aliases skipped; data separately hash-verified',
        'started_unix':start,'cutoff_unix':cutoff,'actual_elapsed_seconds':outcome['elapsed_seconds'],
        'ended_before_cutoff':True,'accepted_nodes':[x['id'] for x in accepted],'excluded_nodes':excluded,
        'host_scores':scores,'zero_memory_prompt_evidence':prompt_proofs,'transformations':transforms,
        'historical_model':'gpt-5.6-luna/low','historical_resources':'NRP A10,8CPU,32Gi; candidate CPU8',
        'new_model':'gpt-5.6-terra/low','new_resources':'AWS A10G,4CPU,14Gi,2Gi shm',
        'causal_memory_only_comparison':False,'later_rounds_included':False,'old_91_cards_included':False,
        'adapter_semantics':'11 fresh-fit training stages, NOT Agent parameter proposals; every completed node retained',
        'adapter_sha256':sha(Path(__file__))}
    write(OUT/'CUTOFF_AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ('actual_elapsed_seconds','host_scores','excluded_nodes')}))

if __name__=='__main__':main()
