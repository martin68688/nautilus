"""Copy a quiescent R2 into a versioned continuation, preserving original clock.
Never edits original LAUNCH, CLOCK, attempts, incumbent or fitted artifacts.
"""
import json,shutil,subprocess,sys,time
from pathlib import Path
from core import sha,atomic_json,RunState,verify_runtime
ROOT=Path('/workspace/autoresearch-leaf-20260918')
SRC=ROOT/'round2-terra-r1memory-001';DST=ROOT/'round2-terra-r1memory-amended-001'
CODE=ROOT/'amendment-v007/code';UNIT='leaf-autoresearch-r2-terra-20260918.service'
manifest_tool=ROOT/'transfer_manifest.py'
assert not DST.exists()
unit_state=subprocess.run(['systemctl','is-active',UNIT],capture_output=True,text=True)
assert unit_state.stdout.strip() in ('inactive','failed')
q=json.loads((ROOT/'R2_QUIESCENCE_001.json').read_text());assert not q['active_training_interrupted']
clock=json.loads((SRC/'CLOCK.json').read_text());launch=json.loads((SRC/'LAUNCH.json').read_text());state=json.loads((SRC/'STATE.json').read_text())
assert clock['started']==1789772208.6579065 and clock['deadline']==1789779408.6579065
assert clock['deadline']>time.time()
assert not subprocess.check_output(['docker','ps','-q','--filter','label=autoresearch.leaf.run='+clock['run_label']],text=True).strip()
verify_runtime(CODE,json.loads((CODE/'RUNTIME_MANIFEST.json').read_text()))
manifest=ROOT/'R2_ORIGINAL_QUIESCENT_MANIFEST.json'
assert not manifest.exists()
subprocess.run([sys.executable,str(manifest_tool),'create',str(SRC),'--output',str(manifest)],check=True)
shutil.copytree(SRC,DST,symlinks=True)
subprocess.run([sys.executable,str(manifest_tool),'verify',str(DST),str(manifest)],check=True)
identity={'contract_sha256':sha(CODE/'CONTRACT.json'),'image':launch['contract']['image'],
    'runtime_sha256':sha(CODE/'RUNTIME_MANIFEST.json'),'memory_sha256':sha(CODE/'memory_specs.json')}
labels={}
for folder in sorted(SRC.glob('attempt-*')):
    f=folder/'MEMORY_DECISIONS.json'
    d=json.loads(f.read_text()) if f.exists() else {}
    labels[folder.name]={'full_card_read_ids':d.get('read_ids',[]),
        'memory_consumption':'full_card_read' if d.get('read_ids') else 'catalogue_only_or_uncompleted_generation',
        'decision_receipt_sha256':sha(f) if f.exists() else None,
        'completed_result':(folder/'RESULT.json').exists()}
amendment={'schema':'leaf_r2_protocol_amendment_v1','authorized_by':'coordinator task 01a0b67d-5a9d-7a50-99d7-5b739733d68e explicit repair authorization',
    'source_root':str(SRC),'continuation_root':str(DST),'source_launch_sha256':sha(SRC/'LAUNCH.json'),
    'source_clock_sha256':sha(SRC/'CLOCK.json'),'source_state_sha256':sha(SRC/'STATE.json'),
    'source_manifest_sha256':sha(manifest),'from_contract':launch['contract'],'to_contract':identity,
    'original_started':clock['started'],'original_deadline':clock['deadline'],'amended_unix':time.time(),
    'changes':['Each proposal query/read >=1 full relevant card; decisions may reject/defer; no architecture forced',
        'Persist raw API response before any JSON parse; historical failed raw responses unrecoverable',
        'Typed nonfinite parameter logging only; metrics/probability finite guards unchanged'],
    'old_attempt_memory_consumption':labels,'inherited_incumbent':state['incumbent'],
    'budget_reset':False,'old_results_invalidated':False,'old_weights_used_to_seed_new_training':False,
    'new_attempts_use_incumbent_code_only':True}
atomic_json(DST/'AMENDMENT.json',amendment)
resumed=RunState(DST,identity,7200,clock=clock)
assert resumed.incumbent==state['incumbent']
assert sha(DST/'LAUNCH.json')==sha(SRC/'LAUNCH.json') and sha(DST/'CLOCK.json')==sha(SRC/'CLOCK.json')
atomic_json(ROOT/'R2_AMENDMENT_PREPARED.json',{'passed':True,'root':str(DST),'amendment_sha256':sha(DST/'AMENDMENT.json'),
    'history_count':len(resumed.history),'incumbent':resumed.incumbent,'clock':clock,'remaining_seconds':clock['deadline']-time.time()})
print('Prepared continuation; original deadline and incumbent preserved')
