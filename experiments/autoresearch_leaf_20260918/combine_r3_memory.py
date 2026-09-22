"""Lossless cumulative R1/R2 packs after unchanged skill builds and audits."""
import importlib.util,json,shutil,tempfile
from pathlib import Path
from core import sha,atomic_json
from memory_bridge import MemoryBridge
R=Path(__file__).resolve().parent;OUT=R/'r3-cumulative-memory-v001'
assert not OUT.exists();OUT.mkdir()
skill=Path('/Users/haoming/.codex/skills/experiment-memory-distillation')
spec=importlib.util.spec_from_file_location('distill',skill/'scripts/distill.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sources=[R/'r1-distillation-v001',R/'r2-distillation-v001'];bindings=[];negative=[]
for d in sources:
 m.verify(d);bindings.append({'root':str(d),'binding_sha256':sha(d/'memory/BINDING.json'),'synthesis_sha256':sha(d/'SYNTHESIS.json')})
 for mutate in ['alter_score','omit_node','omit_trial']:
  with tempfile.TemporaryDirectory(prefix='leaf-r3-audit-') as tmp:
   target=Path(tmp)/'pack';shutil.copytree(d,target);e=json.loads((target/'evidence.json').read_text())
   if mutate=='alter_score':next(n for n in e['nodes'] if n['host_metric'] is not None)['host_metric']+=.1
   elif mutate=='omit_node':e['nodes'].pop()
   else:next(n for n in e['nodes'] if n['trials'])['trials'].pop()
   atomic_json(target/'evidence.json',e)
   try:m.verify(target)
   except AssertionError:negative.append(d.name+':'+mutate)
   else:raise AssertionError('tamper accepted')
hashes={};counts={}
for channel in ['search','inner']:
 packs=[json.loads((d/f'memory/{channel}_memory.json').read_text()) for d in sources]
 assert packs[0]['data_manifest_sha256']==packs[1]['data_manifest_sha256']
 cards=[c for p in packs for c in p['cards']];assert len(cards)==len({c['id'] for c in cards})
 combined={**packs[0],'source_run':'authorized-r1-r2-cumulative','source_run_id':'authorized-r1-r2-cumulative',
  'source_rounds':[p['source_run_id'] for p in packs],'source_runtime_sha256':None,
  'source_bindings':bindings,'transfer_scope':'Complete authorized R1 and R2 summaries/raw observations only; no engineering/probe/code/weights. Preserve each card source provenance.',
  'cards':cards}
 atomic_json(OUT/f'{channel}_memory.json',combined);hashes[channel]=sha(OUT/f'{channel}_memory.json');counts[channel]=len(cards)
bridge=MemoryBridge({k:{'path':str(OUT/f'{k}_memory.json'),'sha256':h} for k,h in hashes.items()},'leaf-classification',packs[0]['data_manifest_sha256'])
for key,card in bridge.cards.items():assert json.loads(bridge.prompt([key]))['cards'][0]['body']==card
binding={'source_rounds':[1,2],'sha256':hashes,'sources':bindings,'complete_node_coverage':True,
 'engineering_scores_excluded':True,'contains_source_code':False,'contains_source_weights':False,
 'counts':counts,'data_manifest_sha256':packs[0]['data_manifest_sha256'],
 'r2_adapter_audit_sha256':sha(R/'r2-adapted-v001/ADAPTER_AUDIT.json'),
 'skill_sha256':sha(skill/'SKILL.md'),'skill_script_sha256':sha(skill/'scripts/distill.py')}
atomic_json(OUT/'BINDING.json',binding)
atomic_json(OUT/'AUDIT.json',{'passed':True,'analyst_lessons_read_against_sources':True,
 'binding_sha256':sha(OUT/'BINDING.json'),'full_card_transport_verified':len(bridge.cards),
 'negative_checks_rejected':negative,'method':'analyst_conditional_trajectory_synthesis',
 'teacher_api_call':False,'limits':'Integrity and transport are not causal benefit or automatic semantic proof; actual R3 Agent probe remains required.'})
print(json.dumps(binding))
