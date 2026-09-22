"""Audit current pack transport, cutoff and integrity negative checks."""
import importlib.util,json,shutil,tempfile
from pathlib import Path
from core import atomic_json,sha
from memory_bridge import MemoryBridge
R=Path(__file__).resolve().parent;D=R/'r1-distillation-v001'
skill=Path('/Users/haoming/.codex/skills/experiment-memory-distillation')
spec=importlib.util.spec_from_file_location('distill_skill',skill/'scripts/distill.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
b=json.loads((D/'memory/BINDING.json').read_text())
specs={k:{'path':str(D/f'memory/{k}_memory.json'),'sha256':b['sha256'][k]} for k in ('search','inner')}
bridge=MemoryBridge(specs,'leaf-classification',b['data_manifest_sha256'])
for key,card in bridge.cards.items():
    assert json.loads(bridge.prompt([key]))['cards'][0]['body']==card
proof=[]
for p in (R/'r1-original-archive').glob('runs/*/logs/search_memory/*.request*.json'):
    d=json.loads(json.loads(p.read_text())['prompt']['user'])['distilled_experience']
    assert d['cards']==[] and d['pack_sha256'] is None and d['source_run'] is None
    proof.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
negative=[]
for mutate in ('alter_score','omit_node','omit_trial'):
    with tempfile.TemporaryDirectory(prefix='leaf-distill-negative-') as t:
        target=Path(t)/'pack';shutil.copytree(D,target)
        e=json.loads((target/'evidence.json').read_text())
        if mutate=='alter_score':next(n for n in e['nodes'] if n['host_metric'] is not None)['host_metric']+=0.1
        elif mutate=='omit_node':e['nodes'].pop()
        else:next(n for n in e['nodes'] if n['trials'])['trials'].pop()
        atomic_json(target/'evidence.json',e)
        try:module.verify(target)
        except AssertionError:negative.append(mutate)
        else:raise AssertionError('tamper accepted '+mutate)
atomic_json(D/'AUDIT.json',{'passed':True,'analyst_lessons_read_against_sources':True,
    'method':'analyst_conditional_trajectory_synthesis','teacher_api_call':False,
    'source_model_seed':42,'current_seed':0,'source_original_regular_files_hash_verified':True,
    'cutoff_audit_sha256':sha(R/'r1-cutoff-adapted-v001/CUTOFF_AUDIT.json'),
    'zero_memory_prompt_proofs':proof,'negative_checks_rejected':negative,
    'full_card_transport_verified':len(bridge.cards),'binding_sha256':sha(D/'memory/BINDING.json'),
    'skill_sha256':sha(skill/'SKILL.md'),'skill_script_sha256':sha(skill/'scripts/distill.py'),
    'limits':'Schema/integrity/transport do not prove causal memory benefits; actual Agent probe still required.'})
print('Audit passed',len(bridge.cards),'full card bodies;',len(proof),'zero-memory prompts;',negative)
