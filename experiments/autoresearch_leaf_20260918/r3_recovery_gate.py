"""One readiness request after tests; never launch/retry against failed service."""
import json,time
from pathlib import Path
from core import sha,atomic_json,verify_runtime
from provider import configure,complete
R=Path('/workspace/autoresearch-leaf-20260918');C=R/'r3-v011/code';OLD=R/'r3-v010/code'
tests=json.loads((R/'R3_V011_TESTS.json').read_text())
assert tests['passed'] and 'Ran 32 tests' in tests['stdout']+tests['stderr']
assert tests['runtime_manifest_sha256']==sha(C/'RUNTIME_MANIFEST.json')
verify_runtime(C,json.loads((C/'RUNTIME_MANIFEST.json').read_text()))
unchanged=['agent.py','provider.py','action_protocol.py','budget_checkpoint.py','host.py','replay.py','sandbox.py','memory_bridge.py','memory_specs.json','task.md','supervise.py','core.py']
for name in unchanged:assert sha(C/name)==sha(OLD/name),name
clock=json.loads((R/'round3-terra-r1r2memory-amended-001/CLOCK.json').read_text())
assert clock['deadline']==1789787716.7019622 and clock['deadline']-time.time()>60
path=R/'R3_RECOVERY_READINESS_001.json';raw=R/'r3-recovery-readiness-001.raw'
assert not path.exists() and not raw.exists()
configure('/etc/autoresearch/apizh-terra-low.env','https://apizh-ai.com/v1')
started=time.time()
try:
 reply,usage=complete([{'role':'user','content':'Readiness check only. Return exactly one JSON object: {"action":"query","terms":["health"]}'}],min(time.time()+45,clock['deadline']),raw_response_path=raw)
 assert usage['model_returned']=='gpt-5.6-terra'
 receipt={'passed':True,'reply':reply,'usage':usage,'started':started,'finished':time.time(),'raw_response_sha256':sha(raw),'formal_score':False}
except Exception as e:
 receipt={'passed':False,'error_type':type(e).__name__,'error':str(e),'started':started,'finished':time.time(),
  'raw_response_sha256':sha(raw) if raw.exists() else None,'next_check_not_before':time.time()+900,'formal_score':False}
atomic_json(path,receipt);print(json.dumps(receipt))
if receipt['passed']:
 gate=json.loads((R/'R3_GATES.json').read_text())
 gate.update(contract_sha256=sha(C/'CONTRACT.json'),runtime_sha256=sha(C/'RUNTIME_MANIFEST.json'),
  memory_specs_sha256=sha(C/'memory_specs.json'),amended_original_clock=clock,
  transport_recovery_tests_passed=True,readiness_sha256=sha(path),
  unchanged_scientific_files={name:sha(C/name) for name in unchanged},
  recovery_note='Original exact runtime scientific/sandbox/probe gates preserved by unchanged hashes; new transport circuit32tests and ONE readiness call. Not a new scientific probe or budget.')
 gate['artifacts'].update({str(path):sha(path),str(R/'R3_V011_TESTS.json'):sha(R/'R3_V011_TESTS.json'),
  str(R/'round3-terra-r1r2memory-amended-001/AMENDMENT.json'):sha(R/'round3-terra-r1r2memory-amended-001/AMENDMENT.json')})
 assert not (R/'R3_RECOVERY_GATES.json').exists();atomic_json(R/'R3_RECOVERY_GATES.json',gate)
