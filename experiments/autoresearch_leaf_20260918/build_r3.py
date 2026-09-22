"""Freeze R3 closure with cumulative skill packs and strict action overlay."""
import json,shutil,tarfile
from pathlib import Path
from core import sha,atomic_json
R=Path(__file__).resolve().parent;OUT=R/'builds/r3-v010';CODE=OUT/'code'
assert not OUT.exists();CODE.mkdir(parents=True)
base=R/'builds/amendment-v007/code'
names=['main.py','supervise.py','core.py','memory_bridge.py','sandbox.py','replay.py','host.py',
 'budget_checkpoint.py','adoption_audit.py','agent_probe.py','memory_compatibility_v2.json','task.md',
 'test_agent.py','test_amendment.py','test_checkpoint.py','test_core.py','test_host.py']
for name in names:shutil.copy2(base/name,CODE/name)
overlay=R/'r3-single-action-v002'
for name in ['agent.py','provider.py','action_protocol.py','test_single_action.py']:shutil.copy2(overlay/name,CODE/name)
shutil.copytree(overlay/'fixtures',CODE/'fixtures')
for fixture in (CODE/'fixtures').iterdir():fixture.chmod(0o444)  # sanitized response-only fixtures, no credentials
# Old test intentionally expected inner JSONDecodeError; new typed failure has
# mechanical same-Agent recovery. Outer JSONDecodeError remains unchanged.
p=CODE/'test_amendment.py';text=p.read_text().replace('from provider import complete,BASE_URL','from provider import complete,BASE_URL\nfrom action_protocol import ActionProtocolError')
text=text.replace('self.assertRaises(json.JSONDecodeError):complete','self.assertRaises((json.JSONDecodeError,ActionProtocolError)):complete');p.write_text(text)
contract=json.loads((base/'CONTRACT.json').read_text());contract.update(protocol_revision='r3-single-action-v010',
 memory_selection='Only audited R1 plus completed R2 two-epoch evidence, unchanged full conditional synthesis; no probe data/code/weights.',
 action_framing='Exactly one strict action JSON; invalid content or unqueried read zero tool execution, literal substring query; same-Agent feedback within unchanged 12-turn/original deadline')
atomic_json(CODE/'CONTRACT.json',contract)
packs=R/'r3-cumulative-memory-v001';binding=json.loads((packs/'BINDING.json').read_text());(CODE/'memory').mkdir()
specs={}
for k in ['search','inner']:
 src=packs/f'{k}_memory.json';assert sha(src)==binding['sha256'][k]
 shutil.copy2(src,CODE/'memory'/src.name);specs[k]={'path':'memory/'+src.name,'sha256':sha(src)}
atomic_json(CODE/'memory_specs.json',specs)
atomic_json(CODE/'RUNTIME_MANIFEST.json',{'files':{str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()},
 'upstream_commit':contract['upstream_commit'],'purpose':'R3 preflight closure, NOT formal launch approval',
 'cumulative_binding_sha256':sha(packs/'BINDING.json'),'overlay_manifest_sha256':sha(R/'R3_SINGLE_ACTION_V002_MANIFEST.json')})
with tarfile.open(OUT/'runtime.tar.gz','w:gz') as t:t.add(CODE,arcname='code')
atomic_json(OUT/'BUILD.json',{'archive_sha256':sha(OUT/'runtime.tar.gz'),'runtime_manifest_sha256':sha(CODE/'RUNTIME_MANIFEST.json')})
print((OUT/'BUILD.json').read_text())
