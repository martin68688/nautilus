import json,tarfile
from pathlib import Path
from core import atomic_json,sha
R=Path(__file__).resolve().parent;D=R/'builds/r3-v011';C=D/'code'
assert not (D/'BUILD.json').exists()
contract=json.loads((C/'CONTRACT.json').read_text());contract['protocol_revision']='r3-provider-backoff-circuit-v011'
contract['provider_recovery']={'transient_http':[502,503,504],'network_timeout':True,'delays_seconds':[15,30,60],
 'consecutive_failure_limit':4,'terminal_reason':'provider_blocked','monitor_cooldown_seconds':900,
 'clock_reset':False,'provider_switch':False,'other_http_fail_closed':True}
atomic_json(C/'CONTRACT.json',contract)
atomic_json(C/'RUNTIME_MANIFEST.json',{'files':{str(p.relative_to(C)):sha(p) for p in C.rglob('*') if p.is_file() and p.name!='RUNTIME_MANIFEST.json' and '__pycache__' not in p.parts},
 'upstream_commit':contract['upstream_commit'],'purpose':'authorized R3 transport-only original-clock recovery',
 'parent_manifest_sha256':sha(R/'builds/r3-v010/code/RUNTIME_MANIFEST.json')})
with tarfile.open(D/'runtime.tar.gz','w:gz') as t:t.add(C,arcname='code')
atomic_json(D/'BUILD.json',{'archive_sha256':sha(D/'runtime.tar.gz'),'runtime_manifest_sha256':sha(C/'RUNTIME_MANIFEST.json'),
 'contract_sha256':sha(C/'CONTRACT.json'),'memory_specs_sha256':sha(C/'memory_specs.json')})
print((D/'BUILD.json').read_text())
