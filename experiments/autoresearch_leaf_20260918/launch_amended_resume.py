"""Launch only the authorized continuation against the original R2 deadline."""
import json,math,subprocess,sys,time
from pathlib import Path
from core import atomic_json,sha
ROOT=Path('/workspace/autoresearch-leaf-20260918');RUN=ROOT/'round2-terra-r1memory-amended-001'
CODE=ROOT/'amendment-v007/code';GATES=ROOT/'R2_AMENDMENT_GATES.json'
UNIT='leaf-autoresearch-r2-terra-amended-20260918.service'
clock=json.loads((RUN/'CLOCK.json').read_text());amend=json.loads((RUN/'AMENDMENT.json').read_text())
assert clock['started']==1789772208.6579065 and clock['deadline']==1789779408.6579065
assert sha(RUN/'LAUNCH.json')==amend['source_launch_sha256'] and sha(RUN/'CLOCK.json')==amend['source_clock_sha256']
gates=json.loads(GATES.read_text());assert gates['runtime_sha256']==sha(CODE/'RUNTIME_MANIFEST.json')
assert gates['contract_sha256']==sha(CODE/'CONTRACT.json') and gates['memory_specs_sha256']==sha(CODE/'memory_specs.json')
assert not (RUN/'RESUME_LAUNCH.json').exists()
old=subprocess.run(['systemctl','is-active','leaf-autoresearch-r2-terra-20260918.service'],capture_output=True,text=True)
assert old.stdout.strip() in ('inactive','failed')
assert not subprocess.check_output(['docker','ps','-q','--filter','label=autoresearch.leaf.run='+clock['run_label']],text=True).strip()
remaining=clock['deadline']-time.time();assert remaining>660,'not enough original budget for meaningful training admission'
image=gates['image']
command=['systemd-run','--unit='+UNIT,'--property=Type=exec','--property=KillMode=control-group',
    '--property=TimeoutStopSec=30','--property=RuntimeMaxSec='+str(math.ceil(remaining)+60),'--property=Restart=no',
    sys.executable,str(CODE/'supervise.py'),'--public',str(ROOT/'inputs-v001/public'),
    '--hidden',str(ROOT/'inputs-v001/host-labels'),'--output',str(RUN),'--gates',str(GATES),
    '--memory-specs',str(CODE/'memory_specs.json'),'--image',image,'--endpoint-url','https://apizh-ai.com/v1',
    '--endpoint-env','/etc/autoresearch/apizh-terra-low.env','--network','none']
subprocess.run(command,check=True)
atomic_json(RUN/'RESUME_LAUNCH.json',{'unit':UNIT,'resumed_unix':time.time(),'original_clock':clock,
    'remaining_at_dispatch':remaining,'budget_reset':False,'amendment_sha256':sha(RUN/'AMENDMENT.json'),
    'gates_sha256':sha(GATES),'runtime_manifest_sha256':sha(CODE/'RUNTIME_MANIFEST.json'),
    'inherited_incumbent':json.loads((RUN/'STATE.json').read_text())['incumbent'],'command':command})
print(json.dumps({'unit':UNIT,'remaining_seconds':remaining,'original_deadline':clock['deadline']}))
