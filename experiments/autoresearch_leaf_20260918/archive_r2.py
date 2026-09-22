"""Quiescent R2 two-epoch archive, no deletions or source modifications."""
import json,subprocess,tarfile,time,hashlib
from pathlib import Path
ROOT=Path('/workspace/autoresearch-leaf-20260918')
OUT=ROOT/'r2-terminal-archive-v001'
roots=['round2-terra-r1memory-001','round2-terra-r1memory-amended-001']
unit='leaf-autoresearch-r2-terra-amended-20260918.service'
assert not OUT.exists()
status=subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','-p','MainPID'],text=True)
assert 'MainPID=0' in status and 'ActiveState=active' not in status,status
clock=json.loads((ROOT/roots[1]/'CLOCK.json').read_text())
assert clock['deadline']==1789779408.6579065
assert (ROOT/roots[1]/'SUPERVISOR.json').exists()
assert not subprocess.check_output(['docker','ps','-q','--filter','label=autoresearch.leaf.run='+clock['run_label']],text=True).strip()
OUT.mkdir()
for name in roots:
 subprocess.run(['python3',str(ROOT/'transfer_manifest.py'),'create',str(ROOT/name),'--output',str(OUT/(name+'.manifest.json'))],check=True)
archive=OUT/'r2-both-epochs.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in roots:t.add(ROOT/name,arcname=name)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
receipt={'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'roots':roots,'status':status,'clock':clock,'writers_absent':True,'created':time.time(),'no_exclusions':True,'dedup_policy':'both raw roots retained byte-exact; logical attempts merged only in explicit distillation adapter'}
(OUT/'ARCHIVE.json').write_text(json.dumps(receipt,indent=2)+'\n')
for f in OUT.iterdir(): f.chmod(0o644)
OUT.chmod(0o755)
print(json.dumps(receipt))
