"""Persist tests run in the exact immutable training dependency image."""
import argparse,json,subprocess
from pathlib import Path
from core import atomic_json,sha
p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--image',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
result=subprocess.run(['docker','run','--rm','--network=none','--read-only','--cap-drop=ALL',
    '--security-opt=no-new-privileges','--cpus=1','--memory=2g','--tmpfs=/tmp:rw,size=536870912',
    '--env=PYTHONDONTWRITEBYTECODE=1','--env=PYTHONPATH=/code','--env=OMP_NUM_THREADS=1','--env=OPENBLAS_NUM_THREADS=1',
    '--mount',f'type=bind,src={a.code.resolve()},dst=/code,readonly','--entrypoint','python',a.image,
    '-m','unittest','discover','-s','/code','-p','test_*.py'],capture_output=True,text=True,timeout=90)
atomic_json(a.output,{'passed':result.returncode==0,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
    'image':a.image,'runtime_manifest_sha256':sha(a.code/'RUNTIME_MANIFEST.json')})
print(result.stdout+result.stderr)
raise SystemExit(result.returncode)
