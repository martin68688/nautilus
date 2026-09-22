"""Create an immutable preflight runtime closure (not a launch approval)."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from core import sha, atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--memory-source',required=True)
    p.add_argument('--memory-sha256',required=True)
    p.add_argument('--inner-memory-source',required=True)
    p.add_argument('--inner-memory-sha256',required=True)
    a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('build name must be a basename')
    here=Path(__file__).resolve().parent
    root=here/'builds'/a.name;root.mkdir(parents=True,exist_ok=False)
    code=root/'code';code.mkdir()
    for f in here.iterdir():
        if f.is_file() and f.suffix in ('.py','.md','.json','.txt') and f.name not in ('transfer-pod.json','memory_compatibility.json'):
            shutil.copy2(f,code/f.name)
    source=Path(a.memory_source).resolve(strict=True)
    expected=a.memory_sha256
    assert expected!='a7a2162307ae9ee02de195468dcdeb8efab4941432ce7a6eb07f8c03e494bc84', 'superseded 91-card memory forbidden'
    assert sha(source)==expected
    (code/'memory').mkdir()
    shutil.copy2(source,code/'memory/search_experience.json')
    assert sha(code/'memory/search_experience.json')==expected
    inner=Path(a.inner_memory_source).resolve(strict=True)
    assert sha(inner)==a.inner_memory_sha256
    shutil.copy2(inner,code/'memory/inner_memory.json')
    atomic_json(code/'memory_specs.json',{'search':{'path':'memory/search_experience.json','sha256':expected},
        'inner':{'path':'memory/inner_memory.json','sha256':a.inner_memory_sha256}})
    manifest={'files':{str(f.relative_to(code)):sha(f) for f in code.rglob('*') if f.is_file()},
              'upstream_commit':'1e9849798006a069d17399f0d4ad0178200ee946',
              'purpose':'preflight only; live launch gates not passed'}
    atomic_json(code/'RUNTIME_MANIFEST.json',manifest)
    archive=root/'runtime.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(code,arcname='code')
    result={'build':a.name,'archive':str(archive),'archive_sha256':sha(archive),
            'runtime_manifest_sha256':sha(code/'RUNTIME_MANIFEST.json'),'file_count':len(manifest['files'])}
    atomic_json(root/'BUILD.json',result);print(json.dumps(result))

if __name__=='__main__':main()
