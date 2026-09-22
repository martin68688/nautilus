#!/usr/bin/env python3
"""Resumable bounded archive upload to the exact lifecycle transfer Pod."""
import argparse, concurrent.futures, hashlib, json, os, pathlib, subprocess, time

p=argparse.ArgumentParser()
p.add_argument('archive',type=pathlib.Path)
p.add_argument('label')
p.add_argument('sha256')
a=p.parse_args()
assert a.label.replace('-','').isalnum()
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
pod='autoresearch-return-20260919-v1'
base='/workspace/.transfer-staging/'+pod+'/'+a.label
def kube(args, data=None):
    r=subprocess.run(['kubectl','-n','ecepxie',*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=600)
    if r.returncode: raise RuntimeError(r.stderr.decode()[-2000:])
    return r.stdout
obj=json.loads(kube(['get','pod',pod,'-o','json']))
assert obj['metadata']['uid']=='d8f88660-c30b-4dc4-ba74-7378a46423e0'
assert obj['status']['phase']=='Running'
assert any(v.get('persistentVolumeClaim',{}).get('claimName')=='haoming-storage' for v in obj['spec']['volumes'])
def remote(code, data=None):
    return kube(['exec','-i',pod,'--','python','-c',code],data)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert digest(a.archive)==a.sha256, 'local archive hash mismatch'
remote('import pathlib; pathlib.Path('+repr(base)+').mkdir(parents=True,exist_ok=True)')
size=a.archive.stat().st_size
chunk=64*1024*1024
def one(i):
    with a.archive.open('rb') as f:
        f.seek(i*chunk); data=f.read(chunk)
    sha=hashlib.sha256(data).hexdigest()
    dest=base+'/part-%05d'%i
    code='import pathlib,hashlib,sys,os; p=pathlib.Path('+repr(dest)+'); data=sys.stdin.buffer.read(); assert hashlib.sha256(data).hexdigest()=='+repr(sha)+'; tmp=p.with_suffix(".partial"); tmp.write_bytes(data); os.replace(tmp,p); print(len(data))'
    check='import pathlib,hashlib; p=pathlib.Path('+repr(dest)+'); print(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(sha)+')'
    if remote(check).strip()==b'True': return i
    for attempt in range(3):
        try: remote(code,data); print('uploaded',i,len(data),flush=True); return i
        except Exception:
            if attempt==2: raise
            time.sleep(2)
count=(size+chunk-1)//chunk
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(one,range(count)))
code='''import pathlib,hashlib,os,json
base=pathlib.Path(BASE)
dest=base/'archive'
temp=base/'archive.partial'
h=hashlib.sha256()
with temp.open('wb') as out:
 for i in range(COUNT):
  with (base/('part-%05d'%i)).open('rb') as src:
   for data in iter(lambda:src.read(8*1024*1024),b''): out.write(data); h.update(data)
assert h.hexdigest()==SHA
os.replace(temp,dest)
print(json.dumps({'status':'archive_verified','path':str(dest),'sha256':h.hexdigest(),'bytes':dest.stat().st_size}))
'''.replace('BASE',repr(base)).replace('COUNT',str(count)).replace('SHA',repr(a.sha256))
receipt=json.loads(remote(code))
receipt.update(local_path=str(a.archive.resolve()),pod_uid=obj['metadata']['uid'],created_unix=time.time())
(pathlib.Path(__file__).parent/(a.label+'-UPLOAD.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
