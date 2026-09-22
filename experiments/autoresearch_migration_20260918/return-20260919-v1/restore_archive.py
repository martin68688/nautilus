#!/usr/bin/env python3
"""Restore a verified transport into a fresh PVC version and verify content."""
import argparse,json,os,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('mappings',help='JSON list of {root,manifest}');a=p.parse_args()
here=pathlib.Path(__file__).parent
receipt=json.loads((here/(a.label+'-UPLOAD.json')).read_text())
payload={'receipt':receipt,'destination':'/workspace/backups/aws/autoresearch-20260918/'+a.label,'script':pathlib.Path('/Users/haoming/.codex/skills/aws-nrp-experiment-lifecycle/scripts/transfer_manifest.py').read_text(),'manifests':[]}
for m in json.loads(pathlib.Path(a.mappings[1:]).read_text() if a.mappings.startswith('@') else a.mappings):
 payload['manifests'].append({'root':m['root'],'text':pathlib.Path(m['manifest']).read_text()})
code='''import json,sys,pathlib,tarfile,hashlib,subprocess,os,time
x=json.load(sys.stdin); src=pathlib.Path(x['receipt']['path']); dest=pathlib.Path(x['destination'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(src)==x['receipt']['sha256']
dest.mkdir(parents=True,exist_ok=False)
data=dest/'restored';data.mkdir()
with tarfile.open(src) as t:
 members=t.getmembers(); links=set(m.name.rstrip('/') for m in members if m.issym() or m.islnk())
 for m in members:
  p=pathlib.PurePosixPath(m.name)
  assert not p.is_absolute() and '..' not in p.parts
  assert m.isfile() or m.isdir() or m.issym(), 'unsupported archive member'
  assert not any(str(parent) in links for parent in p.parents), 'member traverses archive link'
 for m in members:
  if not m.issym():t.extract(m,data)
 for m in members:
  if m.issym():os.symlink(m.linkname,data/m.name)
script=dest/'transfer_manifest.py';script.write_text(x['script'])
verified=[]
for i,m in enumerate(x['manifests']):
 root=pathlib.PurePosixPath(m['root']);assert not root.is_absolute() and '..' not in root.parts
 mf=dest/('manifest-%02d.json'%i);mf.write_text(m['text'])
 r=subprocess.run([sys.executable,str(script),'verify',str(data/root),str(mf)],capture_output=True,text=True)
 result=json.loads(r.stdout);assert r.returncode==0,result
 verified.append({'root':str(data/root),'manifest':str(mf),'manifest_sha256':sha(mf),'verification':result})
out={'status':'PVC_VERIFIED','archive_sha256':x['receipt']['sha256'],'destination':str(dest),'manifests':verified,'verified_unix':time.time()}
(dest/'PVC_RECEIPT.json').write_text(json.dumps(out,indent=2)+'\\n')
print(json.dumps(out))
'''
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
cmd=['kubectl','-n','ecepxie','exec','-i','autoresearch-return-20260919-v1','--','python','-c',code]
r=subprocess.run(cmd,input=json.dumps(payload),text=True,capture_output=True,env=env,timeout=3600)
if r.returncode: raise RuntimeError(r.stderr+'\n'+r.stdout)
out=json.loads(r.stdout);(here/(a.label+'-PVC_RECEIPT.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
