#!/usr/bin/env python3
"""Finish only the three fixed transfers and record verified PVC receipts."""
import pathlib,subprocess,json,time,os,sys,base64,hashlib,traceback
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[2]
LEAF=REPO/'experiments/autoresearch_leaf_20260918/lifecycle-return-v001'
deadline=time.monotonic()+7200
def wait_file(p):
 while not p.exists():
  if time.monotonic()>deadline:raise TimeoutError(str(p))
  time.sleep(15)
 return json.loads(p.read_text())
def run(args):subprocess.run([sys.executable,*map(str,args)],check=True)
def restore(label,mapping):
 if not (HERE/(label+'-PVC_RECEIPT.json')).exists():run([HERE/'restore_archive.py',label,'@'+str(mapping)])
def main():
 for label,mapping in [('leaf-r2-v1','leaf-r2-mappings.json'),('spooky-v1','spooky-mappings.json')]:
  wait_file(HERE/(label+'-UPLOAD.json'));restore(label,HERE/mapping)
 mac=wait_file(LEAF/'MAC_RECEIPT.json')
 # Presence does not replace the independently checked archive SHA or PVC content checks.
 print('Leaf Mac receipt received',flush=True)
 label='leaf-content-v1';archive=LEAF/'transport/leaf-complete-except-r2.tar.gz'
 if not (HERE/(label+'-UPLOAD.json')).exists():run([HERE/'upload_archive.py',archive,label,'f1435498884f0a8c413bd61cee447acef5a35d6e7e0180b2030021cc23b1783d'])
 mapping=HERE/'leaf-content-mappings.json';mapping.write_text(json.dumps([{'root':'content','manifest':str(LEAF/'transport/content.manifest.json')}]))
 restore(label,mapping)
 if (HERE/'leaf-v1-PVC_RECEIPT.json').exists():return
 manifest=(LEAF/'transport.manifest.json').read_text();doc=json.loads(manifest)
 small={row['path']:base64.b64encode((LEAF/'transport'/row['path']).read_bytes()).decode() for row in doc['entries'] if not row['path'].endswith('.tar.gz')}
 payload={'manifest':manifest,'small':small,'content_receipts':[json.loads((HERE/(v+'-PVC_RECEIPT.json')).read_text()) for v in ['leaf-r2-v1','leaf-content-v1']]}
 code='''import pathlib,json,sys,base64,shutil,subprocess,hashlib,time
x=json.load(sys.stdin);dest=pathlib.Path('/workspace/backups/aws/autoresearch-20260918/leaf-v1');dest.mkdir(exist_ok=False)
transport=dest/'transport';transport.mkdir()
for name,data in x['small'].items():
 assert pathlib.PurePath(name).name==name
 (transport/name).write_bytes(base64.b64decode(data))
staging=pathlib.Path('/workspace/.transfer-staging/autoresearch-return-20260919-v1')
for label,name in [('leaf-r2-v1','r2-both-epochs.tar.gz'),('leaf-content-v1','leaf-complete-except-r2.tar.gz')]:shutil.copyfile(staging/label/'archive',transport/name)
mf=dest/'transport.manifest.json';mf.write_text(x['manifest'])
script='/workspace/backups/aws/autoresearch-20260918/leaf-content-v1/transfer_manifest.py'
r=subprocess.run([sys.executable,script,'verify',str(transport),str(mf)],capture_output=True,text=True);assert r.returncode==0,(r.stdout,r.stderr)
receipt={'status':'PVC_VERIFIED','path':str(transport),'manifest_sha256':hashlib.sha256(mf.read_bytes()).hexdigest(),'transport_verification':json.loads(r.stdout),'content_receipts':x['content_receipts'],'verified_unix':time.time()}
(dest/'PVC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
 env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
 r=subprocess.run(['kubectl','-n','ecepxie','exec','-i','autoresearch-return-20260919-v1','--','python','-c',code],input=json.dumps(payload),capture_output=True,text=True,env=env,timeout=3600)
 if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout)
 (HERE/'leaf-v1-PVC_RECEIPT.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
 print('ALL_PVC_VERIFIED',flush=True)
try:main()
except Exception:
 (HERE/'TRANSFER_FAILURE.json').write_text(json.dumps({'time':time.time(),'error':traceback.format_exc()},indent=2)+'\n');raise
