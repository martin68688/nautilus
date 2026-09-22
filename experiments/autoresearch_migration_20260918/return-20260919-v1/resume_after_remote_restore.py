import json,os,pathlib,subprocess,time,sys
base=pathlib.Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
remote='/workspace/backups/aws/autoresearch-20260918/leaf-content-v1/PVC_RECEIPT.json'
code="import pathlib; p=pathlib.Path("+repr(remote)+"); print(p.read_text() if p.exists() else 'PENDING')"
for i in range(150):
 r=subprocess.run(['kubectl','-n','ecepxie','exec','autoresearch-return-20260919-v1','--','python','-c',code],env=env,capture_output=True,text=True,timeout=45)
 if r.returncode==0 and r.stdout.strip()!='PENDING':
  d=json.loads(r.stdout);assert d['status']=='PVC_VERIFIED'
  assert all(m['verification']['status']=='verified' for m in d['manifests'])
  (base/'leaf-content-v1-PVC_RECEIPT.json').write_text(r.stdout)
  print('Recovered remote verified receipt',flush=True)
  subprocess.run([sys.executable,str(base/'finish_transfers.py')],check=True)
  break
 time.sleep(60)
else:raise TimeoutError('Remote restore receipt still pending; inspect existing worker, do not duplicate')
