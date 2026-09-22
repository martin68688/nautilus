"""Docker worker transport: rootless candidate, no network, no Host labels/keys."""
import os,time,json,subprocess,hashlib,re,signal
from pathlib import Path
import numpy as np
import harness as h
IMAGE='docker.io/haomingwang22/mlevolve@sha256:fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc'
CPUS=7;RAM='28g'
def identity():
 info=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
 if len(info.splitlines())!=1 or 'A10G' not in info:raise RuntimeError('required single AWS A10G not found')
 return {'metadata':{'uid':os.environ['AUTORESEARCH_INSTANCE_ID']},'hardware':info,'cpu_limit':CPUS,'memory_limit':RAM}

def container_stage(mode,work,cache,seconds,log):
 name='spooky-ar-'+work.name+'-'+mode
 cmd=['docker','run','--name',name,'--rm','--gpus','all','--cpus',str(CPUS),'--memory',RAM,'--pids-limit','256','--cap-drop','ALL','--security-opt','no-new-privileges','--user','1000:1000','--network','none','--read-only','--tmpfs','/tmp:rw,size=2g,mode=1777','--shm-size','1g','-v',str(work)+':/work:rw','-v',str(h.DATA)+':/data:ro','-v',str(cache)+':/cache:ro','-v',str(h.R/'invoke.py')+':/harness/invoke.py:ro','-w','/work','-e','HOME=/work','-e','HF_HOME=/cache','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','OMP_NUM_THREADS=7','-e','OPENBLAS_NUM_THREADS=7','-e','MKL_NUM_THREADS=7','-e','PYTHONHASHSEED=0',IMAGE,'python','/harness/invoke.py',mode,'/work',str(max(1,seconds-10))]
 start=time.monotonic();timed=False
 with log.open('wb') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:p.wait(timeout=max(.1,seconds))
  except subprocess.TimeoutExpired:timed=True
  finally:
   subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
   p.wait()
 return {'mode':mode,'returncode':p.returncode,'timeout':timed,'elapsed_seconds':time.monotonic()-start}

def execute(code,attempt,out,seconds):
 identity();start=time.monotonic();deadline=start+seconds
 work=out/'worker';work.mkdir();work.chmod(0o777);art=work/'artifacts';art.mkdir();art.chmod(0o777);(work/'train.py').write_text(code)
 # Each arm has a separate public-only cache; historical fitted models never enter it.
 cache=out.parent/'public-cache';cache.mkdir(exist_ok=True);stages=[]
 models=json.loads((out/'public_models.json').read_text()) if (out/'public_models.json').exists() else []
 if len(models)>3 or any(not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',m) for m in models):raise ValueError('invalid public model dependency')
 if models:
  script='from huggingface_hub import snapshot_download; import json; [snapshot_download(repo_id=m) for m in json.loads('+repr(json.dumps(models))+')]'
  name='spooky-ar-download-'+work.name
  cmd=['docker','run','--name',name,'--rm','--cpus','2','--memory','4g','--cap-drop','ALL','--security-opt','no-new-privileges','-v',str(cache)+':/cache:rw','-e','HF_HOME=/cache',IMAGE,'python','-c',script]
  with (out/'public_download.log').open('wb') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
   try:proc.wait(timeout=max(1,deadline-time.monotonic()-90))
   finally:
    subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20);proc.wait()
  if proc.returncode:raise RuntimeError('public dependency download failed')
 for mode in ['train','predict1','predict2']:
  rem=deadline-time.monotonic()
  if rem<=0:break
  stage=container_stage(mode,work,cache,rem-60 if mode=='train' and rem>90 else rem,out/(mode+'.log'));stages.append(stage)
  if stage['returncode'] or stage['timeout']:break
 receipt={'attempt':attempt,'stages':stages,'source_sha256':h.sha(code.encode()),'success':len(stages)==3 and all(not s['returncode'] and not s['timeout'] for s in stages),'elapsed_seconds':time.monotonic()-start};h.save(out/'execution.json',receipt)
 if not receipt['success']:return {'status':'failed','search_loss':None,'reason':'nonzero_or_timeout','seconds':time.monotonic()-start}
 with np.load(work/'predict1.npz',allow_pickle=False) as a,np.load(work/'predict2.npz',allow_pickle=False) as b:metrics=h.validate_predictions(a,b)
 result={'status':'valid',**metrics,'seconds':time.monotonic()-start,'source_sha256':h.sha(code.encode()),'predictions_sha256':h.sha((work/'predict1.npz').read_bytes()),'data_manifest_sha256':h.DM,'host_recomputed':True,'reload_verified':True}
 h.save(out/'HOST_RESULT.json',result)
 (out/'actual_parameters.json').write_bytes((work/'actual_parameters.json').read_bytes())
 return result

def install():
 h.identity=identity;h.execute=execute
 h.DATA=Path(os.environ['AUTORESEARCH_DATA']).resolve();h.LABELS=Path(os.environ['AUTORESEARCH_HOST_LABELS']).resolve()
 assert not h.LABELS.is_relative_to(h.DATA)
 if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('AUTORESEARCH_API_BASE'):raise RuntimeError('private AWS endpoint credentials required; no implicit fallback')
 manifest=h.DATA/'manifest.json';assert h.sha(manifest.read_bytes())==h.DM
 for name,digest in json.loads(manifest.read_text())['files'].items():assert h.sha((h.DATA/name).read_bytes())==digest
