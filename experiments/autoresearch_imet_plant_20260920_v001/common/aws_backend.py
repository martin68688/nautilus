import os,time,json,subprocess,signal,re
from pathlib import Path
import numpy as np
import harness as h
IMAGE='docker.io/haomingwang22/mlevolve@sha256:fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc';CPUS=7;RAM='28g'
def container_stage(mode,work,cache,seconds,log):
    name='ar-'+h.TASK+'-'+work.name+'-'+mode;cmd=['docker','run','--name',name,'--rm','--gpus','all','--cpus',str(CPUS),'--memory',RAM,'--pids-limit','256','--cap-drop','ALL','--security-opt','no-new-privileges','--user','1000:1000','--network','none','--read-only','--tmpfs','/tmp:rw,size=4g,mode=1777','--shm-size','2g','-v',str(work)+':/work:rw','-v',str(h.DATA)+':/data:ro','-v',str(cache)+':/cache:ro','-v',str(Path(__file__).resolve().parent/'invoke.py')+':/harness/invoke.py:ro','-w','/work','-e','HOME=/work','-e','HF_HOME=/cache','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','OMP_NUM_THREADS=7','-e','OPENBLAS_NUM_THREADS=7','-e','MKL_NUM_THREADS=7','-e','PYTHONHASHSEED=0',IMAGE,'python','/harness/invoke.py',mode,'/work',str(max(1,seconds-10))]
    t=time.monotonic();
    with Path(log).open('wb') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=max(.1,seconds))
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','rm','-f',name],capture_output=True);os.killpg(p.pid,signal.SIGKILL);p.wait()
    return {'mode':mode,'returncode':p.returncode,'timeout':p.returncode is None,'elapsed_seconds':time.monotonic()-t}
def execute(code,attempt,out,seconds):
    h.identity();start=time.monotonic();work=Path(out)/'worker';work.mkdir();work.chmod(0o777);(work/'artifacts').mkdir();(work/'artifacts').chmod(0o777);cache=Path(out).parent/'public-cache';cache.mkdir(exist_ok=True);(work/'train.py').write_text(code)
    models=json.loads((Path(out)/'public_models.json').read_text()) if (Path(out)/'public_models.json').exists() else []
    if len(models)>3 or any(not isinstance(m,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',m) for m in models):
        raise ValueError('invalid public model dependency')
    if models:
        script='from huggingface_hub import snapshot_download; import json; [snapshot_download(repo_id=m) for m in json.loads('+repr(json.dumps(models))+')]'
        name='ar-'+h.TASK+'-download-'+work.name
        cmd=['docker','run','--name',name,'--rm','--cpus','2','--memory','4g','--cap-drop','ALL','--security-opt','no-new-privileges','-v',str(cache)+':/cache:rw','-e','HF_HOME=/cache',IMAGE,'python','-c',script]
        with (Path(out)/'public_download.log').open('wb') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            # The public-cache stage is bounded by this candidate's local
            # budget.  `execute` receives `seconds`; there is no outer
            # `deadline` variable in this backend process.
            try:proc.wait(timeout=max(1,seconds-90))
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
        if proc.returncode: raise RuntimeError('public dependency download failed')
    stages=[]
    for mode in ['train','predict1','predict2']:
        rem=seconds-(time.monotonic()-start)
        if rem<=10:break
        stages.append(container_stage(mode,work,cache,rem,out/f'{mode}.log'))
        if stages[-1]['returncode']!=0 or stages[-1]['timeout']:break
    rec={'attempt':attempt,'stages':stages,'source_sha256':h.sha(code.encode()),'success':len(stages)==3 and all(s['returncode']==0 and not s['timeout'] for s in stages),'elapsed_seconds':time.monotonic()-start};h.save(Path(out)/'execution.json',rec)
    if not rec['success']:return {'status':'failed','search_score':None,'reason':'candidate_stage_failed','seconds':rec['elapsed_seconds']}
    with np.load(work/'predict1.npz',allow_pickle=False) as a,np.load(work/'predict2.npz',allow_pickle=False) as b:result={'status':'valid',**h.validate_predictions(a,b),'seconds':rec['elapsed_seconds'],'source_sha256':h.sha(code.encode()),'predictions_sha256':h.sha((work/'predict1.npz').read_bytes()),'host_recomputed':True,'reload_verified':True}
    h.save(Path(out)/'HOST_RESULT.json',result);return result
def install():
    h.execute=execute
    manifest=h.DATA/'manifest.json';assert h.sha(manifest.read_bytes())==os.environ['AUTORESEARCH_DATA_MANIFEST_SHA256']
