"""Trusted root runner. Candidate executes as uid 1000 without secrets/host labels."""
import argparse,os,json,subprocess,signal,time,hashlib,sys
from pathlib import Path

def run_group(cmd,cwd,timeout,log):
    env={k:v for k,v in os.environ.items() if k in ['PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES']}
    env.update(HOME=str(cwd),HF_HOME=str(cwd/'hf-cache'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='0')
    def drop():os.setgroups([]);os.setgid(1000);os.setuid(1000)
    with log.open('wb') as out:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=drop)
        timed=False
        try:p.wait(timeout=max(.1,timeout))
        except subprocess.TimeoutExpired:timed=True
        finally:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
    return {'returncode':p.returncode,'timeout':timed}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);ap.add_argument('--seconds',type=float,required=True);args=ap.parse_args()
    assert args.attempt.replace('-','').replace('_','').isalnum()
    root=Path('/scratch')/args.attempt;root.mkdir(exist_ok=False);os.chown(root,1000,1000)
    code=sys.stdin.read();compile(code,'train.py','exec');(root/'train.py').write_text(code)
    artifacts=root/'artifacts';artifacts.mkdir();os.chown(artifacts,1000,1000)
    wrapper=Path('/opt/autoresearch/invoke.py');start=time.monotonic();deadline=start+args.seconds
    receipt={'attempt':args.attempt,'source_sha256':hashlib.sha256(code.encode()).hexdigest(),'metric':None,'stages':[]}
    for mode in ['train','predict1','predict2']:
        remaining=deadline-time.monotonic()
        if remaining<=0:receipt['error']='deadline';break
        cap=remaining-60 if mode=='train' and remaining>90 else remaining
        result=run_group([sys.executable,str(wrapper),mode,str(root),str(max(1,cap-10))],root,cap,root/(mode+'.log'))
        receipt['stages'].append({'mode':mode,**result})
        if result['returncode'] or result['timeout']:receipt['error']='nonzero_or_timeout';break
    receipt['elapsed_seconds']=time.monotonic()-start;receipt['success']=len(receipt['stages'])==3 and all(not s['returncode'] and not s['timeout'] for s in receipt['stages'])
    # Root-owned receipt is not supplied by candidate stdout.
    target=Path('/opt/autoresearch/receipts');target.mkdir(exist_ok=True);(target/(args.attempt+'.json')).write_text(json.dumps(receipt));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
