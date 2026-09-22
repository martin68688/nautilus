"""AWS engineering-only live Docker timeout/cleanup and deadline recovery probe."""
import argparse,json,subprocess,sys,time,uuid
from pathlib import Path
from core import RunState,atomic_json,run_process,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    name='leaf-timeout-probe-'+uuid.uuid4().hex[:10]
    label='leaf-timeout-probe-'+uuid.uuid4().hex
    cmd=['docker','run','--name',name,'--label','autoresearch.leaf.run='+label,
         '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
         '--cpus','4','--memory','14g','--memory-swap','14g','--shm-size','2g','--gpus','device=0',
         '--entrypoint','python',a.image,'-c',
         'import subprocess,time; subprocess.Popen(["python","-c","import time; time.sleep(120)"]); time.sleep(120)']
    result=None
    try:result=run_process(cmd,a.output,None,a.output/'timeout.log',time.time()+8)
    finally:
        ids=subprocess.check_output(['docker','ps','-aq','--filter','label=autoresearch.leaf.run='+label],text=True).split()
        for ident in ids:subprocess.run(['docker','rm','-f',ident],check=True,capture_output=True)
    assert result['timeout']
    assert not subprocess.check_output(['docker','ps','-aq','--filter','label=autoresearch.leaf.run='+label],text=True).strip()
    clock={'started':time.time()-10,'deadline':time.time()+100}
    identity={'probe':'never a scientific result'}
    state=RunState(a.output/'resume',identity,110,clock=clock)
    folder=state.new_attempt();(folder/'train.py').write_text('print("interrupted engineering probe")\n')
    recovered=RunState(a.output/'resume',identity,110,clock={'started':time.time(),'deadline':time.time()+10000})
    assert recovered.launch['started']==clock['started'] and recovered.launch['deadline']==state.launch['deadline']
    assert recovered.incumbent is None and recovered.history[-1]['status']=='interrupted'
    dependencies=subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint','python',a.image,'-c',
        'import json,sys,numpy,pandas,sklearn,torch,catboost; print(json.dumps({"python":sys.version,"numpy":numpy.__version__,"pandas":pandas.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__,"catboost":catboost.__version__}))'],text=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    atomic_json(a.output/'OPERATIONS.json',{'passed':True,'timeout_and_resume':True,'docker_timeout':result,
        'exact_run_containers_reaped':ids,'deadline_reset_rejected':True,'image':a.image,
        'dependencies':json.loads(dependencies),'gpu':gpu,'engineering_only':True,'script_sha256':sha(Path(__file__))})
    print('Live Docker timeout/cleanup and original-deadline recovery passed')
if __name__=='__main__':main()
