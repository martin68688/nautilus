"""Run engineering-only actual Leaf training on the assigned AWS instance.

No model/API call, no historical memory, no formal A/B state. Emits partial gates
only. Coordinator must still verify provider, memory decisions, lifecycle monitor.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
from core import sha, atomic_json, verify_runtime
from sandbox import execute
from host import verify


def main():
    p=argparse.ArgumentParser()
    for k in ('public','hidden','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--image',required=True);args=p.parse_args()
    here=Path(__file__).resolve().parent
    verify_runtime(here,json.loads((here/'RUNTIME_MANIFEST.json').read_text()))
    contract=json.loads((here/'CONTRACT.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(here/'preflight_candidate.py',args.output/'train.py')
    artifacts=args.output/'artifacts';artifacts.mkdir();artifacts.chmod(0o777)
    now=time.monotonic()
    spec={'node_id':'engineering-smoke','code_sha256':sha(args.output/'train.py'),
          'data_manifest_sha256':contract['data_manifest_sha256'],'contract_sha256':sha(here/'CONTRACT.json'),
          'public_data':'/data','artifact_root':'/artifacts','node_started_monotonic':now,
          'training_deadline_monotonic':now+1200,'execution_deadline_monotonic':now+1800}
    atomic_json(args.output/'spec.json',spec)
    process=execute(args.image,'leaf-engineering-smoke',args.public,args.output,here,time.time()+1800,
                    run_label='engineering-smoke-20260918')
    atomic_json(args.output/'TRAIN_PROCESS.json',process)
    if process['returncode']!=0 or process['timeout']:raise RuntimeError('training preflight failed')
    before={str(f.relative_to(artifacts)):sha(f) for f in artifacts.rglob('*') if f.is_file() and not f.is_symlink()}
    replay=execute(args.image,'leaf-engineering-replay',args.public,args.output,here,time.time()+120,
                   replay=True,run_label='engineering-smoke-20260918')
    atomic_json(args.output/'REPLAY_PROCESS.json',replay)
    if replay['returncode']!=0 or replay['timeout']:raise RuntimeError('replay preflight failed')
    for name,digest in before.items():
        if sha(artifacts/name)!=digest:raise ValueError('replay changed source evidence')
    result=verify(artifacts,args.public,args.hidden,spec,contract['host_labels_sha256'])
    result['engineering_smoke_only']=True
    atomic_json(args.output/'HOST.json',result)
    atomic_json(args.output/'PARTIAL_GATES.json',{
        'contract_sha256':sha(here/'CONTRACT.json'),'runtime_sha256':sha(here/'RUNTIME_MANIFEST.json'),
        'checkpoint_fresh_reload':True,'host_recomputed':True,'gpu_identity_verified':True,
        'actual_leaf_training':True,'provider_generated':False,'formal_launch_approved':False})
    print(json.dumps(result))


if __name__=='__main__':main()
