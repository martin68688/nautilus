"""Authorized preflight: real Luna retrieval/decision/code -> train/reload/Host.
Never use this probe's candidates or fitted weights as either formal arm's seed.
"""
import argparse
import json
from pathlib import Path
import time
from agent import propose
from main import provider_call, check_inputs
from provider import configure
from memory_bridge import MemoryBridge
from core import atomic_json,sha,verify_runtime
from sandbox import execute
from host import verify
from adoption_audit import audit


def main():
    p=argparse.ArgumentParser()
    for k in ('public','hidden','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('image','endpoint-url','endpoint-env'):p.add_argument('--'+k,required=True)
    p.add_argument('--seconds',type=int,default=900)
    args=p.parse_args();here=Path(__file__).resolve().parent
    verify_runtime(here,json.loads((here/'RUNTIME_MANIFEST.json').read_text()))
    contract=json.loads((here/'CONTRACT.json').read_text());check_inputs(args,contract)
    configure(args.endpoint_env,args.endpoint_url)
    args.output.mkdir(parents=True,exist_ok=False)
    specs=json.loads((here/'memory_specs.json').read_text())
    for entry in specs.values():entry['path']=str(here/entry['path'])
    bridge=MemoryBridge(specs,'leaf-classification',contract['data_manifest_sha256'])
    started=time.monotonic();deadline=time.time()+args.seconds
    code,change=propose(lambda messages:provider_call(messages,args.output,deadline,args),args.output,
        (here/'task.md').read_text()+'\nEngineering preflight only: use the SAME mandatory read/decision protocol as formal proposals, then produce a minimal real training candidate whose entire fit should take under 90 seconds on the stated resources. Architecture remains your choice; do not force a historical score. This probe is never a formal score or a seed candidate.',
        '',[],bridge)
    (args.output/'train.py').write_text(code)
    atomic_json(args.output/'PROPOSAL.json',{'change':change,'engineering_probe_only':True})
    receipt=json.loads((args.output/'MEMORY_DECISIONS.json').read_text())
    if not receipt['read_ids']:raise ValueError('actual probe did not read a card')
    artifacts=args.output/'artifacts';artifacts.mkdir();artifacts.chmod(0o777)
    remaining=max(0,deadline-time.time())
    spec={'node_id':'agent-probe','code_sha256':sha(args.output/'train.py'),
          'data_manifest_sha256':contract['data_manifest_sha256'],'contract_sha256':sha(here/'CONTRACT.json'),
          'public_data':'/data','artifact_root':'/artifacts','node_started_monotonic':started,
          'training_deadline_monotonic':time.monotonic()+max(0,remaining-600),
          'execution_deadline_monotonic':time.monotonic()+max(0,remaining-120)}
    atomic_json(args.output/'spec.json',spec)
    result=execute(args.image,'leaf-real-agent-probe',args.public,args.output,here,deadline-120,
                   run_label='agent-probe-20260918')
    atomic_json(args.output/'TRAIN_PROCESS.json',result)
    if result['returncode']!=0 or result['timeout']:raise RuntimeError('actual Agent candidate failed execution')
    before={str(f.relative_to(artifacts)):sha(f) for f in artifacts.rglob('*') if f.is_file() and not f.is_symlink()}
    replay=execute(args.image,'leaf-real-agent-replay',args.public,args.output,here,deadline-15,
                   replay=True,run_label='agent-probe-20260918')
    atomic_json(args.output/'REPLAY_PROCESS.json',replay)
    if replay['returncode']!=0 or replay['timeout']:raise RuntimeError('actual Agent candidate failed reload')
    for name,digest in before.items():
        if sha(artifacts/name)!=digest:raise ValueError('replay changed publication')
    host=verify(artifacts,args.public,args.hidden,spec,contract['host_labels_sha256'])
    host['engineering_probe_only']=True
    atomic_json(args.output/'HOST.json',host);audit(args.output,host)
    atomic_json(args.output/'AGENT_PROBE.json',{'passed':True,'real_provider':True,
        'read_ids':receipt['read_ids'],'decisions':receipt['decisions'],'host_verified':True,
        'elapsed':time.monotonic()-started,'formal_score':False})
    print(json.dumps({'probe_passed':True,'host_score':host['search_logloss'],'formal_score':False}))


if __name__=='__main__':main()
