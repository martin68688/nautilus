"""Leaf adaptation of autoresearch's single proposal -> Python execution -> keep.

Trusted AWS host controller; Docker isolates provider, training and replay.
This is not the MLEvolve searcher. Formal launch requires signed-off live gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from agent import propose
from core import RunState, atomic_json, run_process, sha, exclusive, verify_runtime
from memory_bridge import MemoryBridge
from sandbox import execute
from provider import complete, configure
from adoption_audit import audit as audit_adoption

HERE = Path(__file__).resolve().parent


def provider_call(messages, folder, deadline, args):
    index = len(list(folder.glob('provider-*')))
    exchange = folder/f'provider-{index:03d}'
    exchange.mkdir()
    atomic_json(exchange/'request.json', {'messages':messages,'deadline':deadline,
                                        'expected_base_url':args.endpoint_url})
    content,usage=complete(messages,deadline,expected_base_url=args.endpoint_url,
                           raw_response_path=exchange/'raw_response.bin')
    atomic_json(exchange/'response.json',{'content':content,'usage':usage})
    return content,usage


def check_inputs(args, contract):
    if sha(args.public/'manifest.json') != contract['data_manifest_sha256']:
        raise ValueError('data manifest mismatch')
    for name,digest in json.loads((args.public/'manifest.json').read_text())['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('manifest traversal')
        if sha(args.public/name) != digest: raise ValueError('data file mismatch: '+name)
    if sha(args.hidden/'search.csv') != contract['host_labels_sha256']:
        raise ValueError('hidden label hash mismatch')
    if '@sha256:' not in args.image: raise ValueError('image not pinned')


def run(args):
    clock=json.loads((args.output/'CLOCK.json').read_text())
    args.run_label=clock['run_label']
    # Key is read on the trusted host only, never in Docker ENV or on EBS.
    configure(args.endpoint_env,args.endpoint_url)
    if time.time() >= clock['deadline']: raise TimeoutError('original run deadline expired')
    contract=json.loads((HERE/'CONTRACT.json').read_text())
    verify_runtime(HERE,json.loads((HERE/'RUNTIME_MANIFEST.json').read_text()))
    check_inputs(args,contract)
    gates=json.loads(args.gates.read_text())
    required=['provider_generated','sandbox_negative_tests','checkpoint_fresh_reload',
              'host_recomputed','gpu_identity_verified','memory_source_audit',
              'timeout_and_resume','same_dependencies','input_manifest_verified']
    if any(gates.get(k) is not True for k in required):
        raise ValueError('formal launch gate not passed')
    if gates.get('contract_sha256') != sha(HERE/'CONTRACT.json'):
        raise ValueError('gate contract mismatch')
    if gates.get('runtime_sha256') != sha(HERE/'RUNTIME_MANIFEST.json'):
        raise ValueError('gate runtime mismatch')
    from host import verify
    bridge=None
    if args.memory_specs:
        specs=json.loads(args.memory_specs.read_text())
        for entry in specs.values():
            if not Path(entry['path']).is_absolute(): entry['path']=str(args.memory_specs.parent/entry['path'])
        bridge=MemoryBridge(specs,'leaf-classification',
                            contract['data_manifest_sha256'])
    identity={'contract_sha256':sha(HERE/'CONTRACT.json'),'image':args.image,
              'runtime_sha256':sha(HERE/'RUNTIME_MANIFEST.json'),
              'memory_sha256':sha(args.memory_specs) if args.memory_specs else None}
    state=RunState(args.output,identity,contract['seconds_per_condition'],clock=clock)
    deadline=state.launch['deadline']
    text=(HERE/'task.md').read_text()
    while time.time()+60 < deadline:
        folder=state.new_attempt()
        node_start=time.monotonic()
        node_deadline=min(deadline,time.time()+3600)
        code=''
        if state.incumbent:
            code=(state.root/state.incumbent['attempt']/'train.py').read_text()
        try:
            code,change=propose(lambda m:provider_call(m,folder,node_deadline,args),
                               folder,text,code,state.history,bridge)
            (folder/'train.py').write_text(code)
            atomic_json(folder/'PROPOSAL.json',{'change':change,'generated':time.time()})
            artifacts=folder/'artifacts';artifacts.mkdir();artifacts.chmod(0o777)
            remaining=max(0,node_deadline-time.time())
            spec={'node_id':folder.name,'code_sha256':sha(folder/'train.py'),
                  'data_manifest_sha256':contract['data_manifest_sha256'],
                  'contract_sha256':identity['contract_sha256'],'public_data':'/data',
                  'artifact_root':'/artifacts','node_started_monotonic':node_start,
                  'training_deadline_monotonic':time.monotonic()+max(0,remaining-600),
                  'execution_deadline_monotonic':time.monotonic()+max(0,remaining-60)}
            atomic_json(folder/'spec.json',spec)
            process=execute(args.image,'leaf-train-'+str(os.getpid()),args.public,folder,HERE,node_deadline-45,run_label=args.run_label)
            receipt=None
            if process['returncode']==0 and not process['timeout']:
                # Replay is untrusted too: ensure it cannot replace published evidence.
                before={str(p.relative_to(artifacts)):sha(p) for p in artifacts.rglob('*') if p.is_file() and not p.is_symlink()}
                atomic_json(folder/'PRE_REPLAY_HASHES.json',before)
                replay=execute(args.image,'leaf-replay-'+str(os.getpid()),args.public,folder,HERE,node_deadline-15,replay=True,run_label=args.run_label)
                atomic_json(folder/'REPLAY_PROCESS.json',replay)
                if replay['returncode']==0 and not replay['timeout']:
                    for name,digest in before.items():
                        if sha(artifacts/name)!=digest: raise ValueError('replay changed published evidence')
                    receipt=verify(artifacts,args.public,args.hidden,spec,contract['host_labels_sha256'])
            audit_adoption(folder,receipt)
            state.finish(folder,process,receipt)
        except Exception as error:
            atomic_json(folder/'ERROR.json',{'type':type(error).__name__,'message':str(error)})
            if not (folder/'train.py').exists(): (folder/'train.py').write_text(code)
            if not (folder/'RESULT.json').exists():
                state.finish(folder,{'returncode':None,'timeout':time.time()>=node_deadline})
        print(json.dumps({'attempt':folder.name,'incumbent':state.incumbent}),flush=True)
    atomic_json(state.root/'OUTCOME.json',{'reason':'budget_complete','incumbent':state.incumbent,
                'elapsed_seconds':time.time()-state.launch['started'],'finished':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('public','hidden','output','gates'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--memory-specs',type=Path)
    p.add_argument('--image',required=True)
    p.add_argument('--endpoint-url',required=True)
    p.add_argument('--endpoint-env',required=True)
    p.add_argument('--network',required=True)
    args=p.parse_args()
    with exclusive(args.output):
        run(args)
