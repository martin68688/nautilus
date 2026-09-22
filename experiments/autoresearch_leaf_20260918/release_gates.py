"""Evidence-backed gate aggregation; no model calls and no experiment launch."""
import argparse,json
from pathlib import Path
from core import atomic_json,sha,verify_runtime
from memory_bridge import MemoryBridge

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for key in ('runtime','probe','smoke','operations','source_audit','binding','output'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--amendment-tests',type=Path)
    a=p.parse_args();assert not a.output.exists()
    manifest=read(a.runtime/'RUNTIME_MANIFEST.json');verify_runtime(a.runtime,manifest)
    contract=read(a.runtime/'CONTRACT.json');assert contract['model']=='gpt-5.6-terra'
    probe=read(a.probe/'AGENT_PROBE.json');assert probe['passed'] and probe['real_provider'] and probe['host_verified']
    host=read(a.probe/'HOST.json');assert host['verified'] and host['replay']['fresh_process']
    assert read(a.probe/'TRAIN_PROCESS.json')['returncode']==0
    assert read(a.probe/'REPLAY_PROCESS.json')['returncode']==0
    # This probe must have been run against the same runtime contract/helper surface.
    assert read(a.probe/'spec.json')['contract_sha256']==sha(a.runtime/'CONTRACT.json')
    responses=list(a.probe.glob('provider-*/response.json'));assert responses
    for f in responses:
        usage=read(f)['usage'];assert usage['model_requested']=='gpt-5.6-terra' and usage['effort']=='low'
        assert usage['model_returned']=='gpt-5.6-terra'
        assert read(f.with_name('request.json'))['expected_base_url']=='https://apizh-ai.com/v1'
        if a.amendment_tests:
            assert sha(f.with_name('raw_response.bin'))==usage['raw_response_sha256']
    specs=read(a.runtime/'memory_specs.json')
    binding=read(a.binding)
    assert binding['source_round']==1
    for k,v in specs.items():
        assert v['sha256']==binding['sha256'][k]
        v['path']=str(a.runtime/v['path'])
    bridge=MemoryBridge(specs,'leaf-classification',contract['data_manifest_sha256'])
    decisions=read(a.probe/'MEMORY_DECISIONS.json');assert decisions['read_ids']
    assert bridge.prompt(decisions['read_ids'])==decisions['prompt_body']
    assert all(d['key'] in decisions['read_ids'] for d in decisions['decisions'])
    source=read(a.source_audit);assert source['passed'] and source['analyst_lessons_read_against_sources']
    assert source['binding_sha256']==sha(a.binding)
    ops=read(a.operations);assert ops['passed'] and ops['timeout_and_resume']
    smoke=read(a.smoke/'HOST.json');assert smoke['verified']
    assert read(a.smoke/'PARTIAL_GATES.json')['gpu_identity_verified']
    unchanged=['sandbox.py','replay.py','host.py']
    if a.amendment_tests:
        tests=read(a.amendment_tests)
        assert tests['passed'] and tests['returncode']==0 and tests['image']==ops['image']
        assert tests['runtime_manifest_sha256']==sha(a.runtime/'RUNTIME_MANIFEST.json')
        assert 'Ran 15 tests' in tests['stdout']+tests['stderr']
    else:unchanged.append('budget_checkpoint.py')
    for filename in unchanged:
        # Earlier isolation engineering smoke is reusable only for byte-identical code.
        assert sha(a.runtime/filename)==sha(a.runtime.parents[1]/'preflight-v004/code'/filename)
    result={k:True for k in ('provider_generated','sandbox_negative_tests','checkpoint_fresh_reload',
        'host_recomputed','gpu_identity_verified','memory_source_audit','timeout_and_resume',
        'same_dependencies','input_manifest_verified')}
    result.update(contract_sha256=sha(a.runtime/'CONTRACT.json'),runtime_sha256=sha(a.runtime/'RUNTIME_MANIFEST.json'),
        memory_specs_sha256=sha(a.runtime/'memory_specs.json'),approved_round=2,
        image=ops['image'],engineering_scores_excluded_from_memory=True,
        artifacts={str(f):sha(f) for f in (a.probe/'AGENT_PROBE.json',a.probe/'HOST.json',a.probe/'MEMORY_DECISIONS.json',
            a.smoke/'HOST.json',a.operations,a.source_audit,a.binding)},
        limitation='This gate proves execution and evidence transport, not causal memory benefit or arbitrary-model scientific correctness.')
    if a.amendment_tests:result['artifacts'][str(a.amendment_tests)]=sha(a.amendment_tests)
    atomic_json(a.output,result);print(json.dumps(result))
if __name__=='__main__':main()
