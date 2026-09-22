"""Bind read/decision receipts to executed code and reported runtime parameters.
Reports observation, not causal credit or proof of arbitrary Python semantics.
"""
import json
from pathlib import Path
from core import sha, atomic_json


def audit(folder, host_receipt):
    folder=Path(folder)
    decision_file=folder/'MEMORY_DECISIONS.json'
    if not decision_file.exists():return
    decisions=json.loads(decision_file.read_text())
    events_file=folder/'artifacts/training_events.jsonl'
    events=[json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
    published=[]
    for receipt in sorted((folder/'artifacts').glob('checkpoint-*/receipt.json')):
        published.append({'receipt_sha256':sha(receipt),'metadata':json.loads(receipt.read_text()).get('metadata',{})})
    report={'schema':'leaf_memory_execution_observation_v1','code_sha256':sha(folder/'train.py'),
            'decision_receipt_sha256':sha(decision_file),'host_scored':bool(host_receipt and host_receipt.get('verified')),
            'decisions':decisions['decisions'],'actual_training_events':events,'published_metadata':published,
            'scope':'Code excerpts checked at proposal time; actual candidate-reported runtime parameters and checkpoint metadata retained. Host independently verifies predictions/score, not causal attribution of any memory.'}
    atomic_json(folder/'MEMORY_EXECUTION_EVIDENCE.json',report)
