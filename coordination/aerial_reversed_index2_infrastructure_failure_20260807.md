# Aerial reversed_router index 2 infrastructure failure

- Observed at: 2026-08-07T13:56:28Z
- Namespace: `ecepxie`
- Job: `mlevolve-e2e-aerial-pilot-v23`
- Job UID: `4a49708a-4bb1-4574-8a47-7dcfd3ee9cd4`
- Completion index: `2`
- System: `reversed_router`
- Pod: `mlevolve-e2e-aerial-pilot-v23-2-qfj2q`
- Pod UID: `937e3319-5d29-4ec7-971c-ffe36a3dc7f5`
- Node: `ren-gp-argo-01.madren.org`
- Classification: infrastructure failure before the 21,600-second search budget
- Retry policy: eligible only as a later immutable retry Job after the frozen fairness runner is published and verified; no retry was submitted while the four-slot Aerial Indexed Job remains active

## Kubernetes evidence

- `2026-08-07T13:37:43Z` `NodeNotReady`: node is not ready.
- `2026-08-07T13:42:49Z` `TaintManagerEviction`: pod marked for deletion.
- `2026-08-07T13:45:27Z` `UnexpectedAdmissionError`: no healthy `nvidia.com/a100` devices were available on the node.
- `2026-08-07T13:45:58Z` `FailedKillPod`: Calico could not reach the Kubernetes API while deleting the pod sandbox.
- Job status at observation: `active=4`, `failed=1`, `failedIndexes="2"`.

The pod object had already been removed by the taint-eviction controller when the 13:56 monitor ran. The event evidence above is therefore the retained replacement for the no-longer-readable Pod describe.

## Persistent run evidence

- Logical run: `e2e-pilot-agentic-three-role-v23__aerial-cactus-identification__reversed_router__seed-1`
- Attempt: `attempt-000`
- Journal: `/workspace/experiment-end2end-memory-agent-v23/runs/e2e-pilot-agentic-three-role-v23__aerial-cactus-identification__reversed_router__seed-1/attempt-000/agent/logs/20260807_122959_e2e-pilot-agentic-three-role-v23__aerial-cactus-identification__reversed_router__seed-1/journal.json`
- Journal SHA-256: `875086932d6bcb47bf0610fcaa463d227e58ddd1c821eb489b71d9afcf7aef02`
- Journal bytes: `824921`
- Nodes: `9` including root; latest step: `8`
- Last node creation time: `2026-08-07T13:30:29`
- Best internal ROC-AUC observed before interruption: `0.9999947000914234`
- `MEASUREMENT.json` and `RUN_OUTCOME.json` were not emitted because the node disappeared before wrapper finalization.

All persistent journal/code artifacts remain on the PVC. No active or pending workload was stopped, deleted, overwritten, or resumed during this observation.
