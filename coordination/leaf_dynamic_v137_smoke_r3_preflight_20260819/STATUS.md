# v137-smoke-r3 preflight and launch status

## Immutable identities

- Branch: `codex/dynamic-retrieval-gpt56sol-v137`
- Implementation commit: `03d695ca`
- Release builder/path/cache fixes: `b92447d2`, `7fbc6627`, `6e81e5f2`
- Frozen release commit: `52a257fa`
- Runtime: `/workspace/nautilus-exp-end2end-agent-v137-smoke-r3`
- Source lock: `bfc4e4f80af39c0e563d7dcf772525dc89739cf3ebf918326137e00b574dbff4`
- Source-locked files: 677; actual allowed files: 679; bad/extra: 0/0
- Evaluator: `/workspace/experiment-end2end-leaf-official-evaluator-v137-smoke-r3`
- Output: `/workspace/experiment-end2end-memory-agent-v137-smoke-r3/runs`
- Logical run: `e2e-smoke-leaf-dynamic-retrieval-official-gpt56sol-v137-smoke-r3__leaf-classification__dynamic_hybrid__seed-1`
- Execution manifest: `9102710e8d45437eec4f7372319b0eaf34a9374e602927ee50f8e8691c5a42c5`

## Verification

- Host focused regressions: 54/54 passed.
- Locked runtime focused regressions: 18/18 passed.
- Source lock passed before and after container tests.
- Runtime contains no `*.pyc`, `__pycache__`, `.pytest_cache`, or `._*` files.
- Official evaluator rows: train=990, test=594, sample=594.
- Test SHA256: `c70a539ec7a5e900af69307ed3f630ff2650497870170cb2d130b4158e27eeda`.
- Sample SHA256: `09c16c65e54876a01bafe82b140c0feeefd1b97ca81147f87b33c60879006e1a`.
- Real `run_assignment.build_solver_command` dry-run opened Memory Bundle v8 and the evaluator.
- All solver roles resolved to `gpt-5.6-sol` through `https://apizh.net/v1`.
- Effective fixed holdout disabled; explicit protocol bypass enabled; broad leakage gate disabled; official submission enabled; maximize=false.
- Budget: 5 agent steps, 1 A100, 1 search worker, 16 CPU, 64 GiB.

## Kubernetes

- Staging Pod: `mlevolve-leaf-gpt56sol-v137-smoke-r3-stager`
- Staging Pod UID: `adc851c8-25ea-46a8-99ca-5afc3c6fd805`
- Staging Pod natural terminal phase: Succeeded.
- Smoke Job: `mlevolve-leaf-gpt56sol-v137-smoke-r3-dynamic-hybrid`
- Smoke Job UID: `9c4acb10-c3a7-438c-ac76-6db4c92bfe63`
- Job was created exactly once with `backoffLimit=0`; no Pod/attempt exists yet.
- Current admission blocker: namespace `a100-limit` uses 8/8 requests. Four are the shared unowned `gpu-dev2` Deployment and four are retained user runs (v135 Dynamic plus v136 Flat/No Memory/SOP). The Job is retained for natural controller retry; no existing run will be stopped for quota.
