# ClaudeCode Review Request: Run-Forest Online Memory Pilot

## Purpose

This document records the implementation and experiment setup for the online test of the new **Agentic Run-Forest Memory** system.

User request:

- Launch a Kubernetes Job with **4x RTX A6000, 12 CPU, 64Gi memory**.
- Test the new memory system across multiple tasks.
- Put the memory into both:
  - cold-start memory, and
  - runtime memory before draft/improve/debug/evolution/fusion.
- Keep the original cold-start model template unchanged, so comparison against previous no-RunForest runs is fair.
- Monitor navigator behavior, retrieval quality signals, and adoption rate.
- Compare against previous runs without this memory structure.

## High-Level Design

The memory carrier is now:

```text
Run/journal forest = primary hyperbolic structure
Transition = parent -> child change path
SOP = signpost distilled from transitions
Evidence = code/metric/error proof attached to transitions
```

Runtime uses `RunForestMemoryLayer` read-only:

```json
{
  "matched_run_paths": ["run_x/node_7 -> node_12 -> node_19"],
  "selected_transitions": ["transition_a"],
  "attached_sops": ["sop_x"],
  "risk_warnings": ["sibling branch repeated this error"],
  "evidence_refs": ["evidence_y"]
}
```

## Critical Fairness Fix

The first attempt appended the Run-Forest cold-start pack directly to `cfg.coldstart.description`.

That was **wrong for this experiment**, because the user wants the cold-start model template to remain identical to the previous no-memory run.

Current implementation:

- `coldstart_description` remains the original model-template guidance.
- Run-Forest cold-start memory is stored separately in:
  - `agent.coldstart_external_memory_text`
  - `agent.coldstart_external_ref_ids`
  - `agent.coldstart_external_source`
- Draft prompts inject this separate pack into the external memory section, not into the model-template Option A text.
- Smoke check:

```text
template_contains_runforest False
external_text_contains_runforest True
external_refs 18
source run_forest_memory
```

## Code Diff Overview

Main runtime files:

- `mlevolve/agents/memory/external_skill_memory.py`
  - Adds/uses `RunForestMemoryLayer`.
  - Adds `external_memory_section_intro`.
  - Formats external memory as read-only map path packs.
  - Fixes Run-Forest LLM navigator payload to JSON string before `llm.query`, because nested dict/list/int payloads break `compile_prompt_to_md`.
  - Improves `read_skillgraph_node_text` formatting for Run/RunNode/Transition/Evidence nodes so adoption tracking can judge Run-Forest refs.

- `mlevolve/engine/coldstart/knowledge.py`
  - Adds `_build_run_forest_coldstart_text`.
  - Keeps original model-template text unchanged.
  - Stores Run-Forest cold-start text and ref ids in separate side-channel globals.

- `mlevolve/engine/agent_search.py`
  - Instantiates `RunForestMemoryLayer` when mode/source/path contains `run_forest`.
  - Carries cold-start external memory side-channel into the agent instance.

- `mlevolve/agents/draft_agent.py`
  - Injects separate Run-Forest cold-start map pack plus runtime draft navigator pack into the external memory section.
  - Logs cold-start Run-Forest ref ids separately for adoption tracking.

- `mlevolve/agents/{improve,debug,evolution,fusion,aggregation}_agent.py`
  - Use `external_memory_section_title` and `external_memory_section_intro`.
  - Run-Forest packs are described as map path packs, not generic SOP cards.

- `mlevolve/agents/coder/stepwise_coder.py`
  - Stepwise coder wording now accepts either SOP cards or run-forest map path packs.

Experiment scripts:

- `paper-skills/hyper_memory/run_runforest_online_matrix.py`
  - Runs the memory-enabled condition across multiple tasks.
  - Default tasks:
    - `spooky-author-identification`
    - `aerial-cactus-identification`
    - `leaf-classification`
    - `new-york-city-taxi-fare-prediction`
  - Uses `MLEVOLVE_CONFIG=./config/config_run_forest_agentic.yaml`.
  - Forces:
    - `agent.search.num_gpus=4`
    - `agent.search.parallel_search_num=4`
    - `cpu_number=12`
    - `external_skill_memory.mode=run_forest_agentic`
    - `adoption_tracking.judge_mode=llm-all`
  - Writes `runforest_online_manifest.jsonl`.

- `paper-skills/hyper_memory/summarize_runforest_online_matrix.py`
  - Reads the manifest.
  - Extracts best metrics from new runs.
  - Finds historical same-task no-RunForest runs.
  - Reads `adoption_report.json`.
  - Greps RunForest logs for navigator/fallback/strategy counts.
  - Writes JSON and Markdown summaries.

Kubernetes files:

- `job-runforest-online-a6000x4.yaml`
  - Job, not long-lived pod.
  - Requests and limits:
    - `nvidia.com/rtxa6000: "4"`
    - `cpu: "12"`
    - `memory: "64Gi"`
  - Uses node affinity `NVIDIA-RTX-A6000`.
  - Clones the pushed branch into a fresh workdir on PVC.
  - Symlinks `/workspace/nautilus/mlevolve/.env`.
  - Runs preflight compile/tests.
  - Runs the multi-task matrix.
  - Runs the summarizer.

- `pod-runforest-sync.yaml`
  - Lightweight 1 CPU sync/check pod.
  - Used only to inspect PVC state.

## Tests Run Locally

```bash
python -m py_compile \
  paper-skills/hyper_memory/run_runforest_online_matrix.py \
  paper-skills/hyper_memory/summarize_runforest_online_matrix.py \
  mlevolve/engine/coldstart/knowledge.py \
  mlevolve/engine/agent_search.py \
  mlevolve/agents/draft_agent.py \
  mlevolve/agents/memory/external_skill_memory.py

pytest -q tests/test_run_forest_memory.py tests/test_hyperbolic_memory.py
```

Result:

```text
21 passed
```

YAML sanity:

```text
job-runforest-online-a6000x4.yaml Job runforest-online-a6000x4
requests/limits: cpu=12, memory=64Gi, nvidia.com/rtxa6000=4
```

## PVC Inspection

PVC repo:

```text
/workspace/nautilus
current PVC branch before sync: beta3fullmlebench
current PVC commit before sync: d3993a3
```

Available data checked on PVC:

```text
spooky-author-identification: 5.1M
aerial-cactus-identification: 31M
leaf-classification: 62M
new-york-city-taxi-fare-prediction: 6.9G
denoising-dirty-documents: 239M
mlsp-2013-bird: 2.5G
```

Experiment uses the first four tasks for text/image/tabular coverage and historical baseline availability.

## Job Code Delivery

The user asked whether code can enter PVC by pushing to remote, then letting the Job pull it.

Yes. The Job is designed to:

1. Read the existing tokenized origin remote from `/workspace/nautilus`.
2. Clone branch `codex/hyperbolic-structural-memory` into:

```text
/workspace/nautilus_runforest_online_${RUN_TAG}
```

3. Run from that clean checkout.
4. Write run outputs into:

```text
/workspace/nautilus/mlevolve/runs
```

This avoids mutating the old PVC checkout and keeps the experiment source reproducible by branch/commit.

## Cluster Status Checkpoint

As of the latest readonly monitoring checkpoint, `2026-07-09 06:28:32 CST`, the code has been committed and pushed to:

```text
branch: codex/hyperbolic-structural-memory
commit: cc07cea Add run-forest online memory pilot
```

First submitted Job:

```text
job: runforest-online-a6000x4
```

This Job scheduled onto `gpu00.nrp.hpc.udel.edu` but failed before the container command ran. Kubernetes reported an NVIDIA runtime/CDI device-handle error:

```text
failed to create NVIDIA Container Runtime
failed to get device handle from UUID: Unknown Error
```

No experiment code, PVC checkout, preflight tests, matrix run, navigator, or adoption analysis executed in that failed attempt.

A retry Job was created with the same resource shape and same branch, excluding only the node that produced the NVIDIA runtime/CDI startup failure:

```text
job: runforest-online-a6000x4-r2
pod: runforest-online-a6000x4-r2-bnfdh
resources: 4x RTX A6000, 12 CPU, 64Gi
status at checkpoint: Pending, 0/1, no node assigned, age 179m
```

The retry pod has remained Pending during readonly monitoring from approximately `05:53` through `06:28 CST`. Recent scheduler events still show `FailedScheduling` because no suitable node is currently available under the requested constraints. Per the user's explicit instruction, while this pod is Pending, Codex must not delete/recreate the job, lower resources, switch GPU type, or mutate the job spec. The correct action is to keep waiting and monitor read-only.

Because the retry Job has not reached Running, there are not yet any runtime logs, GPU process state, RunForest navigator traces, adoption reports, matrix summaries, or performance comparison results to review.

Current blocker: external cluster scheduling. The experiment is queued with the requested fixed resource shape; completion now requires the Kubernetes scheduler to assign a suitable 4x RTX A6000 node or the user to explicitly change the requested constraints.

## Latest Resource Retarget: A100x3

The user then updated the requested online pilot resource shape. The latest active target is now:

```text
job yaml: job-runforest-online-a100x3.yaml
job name: runforest-online-a100x3
resources: 3x A100, 6 CPU, 64Gi
gpu resource key: nvidia.com/a100
branch cloned by Job: codex/hyperbolic-structural-memory
```

The previous A6000 Jobs were not deleted or mutated. The new A100 Job keeps the same Run-Forest online memory test design:

- clone the pushed branch from remote into a fresh PVC workdir;
- symlink `/workspace/nautilus/mlevolve/.env`;
- run preflight compile plus `pytest -q tests/test_run_forest_memory.py`;
- run the same four-task matrix through `run_runforest_online_matrix.py`;
- inject Run-Forest memory through cold-start side-channel plus runtime external memory;
- keep the original cold-start model-template text unchanged;
- write the manifest and summary under `/workspace/nautilus/mlevolve/runs`.

YAML validation performed locally before submission:

```text
kind: Job
metadata.name: runforest-online-a100x3
namespace: ecepxie
requests/limits: cpu=6, memory=64Gi, nvidia.com/a100=3
runner args: --num-gpus 3 --cpu-number 6
kubectl apply --dry-run=client: job.batch/runforest-online-a100x3
```

Submission result for the first A100 attempt:

```text
job: runforest-online-a100x3
pod: runforest-online-a100x3-5jsnc
node: node-1-1.sdsc.optiputer.net
status: Failed / pod Error
failure point: git clone before code checkout completed
error: RPC failed; curl 56 Recv failure; early EOF; invalid index-pack output
```

This failure happened before `checked_out_commit`, preflight tests, matrix execution, navigator calls, or adoption analysis. It is a code-delivery/network failure, not a Run-Forest runtime failure.

A retry YAML was added:

```text
job yaml: job-runforest-online-a100x3-r2.yaml
job name: runforest-online-a100x3-r2
resources: 3x A100, 6 CPU, 64Gi
kubectl apply --dry-run=client: job.batch/runforest-online-a100x3-r2
```

The r2 job keeps the same experiment and resource shape, but makes code checkout more robust:

- retry remote shallow clone up to three times;
- if repeated remote shallow clone fails, seed from the existing PVC repo and then `git fetch --depth=1 origin codex/hyperbolic-structural-memory`;
- still run from the remote branch contents, preserving the pushed-branch provenance requirement.

Submission result for the second A100 attempt:

```text
job: runforest-online-a100x3-r2
pod: runforest-online-a100x3-r2-jq6rh
node: node-1-1.sdsc.optiputer.net
status: Failed / pod Error
checked_out_commit: aa11d958c8f47253538e0488174bf86797b69bfc
failure point: preflight pytest path
error: file or directory not found: tests/test_run_forest_memory.py
```

The r2 checkout succeeded, but the command was running from `${WORKDIR}/mlevolve`, while the test file lives at `${WORKDIR}/tests/test_run_forest_memory.py`. Therefore the test path should be `../tests/test_run_forest_memory.py`.

A third retry YAML was added:

```text
job yaml: job-runforest-online-a100x3-r3.yaml
job name: runforest-online-a100x3-r3
resources: 3x A100, 6 CPU, 64Gi
kubectl apply --dry-run=client: job.batch/runforest-online-a100x3-r3
preflight fix: pytest -q ../tests/test_run_forest_memory.py
```

The r3 checkout strategy also prefers the already-successful r2 checkout as a PVC local seed, then fetches the pushed branch from remote. This avoids spending another long A100 allocation on a full remote checkout while still validating the remote branch provenance.

Live r3 checkpoint as of `2026-07-09 14:42:20 CST`:

```text
job: runforest-online-a100x3-r3
pod: runforest-online-a100x3-r3-58772
node: node-1-1.sdsc.optiputer.net
status: Running, 0/1 completions
checked_out_commit: eb754c1671f02b395ac7b2eb9473faacbb7fe186
run_tag: runforest_online_a100x3_r3_20260709_045537
current task: spooky-author-identification
run dir: /workspace/nautilus/mlevolve/runs/20260709_053545_runforest_online_a100x3_r3_20260709_045537_spooky-author-identification_runforest
```

Verified runtime behavior in r3:

- `RunForestMemory` loaded `6666 nodes / 15040 edges`, `scoring=poincare`, `agentic=True`.
- `AgentSearch` enabled external memory with `source=run_forest_agentic_memory`.
- ThreadPool max workers is `3`, matching the requested 3 GPUs.
- Draft navigation fired before each draft:
  - `stage=draft strategy=draft_successful_branches`
  - returned transition refs plus SOP refs, e.g. `sop::sg_0202`, `sop::sg_0204`, `sop::sg_0222`.
- Debug navigation fired after a failed draft:
  - `stage=debug strategy=debug_failure_recovery`
  - returned failure-recovery transition refs plus SOP refs.
- Initial three draft executions were bound to GPU `0`, `1`, and `2` respectively.
- At checkpoint, all three A100s were active in training:
  - GPU0 about `30GiB`, high utilization;
  - GPU1 about `35GiB`, high utilization;
  - GPU2 about `14GiB`, high utilization after debug retry started.

Current online-result status:

- The first task has started and Run-Forest memory is visibly active.
- No successful metric has been reported yet; journal currently contains the root plus one buggy draft.
- `runforest_online_manifest.jsonl` is still empty because the matrix runner writes a row after each task finishes.
- `adoption_report.json` is not present yet; adoption analysis is expected after enough generated nodes / run completion.

Additional live checkpoint as of `2026-07-09 15:00:30 CST` / `07:00:30 UTC`:

```text
job: runforest-online-a100x3-r3
pod: runforest-online-a100x3-r3-58772
status: Running, 0/1 completions
age: about 124 minutes
restarts: 0
node: node-1-1.sdsc.optiputer.net
```

GPU/process status from read-only `kubectl exec`:

```text
GPU0: NVIDIA A100-SXM4-80GB, 30363 MiB used, 100% utilization
GPU1: NVIDIA A100-SXM4-80GB, 35307 MiB used, 96% utilization
GPU2: NVIDIA A100-SXM4-80GB, 13927 MiB used, 96% utilization
top worker processes: three Python workers at roughly 95-96% CPU each
```

Run artifact status at this checkpoint:

```text
matrix dir: /workspace/nautilus/mlevolve/runs/runforest_online_a100x3_r3_20260709_045537_matrix
manifest: runforest_online_manifest.jsonl exists but is still 0 bytes
spooky log: spooky-author-identification.log exists, about 65 KiB
journal: exists, about 145 KiB
journal nodes: 2
metric_count: 0
best_min: None
latest nodes: root plus buggy draft 9074a44d9e64432aa8e2c5347d0d3a8d
```

Interpretation: this is still an active run, not a completed result. The pod is not Pending and does not require any mutation. The three GPUs are busy, so the correct action is continued read-only monitoring until either the first long-running draft finishes or the Job fails/completes.

Follow-up read-only health check at `2026-07-09 15:03:23 CST` / `07:03:23 UTC`:

```text
job status: still Running, 0/1 completions, about 127 minutes old
pod status: Ready 1/1, Running, restarts=0
GPU0: 30363 MiB used, 94% utilization
GPU1: 35307 MiB used, 99% utilization
GPU2: 13927 MiB used, 98% utilization
journal nodes: still 2
metric_count: still 0
manifest: still 0 bytes
adoption_report/adoption_events: not yet present
summary json/md: not yet present
```

Interpretation unchanged: the Job is still spending GPU time on the first task's long-running draft/debug executions. There is no evidence of a scheduler Pending state, pod restart, or completed online result yet.

Additional material checkpoint at `2026-07-09 15:10:17 CST` / `07:10:17 UTC`:

```text
job status: still Running, 0/1 completions, about 132 minutes old
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 3
metric_count: 0
manifest: still 0 bytes
summary/adoption report: not yet present
```

New runtime behavior since the previous checkpoint:

```text
node fe6beb7438c948dc8f5e5c1b1f56c266 finished execution parsing
result: FAIL / is_buggy=True / metric=None
progress: 2/80 steps completed, 3 tasks running
RunForestMemory fired again for debug:
  stage=debug
  strategy=debug_failure_recovery
  refs include transitions:
    run::20260509_154039_spooky-author-identification::transition::dc633aebfe::1852b63b5b
    run::20260509_154039_spooky-author-identification::transition::dc633aebfe::b926644769
    run::20260511_014836_spooky-author-identification::transition::31e12b2fef::5be1911c1a
  refs include SOPs:
    sop::sg_0096, sop::sg_0101, sop::sg_0094, sop::sg_0156
debug patch:
  Successfully applied 6 diff patch(es)
  new child: bd5021e06d034427847f1a9373ef8807
```

GPU/process snapshot immediately after that debug handoff:

```text
GPU0: 897 MiB used, 0% utilization
GPU1: 4 MiB used, 0% utilization
GPU2: 13927 MiB used, 0% utilization
top processes: two Python workers still active, one near 99% CPU and one near 97% CPU
```

Interpretation: this is no longer the earlier "all GPUs busy" state. One more draft has failed, Run-Forest debug navigation is confirmed a second time, and the run appears to be in a CPU/debug handoff or task-transition phase. It is still not a final online result; no metric, adoption report, manifest row, or summary exists yet.

Additional live checkpoint at `2026-07-09 15:14:28 CST` / `07:14:28 UTC`:

```text
job status: still Running, 0/1 completions, about 137 minutes old
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 5
metric_count: 0
manifest: still 0 bytes
summary/adoption report: not yet present
```

New runtime behavior since the previous checkpoint:

```text
node 758c132e22e94135bc7889d0e8e657f7 finished execution parsing
result: FAIL / is_buggy=True / metric=None
failure class in log: RuntimeError; no metric value reported; submission file not found
progress: 3/80 steps completed, 3 tasks running
RunForestMemory fired again for debug:
  stage=debug
  strategy=debug_failure_recovery
  refs include transitions:
    run::20260516_125444_spooky-author-identification::transition::cc9848eb59::39c03723bd
    run::20260513_165253_spooky-author-identification::transition::287ed0c96b::07eef72d46
    run::20260509_185008_spooky-author-identification::transition::0d800b57b4::cbe7d283fe
  refs include SOPs:
    sop::sg_0112, sop::sg_0085, sop::sg_0148, sop::sg_0149
debug patch:
  Successfully applied 2 diff patch(es)
  new child: 356ed2ef23c34618baf2f0dcad95168a
```

The earlier debug child for `fe6beb...` also finished and failed:

```text
node bd5021e06d034427847f1a9373ef8807
parent: fe6beb7438c948dc8f5e5c1b1f56c266
stage: debug
result: FAIL / is_buggy=True / metric=None
failure class in log: TypeError; no metric value reported; submission file not found
progress after parse: 4/80 steps completed, 3 tasks running
RunForestMemory fired again for debug:
  stage=debug
  strategy=debug_failure_recovery
  refs include SOPs/evidence:
    sop::sg_0187, sop::sg_0004, sop::sg_0189, evidence::117dabde73e6
debug patch:
  Successfully applied 2 diff patch(es)
  new child: ddbc3a59afa4427e8928125c25d4b407
```

GPU/process snapshot:

```text
GPU0: 7961 MiB used, 81% utilization
GPU1: 34735 MiB used, 94% utilization
GPU2: 13927 MiB used, 0% utilization
top processes: three Python workers still active, around 98-101% CPU each
```

Interpretation: the first task is still failing to produce a valid metric, but this is useful evidence for the runtime-memory requirement. The Run-Forest navigator has now repeatedly fired in debug mode across multiple failed branches and supplied transition/SOP/evidence refs used before patch generation. Still no online performance comparison or adoption-rate result can be claimed until a valid metric and run summary/adoption artifacts exist.

Additional live checkpoint at `2026-07-09 15:17:12 CST` / `07:17:12 UTC`:

```text
job status: still Running, 0/1 completions, about 141 minutes old
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 6
metric_count: 0
manifest: still 0 bytes
summary/adoption report: not yet present
```

New runtime behavior since the previous checkpoint:

```text
node ddbc3a59afa4427e8928125c25d4b407 finished execution parsing
parent: bd5021e06d034427847f1a9373ef8807
stage: debug
result: FAIL / is_buggy=True / metric=None
failure class in log: RuntimeError; no metric value reported; submission file not found
progress after parse: 5/80 steps completed, 3 tasks running
RunForestMemory fired again for debug:
  stage=debug
  strategy=debug_failure_recovery
  refs include transitions:
    run::20260514_023457_spooky-author-identification::transition::bdcb77ac47::bef409bbb7
    run::20260513_145802_spooky-author-identification::transition::fd4b9d10ce::3b74cb6461
    run::20260514_023457_spooky-author-identification::transition::530dd2b836::126b3fd04d
  refs include SOPs/evidence:
    sop::sg_0187, sop::sg_0001, sop::sg_0085, evidence::117dabde73e6
debug patch:
  Successfully applied 2 diff patch(es)
  new child: 5bb660d469c944bbbb8aca410d6d6078
```

GPU/process snapshot:

```text
GPU0: 1877 MiB used, 0% utilization
GPU1: 34735 MiB used, 96% utilization
GPU2: 13927 MiB used, 0% utilization
top processes: three Python workers still active, around 96-100% CPU each
```

Interpretation: the online run is still alive and actively cycling through debug recovery, but it has not produced any successful metric yet. This strengthens the evidence that runtime Run-Forest memory is wired into repeated debug attempts, while also showing that the current first-task trajectory is struggling to recover from generated-code failures.

Additional live checkpoint at `2026-07-09 15:21:25 CST` / `07:21:25 UTC`:

```text
job status: still Running, 0/1 completions, about 145 minutes old
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 7
metric_count: 0
manifest: still 0 bytes
summary/adoption report: not yet present
```

New runtime behavior since the previous checkpoint:

```text
node cbca06f530e64186b6a70754e3160623 finished execution parsing
parent: 9074a44d9e64432aa8e2c5347d0d3a8d
stage: debug
result: FAIL / is_buggy=True / metric=None
failure class in log: execution error detected; no metric value reported
progress after parse: 6/80 steps completed, 3 tasks running
RunForestMemory fired again for debug:
  stage=debug
  strategy=debug_failure_recovery
  refs include transitions:
    run::20260509_154039_spooky-author-identification::transition::dc633aebfe::1852b63b5b
    run::20260517_151325_spooky-author-identification::transition::d44038102d::0acd2ce065
    run::20260514_023457_spooky-author-identification::transition::11dd8825fa::4c59ca9769
  refs include SOPs:
    sop::sg_0268, sop::sg_0267, sop::sg_0185, sop::sg_0186
debug patch:
  Successfully applied 1 diff patch(es)
  new child: 8949d0497f9e427eb4c0d8f2b4f6b4cb
```

GPU/process snapshot:

```text
GPU0: 7961 MiB used, 77% utilization
GPU1: 35307 MiB used, 3% utilization
GPU2: 4 MiB used, 0% utilization
top processes: three mlevolve worker Python processes still active, plus one short-lived Python from monitoring
```

Interpretation: still no valid metric, but the job remains live. Runtime memory evidence is now extensive for debug-stage navigation, though the first task is stuck in repeated generated-code failure recovery rather than reaching a score.

Additional material checkpoint at `2026-07-09 15:24:07 CST` / `07:24:07 UTC`:

```text
job status: still Running, 0/1 completions, about 148 minutes old
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 8
metric_count: 1
best_min: 0.37155
manifest: still 0 bytes because the task has not finished
summary/adoption report: not yet present
```

First successful metric:

```text
node: 5bb660d469c944bbbb8aca410d6d6078
stage: debug
parent: ddbc3a59afa4427e8928125c25d4b407
is_buggy: False
metric: 0.37155
maximize: False
best updated: node 5bb660d469c944bbbb8aca410d6d6078, metric=0.37155
```

Validation/leakage evidence:

```text
format validation: passed
content quality: valid
metric direction: maximize=False
data leakage check: has_leakage=False, confidence=high
parse result: PASS | metric=0.37155
```

The run then moved from debug recovery into improve:

```text
progress: 7/80 steps completed, 3 tasks running
next stage: normal improve for node 5bb660d469c944bbbb8aca410d6d6078
RunForestMemory fired for improve:
  stage=improve
  strategy=improve_local_best_lineage
  refs include transitions:
    run::20260516_125444_spooky-author-identification::transition::92989935c3::bfbf637cc9
    run::20260511_102550_spooky-author-identification::transition::8fa40b1647::dd6126c1cf
    run::20260516_125444_spooky-author-identification::transition::92989935c3::669be7d1fe
  refs include SOPs/evidence:
    sop::sg_0002, sop::sg_0222, sop::sg_0230, sop::sg_0228, sop::sg_0221, sop::sg_0164, evidence::79141a4233f1
improve path:
  diff improve with two-stage planning and memory
```

GPU/process snapshot:

```text
GPU0: 1001 MiB used, 0% utilization
GPU1: 35307 MiB used, 93% utilization
GPU2: 13393 MiB used, 11% utilization
top processes: mlevolve Python workers still active
```

Interpretation: this is the first online success signal. Runtime Run-Forest memory has now been verified in three modes on the same live task: draft retrieval, repeated debug/failure-recovery retrieval, and improve/local-best-lineage retrieval. The online matrix is still not complete and no task-level manifest/adoption summary exists yet, but the first valid metric and leakage-clean success node are now present.

Additional live health checkpoint at `2026-07-09 15:28:18 CST` / `07:28:18 UTC`:

```text
job status: still Running, 0/1 completions, about 152 minutes old
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 8
metric_count: 1
best_min: 0.37155
manifest: still 0 bytes because the first task has not finished
summary/adoption report: not yet present
```

Recent runtime state:

```text
after first valid metric, the run entered improve on node 5bb660d469c944bbbb8aca410d6d6078
RunForestMemory fired for improve with strategy=improve_local_best_lineage
diff improve used two-stage planning with memory
new improve child: 37243e8669764c6abe3e5de076963c4f
```

GPU/process snapshot:

```text
GPU0: 8065 MiB used, 69% utilization
GPU1: 35307 MiB used, 94% utilization
GPU2: 13393 MiB used, 70% utilization
top processes: three active Python workers around 91-100% CPU
```

Interpretation: no new task-level artifact exists yet, but this checkpoint verifies the online Job is not idle after the first success. The first task is still executing, improve-stage memory is active, and all three requested A100 GPUs are doing work.

Additional material checkpoint at `2026-07-09 15:37:17 CST` / `07:37:17 UTC`:

```text
job status: still Running, 0/1 completions
pod status: Ready 1/1, Running, restarts=0
current task: still spooky-author-identification
journal nodes: 9
metric_count: 2
best_min: 0.37155
manifest: still 0 bytes because the first task has not finished
adoption/summary artifacts: not yet present
```

First improve child completed:

```text
node: 37243e8669764c6abe3e5de076963c4f
stage: improve
parent: 5bb660d469c944bbbb8aca410d6d6078
is_buggy: False
metric: 0.376155
maximize: False
best remained: 0.37155 from node 5bb660d469c944bbbb8aca410d6d6078
improvement relative to best: -0.004605 for a minimize metric
parse result: PASS | metric=0.376155
data leakage check: has_leakage=False, confidence=high
```

Follow-up improve retrieval after that node:

```text
progress: 8/80 steps completed, 3 tasks running
next stage: normal improve for node 37243e8669764c6abe3e5de076963c4f
RunForestMemory fired again:
  stage=improve
  strategy=improve_local_best_lineage
  refs include transitions:
    run::20260516_125444_spooky-author-identification::transition::92989935c3::bfbf637cc9
    run::20260516_125444_spooky-author-identification::transition::644318bb19::7febc5ae80
    run::20260517_151325_spooky-author-identification::transition::16114ca8db::fd8710a1b5
    run::20260517_151325_spooky-author-identification::transition::5fffd41185::16114ca8db
  refs include SOPs:
    sop::sg_0002, sop::sg_0222, sop::sg_0230, sop::sg_0228, sop::sg_0221, sop::sg_0238
new improve child submitted:
  24ba5082e39f4494a065c469520c5457
```

GPU/process snapshot:

```text
GPU0: 1001 MiB used, 0% utilization
GPU1: 35307 MiB used, 89% utilization
GPU2: 13927 MiB used, 5% utilization
top processes: three active Python workers still running around 95-99% CPU
```

Interpretation: this checkpoint verifies the first Run-Forest-guided improve produced a valid, leakage-clean solution but did not beat the debug-derived best. The system continued to use Run-Forest memory for the next improve attempt, so retrieval remains active beyond the first success. Final effect/adoption conclusions still cannot be made until the task/matrix completes and reports are written.

## Review Checklist For ClaudeCode

Please verify:

1. Cold-start model template is not changed by Run-Forest memory.
2. Run-Forest cold-start map pack is injected separately and adoption-logged separately.
3. Runtime RunForest memory still injects before draft/improve/debug/evolution/fusion.
4. `config_path=...` is not used incorrectly; the runner uses `MLEVOLVE_CONFIG`.
5. Latest Job resource requests match user request: 3x A100, 6 CPU, 64Gi. The older A6000 Jobs are historical attempts only.
6. Job main command exits after matrix + summary; no `sleep` keeps the Job alive.
7. Branch cloning into a fresh workdir does not destroy historical runs.
8. Summarizer compares against historical runs that do not contain Run-Forest config.
9. Adoption analysis can fetch text for Transition/Evidence refs, not only SOP refs.
10. DeepSeek agentic navigator receives JSON string payload and can use OpenAI-compatible function calling.

## Known Caveats

- The formal online result is not available until the Kubernetes Job completes.
- Adoption rate with `judge_mode=llm-all` may be slow and API-expensive because every injected memory/code pair is judged.
- If DeepSeek/API fails, `RunForestMemoryLayer` falls back to deterministic stage policy and logs `LLM navigator failed`.
- The comparison uses historical no-RunForest runs on the same PVC, not a freshly rerun baseline in the same Job.
