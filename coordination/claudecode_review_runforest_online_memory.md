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
