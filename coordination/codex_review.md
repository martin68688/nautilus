# Codex Review

## Status

Reviewed Claude Code's pre-implementation plan review on 2026-07-02.

## Summary

Claude Code was right to stop before implementation. My previous experiment
plan was useful as a broad scaffold, but it was too permissive for a paper-grade
"hyperbolic" claim. The review correctly identifies that the current artifact
does not yet contain the data needed for the most important experiments:
negative/conflict edges, failure-mode nodes, a clean gold set, and a real
geometry-vs-flat ablation.

I agree with the main direction of the critique: Phase A should not immediately
build result tables that would look quantitative but fail under reviewer
scrutiny. The next task should be narrowed to prerequisites: clean data audit,
negative-edge/failure-mode extraction, and a true hyperbolic-vs-flat comparison
design.

## Findings

- P0: The current "Hyper-Skill pilot" in the plan is only a feature reranker.
  A win there would show that condition/failure/evidence features help, not that
  hyperbolic geometry helps. Claude's objection is correct. If "hyperbolic" is
  in the title, we need a flat twin baseline with identical features and a true
  Poincare/Lorentz variant.
- P0: The current `graph_build/graph.json` cannot support the conflict
  experiment. It has only `enhance` and `co_occur` edges, no
  `prevents/conflicts_with/prereq`, and no `failure_modes` node field. Claude's
  objection is correct.
- P0: The 15-SOP ground truth is too contaminated to use as the main recall
  denominator without audit. It is Run8-derived and repeatedly uses the suspect
  0.0725 result. My plan flagged this as a warning, but Claude is right that it
  should block paper-grade recall claims.
- P1: "Rare = n_use <= 2" is weak because 80% of graph nodes satisfy it. Rarity
  should be redefined as condition-specific scarcity plus value, not raw support
  count alone.
- P1: Static SkillGraph is a floor baseline, not a serious headline baseline,
  because the single-task init graph is almost a dense two-level graph. A
  stronger adapted baseline such as trace-order prereq is needed, but only after
  the user ratifies option C.
- P1: The report has a small process inconsistency: it says "Changed Files:
  None" while shared memory and decisions were updated. The substance is useful,
  but future reports should distinguish "no code changed" from "coordination
  files changed."

## Corrections To My Prior Plan

- Do not implement the full Phase A benchmark harness yet.
- First implement or specify three prerequisites:
  1. deterministic audit of `small-data-transformer-finetuning` gold SOPs;
  2. negative-edge / failure-mode extraction from pre-merge Trace2Skill patches;
  3. geometry ablation: same features in flat Euclidean scoring vs true
     Poincare/Lorentz scoring.
- Keep the original four experiments as the eventual structure, but mark them
  as blocked until the above prerequisites exist.
- Treat `SkillGraph-static` as a diagnostic floor. Use `SkillGraph-C` only after
  the A/B/C retrieval choice is explicitly ratified.

## Recommended Next Claude Task

Replace the current implementation task with a prerequisite task:

1. Build `allowlist_clean_runs.json` from verified clean runs, or report exactly
   why it cannot be reconstructed.
2. Audit the 15 SOPs and split them into `clean`, `suspect`, and `exclude`.
3. Inspect GLM Trace2Skill patch intermediates and propose a concrete schema for
   `Condition`, `FailureMode`, `prevents`, `conflicts_with`, and `refines`.
4. Prototype a tiny geometry sanity check with identical features:
   flat scorer vs Poincare/Lorentz scorer on a manually labeled mini-set.

## Decision Needed

User decision remains central: keep "hyperbolic" as the load-bearing headline,
or reposition the paper as conflict-aware procedural skill memory with
hyperbolic geometry as an optional structure. If keeping the hyperbolic headline,
the geometry ablation is mandatory.

---

# Codex Acceptance Review — Claude Code General Normalization Patch

## Status

Reviewed Claude Code commit `5424433` on 2026-07-03.

Verdict: **mechanically accepted, semantically not yet clean enough for the final
paper baseline**.

Claude Code implemented the requested optimized SkillGraph-style baseline and
the reported counts reproduce locally. However, the active `universal_general`
layer still contains API/model/library-specific impurities because the
normalizer classifies canonical clusters before demoting API warnings, and some
cluster regexes are too broad.

## What Passed

- New deterministic normalizer exists:
  `paper-skills/distillation/normalize_general_nodes.py`.
- Distillation prompt now limits each batch to at most one `general` and adds a
  required `scope` field for future distillations.
- Graph builder supports:
  `--input`, `--output`, and `--selective-general-enhance`.
- Optimized generated graph exists locally:
  `paper-skills/distillation/graph_build/graph_optimized_skillgraph.json`.
- Numeric acceptance criteria reproduce:
  - nodes: `306 -> 242`
  - active `general`: `113 -> 6`
  - edges: `22358 -> 2165`
  - enhance edges: `21809 -> 1416`
  - optimized faithful retrieval: `6 general + 2 task-specific` for each demo task
  - optimized with `--general-cap 2`: `2 general + 6 task-specific`

This fixes the original visible failure mode where faithful retrieval returned
`8 general + 0 task-specific`.

## Main Finding

The optimized graph passes the count test but the six remaining universal SOPs
are not fully clean.

There is also an important process gap: **Part 1 changed the prompt source code
for future distillations, but the current optimized graph was not produced by
re-running distillation with that prompt**. Local artifacts confirm this:

- `graph_build/raw_nodes.json`: `324` nodes, `118` general, `0` nodes with
  `scope`;
- `graph_build/merged_nodes.json`: `306` nodes, `113` general, `0` nodes with
  `scope`;
- `graph_build/merged_nodes_general_normalized.json`: `242` nodes, `6` general,
  `242` nodes with `scope`.

So the actual current improvement comes from post-hoc normalization and
selective edge construction, not from the tightened distillation prompt. The
prompt change is real in `distill_skillgraph_nodes.py`, but it has not been
validated end-to-end.

In `normalize_general_nodes.py`, `classify()` currently does this:

1. match canonical universal cluster regexes;
2. only then check `DEMOTABLE`.

That priority lets task/API-specific warnings get absorbed into
`universal_general` before the demotion rule can catch them.

Concrete examples currently absorbed into active universal nodes:

- `set DataLoader num_workers=0 to avoid shared memory issues`
- `disable mixed precision (AMP) for DeBERTa-v3 to avoid attention mask overflow`
- `ensure loss function API compatibility with installed PyTorch version`
- `use label smoothing manually when BCEWithLogitsLoss does not support it`
- `use correct attribute name for HuggingFace model config hidden size`
- `use allow_pickle=True when loading numpy arrays saved as object dtype`
- `save model checkpoints with weights_only=False for custom architectures`
- `use correct import path for PyTorch functions`
- `download all required NLTK resources before feature extraction`

These are useful memories, but they should usually be `api_warning` or
`implementation_note`, not broad universal SOPs.

## Why This Matters

For our research story, the global layer should behave like a small set of
general working habits:

- check script order before execution;
- clean generated code;
- validate data-flow contracts;
- avoid leakage;
- smoke-test before expensive runs;
- check tensor/model interfaces;
- check resource budget.

Right now it is partly that, but also partly a basket that swallowed local
implementation accidents. That is dangerous for a top-conference baseline:
reviewers may say the method wins because the baseline's "general" layer is
hand-pruned but semantically inconsistent.

## Required Fix Before Final Acceptance

Ask Claude Code to revise `normalize_general_nodes.py`:

1. Apply an API/model/library denylist before canonical clustering, or add
   priority rules so known local warnings cannot become `universal_general`.
2. Tighten broad canonical regexes:
   - `shape`, `checkpoint`, `dtype`, `device`, `attribute`, `resource`,
     `.npy`, and `feature matrix` should not automatically imply universal SOP.
3. Split mixed clusters where needed:
   - script-order checks,
   - data-flow file/shape checks,
   - tensor/model interface checks,
   - API/library warnings.
4. Add deterministic guard checks or tests:
   - the examples above must not appear inside active `category=general` nodes.
5. Rebuild:

```bash
python3 paper-skills/distillation/normalize_general_nodes.py
python3 paper-skills/distillation/build_edges_levels.py \
  --input paper-skills/distillation/graph_build/merged_nodes_general_normalized.json \
  --output paper-skills/distillation/graph_build/graph_optimized_skillgraph.json \
  --selective-general-enhance
python3 paper-skills/distillation/skillgraph_retrieve.py \
  paper-skills/distillation/graph_build/graph_optimized_skillgraph.json --demo
```

Expected result after cleanup: active general may remain around `6-8`, or drop
below 6. Retrieval should still return at least two task-specific skills without
`--general-cap`, but semantic purity matters more than forcing the number to be
exactly 6.

## Bottom Line

Claude Code completed a useful first pass. It is good enough as a diagnostic
patch and proves the direction is viable. It is **not yet clean enough to freeze
as the optimized SkillGraph baseline** until the universal-general layer is
purified.

---

# Codex Follow-Up Fix — Distillation-Level General Control

## Status

Implemented by Codex after the user pointed out that Claude Code had changed the
prompt source but had not validated the bottom distillation artifact.

Verdict: **the source-level issue is now fixed and validated end-to-end**.

## Code Changes

- `distill_skillgraph_nodes.py`
  - added deterministic guardrails after teacher output:
    - at most one active `general` per batch;
    - only whitelist-like process SOPs may remain `general`;
    - API/library/model/task-specific/architecture/training-technique items are
      demoted before writing `raw_nodes.json`;
    - true SOPs that the teacher marks too narrowly can be promoted to the
      batch's single `universal_general`;
    - partial checkpointing via `raw_nodes.partial.json`;
    - `--resume` support;
    - API retry/timeout handling.
- `merge_nodes.py`
  - preserves `scope`;
  - keeps merged scope deterministic;
  - adds API retry/timeout handling.
- `normalize_general_nodes.py`
  - checks demotion rules before canonical clustering;
  - uses fixed generic SOP principles for universal nodes;
  - adds evidence gating for canonical clusters;
  - raises if API/model/library-specific titles leak into `universal_general`.

## End-to-End Run

Ran full distillation with the new prompt and guardrails:

```bash
python3 -u paper-skills/distillation/distill_skillgraph_nodes.py --all --batch 5
python3 -u paper-skills/distillation/merge_nodes.py
python3 paper-skills/distillation/normalize_general_nodes.py
python3 paper-skills/distillation/build_edges_levels.py \
  --input paper-skills/distillation/graph_build/merged_nodes_general_normalized.json \
  --output paper-skills/distillation/graph_build/graph_optimized_skillgraph.json \
  --selective-general-enhance
```

`raw_nodes.json` is now produced by the new distillation path:

- `309` raw nodes;
- `12` active raw `general` after deterministic guardrail cleanup;
- every raw node has `scope`;
- `bad_general_count = 0` under the current guardrail check;
- `n_guardrail_sanitized = 3` for deterministic cleanup of already-returned
  teacher outputs during this run.

After merge:

- `288` merged nodes;
- `11` active merged `general`;
- all merged nodes preserve `scope`;
- `bad_general_count = 0`.

After normalization:

- `281` normalized nodes;
- active `general`: `11 -> 2`;
- final universal SOPs:
  - `Clean merged or generated code before execution`;
  - `Run a script-order sanity check before execution`.
- demoted low-evidence canonical nodes: `2`;
- graph edges: `1,559` total:
  - `558` enhance;
  - `1,001` co_occur.

## Retrieval Check

Faithful retrieval on `graph_optimized_skillgraph.json` now returns:

- `aerial-cactus-identification`: `2 general + 6 task-specific`;
- `denoising-dirty-documents`: `2 general + 6 task-specific`;
- `leaf-classification`: `2 general + 6 task-specific`;
- `new-york-city-taxi-fare-prediction`: `2 general + 6 task-specific`;
- `spooky-author-identification`: `2 general + 6 task-specific`.

The previous `--general-cap` workaround is no longer needed for the optimized
graph.

## Residual Risk

One large batch returned `0` skills after consuming the output budget. This is a
teacher/API robustness issue, not a graph-cleanliness issue. If this baseline is
used for final paper tables, rerun that batch or add per-batch output validation
that fails closed instead of accepting an empty skill list.

---

# Codex Follow-Up Fix — SkillGraph-C Trace-Order Prereq

## Status

Implemented an adapted prerequisite-edge builder after the user asked whether
Codex could manually add MLE trace-order `prereq` edges.

Verdict: **implemented and verified as an explicit adapted baseline, not as the
faithful static SkillGraph baseline**.

## Code Changes

- `build_edges_levels.py`
  - added `--trace-order-prereq`;
  - added configurable parameters:
    - `--trace-prereq-window` default `3`;
    - `--trace-prereq-min-support` default `1`;
    - `--trace-prereq-max-per-dst` default `3`;
  - builds `prereq` edges from evidence turn order within the same run branch
    and same task category;
  - aggregates repeated direction evidence and skips ties/reversed conflicts;
  - labels graph meta with `trace_order_prereq=true`.
- `skillgraph_retrieve.py`
  - added `--task-seed-limit N` for query-like/narrow-seed retrieval tests;
  - keeps default broad seed behavior unchanged;
  - filters forward-beam expansion to the queried task type to avoid cross-task
    contamination from global enhance edges.

## Generated Graph

Command:

```bash
python3 paper-skills/distillation/build_edges_levels.py \
  --input paper-skills/distillation/graph_build/merged_nodes_general_normalized.json \
  --output paper-skills/distillation/graph_build/graph_skillgraph_c_trace_prereq.json \
  --selective-general-enhance \
  --trace-order-prereq \
  --trace-prereq-window 3 \
  --trace-prereq-min-support 1 \
  --trace-prereq-max-per-dst 3
```

Result:

- nodes: `281`;
- edges: `1,926`;
- edge breakdown:
  - `enhance`: `558`;
  - `co_occur`: `1,001`;
  - `prereq`: `367`;
- levels now form a deeper hierarchy:
  - level `0`: `2`;
  - level `1`: `86`;
  - level `2`: `65`;
  - level `3`: `44`;
  - continuing to level `10`.

## Retrieval Behavior

Default broad seed still has `backward_bfs=0`, because the current faithful
retriever seeds all nodes of the queried task. If every task node is already in
the seed set, BFS has no non-seed ancestor to add.

With query-like narrow seeds:

```bash
python3 paper-skills/distillation/skillgraph_retrieve.py \
  paper-skills/distillation/graph_build/graph_skillgraph_c_trace_prereq.json \
  --demo --task-seed-limit 6
```

Observed backward-BFS:

- `aerial-cactus-identification`: `1`;
- `denoising-dirty-documents`: `0`;
- `leaf-classification`: `6`;
- `new-york-city-taxi-fare-prediction`: `1`;
- `spooky-author-identification`: `3`.

This confirms that the trace-order `prereq` edges are usable, but they matter
only once retrieval is query-like or seed-limited. For live mlevolve injection,
the seed should be the current task/query context, not the entire task category.

## Interpretation

This graph should be reported as:

- `SkillGraph-static`: faithful init, no `prereq`, backward-BFS empty;
- `SkillGraph-optimized`: cleaned global layer, selective enhance;
- `SkillGraph-C` / `SkillGraph-adapted-ordering`: optimized graph plus
  trace-order `prereq` edges.

Do not call SkillGraph-C a faithful reproduction of the original paper's static
baseline.

# Codex Runtime Injection + Live Job Update — 2026-07-03

## Runtime SkillGraph Injection

Codex implemented and pushed commit `54fe371` on `origin/beta2-skillgraph`:

- added `mlevolve/agents/memory/external_skill_memory.py`;
- initialized the layer from `AgentSearch` behind `external_skill_memory.enable`;
- injected retrieved external SOPs into:
  - `draft_agent`;
  - `improve_agent`;
  - `evolution_agent`;
  - `debug_agent`;
  - `fusion_agent`;
  - `aggregation_agent`;
  - planner and stepwise-coder subprompts;
- extended adoption tracking so `source=skillgraph` is counted separately under
  `summary.external_memory`;
- committed `paper-skills/distillation/graph_build/graph_optimized_skillgraph.json`
  so cluster jobs do not need to regenerate the graph.

Important: the current live experiment is **not** a pure external-memory
ablation. User corrected the intended口径: keep original MLEvolve behavior and
add standard SkillGraph external memory on top. Therefore the live job uses:

- `agent.use_global_memory=True`;
- default methodology KB behavior from config;
- `external_skill_memory.enable=True`;
- `external_skill_memory.source_name=skillgraph`;
- `external_skill_memory.graph_path=../paper-skills/distillation/graph_build/graph_optimized_skillgraph.json`;
- `adoption_tracking.enable=True`.

## Live Job

Created Kubernetes job:

```bash
kubectl apply -f job-spooky-skillgraph-l40sx4.yaml
```

Job spec:

- job: `mlevolve-spooky-skillgraph-l40sx4`;
- current pod: `mlevolve-spooky-skillgraph-l40sx4-v7656`;
- namespace: `ecepxie`;
- resources: `4 x nvidia.com/gpu` with node affinity `nvidia.com/gpu.product=NVIDIA-L40S`, `12 CPU`, `32Gi`;
- `priorityClassName=opportunistic`;
- task: `spooky-author-identification`;
- parallel search/gpus: `4`;
- exp name: `spooky-skillgraph-l40sx4`.

Current status update:

- job active: `1`;
- pod phase: `Pending`;
- condition: `PodScheduled=False`, reason `Unschedulable`;
- no internal logs yet because the container has not started;
- first attempt requested `nvidia.com/l40s`, but actual L40S nodes advertise
  `nvidia.com/gpu`; YAML was corrected and job recreated;
- scheduler now reports `3 Insufficient nvidia.com/gpu` among the matching
  `NVIDIA-L40S` nodes, meaning no single L40S node currently has 4 free GPUs.

Prior A100 attempt:

- `mlevolve-spooky-skillgraph-a100x3` was created then deleted;
- it never ran because the namespace A100 quota blocked pod creation.

Next monitoring commands:

```bash
kubectl get pods -n ecepxie -l job-name=mlevolve-spooky-skillgraph-a40x8 -o wide
kubectl describe pod mlevolve-spooky-skillgraph-a40x8-2l848 -n ecepxie | tail -120
kubectl logs -n ecepxie job/mlevolve-spooky-skillgraph-a40x8 --tail=120
```

# A40x8 Replacement Update — 2026-07-03 16:54 CST

User switched the live run to 8×A40, 9 CPU, 32Gi. Codex created and applied
`job-spooky-skillgraph-a40x8.yaml`.

Important resource-format detail:

- A40 nodes expose `nvidia.com/a40`, not generic `nvidia.com/gpu`.
- The active job requests `nvidia.com/a40: 8`, `cpu: "9"`, `memory: "32Gi"`.
- Node affinity targets `nvidia.com/gpu.product=NVIDIA-A40`.

Current status:

- job: `mlevolve-spooky-skillgraph-a40x8`;
- pod: `mlevolve-spooky-skillgraph-a40x8-2l848`;
- node: `bak-hpc1.csub.edu`;
- phase: `Pending`, state `ContainerCreating`;
- `PodScheduled=True`, so the 8×A40 allocation succeeded;
- blocker: PVC `haoming-storage` / CephFS mount, not scheduler capacity.

Relevant pod events:

```text
SuccessfulAttachVolume: AttachVolume.Attach succeeded for volume pvc-133ca65b-9e81-492c-90b3-4320d5e19a94
FailedMount: MountVolume.MountDevice failed ... DeadlineExceeded
FailedMount: MountVolume.MountDevice failed ... operation with the given Volume ID ... already exists
```

No MLEvolve internal logs are available yet because the container has not
started. If this remains stuck past roughly 10 minutes, the next practical move
is likely deleting/recreating the same A40 job to force a fresh mount attempt,
but that may lose the already acquired 8×A40 allocation.

# Leaf 5xRTX3090 Replacement Update — 2026-07-03 17:15 CST

User switched away from spooky and requested 5×RTX 3090, 12 CPU, with a lighter
GPU task. Codex deleted the pending RTX A6000 job and created:

- YAML: `job-leaf-skillgraph-rtx3090x5.yaml`;
- job: `mlevolve-leaf-skillgraph-rtx3090x5`;
- pod: `mlevolve-leaf-skillgraph-rtx3090x5-x9625`;
- task: `leaf-classification`;
- resource request: `nvidia.com/gpu: 5`, `cpu: "12"`, `memory: "32Gi"`;
- node affinity: `nvidia.com/gpu.product=NVIDIA-GeForce-RTX-3090`;
- original MLEvolve memory remains on: `agent.use_global_memory=True`;
- external SkillGraph remains on: `external_skill_memory.enable=True`;
- adoption tracking remains on.

Rationale for task selection: `leaf-classification` has existing clean-ish
historical runs and is lighter than spooky NLP / denoising U-Net. Data paths
use the PVC source:

```bash
data_dir=/workspace/nautilus/mlevolve/data/leaf-classification/prepared/public
desc_file=/workspace/nautilus/mlevolve/data/leaf-classification/prepared/public/description.md
```

Current status:

- pod phase: `Pending`;
- `PodScheduled=False`, reason `Unschedulable`;
- no logs yet because it has not reached a node;
- scheduler reports no matching 3090 node currently has 5 free GPUs + 12 CPU +
  32Gi together.

Relevant monitor commands:

```bash
kubectl get pods -n ecepxie -l job-name=mlevolve-leaf-skillgraph-rtx3090x5 -o wide
kubectl describe pod mlevolve-leaf-skillgraph-rtx3090x5-x9625 -n ecepxie | tail -80
kubectl logs -n ecepxie job/mlevolve-leaf-skillgraph-rtx3090x5 --tail=120
```

# RTX3090 Correction Update — 2026-07-03 17:40 CST

Codex previously made two scheduling mistakes and corrected them:

- do not pin a hostname;
- do not use `nvidia.com/GA102_GEFORCE_RTX_3090` when the intent is to request
  standard RTX3090 nodes broadly. That special key effectively restricts the job
  to rare nodes exposing that resource.

Current correct job:

- YAML: `job-leaf-skillgraph-rtx3090x5.yaml`;
- job: `mlevolve-leaf-skillgraph-rtx3090x5`;
- pod: `mlevolve-leaf-skillgraph-rtx3090x5-67jzq`;
- task: `leaf-classification`;
- resource request: `nvidia.com/gpu: 5`, `cpu: "12"`, `memory: "32Gi"`;
- node affinity: `nvidia.com/gpu.product=NVIDIA-GeForce-RTX-3090`;
- priority class: `opportunistic`.

Current status:

- pod phase: `Pending`;
- `PodScheduled=False`, reason `Unschedulable`;
- scheduler reports the matching standard RTX3090 nodes lack enough free generic
  GPUs / CPU / memory;
- no MLEvolve logs yet because the pod has not reached a node.

Relevant monitor commands:

```bash
kubectl get pods -n ecepxie -l job-name=mlevolve-leaf-skillgraph-rtx3090x5 -o wide
kubectl describe pod mlevolve-leaf-skillgraph-rtx3090x5-67jzq -n ecepxie | tail -80
kubectl logs -n ecepxie job/mlevolve-leaf-skillgraph-rtx3090x5 --tail=120
```
