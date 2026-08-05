# Experiment End2End Smoke Execution Ledger

- Date: 2026-08-04
- Namespace: `ecepxie`
- Experiment label: `experiment-end2end-memory-v1`
- User authorization: explicit instruction `开始smoke`
- Formal Pilot authorized: no

## Pre-Agent launch L0 — retained infrastructure abort

- Job: `mlevolve-e2e-smoke-leaf-v1`
- Job UID: `895bea7b-9883-499f-afac-e57e63094944`
- Created at: `2026-08-04T02:58:28Z`
- Pod: `mlevolve-e2e-smoke-leaf-v1-0-p7csc`
- Completion index: 0
- State at abort decision: Pending, unscheduled, no node, no container state
- Agent/GPU calls: 0
- Logical attempt directories created: 0
- Retention disposition: retained here as a pre-Agent infrastructure launch;
  it is not counted as a logical system attempt and cannot supply a score.
- Abort reason: Kubernetes `activeDeadlineSeconds` is global to an Indexed Job.
  The generated value `3600` incorrectly represented a per-index allowance even
  though ten indexes run with `parallelism=1`; the Job could not reliably run
  the complete 10-system Smoke matrix.
- Cleanup policy: ordinary deletion of this positively-owned Pending Job only;
  no force deletion, no shared-namespace resource deletion.

## Launch-time preflight corrections before L0

1. The pinned container exposes Python at `/usr/local/bin/python` rather than
   `/opt/anaconda3/bin/python`. The Job generator and tests were corrected
   before any Smoke container started.
2. `/workspace/experiment-c-formal-releases-r3` currently and correctly carries
   base evaluator asset binding `668896c0…`; `bee5a525…` is the separate Exp-C
   r9 hardware-control amendment. End2End reuses evaluator assets only, so it
   was rebound to the base release before any Smoke container started.

Further launch and logical-attempt identities will be appended without
overwriting this record.

## Pre-Agent launch L1 — retained user-authorized GPU-selection abort

- Job UID: `8839390d-ea22-46be-9745-ffd608541690`
- Created at: `2026-08-04T03:08:19Z`
- Requested GPU: one `NVIDIA-A40` via `nvidia.com/a40`
- State immediately before deletion: exactly one index-0 Pod, Pending,
  unscheduled, no node, no `containerStatuses`
- Agent/GPU calls: 0
- Logical attempt directories created: 0
- Abort reason: the user prospectively changed the common GPU choice to fixed
  A100 before any container started.
- Cleanup: ordinary deletion after a fail-closed state check; no force delete.
- Experimental effect: none. No system prompt, candidate, result, or score was
  observed. The replacement matrix must use one fixed A100 product for all ten
  Smoke systems and the later Pilot.

## Pre-Agent launch L2 — retained three-minute scheduling abort

- Job UID: `27c143b3-572b-414c-bb1c-1d2bc19dca79`
- Created at: `2026-08-04T03:20:06Z`
- Requested GPU: one `NVIDIA-A100-SXM4-80GB` via `nvidia.com/a100`
- State at `2026-08-04T03:23:35Z`: Pending for more than three minutes,
  unscheduled, no node, no `containerStatuses`
- Scheduler evidence: matching nodes reported insufficient A100, memory, or
  CPU; preemption was not helpful.
- Agent/GPU calls: 0
- Logical attempt directories before launch: 0
- Cleanup: ordinary deletion after a fail-closed state check; no force delete.
- Next disposition: keep the A100 family fixed but request the separate
  `NVIDIA-A100-80GB-PCIe` product pool.

## Pre-Agent launch L3 — retained three-minute scheduling abort

- Job UID: `66aab6df-23e7-445b-b6c5-4a2ba39efeb0`
- Pod UID: `500fcbbc-2c00-45b6-9900-07dda16969a3`
- Created at: `2026-08-04T03:38:52Z`
- Requested GPU: one `NVIDIA-A100-80GB-PCIe` via `nvidia.com/a100`
- State at `2026-08-04T03:41:57Z`: Pending for more than three minutes,
  unscheduled, with no node, start time, Pod IP, init-container status, or
  container status.
- Scheduler evidence: 12 `FailedScheduling` observations were recorded; the
  matching pool reported insufficient A100, memory, or CPU, and preemption was
  not helpful.
- Job counters at retention: active 1, succeeded 0, failed 0, no completed or
  failed indexes.
- Agent/GPU calls: 0
- Logical attempt artifacts: none created by this launch because no container
  started.
- Cleanup policy: ordinary deletion of this positively-owned Pending Job only;
  no force deletion and no child-Pod deletion.
- Next disposition: keep the A100 family fixed but request the separate
  `NVIDIA-A100-PCIE-40GB` product pool.

## Launch L4 — generic A100 Smoke

- Job UID: `5a712637-2cc8-4a05-8315-bf5ef9ea1b13`
- Created at: `2026-08-04T03:55:40Z`
- Initial Pod: `mlevolve-e2e-smoke-leaf-v1-0-hwgnd`, completion index 0
- Requested GPU: one generic A100 via `nvidia.com/a100`; no
  `nvidia.com/gpu.product` constraint
- Matrix: `leaf-classification`, ten frozen systems, seed 1, two MLEvolve
  steps, Indexed Job `parallelism=1`
- Source-lock manifest:
  `cf946c89985312013d3b1de8ac8242c6533c736241079d09c3e4db7db4f74c61`
- Smoke manifest:
  `fb81e57fcaf685128d0533d86825e2f86ddbb5da0ae0986c7365e5cb7f4b6141`
- Interruption policy: completed systems remain on the PVC; the active runner
  forwards termination to MLEvolve for completed-step journal finalization,
  retains the interrupted attempt, and permits only an explicit
  infrastructure retry at the next attempt number.
- Scheduling policy: if the first Pod remains pure unscheduled Pending for
  more than three minutes, retain its exact state and ordinarily delete the
  positively-owned Job before retrying the generic A100 request.
- Final pre-deletion state at `2026-08-04T03:59:01Z`: Pending for more than
  three minutes; no node, start time, Pod IP, init-container status, or
  container status. Job counters were active 1, succeeded 0, failed 0, with no
  completed or failed indexes.
- Pod UID: `2154432a-7032-40a9-86b8-aef4cb15c9d1`
- Scheduler evidence: 20 `FailedScheduling` observations were retained; no
  scheduling or container-start event occurred.
- Agent/GPU calls: 0
- Cleanup disposition: ordinary deletion of the positively-owned Job; no
  force deletion and no independent deletion of its child Pod.

## Pre-Agent launch L5 — retained user-authorized memory resize abort

- Job UID: `f9aa17e8-5be8-4137-a07a-262632e43ec6`
- Pod UID: `ded6dfd7-bdeb-4d0f-9ea5-7be215579041`
- Created at: `2026-08-04T04:00:41Z`
- Requested resources: 16 CPUs, 256 GiB memory, one generic A100
- State at `2026-08-04T04:02:03Z`: Pending, unscheduled, no node, start time,
  Pod IP, init-container status, or container status
- Scheduler evidence: three `FailedScheduling` observations; no scheduling or
  container-start event
- Agent/GPU calls: 0
- Abort reason: the user explicitly changed the memory request to 64 GiB before
  any container started.
- Cleanup disposition: ordinary deletion of the positively-owned Job; no
  force deletion and no independent deletion of its child Pod.

## Pre-Agent launch L6 — retained runtime import failure

- Job UID: `f05e5b00-5c87-413a-9da5-6e5153cf50f4`
- Created at: `2026-08-04T04:08:06Z`
- Requested resources: 16 CPUs, 64 GiB memory, one generic A100
- Scheduling: index 0 reached Running in about three seconds on
  `rci-nrp-gpu-03.sdsu.edu`; observed node product from the cluster node list
  was `NVIDIA-A100-80GB-PCIe`.
- Index 0 Pod: `mlevolve-e2e-smoke-leaf-v1-0-jp59g`, UID
  `490c1f3e-5dfa-493a-880a-425b993b3991`, exit 1, started
  `2026-08-04T04:08:20Z`, finished `2026-08-04T04:08:33Z`.
- Index 1 Pod: `mlevolve-e2e-smoke-leaf-v1-1-2qkv4`, UID
  `3507014e-9b4e-4f19-84f0-5818c24f3b10`, exit 1, started
  `2026-08-04T04:08:40Z`, finished `2026-08-04T04:08:50Z`.
- Both indexes failed identically before any Agent call or attempt-directory
  creation: `ModuleNotFoundError: No module named 'authority'` while importing
  `authority.memory_snapshot` in the runtime Bundle verifier.
- Root cause: the container Job invoked the staged script directly without
  exporting `/workspace/nautilus-exp-end2end/mlevolve` in `PYTHONPATH`.
- Containment: the Job was suspended as soon as the first failure was
  observed. Index 1 had already been created and failed before suspension took
  effect; indexes 2-9 never started.
- Agent/GPU training calls: 0
- Retention: exact Job/Pod UIDs, times, node, product and complete tracebacks
  are retained in this ledger. The common runtime import defect will be fixed
  and tested before an explicit replacement launch.

## Launch L7 — terminal 64-GiB generic-A100 Smoke

- Job UID: `b4bad652-2e71-4a9f-b17c-f6fcc2408310`
- Created at: `2026-08-04T04:25:38Z`
- Initial Pod: `mlevolve-e2e-smoke-leaf-v1-0-kzgcl`, completion index 0
- Requested resources: 16 CPUs, 64 GiB memory, one generic A100
- Scheduling: bound to `gp-argo.usd.edu` and reached Running at
  `2026-08-04T04:25:39Z`, about one second after Job creation
- Source-lock manifest:
  `9e25e11378422e2eae619179a8da632d465f29ad9071e95e383eeec1f81940f4`
- Smoke manifest:
  `00782e0d9c03cf6b7bdec28e7721f4bb2af8e7fee970735c1da1e355f8717a0e`
- Prelaunch remote verification: source lock, runtime import, Leaf Memory
  Bundle, Host binding and evaluator release all passed before submission.

### L7 index 0 — retained Agent failure

- System: `gome_style_port`
- Pod: `mlevolve-e2e-smoke-leaf-v1-0-kzgcl`, UID
  `7eb9a101-4684-4286-9c70-121342dc021c`
- Hardware: `gp-argo.usd.edu`, observed `NVIDIA A100-PCIE-40GB`
- Both frozen MLEvolve steps completed, but the draft and its debug child were
  both buggy, reported no internal metric and produced no submission.
- `RUN_OUTCOME`: status `complete`, completed 2/2, no certified solution
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 249.338617697 seconds; allocated GPU time:
  0.06926072713805555 hours
- Measurement hash:
  `88437bacd5eb5f915f996f77b28708e1e202ee36847110e41264097459353e38`
- Disposition: retained without imputation or retry. This is not an
  infrastructure failure and therefore is ineligible for an explicit attempt
  retry under the frozen policy.

### L7 index 1 — retained Agent failure

- System: `runforest_only`
- Pod: `mlevolve-e2e-smoke-leaf-v1-1-z6gmr`, UID
  `96889427-f94a-416b-962f-212697c19631`
- Hardware: `gp-engine.usd.edu`, observed `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed. The first and debug candidates both
  failed Host protocol preflight and produced no submission.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 296.265944613 seconds; allocated GPU time:
  0.08229609572583332 hours
- Measurement hash:
  `3b4fe7997e86c8ca5bc68308c5df436082fa77477cfb7c064bd52c81b7269c30`
- Disposition: retained without imputation or retry. Index 2 proceeded under
  the original frozen matrix.

### L7 index 2 — retained Agent failure

- System: `reversed_router`
- Pod: `mlevolve-e2e-smoke-leaf-v1-2-57vxs`, UID
  `fd3255cd-0e41-491b-a8d1-d1cf94b0ed79`
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed. Both candidates failed Host protocol
  preflight and produced no submission.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 236.73085221 seconds; allocated GPU time:
  0.06575857005833333 hours
- Measurement hash:
  `a54734157c4797a2adfe5b046204e3baab274ce716436c32e248c6a66818a79d`
- Disposition: retained without imputation or retry. Index 3 (`no_memory`)
  proceeded under the original frozen matrix.

### L7 index 3 — retained No-Memory Agent failure

- System: `no_memory`
- Pod: `mlevolve-e2e-smoke-leaf-v1-3-ddkwp`, UID
  `7ca0e4b1-dcd0-4e10-9bf1-3c582d15b9c8`
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed. The draft was rejected for
  `pre_selection_inference:main`; the debug child was rejected for
  `post_selection_tuning:main:lgb_model.fit,logreg.fit`. Both are explicit
  candidate-source Host protocol violations, not infrastructure failures.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 193.662257045 seconds; allocated GPU time:
  0.05379507140138889 hours
- Measurement hash:
  `f59dbccd65fccb5e70bedc364233e540ae6a7f2b8ed6b3a2344c6a1d78c7f8a0`
- Disposition: retained without imputation or retry. This confirms the repeated
  failure is not specific to a memory-on router; index 4 proceeded.

### L7 index 4 — retained Agent failure

- System: `static_hybrid`
- Pod: `mlevolve-e2e-smoke-leaf-v1-4-t6fmf`, UID
  `3d69d5da-7663-40e7-afd8-205784f188b0`
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 261.731772599 seconds; allocated GPU time:
  0.07270327016638889 hours
- Measurement hash:
  `9487d5bca076e64d22bb5f40749ecbb70059e4907e6ddd09caab8bb9662b8b91`
- Disposition: retained without imputation or retry. Index 5 (`sop_only`)
  proceeded under the original frozen matrix.

### L7 index 5 — scored terminal result

- System: `sop_only`
- Pod: `mlevolve-e2e-smoke-leaf-v1-5-cd4jj`, UID
  `2e8ceebe-824f-41fd-bfbf-d0c0d3c51a8d`, exit 0
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Step 1 candidate passed Host protocol but did not produce an accepted result.
  The step 2 debug child passed label-free fixed-holdout validation with
  internal search metric 0.6644687819678308.
- Candidate set frozen: true; selected candidate:
  `453264b24ed34c78b1f70b674cad4401`
- Terminal evaluator log loss: `0.937787418848125`
- Measurement: status `scored_terminal_result`, failure class `none`,
  completed true; TTFV remained null in the frozen measurement
- Wall time: 981.531952018 seconds; allocated GPU time:
  0.27264776444944444 hours
- Measurement hash:
  `4ecf66cfeedc3d5f6f61f68a8fb33a78805f3584f6a86b4979eeb6def4583cba`
- Terminal report SHA-256:
  `e04efd888d539fc030a835c196fa732e8a793cc165c4a098726e33ec0d0d5c23`

### L7 index 6 — retained Agent failure

- System: `dynamic_hybrid`
- Pod: `mlevolve-e2e-smoke-leaf-v1-6-46v7k`, UID
  `c01064cc-414f-40cf-9f83-5a08bb6849f2`
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed; both candidates failed Host protocol
  preflight and produced no terminal submission.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 235.788814335 seconds; allocated GPU time:
  0.06549689287083334 hours
- Measurement hash:
  `7353a31b980cab7e6589822ade775899ead22847d807e31636655a5d0dfc27a2`
- Disposition: retained without imputation or retry. Index 7
  (`macla_style_port`) proceeded.

### L7 index 7 — retained Agent failure

- System: `macla_style_port`
- Pod: `mlevolve-e2e-smoke-leaf-v1-7-dbgxs`, UID
  `cffa7ec2-9ed4-4e7b-b9d2-b02001d1390a`
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed; neither candidate produced a valid
  terminal submission.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 287.231639369 seconds; allocated GPU time:
  0.07978656649138889 hours
- Measurement hash:
  `601147abdb4b52ae54217c839b0ce5a862c92cde82ce367e3b5565efd02b2a0a`
- Disposition: retained without imputation or retry. Index 8
  (`rcr_router_style_port`) proceeded.

### L7 index 8 — retained Agent failure

- System: `rcr_router_style_port`
- Pod: `mlevolve-e2e-smoke-leaf-v1-8-sdtsq`, UID
  `e892a7d6-34af-4776-8263-2092bac5b95a`
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed; neither candidate produced a valid
  terminal submission.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 230.686025755 seconds; allocated GPU time:
  0.06407945159861111 hours
- Measurement hash:
  `a016159bf36bd1a6033804b425438bfaa503165d1d2cc915c0f87f9410c87809`
- Disposition: retained without imputation or retry. Final index 9
  (`flat_retrieval`) proceeded.

### L7 index 9 — retained Agent failure

- System: `flat_retrieval`
- Pod: `mlevolve-e2e-smoke-leaf-v1-9-5nft2`, UID
  `7bdeba3b-c7eb-4362-abf3-38ee87b5ea48`, exit 1
- Hardware: `node-1-1.sdsc.optiputer.net`, observed
  `NVIDIA A100-SXM4-80GB`
- Both frozen MLEvolve steps completed. Both candidates passed Host protocol
  preflight, but neither produced an accepted submission. The draft failed
  with a candidate `RuntimeError`; the debug child also returned no internal
  metric after three applied diff patches.
- Measurement: status `retained_agent_complete`, failure class `agent`,
  completed false, terminal score null, TTFV null
- Wall time: 441.11225849 seconds; allocated GPU time:
  0.12253118291388888 hours
- Measurement hash:
  `481682c58c294377aa892521911704c045a48ae7851fecd93353781357c03d05`
- Disposition: retained without imputation or retry. This completed the exact
  ten-system frozen Smoke matrix.

## L7 terminal Job state and frozen Smoke Gate result

- Terminal observation at `2026-08-04T05:34:30Z`: Job UID
  `b4bad652-2e71-4a9f-b17c-f6fcc2408310`, active 0, succeeded 1, failed 9,
  completed index `5`, failed indexes `0-4,6-9`.
- Terminal conditions: `FailureTarget=True` and `Failed=True`, both reason
  `FailedIndexes` with message `Job has failed indexes`.
- Exactly 10 immutable `attempt-000/MEASUREMENT.json` artifacts remain on the
  PVC. There were no logical retries because all nine failed conditions were
  classified as Agent failures rather than infrastructure failures.
- Completion: 1/10. Total allocated GPU time: 0.9483555928141666 hours.
  Missing terminal scores remain null; no imputation was performed.
- Frozen Gate validator result: fail closed at `dynamic_hybrid` with
  `ValueError: dynamic_hybrid: no complete terminal-scored Smoke attempt`.
  No `SMOKE_GATE.json` was created, so this release cannot authorize the
  formal 40-run Pilot.
- Endpoint-first analysis completed over all 10 retained measurements:
  terminal summary hash
  `84f69996fe0ad4ff6e9e2e1e3bd002fee43648469c25602dd5ca71fdbad81ba1`.
- Mechanism analysis completed second: mechanism summary hash
  `fd72956428602a3787797752db282f4d7cf78a1e6fee9a3ee5a943a16026b53c`.
  All ten systems observed the same 16-candidate authorized pool; only
  `sop_only` obtained runtime activation (3 candidates). This mechanism result
  is descriptive and exploratory, not causal or statistically significant.
- The terminal Kubernetes Job and child Pods were subsequently absent from
  the namespace inventory, but all logical attempts, measurements, journals,
  preflight reports, and terminal report remain on the persistent PVC.

## Launch L8 — retained Agent-verifier v1 runtime failure

- Release: `end2end-agent-v1`; output root:
  `/workspace/experiment-end2end-agent-runs-v1`
- Frozen order began with index 0 `runforest_only`, followed by index 1
  `macla_style_port`.
- Index 0 ran on `rci-nrp-gpu-02.sdsu.edu`; observed hardware was
  `NVIDIA A100 80GB PCIe`.
- The draft completed generation and code review, then failed before candidate
  execution with `NameError: is_historical_replay_anchor is not defined` at
  `mlevolve/engine/agent_search.py` while applying the Agent verifier guard.
- Root cause: the Agent-verifier cherry-pick retained the function call but a
  conflict resolution omitted the corresponding import from
  `agents.memory.run_forest_replay`.
- Measurement: status `retained_agent_failed`, failure class `agent`, completed
  false, terminal score null, TTFV null, wall time 109.661929979 seconds,
  allocated GPU time 0.03046164721638889 hours.
- Measurement hash:
  `39917014c15655580d9e5dd5cb270a28994baaa7d90a4565c8fcb9682de7b92c`.
- Index 1 reached Running, but had not created an immutable attempt directory
  when the common-runtime defect was diagnosed and the Job was ordinarily
  stopped. No result is invented for that interrupted index.
- `SMOKE_GATE.json` was not created. The formal 40-run Pilot remained blocked.
- All index-0 logs and the measurement remain on the PVC; the v1 output root
  will be retained unchanged. The replacement is a new `end2end-agent-v2`
  release with an AST regression test requiring the call and import together.

## Pre-Agent launch L9 — retained v2 staging transfer abort

- The first full Git-archive stream to
  `/workspace/nautilus-exp-end2end-agent-v2-stage` failed after about 56 MiB
  with a Kubernetes API TCP read timeout.
- The active `/workspace/nautilus-exp-end2end` directory was not touched. The
  incomplete staging directory was deleted, rebuilt by copying the complete v1
  snapshot inside the PVC, and updated with the exact tracked delta from frozen
  source head `a5689cdf7a979783f218b0c2bee400b1e2cd8cfb`.
- Before the atomic switch, v2 passed manifest/source-lock dry-run, Leaf Bundle,
  Host binding, terminal evaluator, and Host SDK hash verification. The old v1
  snapshot was retained as
  `/workspace/nautilus-exp-end2end-agent-v1-archive-20260804`.

## Launch L10 — retained Agent-verifier v2 prerequisite failure

- Job: `mlevolve-e2e-agent-smoke-leaf-v2`, UID
  `a3f5c503-6143-495a-bc6a-31dc18fe1584`.
- Release/output: `end2end-agent-v2` and
  `/workspace/experiment-end2end-agent-runs-v2`.
- Requested resources per index: one generic A100, 16 CPUs, 64 GiB memory;
  indexed parallelism 1, Leaf task, seed 1, two MLEvolve steps.
- Index 0 first remained Pending in `ContainerCreating` for 180 seconds on
  `rci-nrp-gpu-04.sdsu.edu`. The exact-owned child Pod was ordinarily deleted
  under the user-specified stale-Pending policy; Kubernetes retained this as an
  infrastructure failure.
- After the container finally started during its termination grace period, and
  on subsequent indexes, Python failed before Agent search with
  `ImportError: cannot import name 'is_historical_replay_anchor' from
  'agents.memory.run_forest_replay'`.
- Root cause: the verifier commit came from a reference branch whose parent
  already defined the replay-anchor predicate. The earlier conflict repair
  added the import and an AST syntax test, but did not port that prerequisite
  implementation or execute the import in the test.
- The Job was suspended as soon as the common defect was confirmed and then
  ordinarily deleted. Five immutable attempt-000 infrastructure measurements
  remain for `runforest_only`, `gome_style_port`, `static_hybrid`,
  `reversed_router`, and `rcr_router_style_port`; no score was imputed and no
  Agent outcome is claimed.
- Measurement hashes in that order:
  `3aed2421b3620a82b9671ea1811d839dfdc5e1db58893d4cb0ead347590e4152`,
  `15a96942a576a797dbc96eb8507c26d07c7d45f73a269e2404b19a95aa488c28`,
  `fa05c11456b4eb00a74336dc5f67153c8030f5ed9ebaf4548e4a39c0839247fd`,
  `a5b60fd3cb6add78df268efe9db1e820ba9b99615d9d231f7b1702966bcb2c4c`,
  and `e112f9041e0d61d51e133ea20655f488e75f9707cc02186c46a3512d32985844`.
- The formal Pilot remained blocked. Replacement release v3 ports the exact
  prerequisite predicate from the read-only reference branch and adds a real
  import/classification regression test.

## 2026-08-05 Leaf full-system Smoke completion and Dynamic Agent v2

- The resumed Leaf control Job `mlevolve-e2e-leaf-controls-smoke-v15`
  terminated with eight successful indexes and one retained Agent failure.
  `gome_style_port` remained unscored; all other control conditions produced a
  terminal log-loss measurement. Earlier infrastructure attempts remain under
  their original attempt directories on the PVC.
- Retrieval Agent v2 is source version
  `experiment_r_agentic_final_selection_v2`. It adds an exact final-selection
  contract, usable inspect observations, a mandatory final decision step,
  explicit fallback metadata, and direct Agent ownership of the Dynamic
  selected IDs. The deterministic 4+2 router no longer silently chooses the
  IDs after an Agent non-decision.
- Checkpoint commits were `47a6755c` (Agent final selection), `64524ca1`
  (v2 Smoke staging), `9e08a9c6` (scoped workload labels), and `fde8b676`
  (fresh Dynamic v2 logical identity). The PVC runtime snapshot
  `/workspace/nautilus-exp-end2end-agent-v14` matched all 237 source-lock file
  hashes before launch.
- Job `mlevolve-e2e-leaf-dynamic-smoke-v16` is retained as a pre-training
  identity failure. It attempted to attach the new algorithm to attempt 2 of
  an older terminal-scored logical run and was rejected in one second with
  `ValueError: Explicit retry is allowed only after an infrastructure
  failure`. No candidate training or measurement occurred.
- Replacement Job `mlevolve-e2e-leaf-dynamic-smoke-v17` used the fresh logical
  run `e2e-smoke-leaf-dynamic-agent-v2__leaf-classification__dynamic_hybrid__seed-1`.
  It completed on one `NVIDIA A100-SXM4-80GB`, 16 CPUs and 64 GiB memory with
  zero restarts.
- Both memory-on roles made real final decisions. `memory_transfer` finished
  in one Agent call; `novel_exploration` inspected two known candidates and
  finished on call three. Both selected exactly four SOP and two RunForest
  IDs, with `selection_complete=true`, `fallback_used=false`, and
  `deterministic_quota_selection_used=false`.
- The mandatory same-task clean-best invariant transparently replaced
  RunForest ID `9d9f690c...` with `0d9b3a3a...` in the memory-transfer slot.
  This replacement and the original Agent proposal are both retained in the
  routing trace.
- Candidate outcomes: coldstart internal log loss `0.693`; memory-transfer
  internal log loss `0.09166647865848002` and selected; novel-exploration
  retained buggy because every reported training loss was NaN despite a
  generated submission and finite validation predictions.
- Dynamic v2 terminal log loss was `0.11480336659470919`, TTFV
  `358.895870013` seconds, and allocated GPU time `0.12555508805194446`
  hours. Measurement hash:
  `87ed595227d468d9b510aac5779b146f176cd3f4968f2d86ab447d675ccb3eee`.
- The completed ten-system Leaf Smoke has 9/10 terminal-scored conditions.
  `flat_retrieval` ranked first (`0.09353439660745823`), `sop_only` second
  (`0.11256344056236905`), and Dynamic v2 third (`0.11480336659470919`).
  Every scored memory condition improved over No Memory (`0.9072923887645635`);
  `gome_style_port` remains a completion failure rather than an imputed metric.
- Dynamic code-level adoption was partial. The selected program implemented
  dynamic feature-column discovery, train-only scaling, exact
  `submission.csv`, sample-submission column order, and row-sum validation. It
  did not reproduce the clean-best SigLIP2 image plus branch-fusion core, so
  Prompt injection must not be reported as full method adoption.
- These are three-step, temperature-1, Seed-1 exploratory Smoke results. They
  do not establish statistical significance. The formal 40-run Pilot remains
  generated but not submitted pending explicit user authorization.
