# Experiment End2End — Exploratory Pilot Packet

This directory is the executable and analysis packet for the exploratory
10-system × 4-task × seed-1 Pilot.  The formal Pilot has been launched in
task-local blocks; the authoritative live queue is
`../../coordination/end2end_execution_queue_v23.json`.

## Execution mode

Builders do not launch training.
This experiment intentionally uses `experiment_fast_nonblocking_v1`: Bundle
provenance traversal, Host Protocol Preflight/runtime receipts, Authority
enforcement, adoption enforcement and the formal Smoke gate are disabled. The
pre-run check reads only the frozen local packet and confirms the intended
matrix, resources, same-task-best identities, Agent final-selection mode and
condition-level resume semantics.

The current sequence is:

1. the Leaf ten-system Smoke is complete and retained (9 terminal-scored
   outcomes and one real GOME-style Agent failure); it is not a Pilot result;
2. the formal four-task memory source is `memory-direct-v2`: all four tasks use
   the same reviewed four-task graph/index and may retrieve their own clean
   historical best;
3. Leaf is executed first with at most four independent one-A100 Jobs; after
   every Leaf condition has a retained outcome, Aerial, Denoising and Taxi use
   task-local 10-index Jobs with `parallelism: 4`.  Every Pod requests one A100,
   16 CPU and 64 GiB, and the global cap remains four GPUs;
4. `--resume` skips completed/Agent/evaluator outcomes and creates a new
   immutable attempt only for infrastructure interruption. When the interrupted
   attempt has a valid `journal.json` + `RUN_OUTCOME.json`, the new process
   restores the completed RunForest, branch/Top-K state, submissions and the
   completed-step counter, then executes only the remaining MLEvolve steps
   under the unused portion of the original wall-clock budget. Candidate code
   is archived before process launch; when that archive exists, an interrupted
   active candidate is re-executed from its candidate boundary. Older attempts
   without a candidate archive restart from the completed parent node. A
   hard interruption without a final measurement is first retained as
   infrastructure failure. A candidate interrupted inside its own training
   process restarts from that candidate boundary; arbitrary candidate programs
   are not claimed to support epoch-level checkpoints;
5. run terminal analysis first, time-performance export second, and RunForest
   mechanism analysis only after the terminal table is materialized.

Leaf spans frozen v21/v22/v23 releases because the dynamic recipe/router was
upgraded after the first controls launched.  The composite result plan is
explicit: `dynamic_hybrid` and `sop_only` use v22; the other Leaf conditions
use their frozen v21 logical IDs (including immutable recovery attempts written
back to the same condition root); the three remaining tasks use v23.  The v21
Dynamic v24 resume is diagnostic recovery validation and is never selected for
the official Dynamic cell.

## Frozen files

- `systems_v23/`: one common config plus exactly 10 frozen system overlays;
- `manifests_v23/`: system, task, budget, source, Bundle, evaluator, Smoke and
  Pilot locks with canonical SHA-256 self-hashes; earlier releases remain
  retained for provenance;
- `manifests_resume_v23/` and `manifests_resume_v24/`: byte-identical copies of
  the frozen PVC-only adapter manifests used by the Leaf v23/v24 Jobs.  Their
  Pilot and source-lock self-hashes are bound to the retained publication
  records under `infrastructure_attempts/`;
- `jobs/`: retained Smoke definitions, immutable Leaf condition Jobs and the
  three task-local v23 Indexed Jobs; current submission order is frozen in the
  coordination queue;
- `prepare_direct_leaf_memory.py`: copies the already reviewed seed-heldout
  Base directly for Leaf. It deliberately performs no formal child publication,
  domain certification, or transition-to-SOP proof. The Base contains Leaf
  seed-43/44 history; Dynamic Hybrid pins the best eligible same-task RunForest
  item into one existing Prompt slot.
- `prepare_direct_fourtask_memory.py`: wraps the immutable reviewed four-task
  RunForest graph/index in four normal Base/CURRENT layouts, freezes zero
  held-out run exclusions, and records the direction-aware clean
  `positive_eligible` best node for every task;
- `confirm_experiment_intent.py`: one local, human-facing intent summary. It
  opens no Bundle/data, starts no subprocess, calls no model or cluster API,
  and never launches training;
- `run_assignment.py`: finite Job PID 1, Agent subprocess, terminal evaluator
  and immutable failure measurement; hashes/receipts remain metadata only.
  Resume is completed-search-step-level, with immutable attempt lineage,
  pre-execution candidate-source archives and cumulative wall/GPU/TTFV time;
  arbitrary Agent-generated training programs are not claimed to support
  epoch-level checkpoints;
- `validate_smoke_gate.py`: retained as an optional offline diagnostic and is
  not called by Smoke or Pilot Jobs;
- `analysis_v23/analyze_composite_terminal.py`: the official 40-cell composite
  terminal/completion/negative-transfer/TTFV/cost table.  It reports both the
  selected search-lineage cost and operational cost across every retained
  failed/retry attempt;
- `analysis_v23/analyze_time_performance.py`: candidate internal metric versus
  search and operational active time/GPU-hours.  These are diagnostic curves,
  never substitutes for fixed-holdout terminal scores;
- `analysis_v23/analyze_composite_mechanism.py`: routing, suppression, static
  adoption and runtime activation after terminal outcomes.  It reads every
  retained attempt Journal newest-first and deduplicates resumed RunForest
  nodes, so failed-attempt paths are not silently discarded;
- `analysis_v23/audit_pilot_completion.py`: final fail-closed gate for the
  exact 10×4 matrix, retained attempt files, terminal-score consistency,
  journals, memory-on retrieval/Prompt evidence, No-Memory emptiness and the
  exploratory Seed-1 interpretation boundary;
- `build_manifests.py`: deterministic generator and non-mutating `--check`.

## Local validation

```bash
PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/build_manifests.py --check

PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/confirm_experiment_intent.py

PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/run_assignment.py \
  --manifest experiments/end2end_memory_systems_20260804/manifests/smoke_manifest.json \
  --index 0 --dry-run

PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/analysis_v23/analyze_composite_terminal.py \
  --analysis-root /workspace/experiment-end2end-memory-analysis-v23 \
  --allow-incomplete

PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/analysis_v23/analyze_time_performance.py \
  --terminal-summary /workspace/experiment-end2end-memory-analysis-v23/terminal_summary.json \
  --attempt-inventory /workspace/experiment-end2end-memory-analysis-v23/attempt_inventory.json \
  --analysis-root /workspace/experiment-end2end-memory-analysis-v23
```

`--dry-run` loads only the local packet, performs zero Agent calls,
does not open cluster assets and writes no result directory.

## Interpretation boundary

Seed 1 is an engineering/exploratory Pilot. Missing and failed terminal scores
stay null. The mechanism report's static adoption and runtime activation are
deterministic observational proxies; they are not causal attribution and do
not replace terminal outcomes.
