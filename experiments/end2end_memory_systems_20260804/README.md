# Experiment End2End — Fast Experimental Packet

This directory is the executable, not-yet-submitted packet for the exploratory
10-system × 4-task × seed-1 Pilot.

## Execution mode

The packet status is `generated_not_submitted`. Builders do not launch training.
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
3. only after separate explicit authorization submit the one 40-index Pilot
   Job; `parallelism: 1` means the complete matrix occupies one A100 at a time;
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
5. run terminal analysis before mechanism analysis.

## Frozen files

- `systems/`: one common config plus exactly 10 one-axis system overlays;
- `manifests/`: system, task, budget, source, Bundle, evaluator, Smoke and Pilot
  locks with canonical SHA-256 self-hashes;
- `jobs/`: retained Smoke workload definitions and one unsubmitted
  `pilot-all-40-indexed-job.yaml` (`completions: 40`, `parallelism: 1`, A100,
  16 CPU, 64 GiB, `--resume`);
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
- `analyze_results.py`: terminal/completion/negative-transfer/TTFV/cost first,
  routing/suppression/static-adoption/runtime-activation second;
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
```

`--dry-run` loads only the local packet, performs zero Agent calls,
does not open cluster assets and writes no result directory.

## Interpretation boundary

Seed 1 is an engineering/exploratory Pilot. Missing and failed terminal scores
stay null. The mechanism report's static adoption and runtime activation are
deterministic observational proxies; they are not causal attribution and do
not replace terminal outcomes.
