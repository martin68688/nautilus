# Experiment End2End — Fast Experimental Packet

This directory is the executable, not-yet-submitted packet for the exploratory
10-system × 4-task × seed-1 Pilot.

## Execution mode

The packet status is `generated_not_submitted`. Builders do not launch training.
This experiment intentionally uses `experiment_fast_nonblocking_v1`: Bundle
provenance traversal, Host Protocol Preflight/runtime receipts, Authority
enforcement, adoption enforcement and the formal Smoke gate are disabled. The
pre-run check only confirms that the same-task best exists, is visible in the
final Prompt, and the solver/candidate subprocess entrypoints can start.

The current sequence is:

1. run the one-index `Leaf × Dynamic Hybrid` Smoke and inspect retrieval,
   Prompt exposure, generated code, adoption, runtime activation and outcome;
2. only after that diagnostic is accepted, run the remaining Leaf controls;
3. only after separate explicit authorization submit the four task-specific
   10-index Pilot Jobs;
4. create explicit higher-attempt retries only for retained infrastructure
   failures; never delete or overwrite attempt 0;
5. run terminal analysis before mechanism analysis.

## Frozen files

- `systems/`: one common config plus exactly 10 one-axis system overlays;
- `manifests/`: system, task, budget, source, Bundle, evaluator, Smoke and Pilot
  locks with canonical SHA-256 self-hashes;
- `jobs/`: Aerial diagnostics, the completed Leaf Dynamic condition, a
  generated-but-unsubmitted sequential Leaf 9-control Smoke Job, and four
  unsubmitted Pilot Indexed Jobs;
- `prepare_direct_leaf_memory.py`: copies the already reviewed seed-heldout
  Base directly for Leaf. It deliberately performs no formal child publication,
  domain certification, or transition-to-SOP proof. The Base contains Leaf
  seed-43/44 history; Dynamic Hybrid pins the best eligible same-task RunForest
  item into one existing Prompt slot.
- `confirm_experiment_intent.py`: one local, human-facing intent summary. It
  opens no Bundle/data, starts no subprocess, calls no model or cluster API,
  and never launches training;
- `run_assignment.py`: finite Job PID 1, Agent subprocess, terminal evaluator
  and immutable failure measurement; hashes/receipts remain metadata only;
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
