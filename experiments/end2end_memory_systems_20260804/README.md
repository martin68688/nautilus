# Experiment End2End — Frozen Pre-launch Packet

This directory is the executable, not-yet-submitted packet for the exploratory
10-system × 4-task × seed-1 Pilot.

## Launch gate

The packet status is `generated_not_submitted`. Builders do not launch training.
The current pre-Pilot sequence is:

1. run the one-index `Leaf × Dynamic Hybrid` Smoke and inspect retrieval,
   Prompt exposure, generated code, adoption, runtime activation and outcome;
2. only after that diagnostic is accepted, run the remaining Leaf controls;
3. run `validate_smoke_gate.py` over the retained output root; it requires all
   10 terminal-scored outcomes, verifies every infrastructure retry and full
   routing/Prompt trace, then exclusively creates a self-hashed
   `SMOKE_GATE.json`;
4. only after separate explicit authorization submit the four task-specific
   10-index Pilot Jobs;
5. create explicit higher-attempt retries only for retained infrastructure
   failures; never delete or overwrite attempt 0;
6. run terminal analysis before mechanism analysis.

## Frozen files

- `systems/`: one common config plus exactly 10 one-axis system overlays;
- `manifests/`: system, task, budget, source, Bundle, evaluator, Smoke and Pilot
  locks with canonical SHA-256 self-hashes;
- `jobs/`: Aerial/Leaf diagnostic Smoke Jobs and four unsubmitted Pilot Indexed
  Jobs;
- `prepare_direct_leaf_memory.py`: copies the already reviewed seed-heldout
  Base directly for Leaf. It deliberately performs no formal child publication,
  domain certification, or transition-to-SOP proof. The Base contains Leaf
  seed-43/44 history; Dynamic Hybrid pins the best eligible same-task RunForest
  item into one existing Prompt slot.
- `run_assignment.py`: finite Job PID 1, Agent subprocess, candidate freeze,
  terminal evaluator, immutable failure measurement, and a fail-closed Pilot
  dependency on the exact `SMOKE_GATE.json`;
- `validate_smoke_gate.py`: result-derived Smoke gate with distinct No Memory
  and memory-on activation checks;
- `analyze_results.py`: terminal/completion/negative-transfer/TTFV/cost first,
  routing/suppression/static-adoption/runtime-activation second;
- `build_manifests.py`: deterministic generator and non-mutating `--check`.

## Local validation

```bash
PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/build_manifests.py --check

PYTHONPATH=mlevolve /tmp/nautilus-e2e-py311/bin/python \
  experiments/end2end_memory_systems_20260804/run_assignment.py \
  --manifest experiments/end2end_memory_systems_20260804/manifests/smoke_manifest.json \
  --index 0 --dry-run
```

`--dry-run` verifies only the local frozen packet, performs zero Agent calls,
does not open cluster assets and writes no result directory.

After the authorized Smoke Job has reached terminal outcomes, the only supported
way to open the Pilot gate is:

```bash
/usr/local/bin/python -u \
  /workspace/nautilus-exp-end2end/experiments/end2end_memory_systems_20260804/validate_smoke_gate.py \
  --output-root /workspace/experiment-end2end-runs-v1 \
  --output /workspace/experiment-end2end-runs-v1/SMOKE_GATE.json
```

The output is created exclusively (`O_EXCL` semantics); it cannot overwrite an
earlier gate. All Pilot Job manifests pass that exact path to the runner, which
revalidates its self-hash and all Smoke/Pilot/component/source bindings before
opening any PVC-resident runtime asset.

## Interpretation boundary

Seed 1 is an engineering/exploratory Pilot. Missing and failed terminal scores
stay null. The mechanism report's static adoption and runtime activation are
deterministic observational proxies; they are not causal attribution and do
not replace terminal outcomes.
