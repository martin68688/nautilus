# RunForest Composite Benchmark

This directory implements the preregistered T0-T4 benchmark from
`coordination/runforest_composite_benchmark_plan.md`.  It does not modify
production retrieval code and fails closed whenever human labels, semantic
review, runtime provenance, or hidden-holdout evidence are absent.

## Reproduce the completed offline phase

```bash
python build_composite_benchmark.py
python build_adoption_matrix.py
python build_micro_matrix.py
python build_replay_matrix.py
python run_offline_decisions.py --split dev
python run_offline_decisions.py --split test
python run_replay_repairs.py
python run_agent_adoption.py
python run_micro_execution.py --matrix manifests/micro_execution_matrix_v1.json --dry-run
python score_composite_benchmark.py --split test
python audit_memory_coverage.py
python finalize_benchmark.py
```

The MiniLM baseline needs the repository's Python 3.11 sentence-transformer
environment.  Every report records artifact hashes and every scientific claim
remains false until its configured gate is satisfied.

## T2 real-Agent execution

`run_adoption_generation.py --adapter COMMAND` calls an external Agent using a
JSON-over-stdin/stdout contract.  The Agent receives the decision and retrieved
memory cards, never silver/adjudicated gold.  Its output JSON must contain
`code`; model/token metadata and `adoption_outcome` are optional.  Score the
result with:

```bash
python run_agent_adoption.py --candidates reports/adoption_candidates_v1.jsonl
```

Mock outputs are allowed only for infrastructure tests and are excluded from
claims.

For a bounded Claude pilot (not a full claim-bearing run):

```bash
python run_adoption_generation.py --split dev --max-episodes 2 --adapter python claude_json_adapter.py
```

The runner defaults to `dev`. A test call requires the explicit
`--confirm-frozen-test` acknowledgement and must be the single frozen run, not
a pilot.

When a bounded pilot needs retries, retain every first-pass receipt and run
`score_adoption_pilot.py`; it reports first-pass reliability separately from
the consolidated earliest-success candidates.

## T3 replay execution

`build_replay_matrix.py` freezes the 48-case x R0-R3 comparison. R0 rejects,
R1 uses ordinary free-form Debug, R2 uses staged repair without an exposed
preservation contract, and R3 uses staged repair with preservation and runtime
provenance requirements. Generate candidates through the same external Agent:

```bash
python run_replay_generation.py --adapter COMMAND
python evaluate_replay_conditions.py --candidates reports/replay_candidates_v1.jsonl
```

Agent-declared runtime provenance is recorded but never treated as independently
verified. A clean-repair claim therefore remains closed until a separate
isolated executor produces evaluator-owned provenance.

`evaluate_replay_heldout.py` evaluates a separate 16-case challenge authored by
a fresh ClaudeAgent session that was forbidden to inspect the detector or the
existing fixtures. Its result is frozen and must not be used to tune the static
detector. It estimates structural robustness on this challenge, not population
recall.

`run_replay_repairs.py` always audits the 48 defective source programs before
execution and never executes them.  Semantic-review JSONL may be supplied with
`--semantic-reviews`; repaired candidates may be supplied with
`--candidate-repairs`.  A successful candidate must contain the exact ordered
five-stage history, journal-only intermediate stages, clean static and
preservation audits, and runtime provenance showing exactly one final holdout
evaluation.

## T4 real execution

`run_micro_execution.py` sends each adapter only a train path and prediction
output path.  The external `evaluate_hidden_holdout.py` process owns labels and
checks exact one-to-one `sample_id` coverage before scoring.  Then run
`score_downstream.py` on the scored receipts.  Failures remain failures and are
assigned the task's preregistered worst-valid metric and are never silently
dropped.  Before scoring test runs, freeze a dev-only B0 reference with
`freeze_baseline_reference.py`; `score_downstream.py` rejects test-derived or
unfrozen normalization references.  Scientific T4 runs must build the matrix
with `--isolation-mode container_filesystem`; the default process-only mode is
for smoke tests and can never open a downstream claim.

## Current evidence boundary

The generated report is diagnostic while labels remain silver.  Missing clean
coverage is reported as `insufficient_strategy_coverage`; it is not filled with
wrong abstraction levels.  T1 cannot identify portfolio effects because F/P
conditions differ only when an Agent is actually run.  T2/T3/T4 claims require
external model calls, semantic review, and training and therefore cannot be
opened by an offline smoke run.

`benchmark_terminal_report_v1.json` records the v1 terminal decision. The
independently authored replay challenge failed its preregistered safety gate,
so v1 correctly stops before claim-bearing T4. This is a completed fail-closed
negative experiment; it is not a claim that the unrun T2/T3/T4 tiers passed.
