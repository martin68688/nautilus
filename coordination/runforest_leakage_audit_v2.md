# RunForest Leakage Audit V2

> Superseded by `runforest_leakage_audit_v3_implementation_report.md`. This file records the pre-hardening checkpoint at commit `dc6015dc`; its general closed-loop claims must not be used as the current implementation claim.

## Status

This change replaces the previous single LLM boolean gate with a structured audit and memory-admission pipeline.

- Local implementation: complete.
- Unit and integration tests: `34 passed`.
- RunForest graph/index/report: rebuilt locally.
- Running cluster Jobs: not hot-patched; a new source commit and new Job are required to use V2.

## Why V2 Exists

The old path was:

```text
finish GPU training
  -> ask an LLM whether leakage exists
  -> if yes, overwrite node.analysis and discard the metric
```

It had four gaps:

1. Known leaks were detected only after expensive training.
2. LLM failure returned `has_leakage=false`.
3. Negative findings were not stored as reusable memory.
4. Exact replay trusted an allowlist string and code hash, not the code's data flow.

The new path is:

```text
generated/replayed code
  -> deterministic static preflight
      -> hard leak: block before GPU
      -> protocol bias: allow execution, reject positive-memory admission
      -> clean: allow execution
  -> post-run LLM review for task-specific issues
  -> structured leakage_audit on SearchNode
  -> journal + hash registry + negative GlobalMemory
  -> RunForest FailurePattern nodes
  -> positive retrieval filter / debug warning injection
  -> exact replay checks the same audit before loading
```

## Audit Taxonomy

| Classification | Meaning | Execution | Metric | Memory |
|---|---|---|---|---|
| `hard_leakage` | Labels, row identity, future information, broken split indices | Block | Reject | Quarantine |
| `transductive_contamination` | A learned transformer is fit on validation/test inputs | Block | Reject | Quarantine |
| `selection_bias` | A tuning set is reused as the final reported evaluation | Allow with warning | Protocol-biased | Negative only |
| `clean` | No supported violation detected | Allow | Accept | Positive eligible |

An unavailable static audit is never positive-memory eligible. Heuristic cross-fold findings are warnings rather than automatic hard blocks until concrete data flow is established.

## Deterministic Checks

Implemented in `mlevolve/agents/leakage_audit.py`:

- `TRANSFORM_FIT_ON_HOLDOUT`
- `TRANSFORM_FIT_BEFORE_SPLIT`
- `RESET_INDEX_ORIGINAL_ARRAY_MISALIGNMENT`
- `REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION`
- `CROSS_FOLD_SUPERVISED_FEATURE_LEAKAGE`
- `STATIC_AUDIT_PARSE_FAILED`

The taint analysis is lexical-scope aware. It treats `train_test_split` outputs and explicit train-index slices as partition boundaries, avoiding the earlier false positive where safe `X_train` data inherited a `validation` taint from a combined variable name.

## Structured Node Record

Every audited `SearchNode` can now carry:

```json
{
  "schema": "mlevolve_leakage_audit_v1",
  "detector_version": "deterministic_static_v1",
  "code_sha256": "...",
  "status": "blocked|protocol_biased|warning|clean|audit_unavailable",
  "hard_block": true,
  "paper_grade_eligible": false,
  "metric_disposition": "reject|protocol_biased|accept|unverified",
  "memory_disposition": "quarantine|negative_only|positive_eligible",
  "issues": []
}
```

This object is serialized into the journal instead of relying only on prose in `node.analysis`.

## Memory Write Flow

### 1. Journal

All findings, including pre-execution blocks, are journaled with the code hash, issue evidence, remediation, role, and replay status.

### 2. Hash Registry

Each code hash receives an atomic record under:

```text
<workspace>/global_memory/leakage_audits/<code_sha256>.json
```

This prevents a repeated hash from being treated as unknown inside the same workspace and provides an auditable source for replay gating.

### 3. GlobalMemory

Blocked, biased, warning, and unavailable audits are saved as label `-1` records named `leakage_failure`. Their searchable text contains both evidence and required remediation. They are not silently deleted with other buggy nodes.

### 4. RunForest

`build_run_forest_memory.py` re-audits historical source code and emits deduplicated `FailurePattern` nodes keyed by:

```text
code_sha256 + issue_code
```

Edges:

```text
RunNode -> has_failure_pattern -> FailurePattern
FailurePattern -> blocks_adoption_of -> RunNode
```

Draft and improve retrieval admit only `memory_disposition=positive_eligible`. Debug retrieval can inspect rejected nodes and now injects the exact issue, evidence, and repair instruction.

## Replay Gate

`run_forest_replay.py` now checks:

1. source-membership and graph audit metadata;
2. task/run/node identity;
3. journal and graph code SHA256 agreement;
4. current deterministic static audit;
5. hash-registry audit, when present;
6. manifest `audit_status=verified_clean`;
7. `paper_grade_eligible=true`.

Any failed boundary stops before GPU execution.

## Three-Model Target

The historical target `20260509_185008/.../d93b4c2a...` is now:

```text
audit_status: candidate_replay
metric_status: historical_selected_validation_score
memory_disposition: quarantine
```

Detected issues:

- `TRANSFORM_FIT_ON_HOLDOUT`: punctuation `CountVectorizer` fits train + validation + test.
- `REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION`: the same validation rows select weights and report `0.2013`.

The model family remains useful evidence, but exact replay now fails with those issue codes. A clean reimplementation must retain DeBERTa + XGBoost + LogisticRegression while fitting preprocessing on train only and evaluating on an untouched outer holdout.

## Rebuilt Memory Statistics

The current local graph contains:

- code-bearing RunNodes audited: `1324`
- clean positive-eligible: `785`
- blocked: `340`
- protocol-biased: `42`
- warning: `16`
- audit unavailable / parse failure: `141`
- deduplicated FailurePattern nodes: `668`

Audit-unavailable nodes remain available only as negative/debug evidence.

## Verification

Verified behaviors:

- Safe train-only vectorizer code passes.
- Concatenated train/validation/test fit is blocked before execution.
- Validation-only ensemble selection is classified as protocol bias, not target leakage.
- The audit is written to the hash registry and negative memory.
- RunForest debug retrieval returns both historical issue codes and fixes.
- The contaminated three-model node is absent from draft retrieval.
- Exact replay rejects the historical source before GPU launch.
- `PYTHONPATH=mlevolve python -m pytest -q tests`: `34 passed`.
