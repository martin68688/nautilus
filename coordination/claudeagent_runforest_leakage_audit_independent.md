# Claude Agent MCP: RunForest Leakage Audit V2 (Independent Pass)

- Date: 2026-07-10
- Session: `8dafc6f9-ce0c-4b8b-9e95-4676232f2c7f`
- Runtime: Claude Agent MCP
- Mode: `Opus + effort=high`, single reviewer
- Permissions: `Read`, `Glob`, `Grep`
- Mutation permissions: none

This file records the independent pass's final public findings. Intermediate hidden reasoning is intentionally not included.

## Verdict

**APPROVE WITH CHANGES.** The reviewer independently confirmed that d93 is downgraded and excluded from exact replay, but found correctness and auditability gaps that should be fixed before claiming a general leakage-prevention closure.

## Blocker

### Split boundary does not actually clear seeded taint

**Confirmed bug.** In `mlevolve/agents/leakage_audit.py:281-297`, semantic taints are first seeded from variable names. When a split or explicit train partition is detected, the code clears only `assigned_taints`, then updates the already-seeded `taints[name]` with an empty set. Existing taint is never removed.

Example:

```python
df_train_valid, df_test = train_test_split(df, test_size=0.2)
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(df_train_valid)
```

`df_train_valid` is the training-side split, but the token `valid` seeds a validation taint. The split branch is a no-op for that existing taint, so the legal fit can be hard-blocked as `TRANSFORM_FIT_ON_HOLDOUT`.

The existing clean test uses `X_train`, so it does not cover the claimed false-positive fix.

## Majors

### Selection-bias detector is a narrow fingerprint

`REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION` requires a conjunction of specific weight names, validation-probability names, `arange/linspace/GridSearch`, and log-loss tokens. `scipy.optimize.minimize`, alternate report-set names, or alternate scoring functions can implement the same biased protocol without matching.

### Leakage registry writes are not thread-safe

`persist_audit` uses a temporary name derived only from `os.getpid()`. Concurrent worker threads writing the same code hash share the same temporary path. The read-modify-write sequence is not protected, so occurrence metadata can be lost.

### GlobalMemory writes are not protected

`GlobalMemory.save_leakage_audit` and `_save_memory` can rewrite the same JSON file concurrently. Last-writer-wins behavior can drop negative leakage records and weaken audit provenance.

### Missing leakage_audit is positive-memory fail-open

`external_skill_memory._positive_memory_eligible` falls back to `is_buggy is not True` when audit data is absent. Legacy or partially built nodes can therefore enter positive retrieval. Missing audit should be ineligible.

### Corrupt registry fallback is neutralized during merge

`load_registry_audit` returns an unavailable record without `detector_status`. `merge_audits` ignores that missing status and empty issue list, potentially losing registry-only LLM findings. Replay has fresh static audit as a partial safety net; normal retrieval does not.

### TRANSFORM_FIT_BEFORE_SPLIT is asymmetric

The rule can hard-block common pre-CV fitting on training data when argument names contain `train`, while missing `scaler.fit(X)` because `X` lacks a train token. The reviewer recommends either a stronger structural rule or warning-level treatment until semantics are more reliable.

## Minors

- `CROSS_FOLD_SUPERVISED_FEATURE_LEAKAGE` is tied to specific variable names.
- LLM exception output combines `has_leakage=False` with `classification=audit_unavailable`, which is safe only because current downstream code keys off classification.
- reset-index misalignment detection is regex-fragile and misses multiline/loc/value variants.
- `merge_audits` lets the last non-complete detector status win, which can preserve stale unavailable state.
- `_is_split_assignment` can treat unrelated `.split()` calls as data partitions once taint clearing is fixed.
- Concurrency, parse failure, merge conflicts, registry corruption and real-graph d93 debug retrieval lack direct tests.

## Closure Results

### d93 downgrade

**PASS.** The real RunForest graph marks d93 as blocked, non-paper-grade, quarantined and metric-rejected. Both expected issue codes are present, FailurePattern links exist, positive retrieval excludes it, and exact replay raises a leakage-audit error.

### Preventing the same class next time

**PARTIAL FAIL.** Storage and filtering are connected for the known node and exact code. Semantic variants can evade the narrow detector, so the guarantee does not extend to the whole method family.

## Recommended Tests

- A train-side split variable containing `valid` must remain clean.
- Fit-before-split on an uninformative name such as `X` must be classified intentionally.
- Selection bias using `scipy.optimize.minimize` must be detected.
- Renamed cross-fold model leakage must be detected.
- Same-hash concurrent registry persistence must retain both occurrences.
- Concurrent GlobalMemory leakage records must not be lost or corrupt JSON.
- Missing audit must fail closed for positive retrieval.
- Corrupt registry state must survive merge as unavailable.
- The real d93 graph retrieval must surface both issue codes in debug context.
