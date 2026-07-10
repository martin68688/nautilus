# Claude Agent MCP Review: RunForest Leakage Audit V3

## Session

- Claude Agent MCP session: `24774f20-977e-4241-ab51-96e2701a0115`
- Model/mode: Opus, max effort, read-only
- Initial review: four subagents for detector, policy/search, persistence, replay/migration; lead reviewer personally verified severe findings
- Follow-up: same session re-read the fixes, tests, real graph and builder report

## Initial Verdict: REJECT

Claude confirmed two blockers:

1. Standard `model.fit(..., eval_set=[(X_val, y_val)])` monitoring was incorrectly treated as holdout training and hard-blocked.
2. The structural fingerprint normalized every identifier, so a historical `fit(X_val)` pattern could independently block a fresh-clean `fit(X_train)` program with the same syntax skeleton.

Claude also confirmed five major gaps:

- non-clean metrics could still affect reward/UCT;
- child-memory prompts exposed non-clean metrics and analysis;
- non-clean nodes entered `branch_successful_nodes` and aggregation references;
- Journal best filtering did not explicitly preserve audit-enforced mode for all-missing/legacy nodes;
- corrupt GlobalMemory loaded as empty and could overwrite the negative-memory file.

## Repairs Verified By Claude

- Monitoring-only kwargs (`eval_set`, `eval_metric`, `validation_data`, callbacks) no longer contribute training-input taint.
- Structural history can add a hard failure only when a fresh deterministic audit independently reports the same issue code.
- Non-clean nodes receive zero reward, have metrics withheld from child memory, and cannot enter branch-success aggregation.
- Journal serializes `audit_enforced`; AgentSearch sets it from `check_data_leakage`.
- Corrupt GlobalMemory sets `_load_error`, refuses writes, and preserves the original file.
- GlobalMemory persistence locks, reloads and merges disk records before atomic replacement.
- Explicit train partitions retain any taint already present on their RHS.
- Registry records verify their internal code hash and clean temporary files on failure.
- Real graph statistics and the implementation report match exactly.

## Follow-Up Verdict: APPROVE

Claude marked the original two Blockers and five Majors **CLOSED** and returned:

| Gate | Result |
|---|---|
| d93 exact replay | PASS |
| renamed structural gate | PASS |
| certified ranking isolation | PASS |
| mandatory repair closure | PASS |
| concurrent persistence | PASS |
| corrupt-state fail-closed behavior | PASS |

Claude verified that d93 remains rejected for real reasons: the punctuation vectorizer is fit on train/validation/test-derived data, downstream sparse features inherit that contamination, and the same validation set selects and reports ensemble weights. The prior XGBoost `eval_set` false positive is gone.

## Residual Minor Risks

- `fcntl.flock` depends on the shared filesystem's distributed lock semantics; do not run multiple Pods against the same GlobalMemory workspace without confirming Ceph lock behavior.
- Selection-bias and reset-index detectors remain conservative pattern detectors and can miss sufficiently rewritten semantic variants.
- Semantic similarity remains a review trigger rather than an automatic block to avoid uncontrolled false positives.

Claude originally noted that non-clean nodes could still update `local_best_node`. After the APPROVE response, this was additionally closed: audit repair-only nodes now skip improvement and local-best updates, with a regression assertion in the test suite.
