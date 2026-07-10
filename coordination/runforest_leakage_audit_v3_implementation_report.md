# RunForest Leakage Audit V3 Implementation Report

## Build Boundary

- Baseline commit: `dc6015dc`
- Branch: `codex/dual-time-procedural-memory`
- Audit schema: `mlevolve_leakage_audit_v2`
- Detector: `deterministic_static_v2`
- Baseline verification: `34 passed`
- V3 verification before independent review: `44 passed`; post-review final suite: `51 passed`.

## Implemented Controls

- Certified ranking now requires a complete, clean audit whose code SHA matches the current node.
- Blocked, protocol-biased, warning, unavailable, and missing audits cannot update best/top artifacts or enter positive memory.
- Non-clean nodes retain diagnostic metrics and enter a mandatory repair-only branch.
- Repair-only nodes receive zero reward and cannot update local-best, branch-success, aggregation, or child-memory metric context.
- Debug/improve prompts place a leakage repair contract before ordinary optimization instructions.
- Children inherit leakage evidence as lineage context, not as the fresh verdict for their rewritten code.
- Repair branches stop after two unsuccessful repair generations.
- Taint propagation now treats assignments as overwrites and assigns train/validation roles by split output position.
- Holdout fitting is checked for arbitrary fit receivers, including Pipeline and custom transformers.
- Train-only preprocessing before CV is warning-level protocol evidence rather than an automatic hard leak.
- Exact SHA and normalized-AST FailurePatterns participate in pre-execution gating; semantic similarity remains advisory.
- Registry and GlobalMemory writes use locks and unique atomic temporary files.
- Corrupt registry state remains unavailable during audit merge instead of disappearing as an empty clean opinion.

## Real Graph Migration

Audited code-bearing RunNodes: `1324`.

V3 status distribution:

- clean: `693`
- blocked: `280`
- protocol-biased: `55`
- warning: `155`
- audit unavailable: `141`
- FailurePattern nodes: `787`

V1 to V2 transitions:

| V1 | V2 | Count |
|---|---|---:|
| audit_unavailable | audit_unavailable | 141 |
| blocked | blocked | 199 |
| blocked | protocol_biased | 3 |
| blocked | warning | 138 |
| clean | blocked | 65 |
| clean | clean | 693 |
| clean | protocol_biased | 11 |
| clean | warning | 16 |
| protocol_biased | blocked | 1 |
| protocol_biased | protocol_biased | 41 |
| warning | blocked | 15 |
| warning | warning | 1 |

The large `blocked -> warning` movement is primarily the deliberate downgrade of train-only pre-CV transformer fitting. The `clean -> blocked` movement comes from broader holdout aliases and fit-receiver coverage. The post-review rebuild also excludes `eval_set`/`validation_data` monitoring arguments from training-data taint, removing the XGBoost early-stopping false positive identified by Claude Agent.

## d93 Result

The historical three-model node remains blocked and quarantined. Current findings include:

- `TRANSFORM_FIT_ON_HOLDOUT`
- `REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION`

A version produced by renaming local variables has the same structural fingerprint and retrieves both FailurePatterns. Exact replay remains rejected before GPU execution.

## Concurrency Verification

- Twenty concurrent registry writes for the same code hash retain all twenty occurrences.
- Twenty concurrent GlobalMemory negative writes retain all source node ids and produce valid JSON.

## Claim Boundary

Allowed claim:

> RunForest V3 blocks known leaked code and structurally equivalent variants, excludes all non-clean metrics from certified ranking and positive memory, and routes non-clean branches through mandatory repair.

Not allowed:

> RunForest universally recognizes every semantically equivalent leakage implementation.

Semantic similarity remains a review trigger because making it an automatic block would introduce an uncontrolled false-positive surface.
