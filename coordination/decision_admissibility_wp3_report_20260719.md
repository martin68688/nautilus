# Decision Admissibility WP3 Stop-Gate Report

Date: 2026-07-19  
Branch: `codex/dual-time-procedural-memory`  
Remote-verified baseline: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP3 — P4-B SOP Visibility Gateway  
Status: **PASSED**

## Scope completed

WP3 implements a clause-level visibility boundary before SOP ranking and prompt
materialization:

- `SOPClauseV1`, `VisibilityRequest`, and `VisibleSOPPack` are wired into the
  production stage-hybrid layer.
- `SOPVisibilityGateway` evaluates Claim/Operation/stage/protocol/task/policy
  scope before exposing clause retrieval or prompt text.
- `_rank_sops`, layered L1/L2 retrieval, debug attachments, Tree projection,
  RRF, and prompt formatting consume only the enforce-mode clause projection.
- Navigation and adoption edges are separate. Legacy `distills_to` and
  `navigation_attached_to` are navigation-only; adoption requires both
  `authorized_distills_to` and an explicit `allow`/`allow_with_warning`
  outcome.
- Deprecated or forged `attached_sop_ids` cannot construct a retrieval edge.
- Partial-container geometry is disabled unless the bundle explicitly attests
  the projected retrieval-text hash. A bare boolean cannot bless an embedding
  produced from a mixed container.
- Cache keys bind protocol hash, operation, both stage axes, task scope, bundle
  version, policy version, token budget, candidate SOP IDs, relevant Claims,
  evidence paths/Receipts, and frozen decision snapshots.
- Token budgeting occurs after Authority evaluation and before rendering;
  diagnostic/positive content is prioritized over navigation-only warnings.
- Claim ID, exact Claim type, subject artifact, active protocol, task, stages,
  operation, and policy version must all match a frozen decision snapshot.
- Authority requests now also reject a request whose artifact or task context
  does not match the underlying Claim.
- `DISTILL_DIAGNOSTIC` no longer inherits uncertified Debug/Inspect navigation;
  an empty evidence path cannot satisfy `require_trusted_receipts=True`.
- Internal visibility errors suppress high-risk content. Debug fallback retains
  only diagnostic DEBUG_REPAIR/AUDIT_FINDING content as warning-only
  navigation; SCORE content remains suppressed.

## Planned WP3 tests

The five required test modules were added:

```text
tests/authority/test_sop_visibility_gateway.py
tests/authority/test_mixed_value_sop_visibility.py
tests/authority/test_visibility_pre_prompt.py
tests/authority/test_visibility_projection_bypass.py
tests/authority/test_legacy_sop_visibility.py
```

A small deterministic helper module constructs mixed Claim/SOP and temporary
RunForest fixtures without writing source-tree benchmark outputs:

```text
tests/authority/sop_visibility_helpers.py
```

Focused result:

```text
16 passed
```

## Mixed-value acceptance result

The deterministic mixed SOP contains:

```text
C1 DEBUG_REPAIR: align OOF predictions by sample_id
C2 AUDIT_FINDING: historical selection read test labels
C3 SCORE: historical contaminated score 0.92
```

Observed enforce behavior:

| View | C1 repair | C2 audit warning | C3 score |
|---|---|---|---|
| Debug | visible diagnostic | visible warning | suppressed |
| Rank | not a score candidate | not a score candidate | suppressed |
| Inspect | visible warning-only | visible warning-only | visible warning-only |

Debug oracle retention is `2/2 = 100%`. Rank has an empty SOP pack: the score
clause ID is absent from embedding candidates and RRF eligibility, its text is
absent from the formatted prompt, and rendered token count is zero.

Deterministic assertions:

```text
Unauthorized Prompt Exposure = 0
Unauthorized Activation = 0
Oracle-allowed Debug/Repair knowledge retention = 100%
```

## Projection and cache bypass result

The bypass suite covers three rejected edge forms:

1. `navigation_attached_to + quarantine`;
2. legacy `distills_to + allow`;
3. `authorized_distills_to + deny`.

In every case, forged `attached_sop_ids`, transition attachment metadata, Tree
projection, causal attachment, and RRF cannot recreate an adoption path. The
positive control `authorized_distills_to + allow` is consumable.

Frozen-decision cache tests mutate an existing decision ID in place. Changes to
Claim type, artifact binding, or outcome/scope change the cache key and are
re-evaluated fail-closed rather than returning a stale allow.

## Real legacy graph migration coverage

Read-only source artifacts:

```text
paper-skills/hyper_memory/run_forest_graph.json
paper-skills/hyper_memory/run_forest_index.npz
```

Neither artifact was modified or regenerated.

Observed coverage:

| Item | Count |
|---|---:|
| Legacy SOP containers | 281 |
| Materialized legacy clauses | 843 |
| Legacy/quarantine navigation edges | 2,773 |
| Authorized adoption edges | 0 |

Enforce views, using a deliberately large token budget only to measure complete
migration coverage:

| View | Visible SOPs | Visible clauses | Suppressed SOPs | Suppressed clauses | Rendered tokens | Empty pack | Cold evaluation latency |
|---|---:|---:|---:|---:|---:|---|---:|
| Inspect | 281 | 843 | 0 | 0 | 15,289 | no | 6.02 ms |
| Debug | 281 | 843 | 0 | 0 | 15,289 | no | 6.17 ms |
| Rank | 0 | 0 | 281 | 843 | 0 | yes | 2.39 ms |

These latency values are a local snapshot, not a post-hoc pilot threshold.
Normal requests still apply their configured token budget.

After a Rank request, the sum of active SOP-transition adoption links is zero.
After a Debug request, all 2,773 legacy links remain available as warning-only
navigation. Thus quarantine history remains inspectable without becoming
rankable or adoptable.

## Regression and health checks

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q tests/authority
# 77 passed

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q tests/test_stage_aware_hybrid_memory.py
# 48 passed

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority \
  tests/test_stage_aware_hybrid_memory.py \
  tests/test_causal_granularity_benchmark_v2.py \
  tests/test_protocol_repair.py \
  tests/test_run_forest_memory.py
# 274 passed

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests --ignore=tests/test_composite_memory_benchmark.py
# 367 passed

PYTHONPATH=mlevolve .venv/bin/python -m compileall -q mlevolve paper-skills tests
git diff --check
# passed
```

### Pre-existing frozen-lock inconsistency kept untouched

An unfiltered `pytest -q tests` additionally collects the user-untracked file
`tests/test_composite_memory_benchmark.py`. Its test
`test_heldout_replay_set_is_independently_authored_and_frozen` fails because the
tracked frozen lock records detector SHA-256
`ae30aac332b6f62dccc784955ec2952268d8a1381085bcadb4d683b9d8f6a221`, while
`mlevolve/agents/leakage_audit.py` is
`8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`.

The current detector hash is identical to baseline HEAD `b47dab63`; neither the
detector nor the frozen lock is in the WP0-WP3 diff. The inconsistency therefore
exists in the checkpoint itself and is not a WP3 regression. The lock was not
rewritten because doing so would retroactively alter a frozen held-out benchmark
asset. The single failing test is reported rather than hidden.

## Asset-safety result

- No old graph/index, run journal, PPT/PDF, raw dataset, credential file, or
  unrelated user-untracked asset was deleted, moved, staged, or overwritten.
- All new WP3 graph fixtures are written under pytest temporary directories.
- The previously documented causal benchmark report/receipt event remains
  unchanged by WP3.
- No commit, push, PR, Kubernetes Pod/Job, or paper headline change was made.

## Stop-gate decision

- [x] Section 11.10 mixed-value semantics pass.
- [x] Suppressed score has zero embedding/RRF/prompt/token influence in Rank.
- [x] Edge deletion/rejection, projection, cache, and `attached_sop_ids` bypasses
      are closed.
- [x] Unauthorized Prompt Exposure is zero in the deterministic suite.
- [x] Unauthorized Activation is zero in the deterministic suite.
- [x] Oracle-allowed Debug/Repair retention is 100%.
- [x] All 281 legacy SOPs have visible/suppressed/legacy/empty migration
      coverage.
- [x] All 2,773 quarantine edges have zero high-risk adoption consumption while
      Debug/Inspect warning navigation remains functional.
- [x] Latency, token, and empty-pack overhead are reported without choosing a
      favorable threshold after observation.

**WP3 Stop Gate: PASSED. WP4 may begin.**
