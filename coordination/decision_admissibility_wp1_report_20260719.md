# Decision Admissibility WP1 Report

Date: 2026-07-19  
Baseline: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP1 — Authority semantic correctness

## Outcome

WP1 Stop Gate: **PASSED**.

The Authority layer now records orthogonal generation and governance stages,
distinguishes diagnostic/candidate/positive distillation, fails closed on
high-risk internal errors, and validates persisted GlobalMemory decisions by
outcome, operation, stage axes, protocol hash, task scope, and policy version.

## Modified code

- `mlevolve/authority/stage_ontology.py` (new)
- `mlevolve/authority/models.py`
- `mlevolve/authority/authority_engine.py`
- `mlevolve/authority/policy.py`
- `mlevolve/authority/protocol_compiler.py`
- `mlevolve/authority/derivation_guard.py`
- `mlevolve/authority/adapters/mlevolve/runtime.py`
- `mlevolve/authority/__init__.py`
- `mlevolve/agents/memory/global_memory.py`
- `mlevolve/engine/agent_search.py`
- `mlevolve/agents/debug_agent.py`
- `mlevolve/agents/planner/planner_with_memory.py`

## New public semantics

- `GenerationStage`: draft, model design, improve, debug, evolution, fusion.
- `GovernanceStage`: retrieval, branch selection, memory writeback,
  distillation, replay.
- `AuthorityRequest`, `AuthorityDecision`, and `AuthorityScope` carry both axes;
  legacy `DecisionStage` remains serialized for one migration cycle.
- `ClaimType` adds `METHOD_HYPOTHESIS`, `DEBUG_REPAIR`, and `AUDIT_FINDING`.
- `Operation` adds `DISTILL_DIAGNOSTIC`, `DISTILL_CANDIDATE`, and
  `DISTILL_POSITIVE`; legacy `DISTILL` maps to positive-distillation policy.
- `GlobalMemoryLayer.configure_authority(...)` binds the live decision registry,
  active protocol, policy version, and task.
- Positive GlobalMemory records in enforce mode require a persisted or live
  decision whose ALLOW outcome and complete permitted scope match the request.

## Tests added

- `tests/authority/test_stage_ontology.py`
- `tests/authority/test_claim_types.py`
- `tests/authority/test_global_memory_authority_scope.py`
- `tests/authority/test_high_risk_fail_closed.py`

## Verification

All commands ran against an isolated candidate tree so benchmark rebuilds could
not modify user assets.

```text
WP1 focused authority suite: 44 passed in 0.43s
Planned baseline plus WP1 tests: 241 passed in 39.39s
GlobalMemory/leakage regression: 33 passed in 2.18s
Full tracked tests/: 334 passed in 59.01s
compileall: passed
git diff --check: passed
```

## Stop Gate evidence

- Legacy tests do not regress: full tracked suite passed.
- Every runtime stage emitted by SearchNode creation has one explicit dual-axis
  mapping, covered by tests.
- A pseudo decision ref without a decision cannot pass enforce.
- DENY outcome, stale policy, wrong protocol hash, wrong task, wrong operation,
  wrong generation stage, wrong governance stage, or a widened/missing scope is
  suppressed before GlobalMemory ranking.
- High-risk internal exceptions produce a DENY decision and cannot fall back to
  legacy allow, even when the ordinary denial rollback flag is relaxed.

## Known migration risks

- Existing persisted GlobalMemory records do not contain full decision
  snapshots or dual-axis scope and therefore fail closed in enforce mode. Shadow
  remains the default until WP7 disagreement review.
- A promotion decision does not automatically authorize later candidate
  generation. Operation-specific read authority must be issued by later
  visibility/writeback integration.
- Legacy `DecisionStage` and `Operation.DISTILL` remain readable for one
  migration cycle but do not drive new policy decisions.

No Kubernetes resources, commit, push, PR, or paper headline changes were made
during WP1.
