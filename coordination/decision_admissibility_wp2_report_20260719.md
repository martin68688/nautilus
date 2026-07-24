# Decision Admissibility WP2 Report

Date: 2026-07-19  
Baseline: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP2 — Claim decomposition and trusted collectors

## Outcome

WP2 Stop Gate: **PASSED**.

Mixed RunNodes are decomposed into independently governed fact claims, and
high-risk evidence now crosses an explicit host collector boundary. Static
legacy audit translations remain `legacy_static_only`; they can support
diagnostic navigation but cannot satisfy trusted score/ranking obligations.

## Modified code

- `mlevolve/authority/claim_decomposer.py` (new)
- `mlevolve/authority/collectors/base.py` (new)
- `mlevolve/authority/collectors/trusted.py` (new)
- `mlevolve/authority/collectors/__init__.py` (new)
- `mlevolve/authority/adapters/mlevolve/transition_adapter.py` (new)
- `mlevolve/authority/adapters/mlevolve/node_adapter.py`
- `mlevolve/authority/adapters/mlevolve/receipt_bridge.py`
- `mlevolve/authority/adapters/mlevolve/runtime.py`
- `mlevolve/authority/models.py`
- `mlevolve/authority/receipt_collectors.py`
- `mlevolve/authority/evidence_graph.py`
- `mlevolve/authority/protocol_compiler.py`
- `tests/test_causal_granularity_benchmark_v2.py` (asset-safe test outputs)

## Claim decomposition semantics

- Deterministic facts are extracted before any LLM proposal is considered.
- Stable Claim IDs are derived from artifact kind, artifact ID, ClaimType, and
  deterministic fact boundary. The legacy score ID remains
  `node:<node-id>:score` for one migration cycle.
- A mixed node can independently carry `EXECUTED`, `METHOD_HYPOTHESIS`,
  `DEBUG_REPAIR`, `AUDIT_FINDING`, and `SCORE` claims.
- Source artifact and evidence references are bound against deterministic
  catalogs. Unknown refs, invented ClaimTypes, and ambiguous proposals enter a
  quarantine list rather than producing a Claim.
- An LLM proposal may only reword or narrow exactly one existing deterministic
  fact; it cannot create evidence, a score, superiority, or authority.
- Transition decomposition binds both parent and child artifacts and preserves
  parent Claim lineage.
- Operation-specific claim selection prevents an executed-code fact from being
  silently substituted for a score/superiority Claim.

## Trusted collector boundary

- `TrustedCollectorHost` mints in-memory `HostObservation` objects and records
  their complete immutable fingerprint before a collector can consume them.
- A copied capability is insufficient to rebind payload, protocol, artifact,
  source, or observation metadata; unregistered, mutated, cross-host, replayed,
  and rebound observations are rejected.
- Receipt IDs are stable over collector/version/protocol/validated-payload
  identity. Observation IDs are intentionally excluded from the stable ID and
  included in the advancing parent/event hash chain.
- Public `make_receipt(...)` always emits `legacy_static_only`, including when
  an Agent payload contains `verified=true`.
- Trusted collectors validate code execution, method identity, split lineage,
  fit scope, prediction scope, evaluator integrity, selection freeze, seed
  aggregation, replication, static/runtime actuation, counterfactual actuation,
  and derivation facts.
- Hash-bearing collector fields require real SHA-256 syntax. Failed execution,
  overlap, holdout fitting, evaluator tampering, unfrozen selection,
  best-seed selection, unequal replication budgets, and a counterfactual with
  no action/code delta cannot mint affirmative trusted evidence.

## Evidence semantics

- Score, pairwise, and generalization authority now requires prediction scope
  in addition to method, execution, split, fit, evaluator, and selection facts.
- Pairwise authority has a reachable trusted path: paired/preregistered seed
  aggregation plus the protocol-required number of trusted replications.
- Causal and generalization claims also require successful code execution.
- Positive distillation/causal attribution requires runtime and a real
  counterfactual action/code delta.
- Audit contradictions are Claim-scoped: they block score, pairwise,
  generalization, and causal Claims without suppressing valid repair/audit
  Claims from the same RunNode.
- Evidence remains AND-within-path and OR-across-paths. A contaminated
  alternative path is rejected without poisoning an independently complete
  clean support path.
- Every trusted receipt event is appended to the ledger by event hash, even
  when repeated observations resolve to the same stable Receipt ID.

## Tests added or extended

- `tests/authority/test_claim_decomposition.py`
- `tests/authority/test_mixed_value_authority.py`
- `tests/authority/test_trusted_collectors.py`
- `tests/authority/test_receipt_trust_boundary.py`
- `tests/authority/test_core.py`
- `tests/authority/test_mlevolve_adapter.py`
- `tests/authority/test_high_risk_fail_closed.py`
- `tests/test_causal_granularity_benchmark_v2.py`

## Verification

All benchmark-bearing suites ran in a detached candidate worktree. The causal
benchmark's query, gold, manifest, receipt, and report paths were rebound to a
pytest temporary directory before build/evaluate.

```text
Original WP2 planned tests: 12 passed in 0.03s
Expanded WP2 focused authority tests: 34 passed in 0.36s
Complete authority suite: 60 passed in 0.38s
Planned baseline plus WP1/WP2 suite: 257 passed in 86.04s
Full tests/: 350 passed in 107.28s
compileall: passed
git diff --check: passed
Candidate source-tree writeback check: passed
```

## Stop Gate evidence

- Mixed RunNode: one test fixture produces valid repair and audit Claims while
  its contaminated score Claim remains denied with a blocking audit receipt.
- Forged trust: Agent-created `verified=true`, a manually constructed
  `HostObservation`, cross-host reuse, post-capture mutation, and
  capability-preserving payload rebinding cannot create a trusted Receipt.
- No execution-to-superiority upgrade: code execution alone cannot authorize
  SCORE, PAIRWISE_SUPERIORITY, or an incompatible EXECUTED-to-RANK request.
- Positive control: a complete trusted pairwise path does authorize, proving
  that denial is caused by missing/invalid evidence rather than an unreachable
  policy.
- Legacy regression: the complete tracked test suite passed.

## Dirty-worktree and asset safety audit

The original 1,134-path stat snapshot was rechecked at WP2 close. Three paths
are intentionally excluded from the unrelated-asset comparison:

1. `tests/test_causal_granularity_benchmark_v2.py` — an authorized test change
   that prevents source-tree output writes.
2. `paper-skills/eval_composite_memory/reports/causal_granularity_report_v2.json`
   — a regenerable benchmark report.
3. `paper-skills/eval_composite_memory/reports/causal_granularity_receipts_v2.jsonl`
   — a regenerable benchmark receipt stream excluded from the baseline commit.

The remaining 1,131 pre-existing paths match size and mtime exactly:

```text
expected: 089fa7c6ea45cb8f457aa0c63519a536d061814b4f10584e4ae06d253daa1bf6
current:  089fa7c6ea45cb8f457aa0c63519a536d061814b4f10584e4ae06d253daa1bf6
```

During this audit, a prior WP1-era benchmark process was found to have
regenerated the two report/receipt outputs in the main worktree. Three other
benchmark inputs had unchanged Git content and only changed mtimes; their
snapshot mtimes were restored exactly. The report and receipt currently form a
coherent pair for baseline `b47dab63...` with receipt SHA-256
`68cdc1ce098ee18c068d42d5863364cbad493aab49e9028f2f931014f4402e93`.
The original untracked receipt content was not available as a Git object or
filesystem snapshot, so no fake restoration of its old hash/mtime was
performed. The hermetic test fixture prevents recurrence.

## Known migration risks

- The capability boundary is process-local, not a cryptographic signature.
  Collector hosts must remain outside Agent-controlled object construction and
  must not expose observation objects to model/tool payloads.
- Current MLEvolve runtime provenance is still an aggregate host digest/count
  record. WP5 must connect collectors to concrete static/runtime actuation
  events rather than broadening this bridge by assertion.
- Trusted replication and counterfactual collectors are implemented and tested,
  but production emitters arrive with later actuation/experiment work packages.
- Legacy static evidence remains useful for inspect/debug navigation but fails
  closed for high-risk ranking, promotion, code seeding, and positive
  distillation.

No Kubernetes resources, commit, push, PR, or paper headline changes were made
during WP2.
