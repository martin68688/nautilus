# Decision Admissibility schema and API reference

Status: final implementation reference for the WP0–WP8 closeout.

Canonical implementation lives under `mlevolve/authority/`,
`mlevolve/agents/memory/`, `mlevolve/fixed_holdout/`, and
`paper-skills/memory_bundle/`. This document describes public semantic
contracts, not every private helper.

## 1. Governed transaction

The atomic authorization key is:

```text
(Claim, Operation, GenerationStage, GovernanceStage,
 active ProtocolRef, TaskContext, Bundle version)
```

Retrieval relevance never grants authority. The transaction is:

1. propose stage-compatible RunForest/SOP evidence;
2. decompose containers into typed `Claim` / `SOPClauseV1` objects;
3. compile protocol-relative evidence obligations;
4. evaluate authority before embedding, ranking, token allocation, or Prompt;
5. execute the current candidate under host-owned collectors;
6. publish the current result, historical adoption, and causal attribution
   through three different operations and objects.

Unknown stages, protocols, hashes, receipt types, or high-risk obligations fail
closed.

## 2. Canonical stage axes

Defined in `mlevolve/authority/stage_ontology.py`.

### GenerationStage

| Value | Meaning |
|---|---|
| `draft` | initial strategy generation |
| `model_design` | model/feature/loss design |
| `improve` | iterative candidate improvement |
| `debug` | failure diagnosis and repair |
| `evolution` | descendant or memory evolution |
| `fusion` | candidate fusion |

### GovernanceStage

| Value | Meaning |
|---|---|
| `retrieval` | memory proposal and visibility |
| `branch_selection` | ranking/selecting candidates |
| `memory_writeback` | publishing Result/Adoption/Causal objects |
| `distillation` | producing reusable clauses |
| `replay` | protocol repair / Clean Replay |

Use `resolve_stage_axes(...)`. Runtime-stage and one-cycle legacy mappings are
deterministic; callers may override only an explicit axis. Missing or unknown
values raise `ValueError`.

## 3. Claims

Defined by `ClaimType` and `Claim` in `mlevolve/authority/models.py`.

Claim types:

- `executed`
- `score`
- `method_hypothesis`
- `debug_repair`
- `audit_finding`
- `experience_adoption`
- `pairwise_superiority`
- `causal_attribution`
- `generalization`

Required identity fields are `claim_id`, `claim_type`,
`subject_artifact_id`, `task_scope`, `method_fingerprint`,
`protocol_ref`, and `statement`. Provenance is append-only through
`parent_claims`, `source_artifact_refs`, and `evidence_refs`.
`boundary` records non-transferable scope. A mixed SOP or run must be
decomposed; authority is never unioned across its clauses.

## 4. Operations

Defined by `Operation`.

### Read/proposal operations

- `inspect`
- `debug_hypothesis`
- `generate_candidate`
- `repair_seed`

These may retain warnings or request replay, but they cannot inherit source
scores.

### Rank/governance operations

- `rank`
- `select`
- `code_seed`
- `derived_publication`

They are high-risk and require compatible protocol, scope, clean ancestry, and
the compiled trusted evidence path.

### Three writeback operations

| Operation | Subject | Minimum semantic evidence |
|---|---|---|
| `promote_result` | current target node/result | target execution, protocol, split/fit/prediction/evaluator/selection evidence as applicable |
| `publish_adoption` | source experience → current artifact edge | matching source contract plus static and runtime actuation (L2/L3) |
| `publish_causal` | causal source experience → current artifact edge | prior legal Adoption plus counterfactual actuation (L4) |

`promote` is legacy and must not appear in new production writeback.
`canonical_operation()` maps only the legacy `distill` alias; it does not
reinterpret legacy `promote` as a legal result writeback.

### Distillation operations

- `distill_diagnostic`
- `distill_candidate`
- `distill_positive_result`
- `distill_positive_adopted`

`distill_positive` and `distill` are read-compatibility values. New callers
must select Result or Adopted explicitly. Causal wording additionally requires
L4.

## 5. Protocols

`ProtocolSpec` owns task profile, data split, preprocessing, evaluator,
metric, selection, seed, holdout, promotion, and compatibility policies.
`ProtocolRef` is `protocol_id@version#canonical_hash`.

The `ProtocolRegistry` resolves immutable refs.
`ProtocolCompiler.compile(claim, request)` returns
`EvidenceObligations`:

- required receipt types and minimum counts;
- required payload flags / distinct payload values;
- protocol-compatibility and clean-ancestry requirements;
- positive-effect requirement where applicable;
- trusted-receipt requirement.

Claim/operation compatibility is explicit. A score cannot become a Debug repair,
an audit finding cannot be ranked as a score, and a performance contrast cannot
stand in for an experience-causality edge.

## 6. Receipts and trust

`ReceiptType` values:

- execution/protocol: `code_execution`, `split_lineage`, `fit_scope`,
  `prediction_scope`, `evaluator`, `selection_freeze`;
- replication/identity: `seed_aggregation`, `replication`,
  `method_identity`, `derivation`;
- actuation: `static_actuation`, `runtime_actuation`,
  `adoption_publication`, `counterfactual_actuation`.

A `Receipt` binds artifact, run, protocol hash, collector identity/version,
payload hash, timestamp, hash-chain fields, trust status, and supported/blocked
Claim types. Agent prose is not a trusted Receipt. Trusted collector ingestion
validates artifact, protocol, collector capability, payload, event chain, and
cross-artifact isolation.

Receipt types are not interchangeable:

- `code_execution` proves the current code ran;
- static/runtime actuation proves an injected experience appeared and executed;
- counterfactual actuation proves the experience changed action/code under the
  bound comparison.

## 7. Authority request and decision

`AuthorityRequest` requires artifact ID, Claim ID, Operation, active
`ProtocolRef`, `TaskContext`, requesting component, and both canonical stage
axes (or one deterministic compatibility mapping).

`AuthorityEngine.evaluate(request)` returns `AuthorityDecision` with:

- `outcome`: allow, allow-with-warning, quarantine, deny, require-replay, or
  require-human-review;
- `permitted_scope`;
- satisfied paths;
- missing obligations;
- blocking receipt IDs;
- required action;
- policy and request identity fields.

`allowed` is true only for allow / allow-with-warning. High-risk failures
quarantine or deny; protocol mismatch requires method-preserving replay.

## 8. SOP visibility API

Core types: `SOPClauseV1`, `VisibilityRequest`, and `VisibleSOPPack`.

A clause declares Claim/source refs, task/domain/transfer scope, protocol scope,
permitted operations, generation/governance stages, publication class,
Authority decisions, Receipts, derivations, and optional Experience Contract.

`SOPVisibilityGateway` supports `off`, `shadow`, and `enforce`.
Enforce mode calls `authorize_clause_for_visibility()` before retrieval
candidates are embedded or ranked. The returned pack separates positive,
diagnostic, warning, and suppressed clauses and carries a complete visibility
trace. Empty authorized packs remain empty/abstain. Attached SOP IDs, Tree
projection, navigation edges, or cache entries cannot bypass clause decisions.

## 9. Runtime adapter and GlobalMemory

`MLEvolveAuthorityAdapter` is the production integration surface. It binds
node/transition Claims, host Receipts, visibility, rank/select gates, writeback,
distillation, and ledger records to the active protocol/policy/Bundle snapshot.

`GlobalMemoryLayer` consumes only the visible, hash-bound snapshot and checks
outcome, scope, policy, protocol, task and stage before use. Legacy fallback is
available only under explicit compatibility configuration; unknown high-risk
state fails closed.

## 10. Result, Adoption, and Causal persistence

The Session Overlay is append-only and hash-chained. Typed events are:

- Result Fact — current node is independently legal; `derived_from_refs=[]`
  when no experience was adopted;
- Adoption Edge — source Claim/experience was statically and dynamically
  actuated in the current artifact;
- Causal Edge — a legal Adoption is additionally supported by the bound
  counterfactual.

Runtime methods are idempotent and content-addressed. An exposure record,
retrieval hit, report, or score difference never creates an edge.

## 11. Fixed-holdout terminal API

`mlevolve.fixed_holdout.writeback.finalize_result_writeback(...)` is the sole
positive finalizer after sealed terminal scoring. Search and result parsing
write zero positive memory before the score is sealed. The finalizer either
writes exactly one idempotent Result Fact or
`record_terminal_writeback_failure(...)` writes an explicit incomplete status.
Tampering cannot leave a partial success object.

## 12. Bundle and snapshot APIs

`MemorySnapshotLoader` verifies `CURRENT.json`, the immutable Base manifest,
and all declared files. `SessionOverlay` appends fsynced events under a lock.
`SleepTimePublisher`:

1. freezes the Overlay;
2. executes ordered audit/decomposition/distillation/derivation/visibility/
   validation stages;
3. materializes a new non-loadable staging root;
4. verifies every hash and expected parent;
5. atomically swaps `CURRENT.json`.

A pre-swap crash or validator failure leaves CURRENT unchanged. Published
parents and failed attempts are never overwritten.

## 13. Clean Replay API

`ReplayCandidate`, `ReplayQueue`, `ReplayReceiptIngestor`, and
`ReplayAuthorityRecovery` implement deterministic replay selection and
recovery. The verifier freezes model constructors, features, objective/loss,
search space, compute budget, and inference/ensemble behavior.

- method preserved → new Claim and new trusted support path;
- protected method changed → Successor Claim;
- unclassified change → human review;
- historical Claim/score → never mutated or resurrected.

Certified publication creates a new immutable child Bundle and exposes only the
new replay Claim whose semantic-purity and visibility checks pass.

## 14. Stable schemas

Important emitted schema names include:

- `stage_hybrid_memory_pack_v1`
- `layered_strategy_memory_pack_v1`
- `bundle_publication_event_v1`
- `sleep_time_publication_report_v1`
- `clean_replay_queue_entry_v1`
- `clean_replay_queue_manifest_v1`
- `clean_replay_registration_v1`
- `positive_writeback_distillation_plan_v1`
- `positive_writeback_materialization_v1`
- `fixed_holdout_terminal_writeback_status_v1`
- `fixed_holdout_result_fact_v1`
- `decision_admissibility_wp8_tier2_source_snapshot_v2`

Every formal JSON object uses a canonical payload hash excluding only its own
hash field. Manifests bind byte SHA-256 values separately from logical payload
hashes.

## 15. Claim-authority boundary

Current formal status:

- engineering completion: authorized;
- Full training superiority: rejected;
- four available-pair positive deltas: diagnostic only;
- experience causality for those deltas: pending without L4.

Consumers must preserve this boundary in papers, prompts, Bundle metadata, and
downstream reports.

