# Decision Admissibility WP5 Stop-Gate Report

Date: 2026-07-19  
Branch: `codex/dual-time-procedural-memory`  
Remote-verified baseline and current HEAD: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP5 — Actuation 与 Base/Overlay 写回  
Status: **PASSED**

## Authoritative boundary

- WP0–WP5 changes remain uncommitted and unstaged. HEAD and upstream both remain
  the remote-verified baseline above.
- The dirty worktree contains 30 tracked modified paths and 3,159 untracked
  paths at this gate. Unrelated user assets were not deleted, moved, staged, or
  overwritten.
- No Kubernetes Pod/Job, experiment run, commit, push, PR, credential access,
  or paper headline change was performed during WP5.
- The WP4 verified corpus/bundle evidence remains rooted at
  `coordination/decision_admissibility_wp4_real_work_20260719_r2_verified` and
  the persistent PVC path `/output/corrected-r2`; WP5 does not rewrite it.

## Implementation completed

### ExperienceContract and L0–L5 actuation

- `mlevolve/authority/actuation.py` implements deterministic, hash-bound
  `ExperienceContract` compilation with host-observable preconditions,
  preservation constraints, required changes, forbidden dependencies, and
  runtime observations.
- Contract IDs bind the admitted clause, Claim/source refs, active protocol,
  task scope, operation, both stage axes, policy version, and compiler version.
- `ActuationTracker` enforces sequential evidence: actual prompt exposure (L0),
  claimed adoption (L1), trusted static conformance (L2), trusted runtime path
  execution (L3), paired influence (L4), and protocol-legal efficacy (L5).
- Prose cannot self-satisfy a Contract. L2/L3 receipts are emitted only from
  host observations that match every required predicate, and counterfactual
  receipts require an already verified runtime path.
- `mlevolve/authority/paired_replay.py` deep-copies a frozen control context,
  runs memory-off then memory-on, separates action/code influence from outcome
  efficacy, normalizes maximize/minimize deltas, and rejects protocol-illegal
  improvements as L5 evidence.

### Production actuation and writeback boundary

- `SOPVisibilityGateway` compiles contracts only for the effective visible
  clause projection. Contract compilation failure cannot advance a clause
  beyond exposure/navigation.
- Prompt exposure is recorded only after the corresponding SOP/clause ref was
  actually injected; mere retrieval candidates do not count as L0.
- Authority promotion paths bind static/runtime actuation Receipts. Exposure or
  a claimed adoption without L3 remains ineligible for positive promotion.
- Positive SOP, PROMOTE, Code Seed, causal, and effective claims use the
  plan-defined minimum evidence levels: ordinary positive writeback can use L3,
  causal attribution requires L4, and effective-repair attribution requires L5.
- Session Overlay append is permitted only after an Authority `ALLOW` decision
  and an L3-or-higher actuation report. The event binds protocol/policy,
  decision refs, report refs/hashes, Claim types, scope, and audit disposition.

## Base Bundle and Session Overlay invariants

- `MemorySnapshotLoader` resolves `CURRENT.json`, verifies its pointer hash and
  binding to bundle ID/version/manifest, rejects unsafe or staging paths, and
  verifies every declared Base artifact hash.
- Base reads are manifest-declared and hash-checked. A loaded Base detects any
  manifest or artifact mutation, exposes no write API, and is rechecked before
  publication/writeback boundaries.
- Session Overlay events are append-only, sequence-checked, individually
  hashed, parent-hash chained, fsynced, and bound by an atomically replaced
  overlay manifest. Tampering fails closed on reload.
- An unaudited score is Inspect-only even when `claim_type` exists only inside
  a nested clause and an online evaluator attempts to allow Rank.
- Base clauses use the WP4 precompiled key
  `protocol|operation|generation_stage|governance_stage` before gateway
  evaluation, embeddings, RRF, or prompt materialization. Overlay clauses are
  separately evaluated by the current Authority Engine online.
- A direct ranking-path test proves a clause excluded by the precompiled mask
  is absent from embedding and RRF candidate traces and that its forbidden text
  cannot influence the rendered/ranked SOP projection.

## Runtime configuration binding

- `AgentSearch` now loads the immutable Base selected by `CURRENT.json`, passes
  that Base's exact `runforest/graph.json` and `runforest/index.npz` paths to the
  production memory layer, configures the Authority adapter with the same
  snapshot, and writes only to a run-scoped Session Overlay.
- A WP5 audit caught and fixed a split-brain wiring defect where the Base paths
  were resolved but the old configured graph path was still passed to the
  memory-layer constructor.
- Run identity now binds bundle-backed runs to the verified Base manifest hash,
  Base bundle version, exact index hash, and source-run inventory rather than a
  stale standalone graph configuration.

## Sleep-time publication and crash safety

- `mlevolve/authority/bundle_publisher.py` and
  `paper-skills/memory_bundle/publish_sleep_time_bundle.py` implement the
  explicit ordered pipeline: frozen overlay, audit, Claim decomposition,
  distillation, candidate build, derivation validation, visibility validation,
  bundle validation, immutable publish, and atomic `CURRENT.json` swap.
- A file lock plus expected-parent manifest compare-and-swap serializes
  concurrent publishers. Exactly one of two same-parent publications commits.
- Staging, failed, and input directories are not runtime-loadable bundles.
- Validator failure and a simulated crash before the pointer swap leave the
  previous `CURRENT.json` byte-for-byte unchanged. The old Base remains
  available for rollback and is not mutated by a successful publication.
- Publication reports bind parent bundle/manifest, overlay manifest/events,
  candidate bundle manifest, pipeline reports, final pointer hash, and a report
  hash.
- Publication ledger events are append-only, fsynced, sequence-checked, and
  hash-chained. Ledger integrity is now checked before freezing/building a new
  candidate, so ledger corruption fails before producing an orphan bundle or
  changing `CURRENT.json`.

## Required WP5 tests

All six plan-mandated modules exist and pass:

```text
tests/authority/test_experience_contract.py
tests/authority/test_actuation_pipeline.py
tests/authority/test_counterfactual_actuation.py
tests/test_memory_snapshot_overlay.py
tests/test_sleep_time_bundle_publication.py
tests/test_bundle_publication_crash_safety.py
```

Focused result:

```text
28 passed in 1.62s
```

Additional integration coverage includes Base-backed AgentSearch path binding,
Base-backed run identity, nested score handling, precompiled mask ordering,
publication ledger chaining/tamper failure, concurrent compare-and-swap, and
post-load Base mutation detection.

## Regression and health checks

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority \
  tests/test_stage_aware_hybrid_memory.py \
  tests/test_memory_snapshot_overlay.py \
  tests/test_sleep_time_bundle_publication.py \
  tests/test_bundle_publication_crash_safety.py \
  tests/test_run_identity.py
# 160 passed in 15.24s

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests --ignore=tests/test_composite_memory_benchmark.py
# 416 passed in 124.59s

PYTHONPATH=mlevolve .venv/bin/python -m compileall -q \
  mlevolve paper-skills tests

git diff --check
# passed
```

### Preserved frozen-lock baseline failure

The independently run `tests/test_composite_memory_benchmark.py` remains
`18 passed, 1 failed`. The only failure is the pre-existing frozen-lock
inconsistency: the lock stores detector SHA-256
`ae30aac332b6f62dccc784955ec2952268d8a1381085bcadb4d683b9d8f6a221`, while
`mlevolve/agents/leakage_audit.py` is
`8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`.

The current detector hash and the detector at baseline commit `b47dab63` are
both exactly `8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`;
`git diff --exit-code` confirms no detector change. The frozen lock was not
rewritten to manufacture a pass.

## Stop-gate decision

- [x] Deterministic reports can represent and distinguish every L0–L5 level.
- [x] Unexposed, unclaimed, static-only, and otherwise unadopted experience
      cannot satisfy L3 promotion obligations or enter positive writeback.
- [x] Influence (L4) and protocol-legal efficacy (L5) are separate paired
      counterfactual findings.
- [x] Publication validation failure, ledger corruption, and a crash before
      pointer swap do not change `CURRENT.json`.
- [x] Published Base Bundles are hash-verified and immutable; Session Overlay
      is append-only, hash-chained, and tamper-evident.
- [x] Base precompiled masks and online Overlay Authority evaluation occur
      before ranking/RRF/prompt use.
- [x] AgentSearch and run identity consume the same Base selected by CURRENT.
- [x] Required, integration, broad regression, compile, and diff checks pass;
      the frozen-lock baseline discrepancy remains explicitly isolated.

**WP5 Stop Gate: PASSED. WP6 may begin.**
