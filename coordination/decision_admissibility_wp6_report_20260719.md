# Decision Admissibility WP6 Stop-Gate Report

Date: 2026-07-19  
Branch: `codex/dual-time-procedural-memory`  
Remote-verified baseline and current HEAD: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP6 — Clean Replay 与 certified-memory  
Status: **PASSED**

## Authoritative boundary

- WP0–WP6 changes remain uncommitted and unstaged. HEAD and upstream remain
  the remote-verified baseline above.
- No existing Claim, audit failure, journal, WP4 raw-audited bundle, graph,
  index, run artifact, or unrelated user-untracked asset was overwritten.
- No Kubernetes Pod/Job or real online replay experiment was started in WP6.
  The production candidate extractor is ready to run against the immutable WP4
  full bundle on the PVC, but this report does not mislabel a synthetic replay
  as real-corpus evidence. Real replay queue/receipts and outcome evidence are
  generated in the later authorized experiment stage.

## Deterministic replay queue

`mlevolve/authority/clean_replay.py` implements a hash-bound queue policy:

- every candidate must bind task, source run, parent node, child node, code
  SHA-256, a method hypothesis, method family, and static-audit disposition;
- audit-unavailable, method-fatal leakage, incomplete lineage/code, invalid
  hashes, and missing hypotheses fail closed;
- selection is capped at three per task and prefers distinct method families;
- historical metric improvement is used only for deterministic queue priority,
  never emitted as evidence;
- stable candidate ordering and IDs make input-order changes irrelevant;
- `replay_queue.jsonl` and its manifest bind both canonical entry hashes and
  the exact queue-file byte hash.

Two CLIs are provided:

```text
paper-skills/memory_bundle/build_replay_queue.py
paper-skills/memory_bundle/extract_replay_candidates.py
```

The bundle extractor verifies every Base artifact, resolves RunForest
source/parent/child/transition records back to copied raw journals, rehashes the
exact node code, binds method/debug hypotheses, classifies protocol-repairable
versus method-fatal audit findings, and invokes the same queue policy. A
deterministic bundle fixture proves the selected queue binds the exact raw code
and lineage and marks historical metrics as non-evidence.

## MethodFingerprint and protocol-only verifier

`mlevolve/authority/replay_certifier.py` replaces the earlier constructor-only
comparison with a hash-bound protection surface covering:

- model families, constructors, signatures, and hyperparameters;
- feature constructors and feature logic;
- loss/objective;
- hyperparameter search space;
- training/compute budget;
- inference and ensemble family/weights;
- residual method logic outside the declared protocol surface.

Each replay is classified as exactly one of:

```text
METHOD_PRESERVED
SUCCESSOR_METHOD
REQUIRE_HUMAN_REVIEW
```

A declared protocol surface cannot hide model, feature, loss, search-space,
budget, inference, ensemble, or residual-method changes. Unknown calls and
syntax failures require human review. Fold/split structure, fold-local
fit/transform scope, evaluator wiring, selection freeze, seed aggregation,
holdout access, and protocol instrumentation are allowed only when the active
ProtocolSpec version declares them.

The existing `mlevolve-default@1` hash and behavior remain unchanged; it has no
Clean Replay repair surface. New immutable
`mlevolve-default@2#4e54d9e6e3c44af8d92f578ef25b4be489b602e62ccc2ac88fa2113768f7eff2`
declares the explicit replay-only repair surface and names v1 as its parent.
`verify_clean_replay.py` resolves this surface from the registry and emits a
hash-bound report. Recovery rejects a report whose ProtocolRef or surface hash
does not exactly match the registered ProtocolSpec.

During broad regression, the legacy production repair gate exposed two old
surface names (`prediction_scope`, `final_holdout`) and a valid repair pattern
(remove cross-partition concatenate, fit on train, transform holdouts). They
were mapped narrowly to the canonical holdout/preprocessing scopes. The
verifier still treats transformer constructor/settings and feature logic as
protected while allowing only fit/transform data scope to change.

## New Claim and trusted Receipt transaction

`ReplayReceiptIngestor` validates before any graph mutation:

- trusted-host status and stable Receipt ID;
- payload hash and host event hash;
- exact replay artifact binding;
- exact active protocol hash;
- MethodIdentity fingerprint and replay code hash from the verifier report;
- duplicate Receipt rejection.

Historical/source Receipts cannot be reused for the replay artifact.

`ReplayAuthorityRecovery` then performs a new-Claim transaction:

- a method-preserved replay creates a new replay Claim and independent clean
  support path;
- a changed protected surface creates a distinct Successor Claim/path;
- an unclassified replay creates no Claim, Receipt, or path;
- provenance records the predecessor Claim/protocol without making the old
  restricted Claim an Authority parent;
- the original Claim object, protocol, paths, audit outcome, and authority are
  unchanged;
- no registration ever adds a clean path to the old Claim.

The production Authority adapter exposes the same host-owned registration API
and records the immutable transaction in its ledger.

## Certified-memory publication and runtime loading

`mlevolve/authority/certified_bundle.py` uses the WP5 crash-safe publisher to
create a new replay-scoped child bundle. Publication requires:

- a raw-audited or already-certified immutable parent;
- the predecessor Claim to exist in that exact parent;
- a new replay/successor Claim, trusted replay Receipts, verifier report, and
  EvidencePath whose hashes agree;
- the exact ProtocolSpec version to be published and the child
  `protocol_registry_hash` to be recomputed;
- zero old-Claim mutation and zero blanket clause upgrade.

The child appends replay Claims, trusted Receipts, paths, verifier reports,
registrations, a certification report, and complete artifact/checksum hashes.
It leaves RunForest/SOP semantic artifacts byte-identical unless a separate
validated sleep-time distillation stage supplies new material. Existing and
unreplayed score Claims are copied unchanged and receive no new path.

`mlevolve/authority/bundle_authority.py` loads hash-verified Base Claims,
Receipts, paths, and decisions into the live Authority Engine. It verifies
ProtocolRefs, trusted Receipt IDs/event hashes, missing path Receipts, and
record immutability. An end-to-end test publishes a certified child, reloads it
through `CURRENT.json`, authorizes the new clean replay score, and still denies
both the predecessor and an unrelated unreplayed historical score.

## Required WP6 tests

All four plan-mandated modules exist:

```text
tests/authority/test_method_preserving_replay.py
tests/authority/test_method_changing_fake_replay.py
tests/authority/test_replay_successor_claim.py
tests/authority/test_replay_authority_recovery.py
```

Focused result:

```text
17 passed in 0.32s
```

The suite directly covers deterministic queue selection, real-bundle-shaped
candidate extraction, versioned verifier CLI use, method preservation,
model/feature/loss/search/budget/ensemble changes, unknown-call review,
Successor Claim isolation, cross-artifact Receipt rejection, v1→v2 recovery,
new-Claim Rank/Promote, old/unreplayed denial, certified publication, Base
immutability, and runtime authority reload.

## Regression and health checks

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority \
  tests/test_stage_aware_hybrid_memory.py \
  tests/test_memory_snapshot_overlay.py \
  tests/test_sleep_time_bundle_publication.py \
  tests/test_bundle_publication_crash_safety.py \
  tests/test_run_identity.py
# 175 passed in 15.40s

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests --ignore=tests/test_composite_memory_benchmark.py
# 433 passed in 124.56s

PYTHONPATH=mlevolve .venv/bin/python -m compileall -q \
  mlevolve paper-skills tests

git diff --check
# passed
```

The independently tracked frozen-lock inconsistency remains unchanged:
`tests/test_composite_memory_benchmark.py` is `18 passed, 1 failed` because its
frozen detector hash predates the baseline. The current detector and baseline
commit detector are both exactly
`8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`;
the lock was not rewritten to manufacture a pass.

## Stop-gate decision

- [x] Method-preserved replay creates only a new replay Claim and a new complete
      trusted support path; it does not edit the historical Claim.
- [x] Method-changing fake replay creates a Successor Claim/path, while the old
      Claim remains pathless and denied.
- [x] Human-review and cross-artifact/untrusted evidence create no recovery.
- [x] Unreplayed historical scores remain unable to Rank or Promote before and
      after certified child-bundle publication/reload.
- [x] Protocol repair is versioned and bound to an immutable ProtocolSpec
      surface; v1 is not silently widened.
- [x] Parent Base, old Claims, SOP/RunForest artifacts, and user assets remain
      unchanged.
- [x] Required tests, production repair compatibility, broad regression,
      compile, and diff checks pass.

**WP6 Stop Gate: PASSED. WP7 may begin.**
