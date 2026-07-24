# Decision Admissibility WP7 Local Preflight Report

Date: 2026-07-19  
Branch: `codex/dual-time-procedural-memory`  
Remote-verified baseline and current HEAD: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP7 — Shadow → Canary → Enforce  
Status: **LOCAL PREFLIGHT PASSED; WP7 STOP GATE PENDING AUTHORIZED ONLINE EVIDENCE**

## Authoritative boundary

- WP0–WP7 changes remain uncommitted and unstaged. HEAD and upstream remain
  the remote-verified baseline above.
- No existing run, Base Bundle, Session Overlay, corpus, graph, index, report,
  paper artifact, or unrelated user-untracked asset was overwritten or removed.
- No Kubernetes Pod/Job, paid online MLE run, global enforce run, commit, or push
  was performed in WP7. The plan explicitly requires separate authorization
  before a new Kubernetes Job and forbids advancing to WP8 before the WP7 gate.
- The three formal WP4 Bundle pins remain external/PVC evidence. The local
  final manifests identify, among others, task-heldout Bundle
  `mlevolve-be034ec-nonspooky-task-heldout-v1` with manifest SHA-256
  `8a8abd5146be6ba81a9b334140f88632953f7cae2874ef82e18cd713f9ba31c8`.

## Frozen rollout version set

`mlevolve/authority/rollout.py` introduces a hash-bound version set containing:

- rollout ID;
- Authority policy version;
- immutable ProtocolRef including canonical hash;
- trusted collector version;
- Base Bundle ID and manifest SHA-256, or explicit bundle-free `none`.

The MLEvolve adapter binds the actual hash-verified `CURRENT.json` snapshot
before decisions, verifies configured Bundle pins, and freezes the full tuple
before the first authority or visibility decision. A required-bundle canary
refuses to start if no verified Base is loaded. Policy, protocol, collector, or
Bundle changes after freezing fail closed.

## Shadow comparison and disagreement review

Every adapter gate in shadow/enforce records exactly one immutable comparison
per decision ID:

```text
legacy_allowed
authority_allowed
effective_allowed
enforced
operation + generation/governance stage
taxonomy + reason class
missing obligations + blocking receipts
frozen rollout version hash
```

The report distinguishes agreement, `legacy_allow_authority_deny`,
`legacy_deny_authority_allow`, protocol/scope/evidence failures, contradictory
Receipts, and internal-error classes. Repeated helper probes cannot append or
rewrite the original comparison.

Operational review support is provided by:

```text
paper-skills/memory_bundle/build_shadow_review_packet.py
paper-skills/memory_bundle/verify_shadow_review_packet.py
```

They verify the Authority ledger, deterministically stratify disagreement and
internal-error records, bind the sampled evidence by hash, leave reviewer and
disposition fields blank, and verify completed independent review metadata
against the original ledger. The implementation never self-signs a human
review.

## Staged enforcement and fail-safe behavior

`off`, `shadow`, and `enforce` remain selectable. Enforce can be restricted by
operation, GenerationStage, and GovernanceStage. The production canary profile
is staged to Draft/Improve/Debug and refuses an unbound Bundle; the main config
remains shadow and does not silently enable global enforce.

The staged predicate is used consistently by:

- MLEvolve node ranking/selection/promotion/replay gates;
- GlobalMemory retrieval, including complete outcome/scope/policy/protocol and
  two-axis stage validation;
- SOP clause visibility and Dynamic Hybrid before embedding, geometry, RRF,
  token packing, or Prompt rendering.

Visibility traces now bind legacy-visible, full-policy-visible, effective,
retained, suppressed, embedding-candidate, and RRF-eligible clause IDs plus
agreement/disagreement counts. Precompiled Base masks are applied only for an
actually enforced request, so shadow and out-of-window canary requests preserve
legacy behavior. Unknown stage metadata cannot be used to evade an active
GlobalMemory enforcement window.

Internal failures have explicit semantics:

- high-risk operations return DENY/fail-closed regardless of legacy allow;
- Inspect/Debug may return `ALLOW_WITH_WARNING` only as navigation, with no
  permitted scope and an explicit abstain/no-mutation action;
- other low-risk failures return warning + DENY/abstain rather than escaping to
  the Agent or silently allowing action.

## Canary and rollback operations

`evaluate_authority_canary.py` reloads and verifies the hash-chained ledger,
accepts only independently authored boolean oracle labels, evaluates only
records that were actually enforced, and reports:

- legacy, policy, and effective IIR;
- policy and effective VKR;
- unauthorized effective allows;
- effective false-denial count/rate;
- excluded out-of-window shadow decisions;
- frozen rollout and decision-record hashes.

The kill gate uses actual `effective_allowed` actions, requires the configured
minimum decision count, defaults to zero unauthorized allows, and exits nonzero
on failure.

`rollback_memory_bundle.py` verifies the ledger and target immutable Bundle,
compare-and-swaps `CURRENT.json` using the expected current manifest SHA-256,
and appends prepared/committed rollback events. It preserves both Bundle
directories and the complete ledger.

## New WP7-focused tests

```text
tests/authority/test_shadow_rollout.py
tests/authority/test_enforce_rollout.py
tests/authority/test_canary_gate.py
tests/authority/test_bundle_rollback.py
```

Focused WP7 result:

```text
17 passed in 0.54s
```

Coverage includes version freeze/mismatch, required Bundle binding, shadow
legacy parity, both decision traces, visibility comparison IDs, deterministic
review packets, staged node/GlobalMemory/visibility enforcement, high/low-risk
internal errors, canary pass/kill conditions, verified offline ledger loading,
CLI execution, compare-and-swap rollback, and retention of both old/new Bundles
and ledger events.

## Regression and health evidence

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q tests/authority
# 123 passed in 1.80s

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/test_memory_snapshot_overlay.py \
  tests/test_run_identity.py \
  tests/test_stage_aware_hybrid_memory.py \
  tests/authority/test_global_memory_authority_scope.py \
  tests/authority/test_legacy_sop_visibility.py \
  tests/authority/test_visibility_projection_bypass.py
# 86 passed in 15.49s

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests --ignore=tests/test_composite_memory_benchmark.py
# 451 passed in 125.42s

PYTHONPATH=mlevolve .venv/bin/python -m compileall -q \
  mlevolve paper-skills tests

git diff --check
# passed
```

All four WP7 operational CLIs also pass import/`--help` smoke checks. The
independently tracked frozen-lock inconsistency remains exactly unchanged:
`tests/test_composite_memory_benchmark.py` is `18 passed, 1 failed`; its lock
contains detector hash `ae30aac332b6f62dccc784955ec2952268d8a1381085bcadb4d683b9d8f6a221`,
whereas both current and baseline `mlevolve/agents/leakage_audit.py` are
`8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`.
The lock was not edited to manufacture a pass.

## Stop-gate audit

- [x] Policy/protocol/collector/Bundle versions are hash-bound and frozen.
- [x] Shadow records legacy, full-policy, effective, and enforcement outcomes.
- [x] Disagreement taxonomy and independent-review packet/verifier exist.
- [x] Synthetic high-risk enforce and Draft/Improve/Debug visibility enforce
      pass deterministic tests without pre-ranking or Prompt bypass.
- [x] High-risk internal errors fail closed; low-risk errors warn and
      navigate-only or abstain.
- [x] `off/shadow/enforce` profiles and atomic non-destructive Bundle rollback
      are verified.
- [x] Authority, memory integration, broad regression, compile, CLI smoke, and
      diff checks pass.
- [ ] A real shadow run has produced a new rollout ledger and that disagreement
      sample has been independently reviewed. No such online run was authorized.
- [ ] A low-cost online task canary has passed the independently labeled
      effective-IIR/VKR kill gate while pinned to a formal Bundle.
- [ ] The subsequent global main experiment has been authorized and completed.

**WP7 Stop Gate: NOT YET PASSED. WP8 MUST NOT BEGIN.**

The next permitted action requires explicit authorization for a new online
shadow/canary Job. After a shadow ledger exists, an independent reviewer must
complete and verify the generated packet before staged canary enforcement is
submitted. A failed canary must trigger rollback/fix rather than progression.
