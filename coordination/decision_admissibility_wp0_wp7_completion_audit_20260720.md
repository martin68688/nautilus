# Decision Admissibility WP0–WP7 Completion Audit

Date: 2026-07-20  
Plan: `coordination/decision_admissibility_complete_execution_plan_20260719.md`  
Plan SHA-256: `31bd093c62b92eef94df94478cbdd4d9587910fcdf5ae85729aed87cd0971d7f`  
Branch: `codex/dual-time-procedural-memory`  
Baseline/HEAD/upstream/live remote: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`

## Audit decision

- WP0–WP6 engineering Stop Gates remain **PASSED**.
- WP7 steps 1–6 and the real r14 shadow-disagreement review are evidenced.
- WP7 steps 7–8 (low-cost online canary and global main experiment) are not
  evidenced. Therefore the **WP7 Stop Gate is NOT PASSED**.
- WP8 has not begun and must not begin before WP7 passes.
- The full plan Goal remains active. This document does not redefine success
  around the completed subset.

The current Goal explicitly forbids starting enforce/canary and permits only
dev Pods, not Jobs. This audit therefore stops before the two missing WP7
online stages.

## Repository and asset boundary

- `git rev-parse HEAD`, `@{u}`, and a live `git ls-remote` all resolve to
  `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`.
- The pre-dossier audit boundary had 35 tracked modified paths, 3,239
  untracked files after adding the audit-Pod manifest, and zero staged bytes.
- No commit, push, PR, Job, enforce run, paper-headline change, or credential
  read/write was performed by this audit.
- The only cluster resource created by this audit was the CPU-only Pod
  `decision-admissibility-completion-audit-ro-cpu-r1`. It requested no GPU,
  mounted `haoming-storage` read-only, and was deleted after verification.
- The reproducible Pod manifest is
  `deploy/pod-decision-admissibility-completion-audit-ro-cpu-r1.yaml`.
- During the exact baseline test, external/concurrent work created or modified
  WP7 canary-related untracked assets and later a Pod named
  `decision-admissibility-wp7-canary-a100x1-r4` was observed. This task did
  not create, exec, stop, inspect results from, or otherwise use that Pod. The
  concurrent files are treated as user assets and excluded from this task's
  mutation claims.

After that concurrent change settled, the full regression had identical
pre/post fingerprints:

| Fingerprint | Before | After |
|---|---|---|
| Git status SHA-256 | `6ca3a94630e7134a33884579b2e0590416e91a60402c3576d041f737645b5455` | same |
| Tracked diff SHA-256 | `0fd39d53403fe055d0d6b4c6238ca091fd1c675bd9c176c765dcda77324174f4` | same |
| Untracked stat SHA-256 | `9a7dd6aa99d3ee40e13dfb7d9d496547d5e6d3927564bff66a1db46d287dd1e5` | same |
| Staged diff SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | same |

## Current test evidence

Exact plan §20.1 command, with bytecode and pytest cache writes disabled:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mlevolve \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/authority \
  tests/test_stage_aware_hybrid_memory.py \
  tests/test_causal_granularity_benchmark_v2.py \
  tests/test_protocol_repair.py \
  tests/test_run_forest_memory.py

349 passed in 98.94s
```

Complete current regression, excluding only the independently frozen
pre-baseline-lock module documented since WP3:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mlevolve \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests --ignore=tests/test_composite_memory_benchmark.py

485 passed in 158.63s
```

`git diff --check` passed. All 32 plan-named WP1–WP7 test modules are
present; the path/content manifest SHA-256 is
`36736d69f7ba65df2df4859cd211527b66c5c1eaa2f5e9e828545014f1e0a889`.

The excluded module is not silently called green. Its frozen lock still binds
detector SHA-256
`ae30aac332b6f62dccc784955ec2952268d8a1381085bcadb4d683b9d8f6a221`,
while both baseline and current `mlevolve/agents/leakage_audit.py` are
`8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`.
Prior WP3–WP7 reports record the resulting `18 passed, 1 failed` module
result. The frozen lock was not rewritten to manufacture a pass.

## Gate-by-gate evidence

| WP | Evidence | Current determination |
|---|---|---|
| WP0 | Baseline manifest SHA-256 `e677ab753bc5919a5af27cf1341a1645daefeb5d14bfa78c51cecdf7890b3741`; plan hash matches; local/upstream/live remote all match `b47dab63`; 215-test detached baseline recorded and the expanded current §20.1 suite passes 349 tests | PASSED |
| WP1 | Report SHA-256 `ed9d3b86aa5e0997f1bfff25e79bc181caf03a7b6339a3b7800d35bc12770292`; orthogonal stages, Claim/Operation extensions, GlobalMemory scope checks, and high-risk fail-closed tests included in the 485-test run | PASSED |
| WP2 | Report SHA-256 `6659cbb12de169cc16d75f5ea5e06385fcb4b225464ea64b9169d0f640435ad3`; mixed Claim decomposition and host-only Receipt trust boundary tests included | PASSED |
| WP3 | Report SHA-256 `129593e47cc502567a53abb70b3070e9ad81427f573407aefd3c655aceb9013c`; pre-ranking/prompt visibility, projection/cache bypass, 281 legacy SOP, and 2,773 quarantine-edge evidence recorded | PASSED |
| WP4 | Report/sidecar SHA-256 `9b437189b38bc555905bae24e2993cb91527818c7e201a8ef8ace9420a45c8f0`; current PVC ledger reverified below | PASSED |
| WP5 | Report/sidecar SHA-256 `1fa13840ef1968450df6862c0b61c2922c015bfd8aac94044a59c9ae88dd088b`; L0–L5, Base/Overlay, crash-safe CURRENT, and unadopted-promotion tests included | PASSED |
| WP6 | Report/sidecar SHA-256 `35cba2869753762f21a6303294dc149032874361cae50d021a25e0bc1f379e3e`; method-preserved/new Claim, fake replay/Successor Claim, old-score denial, and certified-child tests included | PASSED engineering gate; real-corpus replay outcomes remain a later experiment deliverable |
| WP7 | Preflight report/sidecar SHA-256 `cfbf753463b31b4b319abd8fefddcd01cd23c263fed6618bfcd39ac410d0ec28`; real r14 evidence below completes the previously missing shadow-review item | NOT PASSED: canary and global main experiment remain missing |
| WP8 | Plan requires WP7 to pass first; `tests/test_multigeneration_contamination.py` and `tests/test_decision_admissibility_factorial.py` are currently absent | NOT STARTED by required order |

## WP4 current PVC re-verification

Formal persistent root as seen from the PVC root:

```text
/workspace/decision-admissibility-wp4-20260719-r1/corrected-r2
```

- `WP4_FINAL_SHA256SUMS` SHA-256:
  `3da97def940498353bc48c2846755d61be2292ba5badb484c2ab293168c23ed4`.
- Ledger entries independently rehashed: 7,488 / 7,488.
- Bytes read: 424,548,919.
- Missing files: 0.
- Hash mismatches: 0.
- The verification Pod mounted the PVC read-only and was deleted after the
  check.

This verifies the current persistent files, not merely the historical WP4
report.

## WP7 r14 shadow-review binding

Authoritative root:

```text
/workspace/decision-admissibility-wp7-shadow-aerial-r14
```

Immutable anchors:

| Artifact | SHA-256 / hash |
|---|---|
| Frozen source aggregate (267 files) | `48946d4210e012c23de7b7c0e6ae850cdaa6a79cc8450aab69414fa5be646b10` |
| Same-domain task-heldout Bundle manifest (531 artifacts) | `4b825b3e00005ac64f95e94d0034cf0f6ab07798ad8755ac7139ca46973771de` |
| Authority ledger (2,052 events) | `6fe99309c6c5b8bec593ae7946e183fd75079cc9a4c9020d7ef7d94be9f02b91` |
| Journal | `0886f2edae024d5f87056c7515e17eb2320d020ee3e22033cb36ec881833e751` |
| Original review packet | `68ae05ca9cf2ede7e4601c547e7a62abd66ca4f5fa462f555c9c494568756c07` |
| Reviewed packet | `eeb1aee83ec73612ee5e64bf3749621fa0d25603787f85ea02889296576dae34` |
| Verifier report file | `c043adaae83a46aa70c0052fb1ad993f71d621fcdf57b1500338727edbe3ce45` |
| Verifier report payload hash | `ec4b21a2ba0736d103481ee92a70aad9ecb7b6d0a174d467913821a02f5f1759` |

Independent review result:

- 29 shadow records: 10 agreement-allow, 10 agreement-deny, and 9
  `legacy_allow_authority_deny`.
- Internal errors: 0.
- Forty exposures are same-domain/different-task and target-heldout; invalid or
  cross-domain/unscoped exposures: 0.
- Sixty trusted runtime-scope Receipt emissions bind two complete-scope clean
  artifacts. Their source/executed code, plan, trace, attestation, static audit,
  scope binding, payload, stable Receipt ID, and host event chain were
  independently checked.
- All ten Rank/Select agreement-allow controls are clean and complete.
- Seven reviewed Rank disagreements are failed/timeout artifacts with
  `metric.value=null`, no SCORE Claim, and correct Authority
  `claim_exists` denial.
- Two reviewed Promote disagreements have valid SCORE paths but no
  `static_actuation` or `runtime_actuation` Receipt and are correctly
  quarantined.
- The frozen verifier reports `verified=true`, population/review count 9/9,
  sample coverage 1.0, reviewer
  `codex-wp7-r14-independent-reviewer-no-claude`, and dispositions
  `{"confirmed_legacy_false_allow": 9}`.
- There is no `confirmed_authority_false_denial`, `requires_fix`, internal
  error, or uncertain review result in r14.

The original packet, ledger, journal, rollout report, source, Bundle, and root
audit files remained unchanged after adding the reviewed packet and verifier
report. The original `STATE` file was deliberately not rewritten; review
completion is represented additively by the hash-bound review artifacts.

## Remaining required work

The following evidence is absent and cannot be inferred from deterministic
tests or the r14 shadow run:

1. A separately authorized, actually enforced low-cost online canary, pinned to
   the reviewed frozen versions and independently labeled with boolean oracle
   decisions.
2. A passing canary report on actual `effective_allowed` records, including
   minimum decision count, zero unauthorized effective allows, effective IIR,
   VKR, false-denial rate, exclusions, and rollback behavior.
3. The subsequent authorized global main experiment.
4. Only after items 1–3 pass: WP8 Tier 0/1, multi-generation and factorial
   integration tests, heldout 2×2 episodes, baselines/ablations, IIR–VKR
   statistics, L2/L3/L4 adoption evidence, and paper/PPT/Evidence-Ledger
   updates.

No result from the independently observed external canary Pod is used here.
Using it would violate this task's explicit no-canary boundary and would bypass
the required provenance/oracle review.

## Stop point

The real shadow-review checkbox in the WP7 preflight is now satisfied by r14.
The low-cost canary and global-main checkboxes remain unsatisfied. Consequently:

```text
WP0–WP6: PASSED
WP7 shadow-review subgate: PASSED
WP7 total Stop Gate: NOT PASSED
WP8: MUST NOT BEGIN
Full Goal: ACTIVE
```
