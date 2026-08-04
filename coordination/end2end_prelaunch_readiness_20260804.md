# Experiment End2End Pre-launch Readiness

- Date: 2026-08-04
- Status: `SMOKE_AUTHORIZED_GENERIC_A100_RETRY_PRELAUNCH`
- Current live End2End workloads: 0
- Retained pre-Agent launches: L0-L3 (all unscheduled; zero Agent calls)
- Agent/GPU calls made by this preparation: 0

## Completed gates

1. Version audit and selective-port ledger are complete.
2. The unified `MemorySystem` registry contains exactly 10 systems.
3. All systems share one authorized SOP/RunForest candidate pool, Top-6 and a
   1,536-whitespace-token Prompt budget.
4. No Memory remains Bundle-bound, observes/logs the raw pool, and exposes zero
   Prompt memory.
5. Four task Bundle/CURRENT/graph/index and Host Contract/DataView bindings are
   frozen; the runtime SDK exactly matches the production binding hash.
6. Exp-C evaluator assets are bound transitively through base aggregate release
   `668896c0…`; Exp-C source/config/system/hardware-control manifests are
   excluded. Launch-time PVC verification corrected the earlier accidental use
   of the r9 hardware-amended control binding `bee5a525…` before any Smoke Job
   was submitted.
7. The formal budget is frozen at 80 steps, 21,600 seconds, one A100, one
   parallel search, 16 CPUs and 64 GiB memory. Smoke uses two steps and the
   same one-GPU/one-parallel-search/64-GiB resource coupling.
8. The matrix contains 10 non-formal Leaf Smoke runs and 40 formal exploratory
   Pilot runs with immutable logical IDs and task-local randomized order.
9. The finite PID-1 runner preserves all attempts, permits explicit retries
   only after infrastructure failure, freezes the candidate set before the
   terminal evaluator and never imputes a failed score.
10. `validate_smoke_gate.py` derives an immutable, self-hashed gate from all 10
    retained Smoke conditions. It checks retry chains, terminal reports,
    selected-candidate code/Host runtime, complete routing and suppression,
    the 1,536-token budget, Bundle identity, zero unauthorized exposure,
    No-Memory non-exposure and real memory-on Prompt activation.
11. Every normal Pilot runner invocation fails closed without that gate and
    verifies its exact Smoke, Pilot, source-lock and component bindings before
    opening external runtime assets. Pilot dry-run remains side-effect free.
12. The analyzer reports terminal utility/completion/negative transfer/TTFV/
    cost before routing/suppression/static adoption/runtime activation.

## Frozen launch identities

| Artifact | SHA-256 |
|---|---|
| Complete source lock | `9e25e11378422e2eae619179a8da632d465f29ad9071e95e383eeec1f81940f4` |
| Smoke manifest | `00782e0d9c03cf6b7bdec28e7721f4bb2af8e7fee970735c1da1e355f8717a0e` |
| Pilot manifest | `fd47e5f2bb5ff8c6b042b137a9b9354a30bee65f9956bdaa7b2fb175da6b38f9` |
| Launch packet | `21ecfcf9a64ec4edcb3eb126ec63cfc6d6377b1eb7e1e53c795c6dddd99a4f50` |

## Verification receipts

| Gate | Result |
|---|---|
| New/relevant local suite | 157 passed, 1 skipped before the resource-only resize; 40 passed after freezing 64 GiB and adding the container import path |
| Smoke gate positive/negative coverage | pass, retained infra retry, missing/failed system, No-Memory exposure, memory-on non-activation, Pilot missing-gate all exercised |
| Exact manifest/config/runner dry-run coverage | 50/50 logical assignments |
| Kubernetes client dry-run | 5/5 Jobs accepted |
| Kubernetes server dry-run | 5/5 Jobs accepted |
| End2End workloads after dry-run | 0 Jobs, 0 Pods |
| PVC | `haoming-storage`, Bound, 3 TiB |
| Secrets | solver and collector Secrets present |
| ownership-scoped preflight | no safe-delete candidates for the exact End2End label; no mutation performed |
| A100 scheduling contract | `nvidia.com/a100: 1`; no product-label constraint. Each attempt records its actual node and `nvidia-smi` product. |

The all-repository suite was also run: 933 passed, 1 skipped and 20 failed.
All 20 failures belong to historical WP8/PR8 evidence tests: this worktree does
not contain their old XML, staging manifests or deploy/DevPod YAMLs (the three
amendment verifiers specifically report missing r14/r15 stager and recovery
Pod artifacts). None exercises the End2End code path. Historical evidence was
not copied from a read-only worktree or rehashed to conceal those failures.

## Intentionally pending execution-side gates

These are not preparation defects; they require the user-authorized cluster
execution phase:

1. stage the exact source-lock tree at `/workspace/nautilus-exp-end2end` and
   verify hash `9e25e11378422e2eae619179a8da632d465f29ad9071e95e383eeec1f81940f4`;
2. let the runner open and verify the PVC-resident Bundle/evaluator artifacts;
3. submit only `smoke-leaf-indexed-job.yaml`;
4. generate `/workspace/experiment-end2end-runs-v1/SMOKE_GATE.json` only by
   passing the 10-run validator; no manually asserted gate is accepted;
5. only after that exact gate, submit the four Pilot Jobs, all of which carry
   the gate path and fail closed on identity drift.

Pilot results must always be reported as exploratory seed-1 evidence without
statistical-significance claims.
