# Decision Admissibility WP4 Stop-Gate Report

Date: 2026-07-19  
Branch: `codex/dual-time-procedural-memory`  
Remote-verified baseline and current HEAD: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP4 — Corpus、RunForest 和 SOP-Clause Bundle  
Status: **PASSED**

## Authoritative boundaries

- Formal local evidence root:
  `coordination/decision_admissibility_wp4_real_work_20260719_r2_verified`
- Formal persistent Pod output root: `/output/corrected-r2`
- Source corpus: `/corpus`, mounted read-only from the user PVC.
- Inventory source commit:
  `be034ec81d58e96ca333abb7bda155726aaa3668`.
- The baseline branch/HEAD/upstream remain unchanged at `b47dab63`; WP0–WP4
  implementation and evidence remain uncommitted and unstaged.
- No old graph/index, raw journal, dataset, credential, paper asset, or unrelated
  user-untracked path was moved, deleted, staged, or overwritten.

## Implementation completed

WP4 adds a manifest- and sidecar-driven raw-audited memory-bundle path:

- `paper-skills/memory_bundle/` implements source fingerprinting, corpus
  inventory, deterministic leakage sidecars, split manifests, frozen-response
  SOP distillation, clause binding, container-only merge, RunForest v2,
  immutable bundle publication, archive export, and external validation.
- `paper-skills/distillation/extract_branches.py` consumes the corpus and split
  manifests rather than a hard-coded run allowlist, and emits globally
  resolvable source references plus a trace manifest.
- `paper-skills/hyper_memory/build_run_forest_memory.py` has a manifest-driven
  v2 path; the legacy `--runs-dir` path remains explicitly uncertified.
- Full, seed-heldout, and task-heldout bundles are built separately, bind their
  split/corpus/protocol/model hashes, and are published atomically.
- Raw-audited bundles contain no authorized adoption edges. Audit findings stay
  inspectable without being silently upgraded into Rank/Promote authority.
- The Pod workflow filters AppleDouble files, verifies reviewed-input hashes
  after transfer, supports resumable split builds, and keeps staging/failed
  attempts outside the formal output root.

## Real-corpus inventory and audit

The expected and actual snapshots match exactly; `drift_detected=false` and the
manual drift-review disposition is `accept_no_snapshot_drift`.

| Item | Result |
|---|---:|
| Run directories | 90 |
| Complete formal runs | 79 |
| Partial/excluded directories | 11 |
| Complete non-Spooky tasks | 16 |
| Run nodes | 1,656 |
| Code nodes | 1,577 |
| Scored metric nodes (`metric.value != null`) | 589 |
| Formal runs missing any of four core hashes | 0 |
| Spooky formal/source runs | 0 |
| Code-node audit sidecars | 1,577 / 1,577 |
| Source journals modified by audit | 0 |

All 11 partial directories lack core artifacts and are excluded from every
formal split. All 79 formal runs bind SHA-256 hashes for `journal.json`,
`filtered_journal.json`, `config.yaml`, and `best_solution.py`.

The deterministic audit result is intentionally not collapsed into a single
pass/fail bit:

| Sidecar status | Count |
|---|---:|
| clean | 718 |
| warning | 40 |
| blocked | 655 |
| audit_unavailable | 164 |

Blocked/unavailable evidence remains present for diagnosis, while bundle
authority remains `raw_audited` with zero authorized adoption edges.

### Corpus integrity anchors

| Anchor | SHA-256 |
|---|---|
| Canonical corpus manifest | `5cfd8808d5c352c3ac7993932c7dc999e7080a391bc72ae0fd40c4498230e030` |
| Actual snapshot | `d8790046e5baed0512354952f3ab9e211619ac93553580557b64c0e89d71073f` |
| Source path list | `af06fa511b77f7e13228b01f589643d114ca14c6fe2e03ae336766430f4c113c` |
| Source stat fingerprint | `639f2fed6f9ec2950d066e31d45308a8955582aec25359e1bbc9f8163121541f` |
| Trace manifest | `28e1b61c405704a5e9572265315e44db2826559a2824056cdf5487d005cc3607` |

The source fingerprint covers 91,058 files and is identical before inventory,
after inventory/audit/split extraction, and after all bundle builds.

## Distillation, binding, and merge evidence

- 79 formal runs produced 354 auditable traces.
- All 354 traces produced frozen responses and 354 proposals.
- The networked DeepSeek run had three transient request/parse failures and six
  retries, but ended with all responses saved. No proposal is missing.
- A network-disabled rebuild from the frozen response artifact reproduced the
  exact proposals hash with zero failures and zero retries.
- Frozen responses SHA-256:
  `a775b1037bfa45941cd0e793a9f1948502b3046dbe70ee61666b9d48e870bdd1`.
- Proposals SHA-256:
  `6a256aa4497c40601fd7d0d97ac1672fc4b72c2b7198d7d4c44d919e58f965cc`.
- System prompt SHA-256:
  `5aae2b4a242e9e2e0cf12b5198556fc0e04b1917b1d7c45b999a02f2a63daf30`.

The final binder produced 1,545 clauses, 1,545 Claims, and 1,545 derivations.
Every source reference resolves; quarantine count and scope-widening count are
both zero. Compilation made no publication-class upgrades and 513 conservative
downgrades:

| Publication class | Proposed | Compiled |
|---|---:|---:|
| diagnostic | 773 | 1,278 |
| candidate | 735 | 267 |
| certified | 37 | 0 |

Container merge reduced 1,134 containers to 934 across 132 merge clusters.
The clause payload hash stayed exactly
`624d3e0e6bbdedcc738afc7c2a06d2d98a47e950438a07e6be4e3d7e607c14c7`;
`clause_authority_changed=false`.

## Split-aware bundle evidence

| Split | Source / heldout runs | Audited code nodes | Clauses | Containers | Archive SHA-256 |
|---|---:|---:|---:|---:|---|
| Full | 79 / 0 | 1,577 | 1,545 | 934 | `e3025421d7c79565258fac23fbf7c849edb95adeb5166ee58a8d20104595f163` |
| Seed-heldout | 53 / 26 | 1,026 | 1,058 | 656 | `090ad00de42d87c4310dbe2deee2a434b7477a5fd8a3181fe6cc489c84d3298d` |
| Task-heldout | 58 / 21 | 1,353 | 1,260 | 744 | `daf1a77338599957f40ca3b4e144efa9919e21dca35743a09306541091cc4ea9` |

All three external validation reports have `valid=true`, empty error/warning
lists, zero Spooky nodes, zero heldout references, complete sidecar coverage,
and fully resolvable clause sources. Each build reports atomic publication,
successful secret scan, and `legacy_artifact_overwritten=false`.

Isolation semantics were checked at the dimension each split is intended to
hold out:

- Full: 79 source runs, no heldout runs, and all 11 partial runs excluded.
- Seed-heldout: source and heldout deliberately share the 16 tasks, but have
  zero run or seed-group overlap. The corrected builder keeps 1,058 clauses
  sourced by the 53 source runs and excludes 487 clauses only because their
  source reference is outside the split. Task-scope exclusion is disabled.
- Task-heldout: 12 source tasks versus 4 heldout tasks, with zero task/run
  overlap and zero heldout references in the graph.
- All three RunForest reports have `authorized_edge_count=0`, as required for
  a raw-audited rather than certified-memory bundle.

## Ledger and immutability evidence

- `/output/corrected-r2/WP4_FINAL_SHA256SUMS` contains 7,488 entries.
- A fresh in-Pod `sha256sum -c --quiet` verification passed all 7,488 entries.
- The formal Pod root contains zero partial or staging files.
- Thirteen fetched final metadata files independently match their exact remote
  ledger entries; the reviewed-input checksum manifest also verifies locally.
- The three archive hashes above are directly bound by the final ledger.
- All nine WP0 legacy artifact hashes still match the baseline manifest.
- The old RunForest still contains exactly 281 SOP containers.
- No old graph/index/taxonomy artifact was regenerated or overwritten.

## Preserved rejected and intermediate attempts

No failed attempt was disguised as a successful formal artifact:

1. `coordination/decision_admissibility_wp4_real_work_20260719` is the rejected
   first inventory. Unsafe config interpretation classified all formal tasks as
   unknown and counted 1,656 metrics; its manifest hash differs and it was never
   reviewed, distilled, or published.
2. `coordination/decision_admissibility_wp4_real_work_20260719_r2` preserves the
   corrected inventory transfer. The complete formal evidence root is the
   separate `_r2_verified` directory named above.
3. `binder_pre_nonwidening_fix/` and `merged_pre_nonwidening_fix/` preserve the
   pre-fix binder/merge output. Only the final non-widening binder is present in
   reviewed inputs and formal bundles.
4. The pre-fix seed bundle excluded all 1,545 clauses: 1,058 incorrectly as
   `heldout_task_scope` and 487 as `source_ref_outside_split`. It was moved out
   of the formal root to
   `/output/decision-admissibility-wp4-invalid-seed-pre-fix-v1`. The corrected
   seed bundle includes 1,058 clauses and has no task-scope exclusion.
5. One interrupted reviewed-input transfer is isolated at
   `/output/decision-admissibility-wp4-transfer-attempts`; no staging path
   remains under the formal root.
6. The interrupted local fetch
   `final_reports/full_bundle_manifest.json.partial-eof-1784443306` is retained
   as evidence but is outside the formal ledger. The refetched complete manifest
   matches its remote ledger entry.

## Regression and health checks

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/test_corpus_manifest.py \
  tests/test_corpus_split_isolation.py \
  tests/test_run_forest_bundle_v2.py \
  tests/test_sop_clause_distillation_schema.py \
  tests/test_memory_bundle_validation.py
# 20 passed

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests --ignore=tests/test_composite_memory_benchmark.py
# 387 passed

PYTHONPATH=mlevolve .venv/bin/python -m compileall -q \
  mlevolve paper-skills tests

bash -n \
  deploy/decision_admissibility_wp4_pod_workflow.sh \
  deploy/run_decision_admissibility_wp4_inventory.sh \
  deploy/run_decision_admissibility_wp4_bundles.sh

git diff --check
# all passed
```

An unfiltered run of `tests/test_composite_memory_benchmark.py` remains
`18 passed, 1 failed`. The only failure is the pre-existing frozen-lock mismatch
already documented in WP3: the lock stores detector hash
`ae30aac332b6f62dccc784955ec2952268d8a1381085bcadb4d683b9d8f6a221`,
while `mlevolve/agents/leakage_audit.py` is
`8f52b2ca627944d5716de30cd50f2e371ec641bc784edc95500b01d7f544f460`.
The detector's current hash is identical to baseline HEAD, and neither file is
in the WP0–WP4 diff. The frozen lock was not rewritten to manufacture a pass.

## Stop-gate decision

- [x] Spooky formal/source runs and bundle nodes are zero.
- [x] Every formal run binds all four core artifact hashes.
- [x] Every code node has a deterministic audit sidecar.
- [x] Every included clause source resolves.
- [x] Seed-heldout has zero run/seed-group overlap and zero heldout references.
- [x] Task-heldout has zero run/task overlap and zero heldout references.
- [x] Old 281-SOP graph/index and all nine frozen legacy artifacts are unchanged.
- [x] Three split-aware bundles validate and their archives are ledger-bound.
- [x] Broad regression, focused WP4 tests, compile, shell syntax, diff, and final
      7,488-entry checksum verification pass.

**WP4 Stop Gate: PASSED. WP5 may begin only after the authorized temporary WP4
Pod is deleted; PVC outputs remain persistent.**

## Post-gate cluster cleanup

- At `2026-07-19T15:01:30+08:00`, the authorized workflow deleted Pod
  `ecepxie/decision-admissibility-corpus-inventory-r1` with exit code 0.
- A subsequent `kubectl get ... --ignore-not-found -o name` returned no Pod.
- PVC `ecepxie/haoming-storage` remains `Bound` at 2 TiB, so the verified
  `/output/corrected-r2` artifacts remain persistent after Pod deletion.
