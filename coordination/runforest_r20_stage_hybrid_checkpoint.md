# RunForest R20 Stage-Hybrid Checkpoint

Date: 2026-07-11
Branch: `codex/dual-time-procedural-memory`
Source baseline: `350b8f61`
Job manifest: `deploy/job-runforest-online-a40x7-clean-r20.yaml`

## Purpose

This checkpoint preserves the exact pre-layered-retrieval state before the
Novel Draft L1/L2 upgrade. The running R20 Job remains an old-code observation
and is not hot-updated by later commits.

## Current Behavior

- Draft roles are assigned as `coldstart_baseline`, `memory_reproduction`, and
  `novel_exploration`, but spare parallel workers can still create additional
  root Novel drafts.
- Novel Draft uses one generic Stage Hybrid memory pack. The same pack is
  passed to data processing, model design, training/evaluation, and merge.
- Draft SOP ranking has no explicit strategy/tactic/repair taxonomy. This
  allowed detail-oriented SOPs such as `sg_0227`, `sg_0069`, and `sg_0115` to
  occupy method-selection slots.
- The current held-out execution MRR is `0.3741` for Tree-only and `0.3670` for
  Stage Hybrid. Stage Hybrid has not established a retrieval advantage.
- Poincare has not beaten Flat-Twin, and the deterministic geometry must not be
  described as a learned hyperbolic embedding.

## Known Runtime Risks

- Concurrent root-draft reservations can exceed the configured role count.
- A forced root return can cause repeated non-actionable aggregation attempts.
- A mandatory repair with a protocol-biased but non-hard-block audit can reach
  execution before the audit status is fully clean.
- The R20 manifest's default run tag still contains `r19`; this checkpoint
  records that fact without changing the running behavior.

## Upgrade Boundary

The next commit will add full SOP taxonomy, Novel-only L1 strategy selection,
model-design-time L2 retrieval, strict role isolation, clean execution evidence
gates, scheduler fixes, repair execution gates, tests, and an updated research
note.
