# Decision Admissibility ICLR Draft Evidence Ledger

Last audited: 2026-07-23

This ledger is the source-of-truth companion to `main.tex`. A result may enter
the paper only with one of the statuses below and a concrete repository source.

## Status vocabulary

- `supported`: directly backed by a completed, inspectable artifact.
- `diagnostic`: useful structural or mechanism evidence, but not an end-to-end
  downstream result.
- `smoke-only`: too small or synthetic for a general claim.
- `rejected`: explicitly failed a claim gate or leakage/provenance audit.
- `pending`: implementation exists, but the required completed evaluation is
  not yet available.

## Audited claims

| ID | Claim or observation | Status | Scope and evidence |
|---|---|---|---|
| C1 | The clean RunForest contains 22 allowlisted journals, 1,346 RunNodes, 1,324 Transitions, 1,236 Evidence nodes, 787 FailurePatterns, and 281 SOPs. | supported | `paper-skills/hyper_memory/run_forest_builder_report.json`; source membership and code audit are recorded as verified. |
| C2 | The SOP taxonomy covers 281/281 SOPs: 28 L1 strategies, 101 L2 tactics, and 152 L3 repairs; all 28 L1 entries are in the explicit review list. | supported | `paper-skills/hyper_memory/sop_taxonomy.json`; coverage=1.0, deterministic classifier v1. |
| C3 | Over 120 retrospective stage episodes, the dynamic stage hybrid has Granularity Precision@5=1.0000, Detail Intrusion@5=0.0000, and empty-result rate=0.0000; ungated flat retrieval has 0.4467, 0.5533, and 0.0000. | diagnostic | `paper-skills/eval_composite_memory/reports/causal_granularity_report_v2.json`; the track verifies level selection, not repair execution. |
| C4 | The causal Debug corpus contains 38 real episodes from 12 source runs, with a fixed 13/25 development/test partition; the test split has 12 covered episodes and 13 evidence gaps. | diagnostic | `causal_granularity_report_v2.json`; retrospective=true, blind_test=false, and labels are semantically aligned silver labels. |
| C5 | On the 25-episode causal Debug test partition, the dynamic hybrid reaches route accuracy=0.8000, Selective Decision Accuracy@1=0.7600, Hit@1=0.5833, transition MRR=0.5833, and fallback accuracy=0.9231. Legacy successful-node Tree reaches 0.4800, 0.1600, 0.3333, 0.4028, and 0.0000. | diagnostic | `causal_granularity_report_v2.json`; primary deltas are +0.3200 route accuracy, +0.6000 selective decision accuracy, and +0.1806 MRR. Random, task-only, and lexical transition controls are also lower. |
| C6 | Run-tree coordinates preserve lineage structure better under Poincare than the same coordinates under Euclidean distance for several parent/neighbor diagnostics. | diagnostic | `run_forest_memory_evaluation.json` and `run_tree_hyperbolic_diagnostics.json`; this is a carrier diagnostic, not an online downstream advantage. |
| C7 | Explicit graph expansion recovers debug children and local-best lineage perfectly in the constructed graph diagnostic. | diagnostic | `run_forest_memory_evaluation.json`; explicit edges make this a graph-navigation check rather than a geometry claim. |
| C8 | V3 static audit labels 693 code nodes clean, 280 blocked, 55 protocol-biased, 155 warning, and 141 unavailable; non-clean nodes are excluded from certified ranking and positive memory. | supported | `run_forest_builder_report.json` and `coordination/runforest_leakage_audit_v3_implementation_report.md`. |
| C9 | The historical Spooky three-model replay (`d93b4c2a`) is detected but its reported 0.2013 is not rank-eligible because preprocessing touches report data and ensemble weights are selected on the reporting split. | rejected | V3 audit reports `TRANSFORM_FIT_ON_HOLDOUT` and `REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION`; quarantined source and repeated online traces. |
| C10 | The value 0.0494 is not a valid model score. | rejected | It originated from parsing/diagnostic context, not a clean final evaluation; retained only as a failure-analysis warning. |
| C11 | Staged protocol repair implements capability-conditioned stages: data scope, validation provenance, cross-fitting/OOF, selection freeze, final holdout, and runtime provenance. | supported (implementation) | `mlevolve/agents/protocol_repair.py`, `protocol_repair_runtime.py`, and `tests/test_protocol_repair.py`. |
| C12 | Staged protocol repair has produced a completed, clean, rank-eligible replay score across multiple task families. | pending | No finished artifact currently establishes this. Current online traces are case studies and must not be reported as a completed comparison. |
| C13 | Dynamic stratified memory improves offline routing decisions. | diagnostic | Supported on the retrospective stage and causal Debug tracks; not evidence of executed repair or downstream MLE improvement. |
| C14 | Hyperbolic geometry improves downstream MLE performance. | pending/not evaluated | The v2 causal benchmark uses the Flat-Twin carrier and contains no geometry comparison; structural diagnostics remain non-downstream evidence. |
| C15 | The current structural failure-pattern matcher retains the two known d93 patterns after the covered local-variable-renaming transformation. | supported (implementation) | `tests/test_run_forest_memory.py::test_d93_structural_rename_still_matches_failure_patterns` passes on 2026-07-20. This narrow regression does not establish universal semantic-equivalence detection. |
| C16 | Dynamic Debug routing separates transition ranking from evidence admission: confidence below 0.50 forces SOP-only fallback; otherwise Tree fusion weight is `0.60 * confidence` and never exceeds 0.60. | supported (implementation) | `mlevolve/agents/memory/stage_aware_hybrid_memory.py`; ranking uses failure/lexical/task/attachment/Flat-Twin components, while confidence excludes geometry and is explicitly not a calibrated success probability. |
| C17 | A deprecated fixed-0.75 Tree ablation has higher covered-episode Hit@1/MRR (0.7500/0.7500) but lower route/selective/fallback accuracy (0.7600/0.7200/0.6923) than the dynamic router (0.8000/0.7600/0.9231). | diagnostic | `causal_granularity_report_v2.json`, `causal_tree_fixed_075` and `fallback_ablation`; retained only to disclose the ranking-versus-abstention trade-off, not as the production router. |
| C18 | Claim-level visibility preserves both oracle-legal Debug clauses in the deterministic mixed-value fixture while producing zero unauthorized Prompt exposure/activation and suppressing the contaminated Score from Rank. | supported (deterministic) | `coordination/decision_admissibility_wp3_report_20260719.md`; focused visibility and bypass tests. This is an exact invariant result, not a natural-prevalence estimate. |
| C19 | The separate formal non-Spooky corpus contains 79 complete runs across 16 tasks, 1,656 RunNodes, 1,577 audited code nodes, and 589 scored metric nodes; 11 incomplete directories are excluded. | supported (systems artifact) | `coordination/decision_admissibility_wp4_report_20260719.md` and its hash-verified local evidence root. |
| C20 | The WP4 binder emits 1,545 source-resolved clauses and 934 merged containers; compilation yields 1,278 diagnostic, 267 candidate, and zero certified clauses with no scope widening. | supported (systems artifact) | WP4 binder/merge reports and three externally validated raw-audited Bundle manifests. Zero certified clauses is intentional before Clean Replay. |
| C21 | Seed-heldout has zero run/seed-group overlap; task-heldout has zero task/run overlap and zero heldout references. All raw-audited Bundles have zero authorized adoption edges. | supported (systems artifact) | `coordination/decision_admissibility_wp4_report_20260719.md`, split manifests, RunForest reports, and external Bundle validation. This does not establish task generalization. |
| C22 | Method-preserved Clean Replay creates a new Claim/support path; protected method changes create a Successor Claim; neither mutates or re-authorizes the historical Claim. | supported (deterministic) | `coordination/decision_admissibility_wp6_report_20260719.md` and the four focused Clean Replay test modules. No real-corpus certified score is claimed. |
| C23 | The latest local rollout preflight passes 451 tests with the independently frozen composite benchmark excluded; that module separately has 18 passes and one pre-existing detector-hash lock mismatch. | supported (test status) | `coordination/decision_admissibility_wp7_local_preflight_20260719.md`; the frozen lock is not rewritten to manufacture a pass. |
| C24 | WP7 corrected canary and the declared WP8 evidence tracks, including the nine-block formal Tier-2 execution, are execution-complete and hash-verified. | supported (execution) | `coordination/decision_admissibility_wp7_corrected_canary_report_20260721.md`, the verified Tier-0/Tier-1/multi-generation/canary packets, and `coordination/decision_admissibility_wp8_tier2_formal_joint_inventory_20260723_r1/verification.json`. Execution completeness does not authorize Full superiority or experience-level causality. |
| C25 | Result Fact, Adoption Edge, and Causal Edge are separately authorized and materialized: a clean unexposed node can publish one Result Fact; L3 can publish one Adoption Edge; L4 plus a prior Adoption can publish one Causal Edge. | supported (deterministic implementation) | `mlevolve/authority/adapters/mlevolve/runtime.py`, `mlevolve/authority/bundle_publisher.py`, and `tests/authority/test_result_adoption_causal_writeback.py`. This establishes object and permission separation, not real-world causal benefit. |
| C26 | Fixed-holdout search performs zero positive writeback before scoring and the sealed terminal scorer finalizes exactly one idempotent Result Fact or records an explicit incomplete status. | supported (deterministic implementation) | `mlevolve/fixed_holdout/writeback.py`, `mlevolve/fixed_holdout/score_run.py`, and `tests/test_fixed_holdout_terminal_writeback.py`. |
| C27 | Positive Result SOP and Positive Adopted SOP distillation have separate authority paths; the former uses the target result's evidence without historical actuation, while the latter requires contract-bound L3 evidence and causal language additionally requires L4. | supported (deterministic implementation) | `mlevolve/authority/positive_distillation.py`, `mlevolve/authority/protocol_compiler.py`, and `tests/test_positive_result_vs_adopted_distillation.py`. |
| C28 | Formal Tier-2 retained all 45 assigned online outcomes across nine task-seed blocks: 22 scored selected results and 23 failed outcomes, plus nine host-only Oracle dispositions; successful conditions produced 22/22 independent Result Facts and failed conditions produced none. | supported (formal systems evidence) | `coordination/decision_admissibility_wp8_tier2_formal_joint_inventory_20260723_r1/joint_inventory.json` and `verification.json`; sample unit is task-seed-system assignment. No failed condition, seed, or task was removed. |
| C29 | Full Decision Admissibility improves target-task training performance over No Memory in the formal experiment. | rejected | `coordination/decision_admissibility_wp8_tier2_formal_statistics_20260723_r1/statistics_report.json` and `verification.json`: Full completion is 4/9 versus 6/9 for No Memory; only 4/9 pairs are score-estimable; Taxi has no Full--No Memory scored pair; exact one-sided p=0.0625 and Holm-adjusted p=0.25. The frozen effect gate is false. |
| C30 | Conditional on both systems producing a protocol-legal selected result, the four observed Full--No Memory deltas all favor Full. | diagnostic | Same formal statistics packet: 4 wins, 0 ties, 0 losses; mean Aerial macro-F1 delta is +0.014373 and the single Birds delta is +0.059794. This available-pair pattern is not a population superiority claim because five pairs are unscored failures and Taxi contributes none. |
| C31 | Formal Tier-2 statistics use no Oracle, source-score, other-system, or constant score imputation and no post-assignment exclusions. | supported (analysis integrity) | `coordination/decision_admissibility_wp8_tier2_formal_analysis_policy_addendum_20260723_r1.json` and the formal statistics verification: 45 assigned, 0 imputed, 0 post-assignment exclusions, 23 retained failures. |
| C32 | Injected historical experience caused the conditional Full gains. | pending/not established | The formal system contrast is intention-to-treat evidence, not an experience edge. Static plus runtime actuation is required for Adoption, and a memory-on/off counterfactual is additionally required for Causal attribution; the formal performance contrast alone does not satisfy L3/L4. |

## Evaluation units that must not be conflated

1. **Deterministic authority invariants** test exact Claim/Receipt/visibility,
   replay, publication, and rollback behavior; they do not estimate prevalence
   or downstream utility.
2. **Split-isolated Bundle construction** verifies immutable corpus and heldout
   boundaries; it does not establish model generalization.
3. **Structural diagnostics** evaluate whether a coordinate system or explicit
   graph operation preserves known run-tree relations.
4. **Stage-granularity tests** evaluate whether the returned memory abstraction
   matches the decision stage; they do not test causal repair utility.
5. **Causal Debug transfer tests** evaluate route choice and historical
   transition recovery under source-run exclusion; they do not execute generated
   training code and are retrospective rather than blind.
6. **Online MLE runs** execute generated code but can report internal validation
   metrics that are not automatically independent final-holdout estimates.
7. **Certified replay and formal terminal results** require clean static audit,
   method preservation where certification is claimed, runtime provenance, and
   a rank-eligible final metric. Formal Tier-2 now contributes 22 verified Result
   Facts, but those independent results do not by themselves establish Adoption
   or Causal edges.

## Required next evidence

- A fresh, preregistered formal replication after reducing Authority false
  denials and candidate-runtime failures; the present failed outcomes must not
  be replaced or re-labelled.
- An experience-level L3/L4 study binding Prompt exposure to static actuation,
  runtime actuation, and memory-on/off counterfactuals.
- Additional same-domain held-out tasks and seeds to estimate task-level
  uncertainty, especially a valid tabular Full--No Memory comparison.
- Mechanism ablations not covered by the five-system formal matrix, including
  stage-only, post-prompt warning, and provenance-only controls where required
  by the paper claim.
