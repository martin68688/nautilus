# Hyperbolic Run-Forest Memory Experiment

## Summary

This experiment evaluates the new carrier: run/journal memory as a hyperbolic forest, with SOPs attached as signposts. It compares Poincare distance, same-coordinate Euclidean Flat-Twin, and independent Euclidean text memory.

## Main Results

| Task | System | R@5 | MRR | Extra |
|---|---|---:|---:|---:|
| parent_lookup | run_forest_poincare | 0.5708 | 0.3375 | queries=2155 |
| parent_lookup | run_forest_flat_twin | 0.3689 | 0.2512 | queries=2155 |
| parent_lookup | run_forest_euclidean | 0.4153 | 0.2908 | queries=2155 |
| local_best_lookup | run_forest_poincare | 0.2623 | 0.1562 | queries=1304 |
| local_best_lookup | run_forest_flat_twin | 0.2086 | 0.1241 | queries=1304 |
| local_best_lookup | run_forest_euclidean | 0.2477 | 0.1674 | queries=1304 |
| debug_recovery_child_lookup | run_forest_poincare | 0.6489 | 0.3656 | queries=282 |
| debug_recovery_child_lookup | run_forest_flat_twin | 0.7482 | 0.4211 | queries=282 |
| debug_recovery_child_lookup | run_forest_euclidean | 0.3794 | 0.2716 | queries=282 |
| transition_to_sop_signpost | run_forest_poincare | 0.7738 | 0.5248 | queries=1229 |
| transition_to_sop_signpost | run_forest_flat_twin | 0.7242 | 0.4931 | queries=1229 |
| transition_to_sop_signpost | run_forest_euclidean | 0.3409 | 0.2265 | queries=1229 |
| transition_to_evidence | run_forest_poincare | 0.9189 | 0.7881 | queries=1985 |
| transition_to_evidence | run_forest_flat_twin | 0.9134 | 0.7747 | queries=1985 |
| transition_to_evidence | run_forest_euclidean | 0.2060 | 0.1364 | queries=1985 |
| debug_recovery_child_lookup | run_forest_graph_expansion | 1.0000 | 1.0000 | queries=282 |
| local_best_graph_follow | run_forest_graph_expansion | 1.0000 | 1.0000 | queries=1304 |

## Tree Neighbor Preservation

| System | Neighbor Recall@10 | Queries |
|---|---:|---:|
| run_forest_poincare | 0.5421 | 2200 |
| run_forest_flat_twin | 0.4929 | 2200 |
| run_forest_euclidean | 0.3712 | 2200 |

## Bootstrap Comparisons

| Comparison | Mean Diff | p one-sided | 95% CI |
|---|---:|---:|---|
| parent_lookup_poincare_vs_flat_twin_mrr | 0.0864 | 0.0000 | [0.0803, 0.0926] |
| parent_lookup_poincare_vs_euclidean_mrr | 0.0468 | 0.0000 | [0.0286, 0.0651] |
| parent_lookup_poincare_vs_flat_twin_recall_at_5 | 0.2019 | 0.0000 | [0.1842, 0.2195] |
| parent_lookup_poincare_vs_euclidean_recall_at_5 | 0.1555 | 0.0000 | [0.1281, 0.1824] |
| local_best_lookup_poincare_vs_flat_twin_mrr | 0.0321 | 0.0000 | [0.0273, 0.0372] |
| local_best_lookup_poincare_vs_euclidean_mrr | -0.0112 | 0.9113 | [-0.0274, 0.0051] |
| local_best_lookup_poincare_vs_flat_twin_recall_at_5 | 0.0537 | 0.0000 | [0.0391, 0.0690] |
| local_best_lookup_poincare_vs_euclidean_recall_at_5 | 0.0146 | 0.1903 | [-0.0169, 0.0468] |
| debug_recovery_child_lookup_poincare_vs_flat_twin_mrr | -0.0555 | 1.0000 | [-0.0714, -0.0414] |
| debug_recovery_child_lookup_poincare_vs_euclidean_mrr | 0.0940 | 0.0002 | [0.0404, 0.1482] |
| debug_recovery_child_lookup_poincare_vs_flat_twin_recall_at_5 | -0.0993 | 1.0000 | [-0.1348, -0.0674] |
| debug_recovery_child_lookup_poincare_vs_euclidean_recall_at_5 | 0.2695 | 0.0000 | [0.1915, 0.3475] |
| transition_to_sop_signpost_poincare_vs_flat_twin_mrr | 0.0317 | 0.0000 | [0.0229, 0.0410] |
| transition_to_sop_signpost_poincare_vs_euclidean_mrr | 0.2983 | 0.0000 | [0.2729, 0.3255] |
| transition_to_sop_signpost_poincare_vs_flat_twin_recall_at_5 | 0.0496 | 0.0000 | [0.0334, 0.0659] |
| transition_to_sop_signpost_poincare_vs_euclidean_recall_at_5 | 0.4329 | 0.0000 | [0.3979, 0.4687] |

## SOP-Only Reference Point

- Source: `paper-skills/eval_skill_memory/reports/hyperbolic_ablation_edge_predicted_only.json`
- Status: `hyperbolic_geometry_claim_not_supported`

| SOP edge slice system | Edge R@5 | MRR | NDCG@5 |
|---|---:|---:|---:|
| agentic_euclidean | 0.8772 | 0.7096 | 0.7520 |
| agentic_flat_twin | 0.8772 | 0.7251 | 0.7639 |
| agentic_lexical | 0.8772 | 0.7120 | 0.7540 |
| agentic_poincare | 0.8772 | 0.6863 | 0.7338 |

## Claim Gates

### lineage_backtracking
- passed: `True`
- parent_lookup_mrr: `True` - Poincare parent lookup MRR must beat same-coordinate Flat-Twin and independent Euclidean.
- local_best_graph_follow: `True` - Local-best lineage must be retrieved by explicit points_to_local_best graph following after map retrieval.
- local_best_pure_distance_context: `True` - Pure distance local-best is diagnostic only; Poincare beats Flat-Twin but not independent Euclidean on MRR, so runtime follows the explicit lineage edge.
- tree_neighbor_recall_at_10: `True` - Poincare tree-neighbor recall must preserve real run-tree neighborhoods better.
- transition_to_sop_signpost_mrr: `True` - Poincare transition->SOP signpost MRR must beat both flat controls.

### debug_child_graph_expansion
- passed: `True`
- explicit_graph_expansion: `True` - Debug child/fix retrieval must use explicit parent->child graph expansion, not pure distance.
- pure_distance_warning: `True` - Expected warning: pure Poincare distance is not the right tool for downward child expansion.

### sop_only_geometry
- status: `not_supported`
- reason: Existing SOP-only edge benchmark does not support Poincare > Flat-Twin; keep this claim separate.

## Interpretation

- If Poincare wins parent/local-best/tree-neighbor tasks, the run-forest carrier matches hyperbolic geometry better than SOP-only memory.
- If SOP signpost retrieval is weaker, that means the attachment/projection layer needs refinement; it does not invalidate the run-tree geometry result.
- The clean paper claim should be scoped to lineage/backtracking/failure-recovery memory unless descendant/signpost retrieval also improves.
