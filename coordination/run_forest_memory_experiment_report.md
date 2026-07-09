# Hyperbolic Run-Forest Memory Experiment

## Summary

This experiment evaluates the new carrier: run/journal memory as a hyperbolic forest, with SOPs attached as signposts. It compares Poincare distance, same-coordinate Euclidean Flat-Twin, and independent Euclidean text memory.

## Main Results

| Task | System | R@5 | MRR | Extra |
|---|---|---:|---:|---:|
| parent_lookup | run_forest_poincare | 0.6360 | 0.3775 | queries=1324 |
| parent_lookup | run_forest_flat_twin | 0.4056 | 0.2813 | queries=1324 |
| parent_lookup | run_forest_euclidean | 0.4298 | 0.2929 | queries=1324 |
| local_best_lookup | run_forest_poincare | 0.3014 | 0.1888 | queries=866 |
| local_best_lookup | run_forest_flat_twin | 0.2309 | 0.1480 | queries=866 |
| local_best_lookup | run_forest_euclidean | 0.2875 | 0.1798 | queries=866 |
| debug_recovery_child_lookup | run_forest_poincare | 0.7418 | 0.3870 | queries=213 |
| debug_recovery_child_lookup | run_forest_flat_twin | 0.8263 | 0.4738 | queries=213 |
| debug_recovery_child_lookup | run_forest_euclidean | 0.3991 | 0.2814 | queries=213 |
| transition_to_sop_signpost | run_forest_poincare | 0.8080 | 0.5716 | queries=1229 |
| transition_to_sop_signpost | run_forest_flat_twin | 0.7868 | 0.5435 | queries=1229 |
| transition_to_sop_signpost | run_forest_euclidean | 0.4109 | 0.2815 | queries=1229 |
| transition_to_evidence | run_forest_poincare | 0.8997 | 0.7709 | queries=1236 |
| transition_to_evidence | run_forest_flat_twin | 0.9005 | 0.7711 | queries=1236 |
| transition_to_evidence | run_forest_euclidean | 0.2799 | 0.1753 | queries=1236 |
| debug_recovery_child_lookup | run_forest_graph_expansion | 1.0000 | 1.0000 | queries=213 |
| local_best_graph_follow | run_forest_graph_expansion | 1.0000 | 1.0000 | queries=866 |

## Tree Neighbor Preservation

| System | Neighbor Recall@10 | Queries |
|---|---:|---:|
| run_forest_poincare | 0.5637 | 1346 |
| run_forest_flat_twin | 0.4996 | 1346 |
| run_forest_euclidean | 0.3740 | 1346 |

## Bootstrap Comparisons

| Comparison | Mean Diff | p one-sided | 95% CI |
|---|---:|---:|---|
| parent_lookup_poincare_vs_flat_twin_mrr | 0.0962 | 0.0000 | [0.0861, 0.1063] |
| parent_lookup_poincare_vs_euclidean_mrr | 0.0846 | 0.0000 | [0.0606, 0.1086] |
| parent_lookup_poincare_vs_flat_twin_recall_at_5 | 0.2304 | 0.0000 | [0.2069, 0.2545] |
| parent_lookup_poincare_vs_euclidean_recall_at_5 | 0.2062 | 0.0000 | [0.1730, 0.2402] |
| local_best_lookup_poincare_vs_flat_twin_mrr | 0.0408 | 0.0000 | [0.0335, 0.0484] |
| local_best_lookup_poincare_vs_euclidean_mrr | 0.0089 | 0.2106 | [-0.0126, 0.0295] |
| local_best_lookup_poincare_vs_flat_twin_recall_at_5 | 0.0704 | 0.0000 | [0.0520, 0.0901] |
| local_best_lookup_poincare_vs_euclidean_recall_at_5 | 0.0139 | 0.2578 | [-0.0266, 0.0531] |
| debug_recovery_child_lookup_poincare_vs_flat_twin_mrr | -0.0868 | 1.0000 | [-0.1125, -0.0633] |
| debug_recovery_child_lookup_poincare_vs_euclidean_mrr | 0.1056 | 0.0003 | [0.0446, 0.1661] |
| debug_recovery_child_lookup_poincare_vs_flat_twin_recall_at_5 | -0.0845 | 1.0000 | [-0.1221, -0.0469] |
| debug_recovery_child_lookup_poincare_vs_euclidean_recall_at_5 | 0.3427 | 0.0000 | [0.2582, 0.4272] |
| transition_to_sop_signpost_poincare_vs_flat_twin_mrr | 0.0281 | 0.0000 | [0.0188, 0.0375] |
| transition_to_sop_signpost_poincare_vs_euclidean_mrr | 0.2901 | 0.0000 | [0.2633, 0.3178] |
| transition_to_sop_signpost_poincare_vs_flat_twin_recall_at_5 | 0.0212 | 0.0076 | [0.0041, 0.0382] |
| transition_to_sop_signpost_poincare_vs_euclidean_recall_at_5 | 0.3971 | 0.0000 | [0.3637, 0.4312] |

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
