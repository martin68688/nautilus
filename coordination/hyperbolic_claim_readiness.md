# Hyperbolic SOP Memory Claim Readiness

This report separates engineering readiness from paper-grade geometry claims.

## Status Lights

- GREEN provenance: paper-grade clean provenance present
- GREEN coordinate quality: gate passed
- GREEN benchmark/gold validation: passed
- GREEN ablation claim-grade inputs: ready
- RED hyperbolic geometry claim: not supported yet
- RED hyperbolic vs Euclidean memory claim: not supported yet
- YELLOW online pilot: not run by this offline evidence-chain script

## Current Interpretation

The tuned Poincare run did not pass the geometry gate. Poincare is usable after tuning, but the evidence supports agentic memory behavior rather than a hyperbolic-geometry-specific win.

The independent Euclidean-memory control was also run. This compares flat coordinates + Euclidean distance against hyperbolic coordinates + Poincare distance; it is separate from the same-coordinate Flat-Twin control.

## Key Numbers

- Hyper graph nodes/edges: 865 / 3580
- SOP source evidence: 281 / 281
- Provenance status: clean_certified
- Direction effective rank: 14.783039093017578
- Theta top-2 bin mass: 0.5338078291814946
- Neighbor coherence lift: 0.28327402135231317
- Benchmark queries: 160
- Benchmark by kind: {'rare_partial_clue': 40, 'abstract_failure': 40, 'minimal_context': 40, 'hard_method_set': 40}
- Benchmark split: {'dev': 80, 'test': 80}
- Benchmark query styles: {'partial_clue': 40, 'abstract_failure': 40, 'minimal_context': 40, 'method_partial': 40}
- Benchmark specificity: {'medium_low': 80, 'low': 80}
- Title-token overlap mean/max: 0.03202380952380953 / 0.5
- Title leakage levels: {'low': 155, 'medium': 5}
- Distractor count mean/min: 57.95 / 7
- Tuned Poincare params: {'geometry_distance_norm': 'minmax', 'geometry_distance_weight': 0.35, 'geometry_semantic_weight': 0.8, 'geometry_constraint_weight': 0.5, 'geometry_query_radius_quantile': 0.5}
- Tuning grid size: 432
- Near-best trial count: 1
- Tuned Poincare Rare Recall@5: 1.0
- Tuned Flat-Twin Rare Recall@5: 1.0
- Tuned Euclidean Memory Rare Recall@5: 1.0
- Tuned Poincare R@1 / MRR / NDCG@5: 0.39375 / 0.5045833333333334 / 0.5530917713684479
- Tuned Flat-Twin R@1 / MRR / NDCG@5: 0.41875 / 0.53 / 0.5783622854427838
- Tuned Euclidean Memory R@1 / MRR / NDCG@5: 0.40625 / 0.5238541666666666 / 0.5781141931065685
- Rare Recall@5 mean diff: 0.0
- Paired bootstrap p-value: 1.0
- Rare Recall paired query count: 40
- Condition Precision diff: -0.01625
- Poincare vs Flat-Twin MRR diff / p-value: -0.025416666666666664 / 0.9996000399960004
- Poincare/Flat-Twin top5 overlap: 0.8384672619047618
- Poincare vs Euclidean Rare Recall@5 diff: 0.0
- Poincare vs Euclidean paired bootstrap p-value: 1.0
- Poincare vs Euclidean Condition Precision diff: -0.0225
- Poincare vs Euclidean MRR diff / p-value: -0.01927083333333333 / 0.9571042895710429
- Poincare/Euclidean top5 overlap: 0.7564732142857143

## Guardrail

If Poincare only beats lexical retrieval but does not beat same-coordinate Flat-Twin, report agentic memory gains only. Do not claim the hyperbolic geometry itself is responsible.

