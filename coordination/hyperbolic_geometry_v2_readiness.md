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
- Tuned Poincare Rare Recall@5: 0.9
- Tuned Flat-Twin Rare Recall@5: 0.9
- Tuned Euclidean Memory Rare Recall@5: 0.925
- Tuned Poincare R@1 / MRR / NDCG@5: 0.3875 / 0.46697916666666667 / 0.5016032300099944
- Tuned Flat-Twin R@1 / MRR / NDCG@5: 0.3625 / 0.4535416666666666 / 0.4902325422468078
- Tuned Euclidean Memory R@1 / MRR / NDCG@5: 0.35625 / 0.4552083333333334 / 0.49599765430209863
- Rare Recall@5 mean diff: 0.0
- Paired bootstrap p-value: 1.0
- Rare Recall paired query count: 40
- Condition Precision diff: 0.0025000000000000005
- Poincare vs Flat-Twin MRR diff / p-value: 0.013437500000000002 / 0.010198980101989802
- Poincare/Flat-Twin top5 overlap: 0.9395833333333332
- Poincare vs Euclidean Rare Recall@5 diff: -0.025
- Poincare vs Euclidean paired bootstrap p-value: 1.0
- Poincare vs Euclidean Condition Precision diff: -0.025
- Poincare vs Euclidean MRR diff / p-value: 0.011770833333333335 / 0.1426857314268573
- Poincare/Euclidean top5 overlap: 0.8291666666666668

## Guardrail

If Poincare only beats lexical retrieval but does not beat same-coordinate Flat-Twin, report agentic memory gains only. Do not claim the hyperbolic geometry itself is responsible.

