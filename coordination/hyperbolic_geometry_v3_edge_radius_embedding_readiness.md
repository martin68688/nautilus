# Hyperbolic Geometry V3 Readiness

This report focuses on the V3 diagnostic question: does Poincare distance help on edge SOP retrieval when radius hints are not gold-derived?

## Status Lights

- GREEN provenance: paper-grade clean provenance present
- GREEN edge benchmark: validated
- RED sentence/contrastive direction backend: tfidf_truncated_svd
- RED query-aware coordinate gate: coordinate_quality_null
- RED edge predicted-only geometry claim: coordinate_quality_null
- RED paper-grade V3 geometry claim: not allowed

## Embedding Backend

- Strict embedding experiment did not run: embedding_backend_unavailable: sentence embedding backend unavailable locally: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
- This is correct fail-closed behavior; do not label TF-IDF fallback as sentence embedding.

## Edge Benchmark

- Queries: 57
- By kind: {'edge_api_debug': 19, 'edge_shape_path': 19, 'edge_version_mismatch': 19}
- Split: {'test': 30, 'dev': 27}
- Title-token overlap mean/max: 0.027568922305764406 / 0.4
- Distractor count mean/min: 78.3157894736842 / 21
- Errors: []

## Radius Hint Ablation

- hard + use_gold_hint: status=coordinate_quality_null
  - radius_hint_modes=['use_gold_hint']
  - Poincare Edge Recall@5 / MRR / NDCG@5: 0.5652173913043478 / 0.3710144927536232 / 0.41917370365633483
  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: 0.5652173913043478 / 0.3731884057971014 / 0.42107908412758926
  - Edge Recall diff / p-value: 0.0 / 1.0
  - Edge Condition Precision diff: 0.008695652173913042
  - Edge MRR diff / NDCG diff: -0.0021739130434782605 / -0.0019053804712544124
  - Query-aware quality: coordinate_quality_null
- hard + predicted_only: status=coordinate_quality_null
  - radius_hint_modes=['predicted_only']
  - Poincare Edge Recall@5 / MRR / NDCG@5: 0.782608695652174 / 0.46521739130434786 / 0.5435803608801082
  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: 0.782608695652174 / 0.4688405797101449 / 0.5465944235725694
  - Edge Recall diff / p-value: 0.0 / 1.0
  - Edge Condition Precision diff: -0.008695652173913042
  - Edge MRR diff / NDCG diff: -0.0036231884057971006 / -0.003014062692461169
  - Query-aware quality: coordinate_quality_null
- edge + use_gold_hint: status=coordinate_quality_null
  - radius_hint_modes=['use_gold_hint']
  - Poincare Edge Recall@5 / MRR / NDCG@5: 1.0 / 0.8406432748538011 / 0.881403054276982
  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: 1.0 / 0.8377192982456141 / 0.8791060410564301
  - Edge Recall diff / p-value: 0.0 / 1.0
  - Edge Condition Precision diff: -0.007017543859649123
  - Edge MRR diff / NDCG diff: 0.0029239766081871348 / 0.0022970132205518846
  - Query-aware quality: coordinate_quality_null
- edge + predicted_only: status=coordinate_quality_null
  - radius_hint_modes=['predicted_only']
  - Poincare Edge Recall@5 / MRR / NDCG@5: 0.8771929824561403 / 0.6862573099415205 / 0.7337862126145659
  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: 0.8771929824561403 / 0.7251461988304093 / 0.7638547526015772
  - Edge Recall diff / p-value: 0.0 / 1.0
  - Edge Condition Precision diff: -0.0035087719298245606
  - Edge MRR diff / NDCG diff: -0.03888888888888889 / -0.030068539987011426
  - Query-aware quality: coordinate_quality_null
- hard + learned_predictor: status=coordinate_quality_null
  - radius_hint_modes=['learned_predictor']
  - Poincare Edge Recall@5 / MRR / NDCG@5: 0.6956521739130435 / 0.44710144927536233 / 0.5088087606843719
  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: 0.6956521739130435 / 0.4492753623188405 / 0.5107141411556263
  - Edge Recall diff / p-value: 0.0 / 1.0
  - Edge Condition Precision diff: 0.008695652173913042
  - Edge MRR diff / NDCG diff: -0.0021739130434782605 / -0.0019053804712544124
  - Query-aware quality: coordinate_quality_null
- edge + learned_predictor: status=coordinate_quality_null
  - radius_hint_modes=['learned_predictor']
  - Poincare Edge Recall@5 / MRR / NDCG@5: 0.6842105263157895 / 0.5552631578947368 / 0.5869916587614673
  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: 0.6842105263157895 / 0.5619883040935673 / 0.5924899111871431
  - Edge Recall diff / p-value: 0.0 / 1.0
  - Edge Condition Precision diff: 0.0
  - Edge MRR diff / NDCG diff: -0.006725146198830409 / -0.005498252425675836
  - Query-aware quality: coordinate_quality_null

## Interpretation Guardrail

V3 does not currently allow a paper-grade hyperbolic geometry claim. If Poincare improves over lexical but not Flat-Twin, report agentic memory or coordinate-quality gains only.

Main claim requires: edge benchmark + predicted_only + sentence/contrastive backend + Poincare Edge Recall@5 >= Flat-Twin + 5pp + paired-bootstrap p < 0.05 + Edge Condition Precision/MRR/NDCG not lower.

