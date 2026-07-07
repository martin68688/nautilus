# Hyperbolic SOP Memory Claim Readiness

This report separates engineering readiness from paper-grade geometry claims.

## Status Lights

- GREEN provenance: paper-grade clean provenance present
- GREEN coordinate quality: gate passed
- GREEN benchmark/gold validation: passed
- GREEN ablation claim-grade inputs: ready
- RED hyperbolic geometry claim: not supported yet
- YELLOW online pilot: not run by this offline evidence-chain script

## Current Interpretation

The ablation did not pass the geometry gate. It may still show agentic memory helps, but not that hyperbolic geometry itself helps.

## Key Numbers

- Hyper graph nodes/edges: 865 / 3580
- SOP source evidence: 281 / 281
- Provenance status: clean_certified
- Direction effective rank: 14.783039093017578
- Theta top-2 bin mass: 0.5338078291814946
- Neighbor coherence lift: 0.28327402135231317
- Benchmark queries: 40
- Benchmark by kind: {'rare_condition': 10, 'debug_failure': 10, 'conflict_risk': 10, 'method_set': 10}
- Rare Recall@5 mean diff: -0.725
- Paired bootstrap p-value: 1.0
- Condition Precision diff: -0.1
- Poincare/Flat-Twin top5 overlap: 0.6904761904761904

## Guardrail

If Poincare only beats lexical retrieval but does not beat same-coordinate Flat-Twin, report agentic memory gains only. Do not claim the hyperbolic geometry itself is responsible.

