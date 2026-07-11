# Stage-Aware SOP Gateway + RunForest Readiness

Held-out split: `test`; queries: 56.

| Control | Gateway MRR | Transition R@5 | Execution MRR | Evidence precision | Blocked positive |
|---|---:|---:|---:|---:|---:|
| no_memory | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| sop_only | 0.3899 | 0.1250 | 0.0500 | 0.0438 | 0.00 |
| tree_only | 0.0000 | 0.0000 | 0.3741 | 0.0000 | 0.00 |
| naive_concat | 0.3899 | 0.1250 | 0.1151 | 0.0438 | 0.00 |
| stage_hybrid | 0.3899 | 0.1250 | 0.3670 | 0.0438 | 0.00 |
| flat_twin_hybrid | 0.3899 | 0.1250 | 0.3709 | 0.0438 | 0.00 |
| independent_euclidean | 0.3899 | 0.1250 | 0.3380 | 0.0438 | 0.00 |

## Stage Gates

| Stage | Queries | Best single channel | Hybrid MRR delta | p-value | Allowed |
|---|---:|---|---:|---:|---|
| debug | 31 | tree_only | -0.0065 | 0.5977 | False |
| draft | 6 | tree_only | -0.2320 | 1.0000 | False |
| evolution | 2 | tree_only | -0.0119 | 0.7526 | False |
| improve | 17 | tree_only | 0.0716 | 0.1489 | False |

## Overall Claim Gates

- `offline_retrieval_claim_allowed`: `False`
- `hyperbolic_geometry_claim_allowed`: `False`
- `online_downstream_claim_allowed`: `False`
- `adoption_precision_available`: `False`
- `downstream_metric_available`: `False`

## Geometry Comparisons

- Stage Hybrid minus `flat_twin_hybrid` Execution MRR: `-0.0039` (`p=0.5762`).
- Stage Hybrid minus `independent_euclidean` Execution MRR: `0.0290` (`p=0.1119`).

Offline retrieval results cannot establish adoption precision or downstream task improvement.
A paper-grade geometry claim additionally requires concurrent online controls and both geometry comparisons to pass.
