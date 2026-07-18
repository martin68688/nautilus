# Flat-Twin Stage Hybrid retry

This diagnostic rerun keeps the frozen memory snapshot, candidate universe,
Tree/SOP channels, stage routing, RRF weights, taxonomy gates, and safety gates
unchanged. It changes only the distance function from Poincare to Flat-Twin
Euclidean distance over the identical coordinates.

| System | nDCG@10 | AP@10 | Unsafe escapes |
|---|---:|---:|---:|
| Stage Hybrid, Poincare (F11) | 0.4382 | 0.3519 | 0 |
| Stage Hybrid, Flat-Twin retry (D6) | 0.4431 | 0.3508 | 0 |
| SOP-only diagnostic (D3) | 0.5222 | 0.4256 | 0 |

Flat-Twin improves nDCG@10 over Poincare by 0.0049, while AP@10 is lower by
0.0011. This does not establish a meaningful overall improvement. Both hybrid
variants remain below SOP-only, so the main bottleneck is not hyperbolic
distance alone; Tree projection/fusion and evidence coverage remain the larger
diagnostic targets.

The result is diagnostic only: 70 of 120 episodes are scored because 50 are
explicit coverage-gap episodes, and the labels are the frozen silver labels.
It does not support a downstream training claim.
