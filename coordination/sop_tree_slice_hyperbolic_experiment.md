# SOP Tree Slice Hyperbolic Experiment

## Setup

This diagnostic forces the current edge SOPs into a clean tree slice: `root -> task -> edge_reason -> SOP`.
It checks whether Poincare begins to separate from same-coordinate Euclidean distance when the SOP memory is made more tree-like.

- Unique edge SOP leaves: 19
- Tasks: 3
- Reason nodes: 10
- Total tree nodes: 33

## Distance Preservation

| Metric | Poincare | Flat-Twin Euclidean |
|---|---:|---:|
| Corr with tree distance | 0.7160 | 0.6322 |
| Relative stress | 0.2719 | 0.3017 |

## Branch Retrieval

| Query Level | Poincare R@5 | Flat R@5 | Diff | Poincare MRR | Flat MRR | Diff |
|---|---:|---:|---:|---:|---:|---:|
| reason_parent | 0.7895 | 0.7895 | 0.0000 | 0.6489 | 0.6489 | 0.0000 |
| task_parent | 0.6842 | 0.6842 | 0.0000 | 0.3796 | 0.3796 | 0.0000 |

## Interpretation

If Poincare wins distance preservation here but does not win branch retrieval, the geometry can encode the tree but the query/gold task is under-specified or saturated.
If Poincare does not even win distance preservation on this tree slice, the constructed SOP hierarchy is too small/unbalanced to reproduce the literature advantage.

