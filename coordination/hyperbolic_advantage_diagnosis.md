# Hyperbolic Advantage Diagnosis

## Question

Why does the current SOP memory not show a Poincare advantage, and is the hyperbolic structure useless?

Short answer: hyperbolic geometry is not useless, but it helps mainly when the stored metric is tree-like, low-dimensional, and has many boundary leaves. The current SOP graph is middle-heavy, dense, TF-IDF-directed, and Poincare/Flat-Twin top-k overlap is very high.

## Synthetic Tree Reproduction

Large tree: branching=4, depth=5, nodes=1365, leaf_fraction=0.750.

| System | Corr with Tree Distance | Relative Stress | Neighbor Recall@10 |
|---|---:|---:|---:|
| Poincare 2D tree layout | 0.8647 | 0.0918 | 0.5521 |
| Euclidean same coordinates | 0.6035 | 0.1457 | 0.4570 |

This reproduces the canonical advantage condition: the same radial/angular coordinates preserve tree distances much better when measured with Poincare distance than with Euclidean distance.

## Euclidean Dimensionality Check

Smaller tree for MDS: branching=3, depth=5, nodes=364, leaf_fraction=0.668.

| System | Corr with Tree Distance | Relative Stress | Neighbor Recall@10 |
|---|---:|---:|---:|
| poincare_2d_tree_layout | 0.8850 | 0.1128 | 0.6646 |
| flat_twin_euclidean_same_coords | 0.6419 | 0.1858 | 0.5170 |
| euclidean_mds_2d | 0.7868 | 0.1496 | 0.2992 |
| euclidean_mds_8d | 0.9382 | 0.0839 | 0.5854 |
| euclidean_mds_16d | 0.9726 | 0.0563 | 0.6558 |

Interpretation: hyperbolic 2D beats Euclidean 2D/same-coordinate on tree geometry, but sufficiently high-dimensional Euclidean MDS can catch up or exceed it. This is why our 16D SOP setup should not be expected to show a free win unless the graph is strongly hierarchical and the query actually uses that hierarchy.

## Current SOP Graph Mismatch

- SOP count: 281
- Radius bands: {'core': 56, 'edge': 23, 'middle': 202}
- Edge-band SOP fraction: 0.082
- SOP-SOP edges: 2326 vs tree baseline 280 edges
- SOP-SOP edge density relative to a tree: 8.31x
- Average SOP-SOP degree: 16.56
- SOP-SOP edge kinds: {'co_occur': 1001, 'conflicts_with': 91, 'enhance': 558, 'prereq': 367, 'refines': 309}

## Current Edge Retrieval Diagnostics

- Query-aware status: coordinate_quality_null
- Poincare/Flat-Twin edge top-5 overlap: 0.9415204678362573
- Gold edge pressure: 1.0
- Selected edge rate by system: {'agentic_euclidean': 0.3894736842105263, 'agentic_flat_twin': 0.3929824561403509, 'agentic_lexical': 0.41754385964912283, 'agentic_poincare': 0.3649122807017544}
- Edge Recall@5 diff Poincare-FlatTwin: 0.0 with p=1.0

## SOP Tree Slice Follow-Up

I also forced the current edge SOPs into a clean project-specific tree slice: `root -> task -> edge_reason -> SOP`. This is closer to the paper thesis than a generic synthetic tree, but still uses current auto-seeded edge SOP labels.

- Unique edge SOP leaves: 19
- Tasks: 3
- Reason nodes: 10
- Total tree nodes: 33

| Metric | Poincare | Flat-Twin Euclidean |
|---|---:|---:|
| Corr with tree distance | 0.7160 | 0.6322 |
| Relative stress | 0.2719 | 0.3017 |

Branch retrieval still ties:

| Query Level | Poincare R@5 | Flat R@5 | Diff | Poincare MRR | Flat MRR | Diff |
|---|---:|---:|---:|---:|---:|---:|
| reason_parent | 0.7895 | 0.7895 | 0.0000 | 0.6489 | 0.6489 | 0.0000 |
| task_parent | 0.6842 | 0.6842 | 0.0000 | 0.3796 | 0.3796 | 0.0000 |

Interpretation: Poincare can encode this small SOP tree slightly better than Euclidean distance, but the slice has too few leaves and the branch-retrieval task is saturated/under-specified, so the geometric advantage still does not convert into better SOP retrieval.

## Diagnosis

1. The literature advantage is a geometry-match advantage, not a magic retrieval bonus. It appears when the data metric looks like an exponentially branching tree.
2. Our current SOP graph is not tree-like enough: it has only a small edge band and many SOP-SOP co-occur/enhance/prereq/refines edges, so it is much denser than a tree.
3. Query routing still does not push hard edge queries far enough outward; on edge-only gold, Poincare selected edge SOPs at a much lower rate than the gold pressure.
4. The direction model is TF-IDF-SVD fallback, not sentence embedding or contrastive projection; short abstract failure clues therefore do not reliably point into the right angular sector.
5. Because Poincare and Flat-Twin retrieve almost the same top-5, the experiment is geometry-null: the distance function is not being given a structurally different candidate frontier.
6. Even after forcing a small SOP tree, the advantage appears only as distance preservation, not retrieval. This suggests the next useful test needs a larger boundary-heavy SOP hierarchy, not another scorer tweak.

## Next Reproduction Target

To make the SOP experiment resemble the successful literature setting, build a small claim-grade slice with explicit Skill -> family -> condition -> edge SOP tree labels, train/derive angular sectors from those labels or sentence embeddings, force edge queries to evaluate boundary retrieval, and then rerun Poincare vs Flat-Twin. If Poincare still ties there, the thesis is in real trouble; if it wins only there, the paper claim must be scoped to tree-like procedural memory.
