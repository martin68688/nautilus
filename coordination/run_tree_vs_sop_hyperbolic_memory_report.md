# Run-Tree vs SOP Hyperbolic Memory Diagnostic

## Bottom Line

当前证据支持你的直觉：项目里的真实运行记忆比蒸馏后的 SOP 外置记忆更像双曲空间擅长的对象。

- `journal.json` 里的 MLEvolve 搜索记录天然是 parent -> child 的树/森林。
- 当前 `GlobalMemoryLayer` 保存了节点经验和 parent metric/error，却没有保存 parent_id、depth、branch_id、children，所以跨 run 记忆把树压扁了。
- SOP 记忆有用，但它被蒸馏成稠密语义图：SOP-SOP 边远多于树基线，edge-band SOP 很少，因此不容易体现双曲几何优势。
- 在真实 run tree 上，同坐标下 Poincare 距离比欧氏距离更能保留树距离；这比当前 SOP retrieval 结果更符合双曲结构的经典优势条件。

## Project Evidence

- Journals scanned: `45`
- GlobalMemory record files: `26`
- GlobalMemory records: `462`
- Records with explicit tree topology keys: `0` / `462`
- Records with parent metric/error context: `462` / `462`

Interpretation: GlobalMemory already remembers useful experience, but not the tree shape that produced it.

## Real Run Tree Shape

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| nodes | 48.8889 | 52.0000 | 2.0000 | 110.0000 |
| leaf_fraction | 0.2886 | 0.2593 | 0.1111 | 0.6667 |
| max_depth | 10.3778 | 10.0000 | 1.0000 | 23.0000 |
| avg_depth | 4.2691 | 4.4000 | 0.5000 | 10.4182 |
| boundary_pressure | 0.1876 | 0.1603 | 0.0617 | 0.6667 |
| internal_avg_children | 1.3628 | 1.2857 | 1.0000 | 2.0000 |

## Distance Preservation on Real Run Trees

Same coordinates, only distance function changes.

| Metric | Poincare Mean | Flat-Twin Euclidean Mean | Poincare - Flat |
|---|---:|---:|---:|
| Corr with tree distance | 0.8204 | 0.6389 | 0.1815 |
| Relative stress lower better | 0.2504 | 0.3514 | -0.1010 |
| Neighbor Recall@10 | 0.7570 | 0.7036 | 0.0534 |

- Poincare corr win rate: `0.9545`
- Poincare stress win rate: `0.9778`

This means the geometry advantage appears at the run-tree carrier level, even though it did not appear in the current SOP retrieval benchmark.

## Run-Tree Retrieval Diagnostics

| Task | Poincare Mean | Flat-Twin Euclidean Mean | Poincare - Flat |
|---|---:|---:|---:|
| Parent lookup Recall@5 | 0.8772 | 0.8167 | 0.0604 |
| Parent lookup MRR | 0.7203 | 0.5202 | 0.2001 |
| Subtree leaf Precision@5 diff | n/a | n/a | -0.0509 |
| Child lookup Precision@5 diff | n/a | n/a | -0.0168 |

Important caveat: Poincare is clearly better for preserving lineage distance and finding parents, but it is not automatically better for every retrieval form. Naive subtree-leaf lookup is slightly worse here, so a real run-memory system should target lineage/backtracking/failure-recovery retrieval first, then tune descendant retrieval separately.

## SOP Memory Shape Contrast

- SOP count: `281`
- Radius bands: `{'core': 56, 'edge': 23, 'middle': 202}`
- Edge-band SOP fraction: `0.0819`
- SOP-SOP edges: `2326`
- Tree baseline for same number of SOPs: `280`
- SOP edge density vs tree: `8.3071x`
- Average SOP-SOP degree: `16.5552`
- Edge predicted-only status: `hyperbolic_geometry_claim_not_supported`
- Poincare/Flat-Twin top-5 overlap on edge slice: `0.9415`

Interpretation: current SOP memory behaves more like a dense semantic library than a branching tree. That is good for stable reusable advice, but weak for proving a hyperbolic geometry thesis.

## Largest Run Examples

| Run | Nodes | Leaves | Leaf Fraction | Max Depth | Poincare Corr | Flat Corr | Corr Diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260501_184944_openadmet-expansionrx | 110 | 15 | 0.1364 | 22 | 0.6029 | 0.4817 | 0.1212 |
| 20260509_154039_spooky-author-identification | 81 | 14 | 0.1728 | 20 | 0.6415 | 0.5921 | 0.0494 |
| 20260516_125444_spooky-author-identification | 81 | 21 | 0.2593 | 11 | 0.7977 | 0.2063 | 0.5914 |
| 20260517_132158_spooky-author-identification | 81 | 22 | 0.2716 | 19 | 0.6505 | 0.6133 | 0.0371 |
| 20260517_151325_spooky-author-identification | 81 | 29 | 0.3580 | 13 | 0.7264 | 0.3109 | 0.4154 |
| 20260627_135133_elite-peach-mayfly | 81 | 33 | 0.4074 | 13 | 0.6340 | 0.4852 | 0.1488 |
| 20260701_145201_denoising-dirty-documents | 81 | 9 | 0.1111 | 14 | 0.7810 | 0.5082 | 0.2728 |
| 20260701_145250_aerial-cactus-identification | 81 | 11 | 0.1358 | 21 | 0.6778 | 0.5440 | 0.1337 |
| 20260701_180146_leaf-classification | 81 | 9 | 0.1111 | 23 | 0.6351 | 0.4701 | 0.1650 |
| 20260510_025317_spooky-author-identification | 80 | 13 | 0.1625 | 21 | 0.6705 | 0.4597 | 0.2108 |

## Recommendation

Use a hybrid memory design:

1. Run/journal memory should be the main hyperbolic forest. Store nodes, parent-child transitions, depth, branch id, stage, metric delta, bug/error context, and local-best lineage.
2. SOP memory should remain as distilled procedural knowledge, but act more like landmarks/annotations/references attached to subtrees, not the only geometry-bearing object.
3. The next paper-grade geometry claim should compare run-tree retrieval: Poincare forest vs same-coordinate Flat-Twin vs independent Euclidean memory on parent/child, ancestor, sibling-branch, and failure-recovery retrieval tasks.
4. Distill SOPs from frequent successful subtrees or transition motifs after the run-tree memory is built, instead of forcing every SOP to be a primary hyperbolic point.

Plain metaphor: the run history is the actual family tree; SOP cards are the family recipes copied out afterward. Hyperbolic space is better at storing the family tree than the recipe box.
