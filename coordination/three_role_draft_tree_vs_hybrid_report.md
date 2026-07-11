# Three-Role Draft: Tree-Only vs Stage Hybrid

Split: `test`; root episodes: 2.

The cold-start and replay slots are fixed. Only the novel_exploration retrieval condition changes.
All runs in the evaluated split are excluded from retrieval memory.

| Novel slot | Method MRR | Semantic method Recall@5 | Exact SOP Recall@5 | Task precision@5 | Cross-task contamination | Replay overlap@5 | Blocked positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tree-only | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 |
| Stage Hybrid | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.2000 | 0.0000 | 0.00 |

## Paired Comparisons

- `method_mrr`: delta `0.0000`, p-value `1.0000`.
- `method_recall_at_5`: delta `0.0000`, p-value `1.0000`.
- `task_precision_at_5`: delta `-0.2000`, p-value `1.0000`.

## Development Diagnostic

Dev has 3 episodes and is diagnostic only.

| Novel slot | Method MRR | Semantic method Recall@5 | Task precision@5 |
|---|---:|---:|---:|
| Tree-only | 0.3333 | 0.3333 | 1.0000 |
| Stage Hybrid | 0.0833 | 0.0667 | 0.8000 |

Claim allowed: `False`

## Limitations

- Cold-start and replay roles are fixed protocol slots; this offline test scores only the novel slot retrieval.
- Gold is a multi-SOP method family from all clean root Draft children, not one historical child node.
- All runs in the evaluated split are excluded from positive retrieval.
- A superiority claim requires at least 20 held-out queries; this split has 2.
- Draft root context does not identify one uniquely correct method, so semantic method-family recall is primary and exact SOP-ID recall is diagnostic only.
- Final generated code, training metric, and downstream adoption require an online three-role run.
