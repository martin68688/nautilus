# Architecture Patterns for Small-Data Text Classification

## Multi-Pooling Head

Extract three complementary representations:

1. **CLS token** — captures global sequence-level information.
2. **Mean pooling** — averages all token embeddings.
3. **Attention-weighted pooling** — learns per-token importance.

```python
class MultiPoolingHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        self.cls_mlp = MLP(hidden_size, hidden_size // 2, dropout)
        self.mean_mlp = MLP(hidden_size, hidden_size // 2, dropout)
        self.attn_mlp = MLP(hidden_size, hidden_size // 2, dropout)
        self.attn_pool = AttentionPool(hidden_size)
        self.classifier = nn.Linear(hidden_size // 2 * 3, num_classes)

    def forward(self, hidden_states, attention_mask):
        cls_repr = self.cls_mlp(hidden_states[:, 0])
        mean_repr = self.mean_mlp(mean_pool(hidden_states, attention_mask))
        attn_repr = self.attn_mlp(self.attn_pool(hidden_states, attention_mask))
        combined = torch.cat([cls_repr, mean_repr, attn_repr], dim=-1)
        return self.classifier(combined)
```

## When to Use Multi-Pooling vs Single Pooling

| Scenario | Recommended Pooling |
|---|---|
| Short text, stylistic signal | Multi-pooling |
| Long documents, semantic classification | CLS or mean |
| Very small dataset (< 500 samples) | Single (CLS or mean) |

### Overfitting Check

Multi-pooling adds ~3× head parameters. On datasets smaller than ~2k samples,
verify via OOF log loss that multi-pooling outperforms single CLS pooling.
