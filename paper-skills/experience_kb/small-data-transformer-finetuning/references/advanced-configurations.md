# Advanced Fine-Tuning Configurations

## Scheduler Configuration

**Problem**: Starting with full learning rates causes gradient overshoot in early epochs.

**Solution**: Linear warmup (2 epochs) → cosine annealing to zero.

Use **differential learning rates**: backbone LR ≈ 1e-5, head LR ≈ 1e-4 (5–10× higher).

## Attention + Mean Pooling Head

```python
class AttentionMeanHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_states):
        attn_weights = softmax(self.attention(hidden_states), dim=1)
        attn_pooled = (hidden_states * attn_weights).sum(dim=1)
        mean_pooled = hidden_states.mean(dim=1)
        fused = attn_pooled + mean_pooled
        return self.classifier(self.dropout(fused))
```

## Label Smoothing

**Recommendation**: Use **0.05** label smoothing.

- **0.1** (too aggressive): prevents sufficient fitting; validation log loss plateaus.
- **0.05** (recommended): mild regularization while allowing confident predictions.
- **0.0**: no regularization benefit; higher overfitting risk on small datasets.
