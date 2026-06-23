# Fine-Tuning Configuration Details

## Regularization Toolkit

| Technique | Value | Purpose |
|---|---|---|
| Label smoothing | epsilon=0.05 | Prevent overconfident predictions |
| Weight decay | 0.01–0.1 | L2 regularization |
| Gradient accumulation | 2 steps | Effective batch size increase |
| Gradient clipping | max_norm 0.5–1.0 | Tight clipping for small-data stability |
| Cosine annealing + SWA | T_max=epochs | Smooth LR decay + weight averaging |
| Mixed precision | enabled | Memory efficiency |
| Focal loss | gamma=2.0 | Focus on hard examples |

## Early Stopping

- Monitor validation log loss with patience=2–3.
- Best checkpoint is often within the first 2–3 epochs.
- Always save and return the best checkpoint, not the last.

## Splitting Strategy

- Use 80/20 or 85/15 stratified train/validation split.
- Stratification ensures validation log loss accurately reflects generalization.

## Gradient Checkpointing Conflicts

**DO NOT combine gradient checkpointing with gradient accumulation** — causes
"Trying to backward through the graph a second time" errors. If you must combine
them, set `use_reentrant=False`.

## Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
```
