# Attention Pooling Head and Training Configuration

## Attention Pooling Head

For style/authorship tasks, replace [CLS]-token-only pooling with attention
pooling over all token hidden states.

```python
class AttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, hidden_states):
        attn_weights = torch.softmax(
            self.attention(hidden_states).squeeze(-1), dim=-1
        )
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return self.ffn(pooled)
```

**Result**: Validation log loss of 0.0550 on a 3-class author identification task.

## Differential Learning Rate Parameter Groups

```python
param_groups = [
    {"params": backbone_params, "lr": 2e-5},
    {"params": head_params, "lr": 5e-5},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

## Scheduler Implementation

```python
from transformers import get_cosine_schedule_with_warmup

total_steps = num_epochs * steps_per_epoch
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
)
```

**Do NOT use `CosineAnnealingWarmRestarts`** — sudden LR spikes mid-training
destabilize fine-tuning.

## Additional Regularization

- **Label smoothing**: 0.05 recommended (0.1 only if overfitting is severe)
- **Dropout**: 0.2 in the classification head
- **Gradient clipping**: max norm 1.0
- **Mixed precision**: `torch.cuda.amp` or `fp16` in Trainer
- **Early stopping**: patience=4 on validation log loss
