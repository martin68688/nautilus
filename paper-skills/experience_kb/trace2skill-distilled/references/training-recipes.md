# Training Recipes

## Table of Contents
1. [Focal Loss](#focal-loss)
2. [Word Dropout](#word-dropout)
3. [Gradient Accumulation](#gradient-accumulation)
4. [Selective SWA](#selective-swa)
5. [Multi-Sample Dropout](#multi-sample-dropout)
6. [Differential Learning Rates](#differential-learning-rates)

## Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()
```

## Word Dropout

```python
def word_dropout(input_ids, dropout_prob=0.05, pad_token_id=0):
    mask = (torch.rand_like(input_ids.float()) > dropout_prob) | (input_ids == pad_token_id)
    return input_ids * mask
```

## Gradient Accumulation

```python
accumulation_steps = 4
optimizer.zero_grad()
for step, batch in enumerate(train_loader):
    with torch.cuda.amp.autocast():
        loss = model(batch) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## Selective SWA

Average only top-K best-validation-loss checkpoints:
```python
def selective_swa(checkpoint_paths, val_losses, top_k=3):
    sorted_indices = sorted(range(len(val_losses)), key=lambda i: val_losses[i])
    best_indices = sorted_indices[:top_k]
    avg_state = None
    for idx in best_indices:
        state = torch.load(checkpoint_paths[idx], map_location='cpu')
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()
    for k in avg_state:
        avg_state[k] /= len(best_indices)
    return avg_state
```

## Multi-Sample Dropout

K=8 forward passes through dropout before classification head; average logits across all K.

## Differential Learning Rates

```python
param_groups = [
    {'params': transformer.parameters(), 'lr': 2e-5},
    {'params': classifier.parameters(), 'lr': 2e-4},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

**Backbone LR**: 5e-6 for very large models. **Task heads**: 5e-4. Create optimizer once, outside epoch loop.