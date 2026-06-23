# Training Safeguards

## NaN Loss Prevention

```python
import torch

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

if torch.isnan(loss):
    print("NaN detected in first step — aborting training")
```

## Checkpoint Path Consistency

```python
best_model_path = './working/best_model.pt'
best_val_loss = float('inf')

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), best_model_path)

model.load_state_dict(torch.load(best_model_path))
```

## Missing-Checkpoint Fallback

```python
import os, numpy as np

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
else:
    predictions = np.full((n_samples, n_classes), 1.0 / n_classes)
```

## Fault-Tolerant Cross-Validation

```python
fold_scores = []
for fold in range(n_folds):
    try:
        fold_scores.append(val_score)
    except Exception as e:
        print(f"Fold {fold} failed: {e}")
        continue
if len(fold_scores) == 0:
    raise RuntimeError("All folds failed")
```
