# Transformer Fine-tuning Patterns

## Safe Optimizer Setup

```python
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
```

## Two-Phase Training Loop

### Phase 1: Frozen Encoder (1–2 epochs)
```python
for param in model.base_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW([p for p in model.classifier.parameters()], lr=1e-3)
```

### Phase 2: Full Fine-tuning
```python
for param in model.base_model.parameters():
    param.requires_grad = True
```

## Regularization Stack

| Technique | Purpose | Typical Value |
|-----------|---------|---------------|
| Label smoothing | Calibrated probabilities | 0.05–0.1 |
| Multi-sample dropout | In-model ensemble | K=4 |
| SWA | Flatter loss landscape | last 3-5 epochs |
| EMA weights | Stable inference | decay=0.999 |
| Mixed precision (AMP) | Memory savings | `torch.cuda.amp` |
| Gradient clipping | Training stability | max_norm=1.0 |

## Layer-wise Learning Rate Decay

- **Lower layers (embeddings):** 0.5x base LR
- **Middle layers:** 0.8x base LR
- **Top layers (classifier head):** 1.0x base LR

## Mixup Augmentation

- **Probability:** 50% of batches
- Interpolate both inputs and labels: `x_mixed = λ*x_i + (1-λ)*x_j`

## Numpy Type Serialization Safety

```python
author_mapping = {k: int(v) for k, v in author_mapping.items()}
metrics = {k: float(v) for k, v in metrics.items()}
```

**When to check**: Before any `json.dump`, `json.dumps`, or config-logging call.
