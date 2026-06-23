# Overfitting Prevention for Small-Data Transformer Fine-tuning

## Selective Layer Freezing

Freeze the bottom ~33% of encoder layers to reduce memory usage and training time.

```python
num_layers = len(model.deberta.encoder.layer)
freeze_until = num_layers // 3
for i, layer in enumerate(model.deberta.encoder.layer):
    if i < freeze_until:
        for param in layer.parameters():
            param.requires_grad = False
```

## Multi-Regularization Stack

| Technique | Value | Failure Mode Addressed |
|---|---|---|
| Label smoothing | 0.05–0.1 | Overconfident predictions on hard examples |
| Dropout | 0.1–0.2 | Co-adaptation of hidden units |
| Early stopping | patience 3–5 on val log loss | Training too many epochs past optimum |
| Gradient clipping | max norm 1.0 | Gradient explosion destabilizing training |

## Common Pitfall: Output Object Structure

```python
# WRONG — causes runtime error
probs = torch.softmax(model(input_ids), dim=-1)

# CORRECT — extract .logits first
output = model(input_ids)
probs = torch.softmax(output.logits, dim=-1)
```
