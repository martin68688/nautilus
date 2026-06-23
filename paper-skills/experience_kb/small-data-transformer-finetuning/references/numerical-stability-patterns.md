# Numerical Stability Patterns for Mixed-Precision Fine-tuning

## fp16 Range Constraints

| Precision | Max Value | Notes |
|-----------|-----------|-------|
| float16   | ~65504    | Constants must stay within range under autocast |
| float32   | ~3.4e38   | Full precision |

Constants that **will crash** under fp16: `-1e9`, `1e9`, `-1e10`.

## Safe Attention Masking

```python
# Option A: -inf (PyTorch softmax handles correctly)
attn_scores = attn_scores.masked_fill(attention_mask == 0, -float('inf'))

# Option B: safe constant
attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)

# Option C: cast to float32 before softmax
with torch.cuda.amp.autocast(enabled=False):
    scores_f32 = scores.float()
    attn_weights = torch.softmax(scores_f32, dim=-1).to(scores.dtype)
```

## NaN-Safe Training Loop

```python
best_state_dict = None
best_val_loss = float('inf')

for epoch in range(max_epochs):
    for batch in train_loader:
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = criterion(outputs, batch['labels'])
        if torch.isnan(loss):
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
```

## GradScaler Usage

```python
scaler = torch.cuda.amp.GradScaler()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # at most ONCE per cycle
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Never skip `GradScaler`** when using `autocast()`.

## Prediction Sanitization

```python
def sanitize_predictions(probs, n_classes):
    probs = np.asarray(probs, dtype=np.float64)
    bad_mask = ~np.isfinite(probs).all(axis=1)
    if bad_mask.any():
        probs[bad_mask] = 1.0 / n_classes
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs
```
