# Training Stability and OOM Recovery

## Table of Contents
1. [Prediction Sanitization](#prediction-sanitization)
2. [AMP Gradient Safety](#amp-gradient-safety)
3. [NaN Detection and Guardrails](#nan-detection-and-guardrails)
4. [Gradient Clipping](#gradient-clipping)
5. [Device-Safe Validation Loop](#device-safe-validation-loop)
6. [BatchNorm Singleton-Batch Crash](#batchnorm-singleton-batch-crash)
7. [Smoke Test Template](#smoke-test-template)
8. [Staged Training Transition Checklist](#staged-training-transition-checklist)

## Prediction Sanitization

```python
# PyTorch tensor outputs
probs = torch.softmax(logits, dim=-1)
probs = torch.nan_to_num(probs, nan=1.0/n_classes, posinf=1.0, neginf=0.0)

# NumPy outputs
val_probs = np.nan_to_num(val_probs, nan=1.0/n_classes, posinf=1.0, neginf=0.0)
val_probs = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)  # renormalize
```

## AMP Gradient Safety

```python
scaler = GradScaler()

optimizer.zero_grad()
with autocast():
    loss = criterion(model(batch), batch.labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # unscale ONCE
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)  # auto-skips if NaN/Inf detected
scaler.update()
```

**NEVER call `scaler.unscale_()` twice** between `scaler.update()`. If NaN persists, disable AMP.

## NaN Detection and Guardrails

```python
# Check loss before backward
if torch.isnan(loss) or torch.isinf(loss):
    print(f"NaN/Inf loss at step {step}, epoch {epoch}")
    optimizer.zero_grad()
    continue  # or break

# Check gradients after backward
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
        param.grad.zero_()
```

## Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Apply BEFORE `optimizer.step()`, AFTER `loss.backward()`.

## Device-Safe Validation Loop

```python
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # CRITICAL
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
```

**Never pass a CPU tensor to a GPU model or criterion.**

## BatchNorm Singleton-Batch Crash

Set `drop_last=True` on training DataLoader, or replace BatchNorm with LayerNorm.

## Smoke Test Template

```python
def smoke_test(model, tokenizer, dataset, device, batch_size=4):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    small_batch = dataset[:batch_size]
    try:
        outputs = model(**small_batch, labels=small_batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Smoke test PASSED. Loss: {loss.item():.4f}")
        return True
    except torch.cuda.OutOfMemoryError:
        print("FAILED: CUDA OOM")
        return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False
```

## Staged Training Transition Checklist

Before launching full training with frozen→unfrozen stages:
- [ ] Optimizer reinitialized for Stage 2 with appropriate LR
- [ ] LR scheduler reset
- [ ] `requires_grad` toggled correctly
- [ ] No dangling autograd graph (`model.zero_grad()` at boundary)
- [ ] Gradient checkpointing compatible with Stage 2 loop (no accumulation)

Run a dry-run (1 epoch per stage on tiny subset) before committing to full schedule.