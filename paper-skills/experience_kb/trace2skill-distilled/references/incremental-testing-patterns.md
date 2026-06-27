# Incremental Testing Patterns

## Table of Contents
1. [Baseline-First Strategy](#baseline-first-strategy)
2. [Adding Techniques One at a Time](#adding-techniques-one-at-a-time)
3. [Quick Sanity Check Protocol](#quick-sanity-check-protocol)
4. [Progressive Complexity Checklist](#progressive-complexity-checklist)

## Baseline-First Strategy

```python
# Step 1: TF-IDF + Logistic Regression baseline
vec = TfidfVectorizer(max_features=10000)
X_train = vec.fit_transform(train_texts)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
print(f"Baseline log_loss: {log_loss(val_labels, clf.predict_proba(vec.transform(val_texts)))}")
```

Checkpoint: Baseline must produce a validation log loss before proceeding.

## Adding Techniques One at a Time

1. Standard fine-tuning loop → verify runs and produces val loss.
2. Add learning rate scheduling → sanity check.
3. Enable AMP with gradient safety → sanity check.
4. Add gradient clipping → sanity check.
5. Add label smoothing → sanity check.
6. Add custom loss (Focal Loss) → sanity check.
7. Add SWA → sanity check.
8. Add gradient accumulation → sanity check.

After each addition: run 1–2 epochs on 10% of data. Verify loss decreases and no NaN/Inf.

## Quick Sanity Check Protocol

```python
def quick_sanity_check(model, val_loader, device, num_batches=5):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            assert logits.device == labels.device, "Device mismatch!"
            assert logits.shape[0] == labels.shape[0], "Batch size mismatch!"
            assert torch.isfinite(logits).all(), "NaN/Inf in logits!"
            loss = F.cross_entropy(logits, labels)
            assert torch.isfinite(loss), "NaN/Inf in loss!"
    print("Sanity check passed.")
```

## Progressive Complexity Checklist

### Phase 1: Baseline (must pass)
- [ ] Standard transformer with classification head
- [ ] No AMP, no gradient accumulation
- [ ] Full epoch completes without crash
- [ ] Validation log loss produces finite value

### Phase 2: Add optimizations one at a time
- [ ] LR scheduling (warmup + decay)
- [ ] AMP with gradient safety
- [ ] Gradient clipping
- [ ] Label smoothing

### Phase 3: Advanced techniques
- [ ] Feature engineering
- [ ] Partial layer freezing
- [ ] Custom LR groups
- [ ] Multi-modal architecture

**Rule:** If any addition causes instability, revert to last working configuration.