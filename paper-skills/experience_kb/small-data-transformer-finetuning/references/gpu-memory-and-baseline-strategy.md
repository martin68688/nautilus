# GPU Memory Management and Baseline Strategy

## GPU Memory Tiers and Model Selection

| Free GPU Memory | Recommended Architecture | Max Batch Size |
|-----------------|--------------------------|----------------|
| < 2 GB          | TF-IDF + Logistic Regression (CPU) | N/A |
| 2–4 GB          | DistilBERT-base (66M)    | 8 |
| 4–8 GB          | DeBERTa-v3-base (140M)   | 8 |
| 8–12 GB         | DeBERTa-v3-base (140M)   | 16 |
| 12+ GB          | DeBERTa-v3-large (434M)  | 4–8 |

## OOM Recovery Patterns

```python
def train_with_oom_recovery(dataset, model, max_retries=3):
    batch_size = 16
    for attempt in range(max_retries):
        try:
            args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=max(1, 16 // batch_size),
                fp16=True,
                gradient_checkpointing=(attempt > 0),
                output_dir="./output",
            )
            trainer = Trainer(model=model, args=args, train_dataset=dataset)
            trainer.train()
            return trainer
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
```

## Incremental Complexity Workflow

### Phase 1: Guaranteed Baseline
- TF-IDF + Logistic Regression OR DistilBERT-base
- Goal: valid submission file

### Phase 2: Transformer Baseline
- DeBERTa-v3-base, fp16, gradient clipping

### Phase 3: Scale Up
- DeBERTa-v3-large (only if GPU memory permits)
- Add feature engineering incrementally

### Anti-Patterns
- Selecting DeBERTa-v3-large without checking GPU memory → instant OOM
- Combining 120+ features + large transformer on attempt 1 → crash in 36 seconds
