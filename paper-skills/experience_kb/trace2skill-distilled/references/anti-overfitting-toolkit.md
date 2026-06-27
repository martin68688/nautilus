# Anti-Overfitting Toolkit

## Table of Contents
1. [Incremental Integration](#incremental-integration)
2. [Technique Catalog](#technique-catalog)
3. [Recommended Order](#recommended-order)
4. [Regularization Defaults](#regularization-defaults)
5. [Detection and Remediation](#detection-and-remediation)

## Incremental Integration

Never generate multiple advanced techniques at once. Follow this cycle:
1. Add the technique to a clean, working pipeline.
2. `python -m py_compile` to catch syntax errors.
3. Run a 1-epoch sanity check on a small subset.
4. If it works, keep it; if it breaks, debug in isolation.
5. Commit before adding the next technique.

## Technique Catalog

### Multi-Sample Dropout
- **Purpose:** Stronger regularization, faster convergence.
- **Parameters:** K=8 samples, apply before classification head, average logits.

### Layer-wise Learning Rate Decay (LLRD)
- **Purpose:** Fine-grained control over layer adaptation.
- **Risk:** Custom parameter groups error-prone; print and verify assignments.

### Label Smoothing + Focal Loss
- **Purpose:** Reduces overconfidence; emphasizes hard examples.
- **Parameters:** ε=0.05–0.1, γ=2.0.

### Progressive Unfreezing
- **Purpose:** Gradually adapt pretrained weights.
- **Warning:** Disable gradient checkpointing before unfreezing. Do NOT combine with AMP (causes NaN).

### MLM Augmentation
- **Purpose:** Data augmentation via masked language model.
- **Parameters:** Replace 10–15% of words at ~30% probability per sample.

### Synonym Replacement
- **Purpose:** Expand small datasets.
- **Parameters:** ~0.5 probability, 3–5 words per text, preserve POS tags.

## Recommended Order

1. Base model + standard training loop (confirm it runs)
2. Weight decay + early stopping (low-risk, high-value)
3. Mixed precision (moderate risk, infrastructure-level)
4. Multi-sample dropout (moderate risk, model-level)
5. LLRD (higher risk, optimizer-level)
6. EDA augmentation (higher risk, data-level)

## Regularization Defaults for Small Datasets

| Parameter | Value |
|-----------|-------|
| Dropout | ≥ 0.20 (classifier + hidden) |
| Weight decay | ≥ 0.1 (AdamW) |
| Label smoothing | 0.1 |
| Learning rate | ≤ 2e-5 |
| Warmup ratio | 0.1–0.3 |
| Max epochs | 3–5 |
| Early stopping patience | 2–3 |

## Detection and Remediation

- Train high + val high → underfitting: reduce regularization, simplify signal.
- Train low + val high → overfitting: increase dropout, increase weight decay, reduce LR.
- If val loss rises > 0.05 above best epoch, halt immediately.

**NaN-safe model selection**: Initialize `best_model_state = None`, `best_val_score = float('inf')`. Only update when `math.isfinite(val_score) and val_score < best_val_score`.