# Progressive Unfreezing with Discriminative Learning Rates

## When to Use

- Dataset is small (< 10k samples) and overfitting is dominant risk.
- Full fine-tuning from epoch 1 causes training loss to drop while validation plateaus.

**Warning:** Do NOT combine freezing with mixed precision (autocast) — causes NaN. Either disable AMP or keep all layers trainable with discriminative LRs.

## Unfreezing Schedule

| Phase | Epochs | Layers Unfrozen | Rationale |
|-------|--------|-----------------|----------|
| 1 | 1–2 | Classifier head only | Adapt random head without disturbing pretrained weights |
| 2 | 3–4 | Last 2 transformer layers + head | Gradually adapt highest-level features |
| 3 | 5+ | All layers + head | Full fine-tuning with discriminative LRs |

## Discriminative LR Assignment

```
base_lr = 2e-5  # for classifier head
layer_lr_decay = 0.85  # multiplicative decay from last layer to first
```

## Critical Implementation Notes

- **Disable gradient checkpointing** before unfreezing step (if memory permits).
- **Create optimizer once**, outside epoch loop, with parameter groups pre-configured.
- Use warmup phase (10% of total steps) before applying full discriminative LRs.
- Weight decay (0.01) on all non-bias/non-LayerNorm parameters.

## Two-Phase Retraining

Retraining on a second stratified split after initial phase can yield lower validation loss:
- Phase 1: Train on split A for strong initialization.
- Phase 2: Retrain on split B (different seed) from Phase 1 weights.