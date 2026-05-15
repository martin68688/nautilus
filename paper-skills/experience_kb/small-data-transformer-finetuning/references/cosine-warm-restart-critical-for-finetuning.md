---
title: CosineAnnealingWarmRestarts + Linear Warmup 是微调调度器的最佳选择
confidence: HIGH
evidence: [Run8 top1 vs top2, Run8 top7 vs top1]
---

# CosineAnnealingWarmRestarts + Linear Warmup 是微调调度器的最佳选择

同一分支内，有 CosineWarmRestart 调度器(0.0725) vs 无调度器(0.1457)，差距 2 倍。

## 三种调度器对比

| 调度器 | 最佳成绩 | 特点 | 来源 |
|--------|---------|------|------|
| **CosineAnnealingWarmRestarts + 10% Warmup** | **0.0725** | 周期重启跳出局部最优，warmup保护预训练权重 | Run8 top1 |
| LinearWarmup + Linear Decay | 0.2631 | 稳定但缺乏探索，后期学习率过低无法精细调整 | Run8 top3 |
| 无调度器（恒定LR） | 0.1457 | 训练不稳定，无法精细收敛 | Run8 top2 |
| CosineAnnealing（无Warmup，全参数微调） | 0.2924 | 缺少warmup导致初期不稳定 | Run8 top7 |

## 关键机制

1. **Linear Warmup (10%)**：学习率从0线性上升到目标值，避免训练初期大梯度破坏预训练权重
2. **CosineAnnealing**：学习率周期性衰减，后期小学习率精细调整
3. **WarmRestarts**：周期性重启学习率，帮助跳出局部最优，探索更好的参数空间

## Actionable Guidance

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
)

# In training loop:
if current_step < warmup_steps:
    warmup_factor = current_step / max(1, warmup_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = initial_lr * warmup_factor
else:
    scheduler.step(epoch + current_step / len(train_loader))
```

**Condition**: 微调预训练 Transformer 时，尤其配合部分解冻
**Confidence**: HIGH
