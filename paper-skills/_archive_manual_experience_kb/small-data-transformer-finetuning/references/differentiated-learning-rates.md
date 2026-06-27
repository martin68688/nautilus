---
title: 差异化学习率（backbone 2e-5, head 5e-5）保护预训练层
confidence: MEDIUM
evidence: [Run8 top1, top2]
---

# 差异化学习率（backbone 2e-5, head 5e-5）保护预训练层

backbone 解冻层用较小学习率(2e-5)，分类头用较大学习率(5e-5，2.5倍)。新初始化的分类头需要更快学习，而预训练层只需微调。

## Actionable Guidance

```python
optimizer = AdamW([
    {"params": backbone_unfrozen_params, "lr": 2e-5, "weight_decay": 0.01},
    {"params": head_params, "lr": 5e-5, "weight_decay": 0.01},
])
```

- backbone: 2e-5 (或 1e-5 ~ 3e-5 范围)
- head: 5e-5 (或 3e-5 ~ 1e-4 范围)
- head lr = 2~3x backbone lr

**Condition**: 部分解冻微调时
**Confidence**: MEDIUM
