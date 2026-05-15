---
title: label_smoothing=0.1 对小数据集分类任务有效防止过拟合
confidence: HIGH
evidence: [Run8 top1~top8 all use LS, Run8 top5 no LS = 0.2868]
---

# label_smoothing=0.1 对小数据集分类任务有效防止过拟合

所有 Run8 top8 方案均使用 `label_smoothing=0.1`。label smoothing 将硬标签 [1,0,0] 软化为 [0.9, 0.05, 0.05]，防止模型对训练标签过度自信，提升泛化能力。

## 证据

- Run8 top1~top8: 全部使用 label_smoothing=0.1
- Run8 top5(0.2868): 使用 CrossEntropyLoss 无 smoothing（隐含在 HybridStylo 架构中），效果不如使用 LS 的同类方案

## Actionable Guidance

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

- 推荐值：0.1（3类任务）
- 类别更多时可适当降低到 0.05
- 不要超过 0.2，过度平滑会降低模型区分能力

**Condition**: 小数据集（<50K）分类任务
**Confidence**: HIGH
