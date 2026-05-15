---
title: 完全冻结backbone只训练head效果最差
confidence: HIGH
evidence: [Run8 top17(0.3853)]
---

# 完全冻结backbone只训练head效果最差

Run8 top17 冻结了 DeBERTa 全部参数，只训练注意力分类头，Log Loss 0.3853。

## 原因

- 预训练模型的任务与作者识别不完全匹配，backbone 的表示需要至少部分调整
- 只训练随机初始化的头，相当于用一个线性/浅层模型在固定特征上学习，无法捕捉任务特有模式
- 这本质上等于"特征提取 + 浅层分类器"，远不如微调

## Actionable Guidance

- 至少解冻后 4~6 层
- 完全冻结 = 放弃了微调的核心优势
- 如果显存不够全模型训练，用部分解冻而非全冻结

**Condition**: 任何微调场景
**Confidence**: HIGH
