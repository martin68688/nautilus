---
title: 风格特征提取代码容易变成死代码，需确保数据流完整接入模型
confidence: HIGH
evidence: [Run8 top1: 340行特征代码未接入模型]
---

# 风格特征提取代码容易变成死代码

Run8 top1 包含 340 行特征提取代码（基础统计、词汇、可读性、情感、TF-IDF），这些特征被计算、缩放、保存到 parquet，但**从未输入模型**。数据流在 `train_set.index.tolist()` 处断裂——只取了行索引，回到原始文本。

## 典型断裂模式

```
特征提取 → all_features → train_set.parquet  ← 存盘后无人读取
                                    ↓
train_set.index  ← 只取行号！
        ↓
train_texts_orig[index]  ← 回到原始文本
        ↓
SpookyDataset(texts, labels, tokenizer)  ← 无特征输入
        ↓
model(input_ids, attention_mask)  ← forward 无特征参数
```

## Actionable Guidance

- 如果代码有特征提取，**必须**验证 forward 方法签名包含特征参数
- Dataset 类必须接受并返回特征 tensor
- 添加断言：`assert feature_tensor is not None, "Features not passed to model"`
- 或者：如果不使用特征，就**删除**特征提取代码，避免浪费运行时间（top1 浪费了 ~3分钟）

**Condition**: 任何同时包含特征提取和深度学习模型的代码
**Confidence**: HIGH
