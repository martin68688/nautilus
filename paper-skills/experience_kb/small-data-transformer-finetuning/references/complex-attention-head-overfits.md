---
title: 复杂注意力头+多路池化在小数据上过拟合严重
confidence: MEDIUM
evidence: [Run8 Branch4 HybridStylo: 0.28~0.37]
---

# 复杂注意力头+多路池化在小数据上过拟合严重

Run8 Branch4 的 HybridStylo 架构（CLS+Mean+Max池化拼接→投影→注意力→MLP），5个变体全部在 0.28~0.37，无一进入 top3。

## 问题

- CLS+Mean+Max 拼接产生 3072 维向量，投影回 1024 维的 Linear 层有 ~300万参数
- MultiheadAttention 在 19K 样本上过拟合
- 多路池化增加的不是信息量，而是参数量和过拟合风险

## Actionable Guidance

- 不要使用多路池化拼接（CLS+Mean+Max），单用 CLS 即可
- 不要在分类头内加 MultiheadAttention
- 如果要融合特征，用独立的分支模型+集成，而非塞进一个模型

**Condition**: 训练样本 < 50K
**Confidence**: MEDIUM
