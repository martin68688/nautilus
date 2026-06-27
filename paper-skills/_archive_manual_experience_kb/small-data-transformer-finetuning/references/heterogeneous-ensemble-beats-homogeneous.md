---
title: 异构集成（DeBERTa+XGBoost+LR）优于同构集成（3个Transformer）
confidence: MEDIUM
evidence: [Run4 top1 vs Run8 top9]
---

# 异构集成（DeBERTa+XGBoost+LR）优于同构集成（3个Transformer）

## 证据

| 集成类型 | 组成 | Log Loss | 来源 |
|---------|------|----------|------|
| **异构** | DeBERTa + XGBoost + LR | **0.2013** | Run4 top1 |
| 同构 | DeBERTa-large + DeBERTa-small + DistilBERT | 0.3016 | Run8 top9 |
| 混合 | DeBERTa+Stylo + LightGBM | 0.3061 | Run8 top10 |

## 原因

- **异构模型互补性强**：深度学习捕捉语义，XGBoost/LR基于统计特征，错误模式不同
- **同构模型高度相关**：三个Transformer学到的特征模式相似，集成增益有限
- **异构集成性价比高**：XGBoost/LR训练快，不占GPU

## Actionable Guidance

- 集成应选择根本不同类型的模型：NN + 树模型 + 线性模型
- 不要集成多个同架构不同规模的 Transformer
- 集成权重用网格搜索优化，步长 0.05

**Condition**: 追求极限性能时
**Confidence**: MEDIUM
