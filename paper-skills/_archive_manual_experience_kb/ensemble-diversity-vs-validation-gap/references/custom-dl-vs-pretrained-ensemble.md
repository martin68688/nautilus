---
title: 定制深度学习vs预训练集成：验证集好但测试集不如
confidence: MEDIUM
evidence: [Run1 val=0.1859 vs Run4 val=0.2013, Run4测试集更优]
---

# 定制深度学习 vs 预训练集成

## 证据

| 方案 | 架构 | 代码行数 | 验证集 | 测试集排名 |
|------|------|---------|--------|-----------|
| Run1 | TF-IDF+Style+CharCNN+CrossAttn+PerAuthorAttn | 1131 | 0.1859 | #2 |
| **Run4** | **DeBERTa微调+XGBoost+LR集成** | **580** | **0.2013** | **#1** |

## Run1 方案详情

- 3路特征：TF-IDF(13000维) + 风格统计(40+维) + 字符CNN(48维嵌入+4卷积核)
- CrossAttentionFusion + PerAuthorAttention
- StochasticDepth + OneCycleLR + 高Dropout(0.6)
- 伪标签再训练(置信度>0.8)

## 为什么Run4测试集更优

1. **预训练知识**：DeBERTa在海量文本上预训练，语义理解远超从头训练的TF-IDF分支
2. **集成多样性**：Run1是单模型(虽复杂)，Run4是3种异构模型集成
3. **过拟合风险**：Run1的1131行定制代码在小数据上更容易过拟合，验证集0.1859可能已过拟合
4. **泛化瓶颈**：TF-IDF+CharCNN不能捕获深层语义，DeBERTa可以

## Run1的优点（不应忽视）

- 风格特征设计非常详细（40+维度，包括n-gram熵、句子统计、标点二元组等）
- 验证集确实比Run4好（0.1859 vs 0.2013）
- 伪标签再训练是一个有价值的策略

## Actionable Guidance

- **最佳方案 = 预训练语义 + 定制风格特征 + 集成**：把Run1的风格特征接入Run4的XGBoost分支
- 不要从零训练语义理解组件，应使用预训练模型
- 定制架构的复杂度应控制：单模型<500行，集成<800行
- CharCNN在这种短文本任务上贡献有限，不如字符n-gram(LR分支)

**Condition**: 文本分类，小数据集
**Confidence**: MEDIUM
