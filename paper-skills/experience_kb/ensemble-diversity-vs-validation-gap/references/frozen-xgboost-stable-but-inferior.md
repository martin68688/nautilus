---
title: 冻结Transformer+XGBoost方案稳定但测试集不如微调+集成
confidence: MEDIUM
evidence: [Run2 val=0.193 vs Run4 val=0.2013 测试集Run4更优]
---

# 冻结Transformer+XGBoost方案稳定但不如微调+集成

## 证据

| 方案 | 模型 | 特征 | 验证集 | 测试集排名 |
|------|------|------|--------|-----------|
| Run2 | 3个冻结Transformer+XGBoost | 3×768 CLS嵌入 | 0.193 | #3 |
| **Run4** | **DeBERTa微调+XGBoost+LR** | **嵌入+手工+n-gram** | **0.2013** | **#1** |

## Run2 方案详情

- 3个冻结模型：deberta-v3-base (184M) + roberta-base (125M) + distilbert-base-uncased (66M)
- 特征：3×768=2304维 [CLS]嵌入拼接
- 分类器：XGBoost (max_depth=4, lr=0.05, 5折CV)
- 优点：代码简洁(280行)、训练稳定、无需微调
- 缺点：特征单一(仅CLS嵌入)、模型冻结无法适应任务

## Run2 vs Run4 关键差异

1. **模型规模**：Run2用base模型，Run4用large模型(435M)——单模型更强
2. **特征多样性**：Run2仅CLS嵌入(2304维)，Run4有嵌入+30维风格+4维可读性+5维POS+n-gram+标点
3. **模型微调**：Run2完全冻结，Run4全参数微调(40 epochs)
4. **集成异构性**：Run2仅1个XGBoost，Run4有3种完全不同的模型

## Actionable Guidance

- 冻结方案适合快速baseline，但不应作为最终方案
- 如果用冻结策略，至少应该：冻结特征+手工特征拼接后再入XGBoost
- 微调DeBERTa-large > 冻结3个base模型集成
- 最优方案是：微调DeBERTa-large的嵌入 + 手工特征 → XGBoost + LR + DeBERTa概率集成

**Condition**: 小数据集NLP分类
**Confidence**: MEDIUM
