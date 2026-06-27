---
title: 异构集成在真实测试集上最优，验证集指标可能严重误导
confidence: HIGH
evidence: [Run4 val=0.2013 测试集最优 vs Run8 val=0.0725 测试集较差]
---

# 异构集成在真实测试集上最优

## 证据

| 方案 | 验证集 Log Loss | 真实测试集排名 | 集成类型 |
|------|----------------|---------------|---------|
| **Run4: DeBERTa微调+XGBoost+LR** | 0.2013 | **#1** | 异构3模型 |
| Run1: MultiInputClassifier | 0.1859 | #2 | 单模型 |
| Run2: 冻结Transformer+XGBoost | 0.193 | #3 | 弱异构 |
| Run8: DeBERTa部分解冻 | 0.0725 | 较差 | 单模型 |

## 原因

- **异构模型错误去相关**：DeBERTa捕捉语义(可能过拟合验证集风格)，XGBoost基于风格统计特征(泛化更好)，LR基于n-gram模式(第三种信息源)
- **验证集偏差**：单模型可以"记住"验证集的特定模式，导致验证集极低但测试集差距大
- **集成自带正则化**：多个模型投票天然限制了任何单一模型的过拟合倾向

## 方案对比

Run4 的集成方案代码(infer_0509_185008_0201.py)包含：
1. **DeBERTa-v3-large 全参数微调**：max_length=512, 40 epochs, label_smoothing=0.1, early stopping
2. **XGBoost**：基于DeBERTa嵌入(1024维) + 30维风格特征 + 4维可读性特征 + 5维POS特征
3. **Logistic Regression**：基于多尺度字符n-gram(2-4, 4-6, 5-7) + 词n-gram(1-3) + 标点序列
4. **网格搜索权重优化**：步长0.05搜索3个模型的融合权重

## Actionable Guidance

- **永远做异构集成**，即使单模型验证集更好
- 集成应包含至少3种根本不同类型的模型：深度语义 + 树模型统计 + 线性模型模式
- 不要仅凭验证集指标选择最终方案，要看集成多样性
- 验证集 Log Loss 低于 0.1 的单模型方案应警惕过拟合

**Condition**: NLP分类任务，尤其是小数据集
**Confidence**: HIGH
