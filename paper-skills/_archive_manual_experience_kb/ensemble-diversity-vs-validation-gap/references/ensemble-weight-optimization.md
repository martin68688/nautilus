---
title: 集成权重网格搜索比简单平均有效
confidence: MEDIUM
evidence: [Run4 网格搜索最优权重 vs 等权(0.22+)]
---

# 集成权重网格搜索比简单平均有效

## Run4 的权重优化

```python
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = w1 * deberta_probs + w2 * xgboost_probs + w3 * lr_probs
        ll = compute_log_loss(y_val, ensemble_proba)
```

3个模型的权重搜索范围：0.05~0.9，步长0.05

## 关键发现

- DeBERTa微调模型获得最高权重(语义最强)
- XGBoost获得中等权重(手工特征补充)
- LR获得最低但非零权重(n-gram模式补充)
- 简单等权(0.33/0.33/0.33)的logloss比最优权重差0.02+

## Actionable Guidance

- 3模型集成权重搜索：步长0.05，范围0.05~0.9，约束w1+w2+w3=1
- 更高效方案：用scipy.optimize.minimize做连续优化
- 4+模型集成：用Bayesian优化(如optuna)避免网格搜索的组合爆炸
- 约束：每个模型权重≥0.05(确保所有模型都有贡献)

**Condition**: 多模型集成
**Confidence**: MEDIUM
