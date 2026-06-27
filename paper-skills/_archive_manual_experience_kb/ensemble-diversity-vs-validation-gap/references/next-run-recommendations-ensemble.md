---
title: 下次运行建议：以异构集成为起点
confidence: HIGH
evidence: [Run4集成模板已替换coldstart, 覆盖所有历史insights]
---

# 下次运行建议

## 当前coldstart模板

Run4 的 DeBERTa微调+XGBoost+LR集成方案已替换coldstart NLP Code_template。
该模板在真实测试集上表现最优，包含所有验证过的有效策略。

## 优先级1：多折训练降低验证集过拟合（预期提升显著）

当前模板用单次train_test_split(0.1)，验证集过拟合风险高。

改进：
```python
# 替换单次划分为5折
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 每折训练一个完整的集成模型
# 测试预测 = 5个集成的概率平均
```

## 优先级2：增加LightGBM分支（预期小幅提升）

4模型集成：DeBERTa + XGBoost + LR + LightGBM

```python
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multiclass', num_class=3,
    random_state=42, verbose=-1,
)
lgb_model.fit(lgb_train_features, y_train_labels,
              eval_set=[(lgb_val_features, y_val_labels)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
```

LightGBM特征：用TF-IDF稀疏特征(与LR不同的特征子集)

## 优先级3：集成权重用贝叶斯优化

替代网格搜索，更高效地找到最优权重：
```python
from scipy.optimize import minimize

def objective(weights):
    w = weights / weights.sum()  # 归一化
    ensemble = w[0]*deberta + w[1]*xgboost + w[2]*lr + w[3]*lgb
    return compute_log_loss(y_val, ensemble)

result = minimize(objective, x0=[0.4, 0.3, 0.2, 0.1],
                  bounds=[(0.05, 0.9)]*4,
                  method='L-BFGS-B')
```

## 关键：保护集成代码不被diff破坏

1. coldstart模板生成的draft节点应lock=True
2. improve操作应创建新的单模型分支，不修改集成模板
3. 集成应在搜索后期(>50%时间)通过fusion操作构建
4. 如果必须修改集成代码，使用full rewrite而非diff模式

## 不要做的方向

| 方向 | 原因 | 证据 |
|------|------|------|
| 追求单模型验证集极低值 | 过拟合 | Run8 val=0.0725测试集差 |
| 去掉LR或XGBoost分支 | 降低多样性 | 集成>单模型 |
| 用diff修改集成代码 | 必然破坏 | Run4 Branch4 |
| ModernBERT | 0.34~0.35 | Run5/7 |
| 全参数微调(无集成) | 天花板0.26 | Run8 |
| 完全冻结backbone | 最差 | Run8 top17 |

## 预期结果

- 优先级1(多折)：Log Loss 预期降低 10-20%
- 优先级2(LightGBM)：Log Loss 预期降低 2-5%
- 优先级3(贝叶斯优化)：Log Loss 预期降低 1-3%
- 三者结合：预期比当前最优再提升 15-25%
