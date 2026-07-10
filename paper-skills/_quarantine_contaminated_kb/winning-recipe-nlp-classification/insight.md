---
category: winning-recipe-nlp-classification
source: mlevolve-evolution (Run 20260516_091845 top4, val=0.2517)
runs_analyzed: 12 (Run1~Run8, Run 0508, 0509_042918, 0509_185008, 20260516_091845)
---

# Insights: Winning Recipe for Small-Data NLP Classification

**Source**: MLEvolve 12次运行总结，以 Run 20260516_091845 top4 (val=0.2517) 为当前最优方案
**Task**: Spooky Author Identification (3-class, ~19K train samples, Log Loss metric)
**Core finding**: 小数据NLP分类的获胜配方 = 部分冻结微调 + 差异学习率 + 多尺度特征工程 + 异构多模型集成 + WeightedRandomSampler + Multi-Sample Dropout。每个组件缺一不可，单独优化任何一项都无法达到最优。

| # | Insight | Evidence | Confidence | File |
|---|---------|----------|------------|------|
| 1 | 部分冻结微调(后8层)+差异学习率(backbone 2e-5, head 5e-5)是小数据集DeBERTa微调的最佳策略 | Run8 top1(0.0725) vs 全参数微调天花板0.26 vs 全冻结0.3853 | HIGH | references/partial-unfreezing-differentiated-lr.md |
| 2 | Multi-Sample Dropout (K=4) 对分类任务有效：4次独立dropout取平均logits，等价于隐式集成，提升泛化且无额外推理成本 | Run 091845 top4 MSD(K=4) val=0.2517 vs 无MSD方案 val=0.2653 | HIGH | references/multi-sample-dropout-effective.md |
| 3 | 多尺度特征工程是异构集成的基石：stylo(30维)+readability(4维)+POS(5维)+n-gram(13500维→chi2 10000维)各有分工 | Run4 集成中 XGBoost 用 dense 特征，LR 用 sparse 特征，两者互补 | HIGH | references/multi-scale-feature-engineering.md |
| 4 | Chi-squared特征选择(MaxAbsScaler+SelectKBest k=10000)对稀疏n-gram特征至关重要：降噪+降维+防过拟合 | Run 091845 top4 加入chi2后 val=0.2517 vs 无chi2 val=0.2653 | HIGH | references/chi2-feature-selection-ngram.md |
| 5 | WeightedRandomSampler处理类别不平衡比shuffle更稳定，保证少数类每epoch充分采样 | Run 091845 top4 使用WRS vs 早期方案用shuffle | MEDIUM | references/weighted-random-sampler-imbalance.md |
| 6 | Label Smoothing 0.1 + 梯度裁剪(max_norm=1.0) + AMP混合精度训练三者协同：LS防过拟合，裁剪防梯度爆炸，AMP加速+正则化 | 所有top方案均同时使用三项 | HIGH | references/training-stability-trifecta.md |
| 7 | 异构集成权重网格搜索(步长0.05)优于简单平均，DeBERTa通常获最高权重 | Run4 网格搜索 vs 等权差距0.02+ | MEDIUM | references/ensemble-weight-grid-search.md |
| 8 | 特征fit严格遵守train-only原则：scaler/vectorizer/chi2全部仅在训练集fit，val和test只transform | Run 091845 top4 标点vectorizer曾联合fit(泄露)，修复后推理脚本更可靠 | HIGH | references/train-only-fit-no-leakage.md |
| 9 | CosineAnnealingWarmRestarts或LinearWarmupDecay均可，关键是要有warmup(10%)，无调度器则训练不稳定 | Run8 top1(0.0725)有调度 vs top2(0.1457)无调度 | HIGH | references/scheduler-warmup-essential.md |
| 10 | DeBERTa-v3-large [CLS] embedding (1024维) 拼接手工特征后喂XGBoost，比单独用任一特征集效果都好 | Run4/Run 091845 XGBoost输入=hstack[stylo, read, pos, deberta_emb] | HIGH | references/deberta-cls-xgboost-synergy.md |
| 11 | 5折交叉验证概率平均是低风险高收益的下一步优化方向，可降低方差且不改架构 | 所有运行均为单折，方差是当前瓶颈 | MEDIUM | references/kfold-probability-averaging.md |
| 12 | 标点序列n-gram特征(Punctuation Sequence)对作者归属任务有独特贡献：标点习惯是最难伪造的写作指纹 | Run4 LR分支标点特征贡献独立信号 | MEDIUM | references/punctuation-sequence-authorship.md |

---

## 获胜配方：6层架构

从底层到顶层，当前最优方案的完整架构：

```
Layer 6: Ensemble Weight Optimization (Grid Search, step=0.05)
Layer 5: Three-Model Heterogeneous Ensemble (DeBERTa + XGBoost + LR)
Layer 4: Model-Specific Feature Routing
  ├─ DeBERTa ← raw text (tokenized)
  ├─ XGBoost ← dense features [stylo(26) + readability(4) + POS(5) + DeBERTa CLS embedding(1024)]
  └─ LR      ← sparse features [chi2-selected n-grams(10000)]
Layer 3: Feature Engineering Pipeline
  ├─ Dense:  Stylometric(30→26) + Readability(4) + POS(5)
  ├─ Sparse: char-ngram(2-4,4-6,5-7) + word-ngram(1-3) + punct-ngram(2-4)
  └─ Selection: MaxAbsScaler → Chi2(k=10000)
Layer 2: DeBERTa Fine-Tuning
  ├─ Partial Unfreezing (last 8/24 layers)
  ├─ Differentiated LR (backbone 2e-5, head 5e-5)
  ├─ Multi-Sample Dropout (K=4)
  ├─ WeightedRandomSampler
  └─ Label Smoothing 0.1 + GradClip 1.0 + AMP
Layer 1: Data Pipeline
  ├─ Stratified Split (90/10)
  ├─ Train-only fit (all scalers/vectorizers/selectors)
  └─ Probability clip + row normalization
```

## 各组件消融估计

| 移除组件 | 预计影响 | 理由 |
|----------|---------|------|
| 去掉 MSD | +0.01~0.02 | 失去隐式集成正则化 |
| 去掉 Chi2 | +0.01~0.02 | 稀疏特征噪声增大，LR过拟合 |
| 去掉 WeightedRandomSampler | +0.005~0.01 | 少数类欠采样 |
| 去掉 XGBoost 或 LR 之一 | +0.02~0.05 | 集成退化，错误相关性上升 |
| 全参数微调替代部分冻结 | +0.05~0.10 | 天花板0.26 |
| 去掉差异学习率 | +0.01~0.03 | backbone被大梯度破坏 |
| 去掉手工特征 | +0.03~0.05 | XGBoost退化为纯embedding模型 |

## 不要做的方向

| 方向 | 原因 | 证据 |
|------|------|------|
| 全参数微调DeBERTa | 天花板~0.26 | Run8 Branch2 |
| ModernBERT | 0.34~0.35 | Run5/7 |
| DeBERTa-small/base | 不如large | Run8 |
| 复杂注意力分类头 | 过拟合 | Run8 Branch4 |
| 同构Transformer集成 | 增益有限 | Run8 top9 |
| 仅凭验证集指标选方案 | 选择偏差严重 | Run8 val=0.0725测试集更差 |
| 在全数据上fit特征 | 数据泄露 | Run4 标点vectorifier泄露 |
| 多折训练时预fit转换器 | 折间泄露 | 见 references/kfold-probability-averaging.md |

## 下次运行建议

1. **5折StratifiedKFold + 概率平均**：降低方差，预期提升5-10%（详见 references/kfold-probability-averaging.md 的防泄露5条规则）
2. **增加LightGBM分支**：4模型集成，与XGBoost错误不完全相关
3. **贝叶斯权重优化**：替代网格搜索，连续空间更精确
4. **验证集比例提升至0.15**：减少early stopping选择偏差
5. **MSD K值扫描**：尝试K=2/4/6/8，找最优pass数
6. **多折训练严禁预fit**：所有 scaler/vectorizer/selector 每折单独 fit，绝不能在全数据上先 fit 再 split
