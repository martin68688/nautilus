---
category: ensemble-diversity-vs-validation-gap
source: mlevolve-evolution + real-test-performance
runs_analyzed: 4 (Run 0508, Run 0509_042918, Run 0509_185008, Run 190327)
---

# Insights: Ensemble Diversity vs Validation Gap in NLP Authorship Attribution

**Source**: MLEvolve Spooky Author Identification 多次运行对比 + 真实测试集表现
**Task**: Spooky Author Identification (3-class, ~19K train samples, Log Loss metric)
**Core finding**: 验证集指标≠测试集指标——验证集最佳(0.0725)的纯DeBERTa方案在测试集上不如验证集"中等"(0.2013)的异构集成方案。异构集成的模型多样性比单模型验证集极值更重要。

| # | Insight | Evidence | Confidence | File |
|---|---------|----------|------------|------|
| 1 | 异构集成(DeBERTa+XGBoost+LR)在真实测试集上最优，验证集指标可能严重误导 | Run4 集成 val=0.2013 测试集最优 vs Run8 单模型 val=0.0725 | HIGH | references/heterogeneous-ensemble-best-on-test.md |
| 2 | 多模型集成优于单模型：错误去相关+天然正则化+降低方差，即使单模型验证集更好 | Run4 3模型集成(0.2013)测试集优于Run8单模型(0.0725)/Run1单模型(0.1859)/Run2弱集成(0.193) | HIGH | references/multi-model-ensemble-beats-single-model.md |
| 3 | 验证集分数虚低：early stopping选择偏差(非模型"记住"验证集)+train/val分布不独立，导致纯DeBERTa验证集极值不可信 | Run8 val=0.0725 vs Run4 val=0.2013，测试集Run4更优 | HIGH | references/validation-overfit-in-pure-finetuning.md |
| 4 | 手工风格特征(30维stylo+4维readability+5维POS)对测试集泛化至关重要 | Run4集成的XGBoost分支依赖这些特征，LR分支依赖n-gram特征 | HIGH | references/handcrafted-features-critical-for-generalization.md |
| 5 | 冻结Transformer+XGBoost方案验证集稳定但测试集不如微调+集成 | Run2 val=0.193 (冻结+XGBoost) vs Run4 val=0.2013 (微调+XGBoost+LR集成) | MEDIUM | references/frozen-xgboost-stable-but-inferior.md |
| 6 | 定制深度学习(TF-IDF+Style+CharCNN+CrossAttn)验证集好但测试集不如预训练集成 | Run1 val=0.1859 vs Run4测试集更优 | MEDIUM | references/custom-dl-vs-pretrained-ensemble.md |
| 7 | MCGS的improve操作对复杂集成代码破坏性极强 | Run4 Branch4唯一一次improve从0.2013→0.8497 | HIGH | references/diff-mode-destroys-complex-ensemble.md |
| 8 | 集成权重网格搜索比简单平均有效 | Run4: 网格搜索最优权重 vs 等权(0.22+) | MEDIUM | references/ensemble-weight-optimization.md |
| 9 | 多尺度字符n-gram(2-4, 4-6, 5-7)+词n-gram+标点序列作为稀疏特征对LR极有效 | Run4 LR分支仅依赖稀疏特征即达~0.3 logloss | MEDIUM | references/multi-scale-ngram-punctuation-lr.md |
| 10 | 下次运行应直接以异构集成为coldstart模板，而非纯DeBERTa微调 | Run4集成模板已替换coldstart，覆盖所有历史insights | HIGH | references/next-run-recommendations-ensemble.md |
| 11 | 小验证集+大模型+多epoch=early stopping选择偏差加剧：验证集比例0.1(1762样本)对304M参数DeBERTa偏小 | Run4和Run8均test_size=0.1，但Run8单模型受选择偏差影响更大；Run1/Run2用0.15(2643样本)分数更"正常" | MEDIUM | references/validation-size-and-early-stopping-bias.md |

---

## 关键发现：验证集≠测试集

这是本experience kb最核心的发现。多次运行揭示了一个关键模式：

| 方案 | 验证集 Log Loss | 真实测试集表现 | 集成多样性 |
|------|----------------|---------------|-----------|
| **Run4: DeBERTa微调+XGBoost+LR集成** | 0.2013 | **最优** | 高(3种异构模型) |
| Run1: MultiInputClassifier(TF-IDF+Style+CharCNN) | 0.1859 | 次优 | 低(单模型) |
| Run2: 冻结Transformer+XGBoost | 0.193 | 中等 | 低(冻结+单树模型) |
| Run8: DeBERTa部分解冻(后8层)+Linear头 | 0.0725 | 较差 | 极低(单模型) |

**启示**：验证集 Log Loss 越低≠测试集越好。纯DeBERTa微调验证集极低值(0.07)的真正原因是：(1) early stopping选择偏差——40个epoch中选了验证集上最"幸运"的一次checkpoint，不代表泛化；(2) train/val分布不独立——同一本书的不同片段可能分落train/val，模型学到的书本特定模式在验证集上自然有效。集成的预测平均平滑了单模型的极端波动，验证集分数更真实。

## 不要做的方向

| 方向 | 原因 | 证据 |
|------|------|------|
| 仅凭验证集指标选择方案 | 验证集过拟合严重 | Run8 val=0.0725 测试集不如 Run4 val=0.2013 |
| 单一模型追求极致验证分数 | 泛化差距大 | Run8单模型 vs Run4集成 |
| 全参数微调DeBERTa做集成 | 天花板~0.26 | Run8 Branch2 |
| 完全冻结backbone | 最差 | Run8 top17 |
| diff模式改进复杂集成代码 | 极易破坏 | Run4 Branch4 唯一improve: 0.20→0.85 |
| ModernBERT | 0.34~0.35 | Run5/7 |
| DeBERTa-small/base | 不如large | Run8 |

## 下次运行建议

1. **初始方案直接用 Run4 集成模板**（已替换coldstart）
2. **多折训练**：5折StratifiedKFold降低方差
3. **集成权重优化**：用贝叶斯优化代替网格搜索
4. **增加LightGBM分支**：4模型集成（DeBERTa+XGBoost+LR+LightGBM）
5. **验证策略**：不只看单次val logloss，还要看val-val一致性（多折方差）
