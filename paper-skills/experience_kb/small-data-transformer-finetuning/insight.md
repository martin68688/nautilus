---
category: small-data-transformer-finetuning
source: mlevolve-evolution
runs_analyzed: 10 (Run1~Run8, 66 steps in Run8)
---

# Insights: Small-Data Transformer Fine-Tuning

**Source**: MLEvolve Run8 (20260512) evolution tree + historical runs
**Task**: Spooky Author Identification (3-class, ~19K train samples, Log Loss metric)
**Core finding**: 在小数据集上微调大 Transformer，"少即是多"——部分解冻 + 简单头 + 调度器 >> 全参数微调 + 复杂头 + 集成

| # | Insight | Evidence | Confidence | File |
|---|---------|----------|------------|------|
| 1 | 部分解冻（后8层）远优于全参数微调，小数据集下预训练知识是最好的正则化 | Run8: partial 0.0725/0.1457 vs full 0.2631~0.34 (8个变体) | HIGH | references/partial-unfreezing-beats-full-finetuning.md |
| 2 | 简单 Linear 分类头优于复杂注意力/MLP头，复杂头噪声梯度干扰微调 | Run8 top1(0.0725) Linear vs top2(0.1457) AttnHead+MLP | HIGH | references/simple-head-beats-complex-head.md |
| 3 | CosineAnnealingWarmRestarts + Linear Warmup 是微调调度器的最佳选择 | Run8 top1(0.0725) 有调度 vs top2(0.1457) 无调度，同分支差距2倍 | HIGH | references/cosine-warm-restart-critical-for-finetuning.md |
| 4 | label_smoothing=0.1 对小数据集分类任务有效防止过拟合 | 所有 top8 方案均使用 | HIGH | references/label-smoothing-prevents-overfitting.md |
| 5 | DeBERTa-v3-large 显著优于 ModernBERT/DeBERTa-small/base/DistilBERT | Run8: large 0.0725 vs small 0.3741 vs base 0.3727 vs ModernBERT 0.34~0.35 | HIGH | references/deberta-v3-large-best-backbone.md |
| 6 | 差异化学习率（backbone 2e-5, head 5e-5）保护预训练层 | Run8 top1/top2 均使用 | MEDIUM | references/differentiated-learning-rates.md |
| 7 | 异构集成（DeBERTa+XGBoost+LR）优于同构集成（3个Transformer） | Run4 top1(0.2013) 异构 vs Run8 top9(0.3016) 同构 | MEDIUM | references/heterogeneous-ensemble-beats-homogeneous.md |
| 8 | 全参数微调存在硬天花板，~0.26 无论怎么调参都无法突破 | Run8 Branch2: 8个变体全部 0.26~0.34 | HIGH | references/full-finetuning-hits-ceiling.md |
| 9 | 复杂注意力头+多路池化在小数据上过拟合严重 | Run8 Branch4 HybridStylo: 0.28~0.37，无一突破 | MEDIUM | references/complex-attention-head-overfits.md |
| 10 | 渐进解冻（8→4→0层）不如直接部分解冻稳定 | Run8 top14(0.3701) 渐进 vs top1(0.0725) 直接解冻后8层 | MEDIUM | references/gradual-unfreezing-less-stable.md |
| 11 | 风格特征提取代码容易变成死代码，需确保数据流完整接入模型 | Run8 top1: 340行特征代码未接入模型 | HIGH | references/feature-extraction-dead-code-bug.md |
| 12 | 完全冻结backbone只训练head效果最差 | Run8 top17(0.3853) 全冻结 | HIGH | references/fully-frozen-backbone-worst.md |
| 13 | 最差节点变异可能产生最大突破，不要过早剪枝低分分支 | Run8: 0.4738 → 0.1457，降幅69% | HIGH | references/worst-node-can-yield-breakthrough.md |
| 14 | FAIL率~40%，代码bug是演化效率的主要瓶颈 | Run8: 66步中约40步FAIL | MEDIUM | references/code-bugs-drain-evolution-efficiency.md |
| 15 | 下次运行建议：top1策略+风格特征正确接入→多折训练→异构集成 | Run8 top1为基线，特征数据流断裂bug待修复 | HIGH | references/next-run-recommendations.md |

---

## 下次运行建议

### 优先级1：top1策略 + 风格特征正确接入（预期 < 0.05）

复用 top1(0.0725) 训练策略（解冻后8层 + 简单Linear头 + CosineWarmRestart + LS），但修复特征数据流断裂 bug，在 forward 中正确接入风格特征：

- `feature_proj = Linear(150→64) + LayerNorm + GELU + Dropout`
- `head = Linear(hidden_size+64, num_classes)` — 仍然简单Linear，不加注意力/MLP
- Dataset 必须返回 features tensor，forward 签名必须包含 features 参数
- 详细代码见 references/next-run-recommendations.md

### 优先级2：多折训练 + 概率平均（预期 0.03~0.05）

用 top1 策略训练 5折 StratifiedKFold，测试时 5 个模型 softmax 概率算术平均，不改架构即减少方差。

### 优先级3：异构集成（预期 < 0.03）

Model A: DeBERTa(优先级1) + Model B: XGBoost(风格特征) + Model C: LightGBM(TF-IDF)，权重网格搜索。

### 不要做的方向（已验证无效）

| 方向 | 原因 | 证据 |
|------|------|------|
| ModernBERT | 性能差 0.34~0.35 | Run5/7 |
| DeBERTa-small/base | 不如 large | Run8 top14/16 |
| 全参数微调 | 天花板 ~0.26 | Run8 Branch2 |
| 复杂注意力分类头 | 过拟合 | Run8 Branch4 |
| 同构Transformer集成 | 增益有限 | Run8 top9 |
| 渐进解冻 | 不如直接部分解冻 | Run8 top14 |
| 无调度器 | 训练不稳定 | Run8 top2 |

### 演化策略建议

1. **初始方案直接用 top1 策略**，不要再从 coldstart 全参数微调模板开始
2. **至少2个分支做架构探索**：1个在top1基础上微调（特征接入/多折），1个尝试全新方向
3. **FAIL 率控制**：每个变体先做 dry-run 验证，再投入完整训练
4. **解冻层数扫描**：尝试后10层/12层，看是否有进一步提升空间
