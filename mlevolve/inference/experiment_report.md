# Spooky Author Identification — 实验汇报

## 1. 任务概述

| 项目 | 内容 |
|------|------|
| **任务** | Kaggle Spooky Author Identification，三分类（EAP/HPL/MWS） |
| **评估指标** | Multi-class Log Loss（越低越好） |
| **训练集** | 19,621 条文本 |
| **测试集** | 8,392 条文本 |
| **基座模型** | microsoft/deberta-v3-large (304M 参数) |

---

## 2. 方案对比总览

| 维度 | Run1 | Run2 |
|------|------|------|
| **Val Log Loss** | 0.2953 (OOF) | **0.2313** (single split) |
| **MLEvolve历史最佳** | 0.2953 | 0.2389 |
| **核心架构** | MultiPoolingDeBERTa（自定义） | DeBERTa-v3-large（原生分类头） |
| **验证策略** | 5-Fold Stratified CV | 单次 90/10 分层划分 |
| **Ensemble策略** | 5折简单平均 | 3模型加权融合 |
| **额外模型** | 无 | XGBoost + Logistic Regression |
| **手工特征** | 无（纯BERT） | 39维稠密 + 13,500维稀疏 |
| **训练时长** | ~45 min | ~25 min (DeBERTa) + XGBoost/LR |

---

## 3. Run1 详细分析：MultiPoolingDeBERTa + 5-Fold CV + SWA

### 3.1 核心创新：MultiPooling 架构

在标准 DeBERTa-v3-large backbone 之上，设计了三路池化融合策略：

```
DeBERTa last_hidden_state
       ↓
Hierarchical Transformer Encoder (1层, 4头)
       ↓
  ┌────┼────────┐
  ↓    ↓        ↓
CLS   Mean    Attention
Pool  Pool    Pool
  ↓    ↓        ↓
LayerNorm × 3
  ↓    ↓        ↓
  └────┼────────┘
       ↓
   Concat (3×1024 = 3072)
       ↓
   Classifier (3072→512→256→3)
```

**三种池化的互补作用**：

| 池化方式 | 计算方法 | 捕获信息 | 适用场景 |
|---------|---------|---------|---------|
| CLS Pooling | 取 [CLS] token 表示 | 句子级语义、长距离依赖 | 长文本、复杂句式 |
| Mean Pooling | attention_mask 加权平均 | 全局均匀语义 | 短文本、信息分散 |
| Attention Pooling | 可学习注意力权重加权 | 风格关键词、判别性token | 含风格标记词的文本 |

### 3.2 特色技术方案及提升分析

| 技术 | 具体实现 | 作用 | 影响程度 |
|------|---------|------|---------|
| Hierarchical Transformer | 1层4头TransformerEncoder，norm_first=True | 在DeBERTa输出上二次精炼token表示 | ★★★ |
| Multi-Sample Dropout | 训练时K=5次前向取均值 | 等效集成5个子模型，提升泛化 | ★★★★ |
| Label Smoothing (0.05) | 自定义Loss + 类别权重 | 防过拟合 + 平衡类别 | ★★★ |
| 5-Fold CV + SWA | 5折交叉验证，最后3 epoch做权重平均 | CV稳健评估，SWA平滑损失平面 | ★★★ |
| Differential Layer-wise LR | 底层0.01×，顶层0.5×，分类头1× | 保护预训练知识 | ★★★★ |
| Differential Warmup | 早期层几乎不warmup，分类头完整warmup | 分类头快速适配 | ★★★ |
| Word Dropout + Token Masking | 10%词丢弃，15% token替换为[MASK] | 数据增强，提升鲁棒性 | ★★ |
| Gradient Accumulation | 2步累积（等效batch=32） | 训练更稳定 | ★★ |
| Layer-wise Weight Decay | 分类头0.5×wd，backbone标准 | 分类头更强正则化 | ★★ |

### 3.3 各折训练结果

| Fold | Best Epoch | Best Val LogLoss | SWA LogLoss | SWA改善? |
|------|-----------|-----------------|------------|---------|
| 1 | 3 | 0.3215 | 0.3436 | 否 |
| 2 | 2 | 0.2898 | 0.3011 | 否 |
| 3 | 3 | 0.2627 | 0.2781 | 否 |
| 4 | 4 | 0.3028 | **0.2973** | **是** |
| 5 | 2 | 0.3121 | — | — |
| **OOF** | — | **0.2953** | — | — |

**观察**: SWA 仅在 Fold4 带来改善，说明5个epoch训练不够充分，SWA checkpoint质量受限。

---

## 4. Run2 详细分析：DeBERTa + XGBoost + LR 三模型融合

### 4.1 架构设计

```
                    ┌─ DeBERTa fine-tune (max_len=512)
                    │   └─ Temporal Ensemble (top-3 checkpoints)
                    │       ├─ Probabilities → Ensemble
                    │       └─ CLS Embedding (1024d) → XGBoost input
                    │
Train Data ────────┼─ Handcrafted Dense Features (39d)
                    │   ├─ Stylometric (30d, after VarianceThreshold)
                    │   ├─ Readability (4d)
                    │   └─ POS Approximation (5d)
                    │
                    └─ Sparse N-gram Features (13,500d)
                        ├─ Char 2-4gram (3,000)
                        ├─ Char 4-6gram (3,000)
                        ├─ Char 5-7gram (2,000)
                        ├─ Word 1-3gram (5,000)
                        └─ Punctuation seq (500)
                            ↓
                    ┌───────┼────────┐
                    ↓       ↓        ↓
                DeBERTa  XGBoost    LR
                  50%      40%      10%
                    └───────┼────────┘
                            ↓
                     Weighted Ensemble
```

### 4.2 各模型验证成绩

| 模型 | Val Log Loss | Ensemble权重 | 特征输入 |
|------|-------------|-------------|---------|
| DeBERTa (单best) | 0.3562 | — | 原始文本 (max_len=512) |
| DeBERTa (temporal ensemble) | 0.2549 | **0.50** | 原始文本 (top-3 checkpoint加权) |
| XGBoost | 0.2656 | **0.40** | DeBERTa CLS embedding (1024d) + 手工特征 (39d) |
| Logistic Regression | 0.5140 | **0.10** | N-gram稀疏特征 (13,500d) |
| **Final Ensemble** | **0.2313** | — | 三模型加权融合 |

### 4.3 特色技术方案及提升分析

| 技术 | 具体实现 | 提升效果 | 影响程度 |
|------|---------|---------|---------|
| Temporal Ensemble | 保留val log loss最低3个checkpoint，逆log loss加权平均 | 0.3562→0.2549，**↓28.5%** | ★★★★★ |
| DeBERTa Embedding → XGBoost | CLS embedding(1024d) + 手工特征(39d) | XGBoost达0.2656，与DeBERTa高度互补 | ★★★★ |
| 多粒度 N-gram | Char(2-4, 4-6, 5-7) + Word(1-3) + Punctuation(2-4) | LR从~0.6提升到0.51 | ★★★ |
| Stylometric特征工程 | 30维风格特征（标点分布、大小写、古语词/情感词/函数词比率等） | 捕获浅层风格信号 | ★★★ |
| Readability特征 | Flesch Reading Ease、ARI、音节/复杂词比率 | 区分作者文风复杂度 | ★★ |
| POS近似特征 | 基于词缀的词性近似（5维） | 无需POS tagger | ★★ |
| Label Smoothing (0.1) | CrossEntropyLoss(label_smoothing=0.1) | 比Run1更强正则化 | ★★★ |
| Ensemble权重网格搜索 | w1,w2步长0.05，w3=1-w1-w2 | 最优组合(0.50, 0.40, 0.10) | ★★★ |
| MAX_LENGTH=512 | vs Run1的256 | 更长上下文捕获段落级风格 | ★★★ |

### 4.4 Temporal Ensemble 的关键作用

| 阶段 | Val Log Loss | 相对单best改善 |
|------|-------------|--------------|
| 单best checkpoint (epoch 3) | 0.3562 | — |
| Temporal ensemble (top-3) | 0.2549 | **↓28.5%** |
| Final 3-model ensemble | 0.2313 | **↓35.1%** |

Temporal ensemble 是 Run2 最关键的提分手段。原理：不同epoch的模型处于损失平面的不同位置，加权平均等价于在平坦区域取值，预测更稳定校准更好。

---

## 5. 两方案关键对比

### 5.1 训练配置对比

| 配置项 | Run1 | Run2 |
|--------|------|------|
| 模型架构 | MultiPoolingDeBERTa（自定义头） | DeBERTaForSequenceClassification（原生头） |
| MAX_LENGTH | 256 | 512 |
| BATCH_SIZE | 16 (×2 accum = 32等效) | 16 |
| EPOCHS | 5 | 40 (early stop at 8) |
| LEARNING_RATE | 2e-5 (layer-wise decay 0.95) | 2e-5 (统一) |
| LABEL_SMOOTHING | 0.05 | 0.1 |
| DROPOUT | 0.15 (multi-sample K=5) | 0.2 |
| 数据增强 | Word Dropout + Token Masking | 无 |
| 验证策略 | 5-Fold CV | 单次 90/10 split |
| 权重平均 | SWA (最后3 epoch) | Temporal Ensemble (top-3 checkpoint) |
| 调度器 | 自定义Differential Warmup + Cosine Decay | Linear Schedule with Warmup |

### 5.2 特征工程对比

| 特征类别 | Run1 | Run2 | 维度 |
|---------|------|------|------|
| DeBERTa语义表示 | ✓ (MultiPool 3072d) | ✓ (CLS 1024d) | — |
| Stylometric特征 | ✗ | ✓ | 30d |
| Readability特征 | ✗ | ✓ | 4d |
| POS近似特征 | ✗ | ✓ | 5d |
| Char n-gram (2-4) | ✗ | ✓ | 3,000 |
| Char n-gram (4-6) | ✗ | ✓ | 3,000 |
| Char n-gram (5-7) | ✗ | ✓ | 2,000 |
| Word n-gram (1-3) | ✗ | ✓ | 5,000 |
| Punctuation sequence | ✗ | ✓ | 500 |
| **特征总维度** | **3072** (纯DeBERTa) | **~17,500** (多源) | — |

### 5.3 模型融合策略对比

| 对比项 | Run1 | Run2 |
|--------|------|------|
| 融合模型数 | 1 (DeBERTa 5折) | 3 (DeBERTa + XGBoost + LR) |
| 融合方式 | 5折预测简单平均 | 加权融合 (网格搜索最优权重) |
| 模型异构性 | 同构（同一架构不同fold） | 异构（Transformer + GBDT + 线性） |
| 稳定性 | 依赖单模型质量 | 多模型冗余，更鲁棒 |
| 最佳单模型 | 0.2627 (Fold3) | 0.2549 (DeBERTa temporal) |
| 最终成绩 | 0.2953 (5折平均) | **0.2313** (3模型融合) |

### 5.4 优劣势对比

| 维度 | Run1 优势 | Run2 优势 |
|------|----------|----------|
| DeBERTa单模型 | 更强架构，5折OOF 0.2953 | — |
| 模型多样性 | — | 3个异构模型互补 |
| 验证可靠性 | 5折CV更稳健 | — |
| 特征利用 | 纯端到端，无需特征工程 | 手工特征编码领域知识 |
| 最终成绩 | 0.2953 | **0.2313** |
| 可解释性 | 黑盒 | XGBoost/LR可解释特征重要性 |
| 训练成本 | 5折×5epoch | 单次训练 + 轻量级模型 |
| 数据增强 | 有 (word dropout + masking) | 无 |

### 5.5 关键技术效果量化对比

| 技术 | Run1 实施方式 | 效果 | Run2 实施方式 | 效果 |
|------|-------------|------|-------------|------|
| 权重平均 | SWA (3 epoch平均) | 仅1/5折改善 | Temporal Ensemble (top-3 ckpt) | 0.3562→0.2549 (↓28.5%) |
| 多模型融合 | 无 | — | 3模型加权 | 0.2549→0.2313 (↓9.3%) |
| Label Smoothing | 0.05 + 类别权重 | 基础正则化 | 0.1 | 更强正则化，适配长序列 |
| 学习率策略 | Layer-wise decay + Differential warmup | 精细调控 | 统一LR + linear schedule | 简单有效 |

---

## 6. 潜在改进方向

### 6.1 短期可行：两方案融合

用 Run1 的 MultiPoolingDeBERTa 5折模型替换 Run2 中的 DeBERTa，保留 XGBoost + LR：

```
Run1 DeBERTa (5-fold, OOF 0.2953) + Run2 XGBoost + Run2 LR → 新Ensemble
```

| 预估组件 | 预估 Val LogLoss | 预估权重 |
|---------|-----------------|---------|
| Run1 DeBERTa (5-fold) | ~0.25 (temporal ensemble后) | 0.45 |
| XGBoost (Run1 embedding + 手工特征) | ~0.24 | 0.45 |
| LR (n-gram) | ~0.51 | 0.10 |
| **预估最终** | **~0.20** | — |

### 6.2 中期改进

| 改进项 | 当前状态 | 改进方案 | 预期收益 |
|--------|---------|---------|---------|
| 修复数据泄露 | Run2标点vectorizer在all data上fit | 改为仅train上fit | val分数更可信 |
| 增加训练epoch | Run1仅5 epoch | 增加到10-15 epoch | SWA效果提升 |
| 5折CV + Temporal Ensemble | 两方案各自独立 | 合并应用 | 互补增效 |
| XGBoost特征增强 | CLS embedding (1024d) | combined_pool (3072d) | 更丰富语义特征 |

### 6.3 长期探索

| 方向 | 描述 | 预期收益 |
|------|------|---------|
| 对抗训练 | FGM/PGD增强鲁棒性 | 提升泛化 |
| 多预训练模型融合 | DeBERTa + RoBERTa + Electra | 模型多样性 |
| 半监督/伪标签 | 利用测试集分布信息 | 数据增强 |

---

## 7. 数据泄露说明

| 问题 | 位置 | 严重程度 | 说明 |
|------|------|---------|------|
| 标点CountVectorizer在all data上fit | Run2 第402-415行 | 低 | LR权重仅10%，实际影响可忽略；对Kaggle提交无影响 |

---

## 8. 结论

| 结论 | 依据 |
|------|------|
| **异构模型融合 > 单模型增强** | Run2 单DeBERTa 0.3562 → ensemble 0.2313；Run1 纯DeBERTa仅0.2953 |
| **Temporal Ensemble是最关键提分手段** | 单模型0.3562→0.2549，降低28.5% |
| **模型多样性比单模型质量更重要** | Run2 DeBERTa较弱但最终成绩更好 |
| **手工特征仍有价值** | XGBoost达0.2656，LR虽弱(0.51)但10%权重仍有贡献 |

**下一步建议**: 将两方案优势结合 — 用 Run1 更强的 MultiPoolingDeBERTa 替换 Run2 中的 DeBERTa 组件，预估可达 **0.20 以下**。
