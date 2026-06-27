---
title: 简单 Linear 分类头优于复杂注意力/MLP头
confidence: HIGH
evidence: [Run8 top1 vs top2, Run8 Branch4 all HybridStylo]
---

# 简单 Linear 分类头优于复杂注意力/MLP头

单层 `Linear(1024→3)` 分类头（Log Loss 0.0725）远优于多组件复杂头（0.1457~0.37）。

## 为什么复杂头反而差

1. **噪声梯度反传**：复杂头大量随机初始化参数在训练初期输出噪声，反传错误梯度到已解冻的 DeBERTa 层，干扰微调
2. **过拟合风险**：小数据集（~19K）下，注意力头、MLP、投影层的参数量远超必要，容易记住训练集噪声
3. **CLS token 已经足够**：DeBERTa 的 CLS token 是模型专门训练的句子级表示，简单线性变换足以映射到类别空间

## 证据

| 分类头 | Log Loss | 参数量(额外) | 来源 |
|--------|----------|-------------|------|
| Linear(1024→3) | **0.0725** | ~3K | Run8 top1 |
| 投影(3072→1024)+Attn(8头)+MLP(4层) | 0.1457 | ~5M | Run8 top2 |
| HybridStylo(DeBERTa+风格投影+3层MLP) | 0.3061 | ~3M | Run8 top10 |
| HybridStylo 变体 | 0.28~0.37 | ~3M | Run8 Branch4 |

## Actionable Guidance

- 分类头用 `nn.Linear(hidden_size, num_classes)` 即可
- 如果一定要加层：最多 `Linear → LayerNorm → GELU → Dropout → Linear`，不要超过2层
- 不要在分类头中使用 MultiheadAttention 或多路池化拼接
- 特征融合应通过独立分支+集成实现，而非塞进单个模型的分类头

**Condition**: backbone 已提供高质量句子表示（DeBERTa/RoBERTa 的 CLS token）
**Confidence**: HIGH
