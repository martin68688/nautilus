---
title: 下一次运行建议方向
confidence: HIGH
based_on: Run8 evolution tree analysis + 10 runs historical data
---

# 下一次运行建议方向

基于 Run8 演化树和全部历史运行的分析，以下方向最有可能突破当前 0.0725 的纪录。

## 优先级 1：当前最佳策略 + 风格特征正确接入（预期 < 0.05）

**方案**：复用 top1(0.0725) 的训练策略（解冻后8层 + 简单Linear头 + CosineWarmRestart + LS），但在 forward 中正确接入风格特征。

```python
class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=150, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large", num_labels=num_authors,
            output_hidden_states=True, hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # 冻结前16层，解冻后8层
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        # 风格特征投影（小网络，避免过拟合）
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        # 融合分类头：CLS + 风格特征
        self.head = nn.Linear(hidden_size + 64, num_authors)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.backbone.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_pool = outputs.last_hidden_state[:, 0, :]
        feat_embed = self.feature_proj(features)
        combined = torch.cat([cls_pool, feat_embed], dim=1)
        logits = self.head(combined)
        return logits
```

**关键**：
- Dataset 必须返回 `features` tensor
- feature_proj 只用 64 维，避免大投影层过拟合
- head 仍然是简单 Linear，不加注意力/MLP
- 保持 CosineWarmRestart + label_smoothing + 差异化学习率

## 优先级 2：多折训练 + 概率平均（预期 0.03~0.05）

用 top1 策略训练 5 折，每折保存模型，测试时 5 个模型的概率取平均。这在不改架构的情况下减少方差。

- 5折 StratifiedKFold
- 每折独立训练，early stopping
- 测试预测 = 5个模型 softmax 概率的算术平均

## 优先级 3：异构集成（预期 < 0.03）

用优先级1的 DeBERTa 模型 + XGBoost(风格特征) + LightGBM(TF-IDF) 做异构集成。

- Model A: DeBERTa-v3-large 部分解冻 + 风格特征 (优先级1)
- Model B: XGBoost 基于 150 维风格统计特征
- Model C: LightGBM 基于 TF-IDF 字符/词 n-gram
- 集成权重网格搜索

## 不要做的方向（已验证无效）

| 方向 | 原因 | 证据 |
|------|------|------|
| ModernBERT | 性能差，0.34~0.35 | Run5, Run7 |
| DeBERTa-small/base | 不如 large | Run8 top14/16 |
| 全参数微调 | 天花板 ~0.26 | Run8 Branch2 |
| 复杂注意力分类头 | 过拟合 | Run8 Branch4 |
| 同构Transformer集成 | 增益有限 | Run8 top9 |
| 渐进解冻 | 不如直接部分解冻 | Run8 top14 |
| 完全冻结backbone | 最差 | Run8 top17 |
| 无调度器 | 训练不稳定 | Run8 top2 |

## 演化策略建议

1. **初始方案直接用 top1 策略**，不要再从 coldstart 的全参数微调模板开始
2. **至少保留 2 个分支做架构探索**：1个在top1基础上微调（特征接入/多折），1个尝试全新方向（如对比学习）
3. **FAIL 率控制**：每个变体生成后先做 dry-run 验证，再投入完整训练
4. **解冻层数扫描**：在 top1 基础上尝试解冻后10层/12层，看是否有进一步提升空间
5. **max_length 实验**：当前 512，可尝试 384（加速）或 768（更长上下文），对比效果
