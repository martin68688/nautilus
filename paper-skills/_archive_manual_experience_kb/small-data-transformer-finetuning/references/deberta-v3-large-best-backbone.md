---
title: DeBERTa-v3-large 显著优于 ModernBERT/DeBERTa-small/base/DistilBERT
confidence: HIGH
evidence: [Run8 top1 vs top14/16, Run5/7 vs Run8]
---

# DeBERTa-v3-large 显著优于 ModernBERT/DeBERTa-small/base/DistilBERT

作者识别任务核心是捕捉细微写作风格差异，大模型的解耦注意力机制(disentangled attention)对此特别有效。

## 证据

| 模型 | 最佳 Log Loss | 参数量 | 来源 |
|------|-------------|--------|------|
| **DeBERTa-v3-large** | **0.0725** | 435M | Run8 top1 |
| DeBERTa-v3-base | 0.3727 | 184M | Run8 top14 |
| DeBERTa-v3-small | 0.3741 | ~86M | Run8 top16 |
| ModernBERT | 0.3368 | — | Run5/Run7 |
| DistilBERT | 0.3016 (集成) | ~66M | Run8 top9 |

**小模型即使用3模型集成(0.3016)也不如单模型 large 部分解冻(0.0725)。**

## Actionable Guidance

- 首选 `microsoft/deberta-v3-large`，不要用 small/base 版本"节省时间"
- ModernBERT 在作者识别上表现不佳，不建议使用
- 如果显存不足，优先降低 batch_size / max_length，不要换小模型

**Condition**: 作者识别/文本风格分类任务，GPU 显存 >= 16GB
**Confidence**: HIGH
