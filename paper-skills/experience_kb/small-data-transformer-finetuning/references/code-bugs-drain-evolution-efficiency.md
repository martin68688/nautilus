---
title: FAIL率~40%，代码bug是演化效率的主要瓶颈
confidence: MEDIUM
evidence: [Run8: 66步中约40步FAIL]
---

# FAIL率~40%，代码bug是演化效率的主要瓶颈

Run8 共 66 步，约 40 步产生 FAIL（代码执行错误），成功率仅约 40%。大量步数浪费在代码 bug 上。

## 常见 FAIL 原因

1. **特征未接入模型**：提取了特征但 forward 签名缺少参数（top1 的死代码 bug）
2. **Tensor device 不匹配**：部分 tensor 在 CPU，部分在 GPU
3. **维度不匹配**：分类头输入维度与 backbone 输出不一致
4. **API 错误**：transformers 库方法调用错误
5. **OOM**：模型太大或 batch_size 过大

## Actionable Guidance

- 每个变体在完整训练前，先做一个 dry-run（1 batch forward+backward）验证无报错
- 代码生成时强制检查：forward 参数 = Dataset 返回的 key 集合
- 设定 `torch.cuda.empty_cache()` 和梯度累积降低 OOM 风险
- 减少 FAIL 率比优化成功节点的增量更重要

**Condition**: LLM 代码生成的演化优化
**Confidence**: MEDIUM
