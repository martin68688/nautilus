---
title: MCGS的diff模式对复杂集成代码破坏性极强
confidence: HIGH
evidence: [Run4 Branch4 唯一improve: 0.2013→0.8497, 后续buggy]
---

# MCGS的diff模式对复杂集成代码破坏性极强

## 证据

Run4 Branch4 的完整演化链：

| 步骤 | 节点 | 阶段 | Log Loss | 结果 |
|------|------|------|----------|------|
| Step 21 | d93b4c2a | draft | 0.2013 | 最佳，全局best |
| Step 29 | d056f962 | improve(diff) | 0.8497 | 严重退化(+0.65) |
| Step 30 | a1eefcda | improve(diff) | None | RuntimeError，强制回传 |

Branch4 在3个节点后终止，后续再未被选中。

## 原因分析

1. **代码复杂度高**：Run4集成方案580行，包含3个完全独立的模型训练流程(DeBERTa微调+XGBoost+LR) + 特征提取(3个函数) + 集成权重搜索 + 数据加载
2. **diff模式局限**：SEARCH/REPLACE只做局部文本替换，无法理解全局代码逻辑。对复杂集成代码，一个看似小的替换可能破坏数据流
3. **跨模块依赖**：特征提取函数的输出维度与下游模型输入维度绑定，修改一处会导致维度不匹配
4. **执行时间长**：集成方案执行~90分钟，一个diff错误导致90分钟白跑，且MCGS不会重试

## Actionable Guidance

- **集成方案应标记为lock=True**：作为draft节点后锁定，不让improve操作修改
- **improve操作应优先对单模型分支执行**，而非集成模板
- **复杂集成应作为独立脚本**：不参与MCGS演化，在搜索结束后手动构建
- **如果必须用diff改进集成**：限制只修改超参数(学习率、epochs等)，不改代码结构
- **代码行数>300行的方案**：建议使用full rewrite而非diff模式

## 对coldstart模板的影响

Run4集成方案(580行)已替换coldstart NLP Code_template。当MLEvolve使用此模板生成draft时：
- draft阶段会直接使用此代码(verbatim copy分支)
- improve阶段会尝试diff修改——**这几乎必然失败**
- 建议：在coldstart模板中添加注释标记哪些部分可以修改、哪些不能

**Condition**: MCGS搜索中使用diff模式 + 复杂集成代码
**Confidence**: HIGH
