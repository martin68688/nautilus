---
title: 手工风格特征对测试集泛化至关重要
confidence: HIGH
evidence: [Run4 XGBoost分支依赖手工特征，集成测试集最优]
---

# 手工风格特征对测试集泛化至关重要

## 证据

| 特征类型 | Run4 使用 | Run1 使用 | Run8 使用 | 对测试集贡献 |
|---------|----------|----------|----------|------------|
| 30维风格统计 | Yes (XGBoost) | Yes (stylo_branch) | No | **高** |
| 4维可读性 | Yes (XGBoost) | No | No | **高** |
| 5维POS近似 | Yes (XGBoost) | No | No | **中** |
| n-gram稀疏特征 | Yes (LR) | Yes (TF-IDF) | No | **高** |
| 标点序列 | Yes (LR) | 部分 | No | **中** |
| DeBERTa嵌入 | Yes (XGBoost输入) | No | Yes (CLS) | 基础 |

## Run4 的特征工程详情

### 30维风格特征 (extract_stylometric_features)
- 基础统计：文本长度、词数、句数、平均词长、平均句长
- 字符比例：大写、小写、数字、空白
- 标点频率(12种)：逗号/句号/分号/冒号/感叹号/问号/破折号/连字符/引号/括号
- 高级特征：字符多样性、长词比例、首字母大写比例、全大写比例
- 句子特征：长度变异系数、方差系数
- 词汇特征：功能词比例、古词比例、情感词比例、Lovecraft词比例、从属连词比例

### 4维可读性特征 (create_readability_features)
- Flesch Reading Ease
- Automated Readability Index (ARI)
- 平均音节数
- 复杂词比例(≥3音节)

### 5维POS近似特征 (create_pos_tag_approximation)
- 名词/动词/形容词/副词后缀比例
- 实词比例

## 为什么对泛化重要

- **风格特征是作者识别的核心**：此任务本质是识别写作风格而非语义内容
- **手工特征泛化性好**：统计特征不依赖特定文本内容，不易过拟合
- **DeBERTa可能过拟合语义**：预训练模型擅长语义理解，但可能"记住"训练集的语义模式
- **XGBoost+手工特征提供互补信号**：树模型擅长捕捉统计规则，与DeBERTa的语义信号正交

## Actionable Guidance

- 作者识别/风格分类任务**必须**包含手工风格特征
- 风格特征应作为独立模型(XGBoost)的输入，而非简单拼接到DeBERTa嵌入后
- 古词/Lovecraft词比例对此特定任务有额外价值(3位作者的时代和风格差异)
- 标点序列n-gram对LR分支特别有效

**Condition**: 作者识别/风格分类任务
**Confidence**: HIGH
