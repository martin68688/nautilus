---
title: 多尺度字符n-gram+词n-gram+标点序列作为LR稀疏特征极有效
confidence: MEDIUM
evidence: [Run4 LR分支仅稀疏特征即达~0.3 logloss]
---

# 多尺度字符n-gram+词n-gram+标点序列

## Run4 的稀疏特征设计

| 特征类型 | 分析器 | n-gram范围 | max_features | 作用 |
|---------|--------|-----------|-------------|------|
| 字符短n-gram | char | (2,4) | 3000 | 字符模式、拼写习惯 |
| 字符中n-gram | char | (4,6) | 3000 | 词根、后缀模式 |
| 字符长n-gram | char | (5,7) | 2000 | 短语风格 |
| 词n-gram | word | (1,3) | 5000 | 词汇选择、搭配 |
| 标点序列 | char | (2,4) | 500 | 标点使用习惯 |

总稀疏特征维度：~13500

## 关键设计决策

1. **3个独立字符n-gram器**：不同粒度捕获不同层次的模式
2. **sublinear_tf=True**：用1+log(tf)代替原始tf，降低高频词影响
3. **norm="l2"**：L2归一化，对LR分类器更友好
4. **min_df=3, max_df=0.85** (词n-gram)：过滤极罕见和极常见词
5. **标点序列提取**：仅保留标点字符，用CountVectorizer分析标点模式
6. **hstack拼接**：所有稀疏特征拼接后输入LR

## Actionable Guidance

- 字符n-gram必须分多尺度：2-4(字符级) + 4-6(词根级) + 5-7(短语级)
- 标点序列是作者识别的独特信号，不要忽略
- LR比XGBoost更适合稀疏高维特征(不需要降维)
- n-gram特征与DeBERTa嵌入是正交信号，集成效果最佳

**Condition**: 文本分类，尤其作者识别
**Confidence**: MEDIUM
