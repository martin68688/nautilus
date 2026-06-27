---
title: INDEX_BUG验证集泄露：pandas reset_index后用原DataFrame索引导致标签错位
confidence: CRITICAL
evidence: [Run171209 val=0.00864, Run183931 val=0.04434, Run190327 val=0.01097]
---

# INDEX_BUG验证集泄露：pandas reset_index后用原DataFrame索引导致标签错位

## 严重性：CRITICAL — 这是MLEvolve运行中最频繁、最隐蔽的bug，已导致3+次运行的验证集分数完全失真

## Bug根因

代码执行了 `reset_index(drop=True)` 后，又用 `train_set.index.tolist()` 作为索引去访问**原始 `train_df`** 的数据，导致索引错位。

### 错误代码模式（出现于 Run171209, Run183931）

```python
# Step 1: 用skf.split获取split索引
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

# Step 2: iloc切片 + reset_index — 这里index变成了0,1,2,...
train_set = train_df.iloc[train_idx].reset_index(drop=True)  # index: 0,1,2,...
val_set = train_df.iloc[val_idx].reset_index(drop=True)      # index: 0,1,2,...

# Step 3: 用train_set.index作为索引去访问原始train_df — BUG!
train_indices = train_set.index.tolist()  # [0, 1, 2, 3, ...]
val_indices = val_set.index.tolist()      # [0, 1, 2, 3, ...]

# Step 4: 用这些0-based索引访问原始DataFrame
train_texts_orig = train_df["text"].values       # 长度=19579
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])

train_texts_final = train_texts_orig[train_indices]   # 取的是前N条！
train_labels_final = train_labels_orig[train_indices]  # 标签与文本不匹配！
val_texts_final = train_texts_orig[val_indices]        # 取的是中间M条！
val_labels_final = train_labels_orig[val_indices]      # 标签完全错位！
```

### 为什么这是泄露

`reset_index(drop=True)` 后，`train_set.index` 变成 `[0, 1, 2, ..., N-1]`，但原始 `train_df` 的索引是全局的。当用 `[0, 1, 2, ...]` 去索引 `train_df` 的 `.values` 数组时：
- **训练文本**：取的是 `train_df` 的前 N 行文本（这些恰好是训练集的文本）
- **训练标签**：取的是 `train_df` 的前 N 行标签（但这些标签对应的是**不同的行**）
- **验证文本**：取的是 `train_df` 的第 N 到 N+M 行文本
- **验证标签**：同样错位

由于标签和文本来自不同的行，模型在训练时看到的标签分布与真实分布不同，验证时计算的 log_loss 完全失真。最恶劣的情况是标签泄露到了训练集中，导致验证集上的 log_loss 虚假极低（0.008~0.05）。

### 正确写法

```python
# 方案A：不要reset_index，直接用iloc位置索引
train_set = train_df.iloc[train_idx]  # 保留原始index
val_set = train_df.iloc[val_idx]

# 用.iloc位置索引，不用.index
train_texts_final = train_set["text"].values
train_labels_final = np.array([author_map[a] for a in train_set["author"].values])

# 方案B：reset_index后直接从子DataFrame取数据
train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

# 直接从train_set/val_set取数据，不要回溯train_df
train_texts_final = train_set["text"].values
train_labels_final = np.array([author_map[a] for a in train_set["author"].values])

# 方案C：用numpy数组直接索引
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
# 直接用skf.split返回的numpy索引
train_texts_final = train_texts_orig[train_idx]
train_labels_final = train_labels_orig[train_idx]
val_texts_final = train_texts_orig[val_idx]
val_labels_final = train_labels_orig[val_idx]
```

## 已确认受影响的运行

| Run | 验证集 Log Loss | 是否INDEX_BUG | 推理脚本注释 |
|-----|----------------|-------------|------------|
| Run171209 | 0.00864 | **是** | "INDEX_BUG - 验证集泄露" |
| Run183931 | 0.04434 | **是** | "INDEX_BUG - 验证集泄露" |
| Run190327 | 0.01097 | 索引正确但仍0.01级别 | 可疑 |
| Run190327 Top3 | 0.05066 | 部分影响 | "有INDEX_BUG但不影响推理" |

## 泄露审查阈值修复

原 `should_check_data_leakage` 函数阈值：
```python
# 修改前：只有metric==0.0才触发，INDEX_BUG最低0.00864，完全无法拦截
is_extreme = (metric_value == 0.0)  # minimize

# 修改后：metric<=0.1即触发审查，可拦截所有已知的INDEX_BUG案例
is_extreme = (metric_value <= 0.1)  # minimize
```

## Actionable Guidance

- **🔴 CRITICAL: 永远不要在 `reset_index(drop=True)` 后用 `.index` 回溯原始DataFrame**
- 切分数据后，直接从子DataFrame取数据，或直接用skf.split返回的numpy索引
- 验证集 Log Loss < 0.1 在此任务上极大概率是bug/泄露
- 代码review时必须检查 train/val split 后的数据对齐
