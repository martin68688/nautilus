# Pandas Data Splitting Patterns for ML Pipelines

## Correct Indexing: .loc vs .iloc

`.iloc[]` is **strictly positional**. Passing a boolean Series raises `NotImplementedError`.

```python
# WRONG — crashes
train_rows = train_df.iloc[train_df["text"].isin(X_train_texts)]

# CORRECT — use .loc or bracket indexing
mask = train_df["text"].isin(X_train_texts)
train_rows = train_df.loc[mask]
```

## Preserving Split Indices

```python
train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.2, stratify=train_df["label"], random_state=42
)
train_subset = train_df.loc[train_idx]
val_subset = train_df.loc[val_idx]
assert len(set(train_idx) & set(val_idx)) == 0
```

## Staged Validation Checklist

```python
assert df.shape[0] > 0
assert "text" in df.columns
assert not df["text"].isna().any()
assert len(X_train) > 0 and len(X_val) > 0
assert len(set(train_idx) & set(val_idx)) == 0
```
