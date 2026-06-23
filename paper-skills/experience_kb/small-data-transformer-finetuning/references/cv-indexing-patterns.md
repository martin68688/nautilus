# Cross-Validation Indexing Patterns

## Safe Global Feature Extraction

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_full = vectorizer.fit_transform(df["text"])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    X_train = X_full[train_idx]   # positional indexing — safe
    X_val = X_full[val_idx]
```

## Safe Per-Fold Feature Extraction

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_full)):
    df_train_fold = df.iloc[train_idx]
    df_val_fold = df.iloc[val_idx]
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_fold = vectorizer.fit_transform(df_train_fold["text"])
    X_val_fold = vectorizer.transform(df_val_fold["text"])
```

## Common Pitfall: Index Space Mismatch

```python
# DANGEROUS: recomputing features inside loop but using outer indices
X_val = X_train_fold.toarray()[val_idx]  # IndexError if val_idx >= 14096!
```

**Fix**: Use `.iloc` to subset the DataFrame first, or use global extraction with positional indexing.

## Pre-Training Validation Checklist

1. `assert max(val_idx) < X_val.shape[0]`
2. `assert X_train.shape[0] == len(train_idx)`
3. `assert len(y_train) == X_train.shape[0]`
4. `assert len(set(train_idx) & set(val_idx)) == 0`
