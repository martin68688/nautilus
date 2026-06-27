# Pipeline Ordering and Code Structure

## Table of Contents
1. [Variable Lifetime Tracing](#variable-lifetime-tracing)
2. [Fold-Dependent vs Fold-Agnostic Preprocessing](#fold-dependent-vs-fold-agnostic-preprocessing)
3. [Label Encoding Consistency](#label-encoding-consistency)
4. [Fitted Object Checklist](#fitted-object-checklist)
5. [Data Handoff Verification](#data-handoff-verification)
6. [Modularization Guidelines](#modularization-guidelines)

## Variable Lifetime Tracing

Before executing, trace each key variable from definition to usage:
- Where it is assigned (which code block)
- Where it is used (all subsequent references)
- Whether any usage precedes assignment (flag as fatal)

Key variables: `train_indices`, `val_indices`, `train_loader`, fitted vectorizers, label encoders.

## Fold-Dependent vs Fold-Agnostic Preprocessing

### Fold-Dependent (inside CV loop)
```python
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold = X.iloc[train_idx]
    vectorizer = TfidfVectorizer(...)
    X_train_vec = vectorizer.fit_transform(X_train_fold)
    X_val_vec = vectorizer.transform(X.iloc[val_idx])
```

### Fold-Agnostic (before CV loop)
```python
vectorizer = TfidfVectorizer(...)
X_vec = vectorizer.fit_transform(X)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_vec)):
    # subset already-vectorized data
```

**Decide explicitly** which approach to use. Mixing causes temporal dependency bugs.

## Label Encoding Consistency

```python
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
val_labels_encoded = le.transform(val_labels)  # transform, NOT fit_transform

dataset = TextDataset(texts, train_labels_encoded)  # NOT raw strings
```

Verify: `assert train_labels_encoded.dtype in (np.int32, np.int64)`

## Fitted Object Checklist

- [ ] Scalers: fitted on training fold only, `.transform()` on validation
- [ ] LabelEncoders: fitted on training labels, `.transform()` on validation
- [ ] TF-IDF vectorizers: fitted on training texts only
- [ ] All fitted objects stored as attributes or passed explicitly

## Data Handoff Verification

```python
# Verify saved files exist and have correct shapes
for f in ['X_train.npy', 'X_test.npy', 'y_train.npy']:
    assert os.path.exists(f'./working/{f}'), f'Missing: {f}'

X_train = np.load('./working/X_train.npy')
assert X_train.shape[0] == y_train.shape[0], 'Row mismatch!'
assert X_train.shape[1] == X_test.shape[1], 'Feature mismatch!'
assert not np.any(np.isnan(X_train)), 'NaN in X_train!'
```

## Modularization Guidelines

- Break complex pipelines into discrete functions: `build_features()`, `train_transformer()`, `ensemble_predictions()`.
- Avoid monolithic scripts exceeding ~300 lines.
- Do not add multiple ambitious modifications simultaneously without testing each incrementally.
- Prefer full-file rewrites over stacking many SEARCH/REPLACE edits for complex code.