# Ensemble and Stacking Patterns

## Leakage-Safe Stacking

**Observed**: Training XGBoost on all 3 folds' OOF and validating on a slice yielded
log loss 0.0102 (leaked). Restricting to fold-1 training data and validating on fold-1
held-out data yielded trustworthy log loss 0.1975.

**Correct Pattern**: Train meta-model ONLY on fold k's training OOF. Validate meta-model
ONLY on fold k's held-out OOF.

## Weighted Ensemble Grid Search

```python
def grid_search_ensemble_weights(preds_list, y_val, weight_grid=np.arange(0, 1.01, 0.05)):
    n_models = len(preds_list)
    best_loss = float('inf')
    best_weights = None
    for weights in product(weight_grid, repeat=n_models):
        s = sum(weights)
        if s == 0: continue
        w = np.array(weights) / s
        ensemble_preds = sum(w[i] * preds_list[i] for i in range(n_models))
        loss = log_loss(y_val, ensemble_preds)
        if loss < best_loss:
            best_loss = loss
            best_weights = w
    return best_weights, best_loss
```

## Index-Based Splitting

```python
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, stratify=df['label'], random_state=42)
tfidf_train, tfidf_val = tfidf_matrix[train_idx], tfidf_matrix[val_idx]
```

## Transformer Fine-tuning Settings

| Parameter              | Value   |
|------------------------|---------|
| Learning rate          | 2e-5    |
| Label smoothing        | Enabled |
| Gradient clipping      | Enabled |
| Mixed precision        | Enabled |
| Early stopping patience| 5       |

## XGBoost Early Stopping API Pitfall

Place `early_stopping_rounds` in the **constructor**, not in `.fit()`.

```python
model = XGBClassifier(n_estimators=500, early_stopping_rounds=50, eval_metric='mlogloss')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```
