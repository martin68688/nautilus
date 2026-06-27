# Ensemble Strategies for Small-Data Text Classification

## Table of Contents
1. [Multi-Model Architecture](#multi-model-architecture)
2. [Ensemble Weight Optimization](#ensemble-weight-optimization)
3. [Checkpoint Ensembling](#checkpoint-ensembling)
4. [Stylometric Features](#stylometric-features)
5. [Pseudolabeling](#pseudolabeling)
6. [Try-Except Isolation](#try-except-isolation)
7. [XGBoost Early Stopping](#xgboost-early-stopping)

## Multi-Model Architecture

| Model | Features | Signal Type |
|-------|----------|-------------|
| Fine-tuned transformer | Raw text | Contextual semantics |
| XGBoost | CLS embeddings + stylometric | Semantic + stylistic |
| Logistic Regression | TF-IDF n-grams | Sparse lexical patterns |

## Ensemble Weight Optimization

```python
from scipy.optimize import minimize

def ensemble_log_loss(weights, preds_list, y_val):
    weighted = sum(w * p for w, p in zip(weights, preds_list))
    weighted = np.clip(weighted, 1e-15, 1 - 1e-15)
    return log_loss(y_val, weighted)

result = minimize(ensemble_log_loss, x0=[1/len(preds_list)]*len(preds_list),
    args=(preds_list, y_val), method='SLSQP',
    bounds=[(0, 1)]*len(preds_list),
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
```

## Checkpoint Ensembling

- Save checkpoints at each epoch.
- Select top-K by validation log loss (K=3 default).
- Average predictions weighted by inverse validation log loss.

## Stylometric Features

```python
def extract_stylometric_features(texts):
    features = []
    for text in texts:
        words = text.split()
        n_words = len(words)
        feat = [n_words, len(text), len(text)/max(n_words,1),
                len(set(words))/max(n_words,1)]
        features.append(feat)
    return np.array(features)
```

## Pseudolabeling

1. Train initial model on handcrafted features.
2. Predict labels on test set.
3. Retain only samples with max probability > 0.95.
4. Append to training set.

## Try-Except Isolation

```python
trained_models = {}
try:
    trained_models['deberta'] = train_deberta(...)
except Exception as e:
    print(f"DeBERTa failed: {e}")
try:
    trained_models['xgb'] = train_xgboost(...)
except Exception as e:
    print(f"XGBoost failed: {e}")
assert len(trained_models) > 0, "All models failed"
```

## XGBoost Early Stopping

**DO NOT** pass `early_stopping_rounds` to `.fit()`. Use callbacks or model init:
```python
from xgboost.callback import EarlyStopping
model.fit(X, y, eval_set=[(X_val, y_val)],
    callbacks=[EarlyStopping(rounds=50, save_best=True)])
```