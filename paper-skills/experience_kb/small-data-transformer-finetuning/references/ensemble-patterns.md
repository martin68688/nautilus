# Ensemble Patterns for Small-Data Text Classification

## Rationale

Different model families make different assumptions about data, so their
prediction errors are partially decorrelated.

## Component Models

### 1. Transformer Fine-tuning
- Use DeBERTa-v3-large with standard classification head.
- Extract CLS embeddings after fine-tuning for use in GBT component.

### 2. Gradient-Boosted Trees (XGBoost)
- Input: concatenate DeBERTa CLS embeddings with handcrafted stylometric features.
- Tune with early stopping on validation log loss.

### 3. Linear Model on TF-IDF
- Use Logistic Regression on TF-IDF n-gram features.
- Provides complementary sparse-text signal.

## Weight Grid Search

```python
import itertools
import numpy as np
from sklearn.metrics import log_loss

def grid_search_weights(preds_dict, y_val, step=0.05):
    names = list(preds_dict.keys())
    best_loss = float('inf')
    best_weights = None
    weight_values = np.arange(0.0, 1.0 + step, step)
    for combo in itertools.product(weight_values, repeat=len(names)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        blended = sum(w * preds_dict[n] for w, n in zip(combo, names))
        loss = log_loss(y_val, blended)
        if loss < best_loss:
            best_loss = loss
            best_weights = dict(zip(names, combo))
    return best_weights, best_loss
```

**Key insight**: The optimal weighting may be non-obvious. Always let the data decide.
