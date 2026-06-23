# Proven Strategies for Small-Data Transformer Fine-tuning

Detailed implementation guidance for the core workflow steps. Each strategy
below was decisive in achieving strong validation log loss on small text
classification tasks.

## Table of Contents
1. [Multi-Task Auxiliary Regression Head](#multi-task-auxiliary-regression-head)
2. [Metric-Aligned Checkpoint Selection](#metric-aligned-checkpoint-selection)
3. [Post-Hoc Probability Calibration](#post-hoc-probability-calibration)
4. [Diverse Multi-Stream Ensemble](#diverse-multi-stream-ensemble)
5. [Temporal Checkpoint Ensembling](#temporal-checkpoint-ensembling)
6. [High-Confidence Pseudolabeling](#high-confidence-pseudolabeling)

---

## Multi-Task Auxiliary Regression Head

**Goal**: Regularize the transformer by forcing the CLS embedding to encode interpretable stylistic information.

**Architecture**:
- Transformer backbone (e.g., DeBERTa-v3-base)
- Classification head: label-smoothed cross-entropy on target labels
- Regression head: MSE loss predicting handcrafted features from the same CLS embedding

**Loss**: `total_loss = cls_loss + alpha * reg_loss`
- Use a **decaying alpha** (e.g., start at 0.5, decay to 0.01).

---

## Metric-Aligned Checkpoint Selection

**Problem:** Training cross-entropy and validation log loss diverge as
transformer models become overconfident.

**Action:** Evaluate validation log loss at each epoch and save the checkpoint
that minimizes that metric.

```python
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate_log_loss(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, path='best_model.pt')
```

---

## Post-Hoc Probability Calibration

Apply lightweight, post-hoc calibration to model outputs:

- **Platt scaling:** Fit logistic regression on logits.
- **Temperature scaling:** Optimize a single temperature parameter on
  validation log loss using L-BFGS.

---

## Diverse Multi-Stream Ensemble

Combine three complementary signal streams:

| Stream | Model Example | Signal Captured |
|---|---|---|
| Deep contextual | DeBERTa-v3 | Semantic context |
| Gradient-boosted trees | XGBoost on embeddings | Non-linear feature interactions |
| Sparse n-grams | TF-IDF + Logistic Regression | Lexical patterns |

### Weight Optimization via SLSQP or Nelder-Mead

```python
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def ensemble_log_loss(weights, preds_list, y_true):
    weighted = sum(w * p for w, p in zip(weights, preds_list))
    return log_loss(y_true, weighted)

constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},)
bounds = [(0, 1)] * len(preds_list)

result = minimize(ensemble_log_loss, x0=np.ones(len(preds_list))/len(preds_list),
    args=(preds_list, y_val), method='SLSQP', bounds=bounds, constraints=constraints)
```

---

## Temporal Checkpoint Ensembling

1. Save checkpoints at regular intervals during extended fine-tuning.
2. Select the **top-3** by lowest validation log loss.
3. Combine using **inverse-loss weighting**: `weight_i = (1/loss_i) / Σ(1/loss_j)`.

---

## High-Confidence Pseudolabeling

1. Train a preliminary model.
2. Predict probabilities on the test set.
3. Keep only samples where `max(predicted_probabilities) > 0.8–0.95`.
4. Use **soft-label probability vectors** as training targets to preserve uncertainty.
5. Retrain on combined original + pseudo-labeled data with reduced max_lr and increased weight_decay.

**Observed result:** Validation log loss reduced from 0.4815 to 0.1859 by
selecting 1714 of 1958 test samples above the 0.8 threshold.