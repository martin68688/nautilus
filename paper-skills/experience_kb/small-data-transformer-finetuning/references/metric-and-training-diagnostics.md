# Metric and Training Diagnostics

## Custom Log-Loss Metric Validation

```python
from sklearn.metrics import log_loss

def validate_metric(compute_fn):
    uniform = np.full((4, 3), 1.0 / 3)
    assert np.isfinite(compute_fn(uniform)), "NaN on uniform inputs"
    extreme = np.array([[0.999, 0.0005, 0.0005], ...])
    assert np.isfinite(compute_fn(extreme)), "NaN on extreme inputs"
```

## Root-Cause Debugging for NaN Metrics

| Symptom-Based Fix (WRONG) | Root-Cause Fix (CORRECT) |
|---|---|
| Initialize `best_model_state` to empty dict | Fix normalization division in metric function |
| Skip epochs where metric is NaN | Add epsilon clipping before log computation |

## Training Divergence Detection

When training loss drops to near zero while validation loss increases:
1. **Reduce learning rate.** Halve or quarter the LR.
2. **Increase regularization.** Raise dropout, increase weight decay.
3. **Reduce training epochs.** Use early stopping with patience 2–3.
4. **Reduce model capacity.** Freeze lower layers, use smaller model.
5. **Add label smoothing.** Use 0.05 in the loss function.

## Recommended Defaults

| Parameter | Conservative Starting Value |
|---|---|
| Learning rate | ≤ 2e-5 |
| Dropout | ≥ 0.1 |
| Weight decay | ≥ 0.01 |
| Batch size | 16–32 |
| Epochs | 3–5 with early stopping |
| Label smoothing | 0.05 |
