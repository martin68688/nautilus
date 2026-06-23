# Code Structure and Validation

## Correct Script Ordering

```
1. Imports
2. Config / hyperparameters
3. Data loading
4. Label encoding
5. Cross-validation split
6. FOR EACH FOLD:
   a. Subset train/val data
   b. Feature engineering
   c. Tokenization
   d. Model initialization
   e. Training loop
   f. Evaluation
7. Aggregate results
8. Save outputs
```

## Incremental Build Checklist

| Iteration | What to add | Validate |
|-----------|-------------|----------|
| 1 | Baseline fine-tune | End-to-end run completes |
| 2 | + Custom loss | Loss decreases, predictions valid |
| 3 | + Feature fusion | Correct shape, no ordering bugs |
| 4 | + SWA / weight averaging | Weights load correctly |

**Rule**: Never bundle changes from multiple rows into a single iteration.

## Before-Execution Validation

1. **Variable lifecycle trace**: Scan top-to-bottom.
2. **Import completeness**: Every class/function used is imported.
3. **Function signature consistency**: After modifying signatures, update ALL call sites.
4. **Shape consistency**: Tensor/dataframe shapes compatible at each step.
5. **Fold-index scope**: Code using fold indices is inside the fold loop.
