# Ensemble Weight Grid Search

## Finding
Grid search (step=0.05) over 3-model weights outperforms equal averaging by ~0.02+ logloss.

## Evidence
- Run4: grid search vs equal averaging → gap ~0.02+
- DeBERTa typically: ~0.55-0.70, XGBoost: ~0.20-0.35, LR: ~0.05-0.15

## Mechanism
- w1 ∈ [0.1, 0.9), w2 ∈ [0.1, 0.9), w3 = 1 - w1 - w2, constraint 0.05 ≤ each ≤ 0.9
- Step 0.05 → ~289 valid combinations
- Objective: minimize validation logloss

## Future
Bayesian optimization with Dirichlet prior for continuous search.

## Condition
Heterogeneous ensemble with 3+ models.
