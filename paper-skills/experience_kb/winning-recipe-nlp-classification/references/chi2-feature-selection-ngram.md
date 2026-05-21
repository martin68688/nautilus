# Chi-Squared Feature Selection for N-gram Features

## Finding
MaxAbsScaler + SelectKBest(chi2, k=10000) on 13,500-dim sparse n-gram features reduces noise and prevents LR overfitting.

## Evidence
- Run 091845 top4: with chi2 → val=0.2517
- Earlier runs without chi2: val=0.2653+
- LR standalone with chi2: ~0.3 val logloss

## Mechanism
- Chi-squared test measures feature-class independence
- Low chi2 = noise (randomly distributed across classes)
- MaxAbsScaler required before chi2 (needs non-negative values)
- MaxAbsScaler preserves sparsity (unlike StandardScaler)
- k=10000 removes 26% least discriminative features

## Condition
Sparse n-gram features with Logistic Regression or linear models. Must MaxAbsScaler first.
