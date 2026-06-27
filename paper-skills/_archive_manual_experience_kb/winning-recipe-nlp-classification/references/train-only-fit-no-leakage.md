# Train-Only Fit Prevents Feature-Level Data Leakage

## Finding
All feature transformations must be fit on training data ONLY. Val/test are only transformed.

## Evidence
- Run4 original: punctuation CountVectorizer fit on train+val+test → leakage
- Fixed in Run 091845 inference script → more reliable

## Leakage Checklist
| Component | Correct | Wrong |
|---|---|---|
| StandardScaler | fit_transform(train) → transform(val/test) | fit on all |
| VarianceThreshold | fit_transform(train) → transform(val/test) | fit on all |
| TfidfVectorizer | fit_transform(train) → transform(val/test) | fit on all |
| CountVectorizer | fit_transform(train) → transform(val/test) | fit on all |
| MaxAbsScaler | fit_transform(train) → transform(val/test) | fit on all |
| SelectKBest(chi2) | fit(train, y_train) → transform(val/test) | fit on all |

## Condition
Any ML pipeline with feature engineering. Mandatory.
