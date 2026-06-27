# Multi-Scale Feature Engineering is the Foundation of Heterogeneous Ensemble

## Finding
Three categories of features, each routed to the model that exploits them best, form the backbone of the winning ensemble.

## Feature Categories and Model Routing

### Dense Features → XGBoost
| Feature Group | Raw Dims | After Processing | Key Signals |
|---|---|---|---|
| Stylometric | 30 | 26 (VarianceThreshold 0.001) | punctuation frequencies, vocabulary richness, archaic/emotional/Lovecraft ratios |
| Readability | 4 | 4 (StandardScaler) | Flesch score, ARI, avg syllables, complex word ratio |
| POS Approximation | 5 | 5 (StandardScaler) | noun/verb/adj/adv suffix ratios, content word ratio |
| DeBERTa [CLS] | 1024 | 1024 (no processing) | High-level semantic representation |

Total XGBoost input: ~1059 dimensions

### Sparse Features → Logistic Regression
| Vectorizer | ngram_range | max_features |
|---|---|---|
| char_short | (2,4) | 3000 |
| char_med | (4,6) | 3000 |
| char_long | (5,7) | 2000 |
| word | (1,3) | 5000 |
| punct | (2,4) | 500 |

Total: 13,500 → chi2(k=10000) → 10,000 dimensions

### Raw Text → DeBERTa
Tokenized sequences, max_length=512.

## Condition
NLP classification with ensemble, especially authorship attribution / style classification.
