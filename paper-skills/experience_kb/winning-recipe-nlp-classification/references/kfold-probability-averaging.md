# 5-Fold Cross-Validation Probability Averaging

## Finding
5-fold StratifiedKFold training with softmax probability averaging is the highest-priority, lowest-risk next optimization.

## Evidence
- All current runs use single 90/10 split → high variance
- CV averages out lucky/unlucky split effect

## Mechanism
- Train 5 independent models (80/20 each fold)
- Test: arithmetic mean of 5 softmax probability vectors
- Reduces variance without architecture changes
- Each fold uses same strategy (partial unfreezing + MSD + WRS + LS 0.1)

## Anti-Leakage Rules for K-Fold

Every fold must strictly isolate its validation split. Violating any of these rules causes the validation score to be untrustworthy.

### Rule 1: Feature transformers fit on training fold only
Each fold's scaler, vectorizer, and selector MUST be fit ONLY on that fold's training portion. Do NOT pre-fit on the full dataset before splitting.

```python
# WRONG: fit on full data, then split — LEAKAGE
scaler = StandardScaler().fit(X_full)     # sees all data including fold val
X_full_scaled = scaler.transform(X_full)  # val information leaked into scaler
for fold, (train_idx, val_idx) in enumerate(skf.split(...)):
    X_train, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]

# CORRECT: split first, then fit on train fold only
for fold, (train_idx, val_idx) in enumerate(skf.split(...)):
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
```

### Rule 2: Same applies to ALL transformers
| Transformer | WRONG (leakage) | CORRECT |
|---|---|---|
| StandardScaler | fit on full data | fit on fold train only |
| TfidfVectorizer | fit on full data | fit on fold train only |
| CountVectorizer | fit on full data | fit on fold train only |
| MaxAbsScaler | fit on full data | fit on fold train only |
| SelectKBest(chi2) | fit on full data | fit on fold train only |
| VarianceThreshold | fit on full data | fit on fold train only |
| LabelEncoder | fit on full data | fit on fold train only |

### Rule 3: Do not use validation data for any training decision
- Early stopping must use the fold's validation set, not an external holdout
- Ensemble weight search must use the fold's validation set
- Do NOT tune hyperparameters based on aggregated CV scores then retrain on full data — this is a form of leakage

### Rule 4: Test set is never touched during CV
- Test predictions from each fold model are averaged
- Test set must NOT be used in any fit/transform/train step
- The full training data is only split into fold train + fold val; test is separate

### Rule 5: DeBERTa tokenizer is safe to share
- Tokenizer vocabulary is fixed (pretrained), no data leakage
- Can use one tokenizer instance across all folds

## Complete Implementation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_test_probs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_texts, y_labels)):
    X_tr, X_va = X_texts[train_idx], X_texts[val_idx]
    y_tr, y_va = y_labels[train_idx], y_labels[val_idx]

    # --- Fit ALL transformers on fold training data ONLY ---
    stylo_scaler = StandardScaler().fit(train_stylo)
    train_stylo_s = stylo_scaler.transform(train_stylo)
    val_stylo_s = stylo_scaler.transform(val_stylo)

    char_vec = TfidfVectorizer(...).fit(X_tr)
    train_char = char_vec.transform(X_tr)
    val_char = char_vec.transform(X_va)

    sparse_scaler = MaxAbsScaler().fit(train_sparse)
    chi2_sel = SelectKBest(chi2, k=10000).fit(
        sparse_scaler.transform(train_sparse), y_tr)

    # --- Train DeBERTa on fold training data ---
    model = train_deberta(X_tr, y_tr, X_va, y_va)

    # --- Extract embeddings and predictions ---
    train_emb = extract_embeddings(model, X_tr)
    val_emb = extract_embeddings(model, X_va)
    test_emb = extract_embeddings(model, X_test)  # transform only, no fit

    # --- Train XGBoost and LR on fold features ---
    xgb_model = xgb.XGBClassifier(...).fit(
        xgb_train_features, y_tr,
        eval_set=[(xgb_val_features, y_va)])

    lr_model = LogisticRegression(...).fit(train_sparse_selected, y_tr)

    # --- Get test predictions ---
    test_probs = get_ensemble_test_probs(model, xgb_model, lr_model, X_test)
    all_test_probs.append(test_probs)

# Average probabilities across folds
final_test_probs = np.mean(all_test_probs, axis=0)
```

## Expected Impact
- 5-10% test set improvement
- Training cost: 5x (parallelizable across GPUs)

## Condition
When single-model variance is the bottleneck. Compatible with all existing techniques.
