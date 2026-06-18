import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import torch

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode labels
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
test_ids = test_df["id"].values
n_classes = len(le.classes_)

# ============================================================
# SENTENCE TRANSFORMER FEATURE EXTRACTION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load lightweight sentence transformer
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Extract embeddings for all texts
print("Extracting sentence embeddings...")
train_embeddings = st_model.encode(
    train_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True,
)
test_embeddings = st_model.encode(
    test_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True,
)
print(
    f"Embedding shapes - Train: {train_embeddings.shape}, Test: {test_embeddings.shape}"
)

# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================
print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=10000,
    sublinear_tf=True,
    min_df=5,
    max_df=0.85,
    strip_accents="unicode",
)

train_tfidf = tfidf.fit_transform(train_df["text"])
test_tfidf = tfidf.transform(test_df["text"])

# Reduce dimensionality with SVD
svd = TruncatedSVD(n_components=150, random_state=42)
train_tfidf_svd = svd.fit_transform(train_tfidf)
test_tfidf_svd = svd.transform(test_tfidf)
print(
    f"TF-IDF SVD shapes - Train: {train_tfidf_svd.shape}, Test: {test_tfidf_svd.shape}"
)

# ============================================================
# CONCATENATE FEATURES
# ============================================================
X_train = np.hstack([train_embeddings, train_tfidf_svd])
X_test = np.hstack([test_embeddings, test_tfidf_svd])
y_train = train_df["author_encoded"].values

print(f"Combined feature shapes - Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# CROSS-VALIDATION TRAINING - XGBoost only (faster, completed within timeout)
# ============================================================
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), n_classes))
test_preds = np.zeros((len(test_df), n_classes))

print(f"\nStarting {n_folds}-fold cross-validation with XGBoost...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold + 1}/{n_folds}")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model_xgb.fit(
        X_fold_train,
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False,
    )

    val_probs = model_xgb.predict_proba(X_fold_val)
    oof_preds[val_idx] = val_probs
    test_preds += model_xgb.predict_proba(X_test) / n_folds

    val_ll = log_loss(y_fold_val, val_probs)
    print(f"  Validation log loss: {val_ll:.4f}")

# ============================================================
# CALCULATE FINAL OOF SCORE
# ============================================================
oof_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
oof_normalized = oof_clipped / oof_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(y_train, oof_normalized)

print(f"\n{'='*50}")
print(f"Final OOF Validation Score: {final_val_score:.6f}")
print(f"{'='*50}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
test_clipped = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_normalized = test_clipped / test_clipped.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_normalized[:, 0],
        "HPL": test_normalized[:, 1],
        "MWS": test_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Submission columns: {submission.columns.tolist()}")
print(f"Final Validation Score: {final_val_score}")
