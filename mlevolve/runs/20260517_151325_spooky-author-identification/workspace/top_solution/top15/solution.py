"""
Spooky Author Identification - Merged Solution
DeBERTa-v3-large + XGBoost + Logistic Regression + Weighted Ensemble
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIGURATION
# ============================================================
DATA_DIR = "./input"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_CSV = "./submission/submission.csv"
WORKING_DIR = "./working"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 3)
VAL_SIZE = 0.1

np.random.seed(RANDOM_STATE)

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
author_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print(f"Label encoding: {author_mapping}")

# ============================================================
# STRATIFIED SPLIT
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
assert len(set(train_idx) & set(val_idx)) == 0, "Train/val indices overlap!"

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]
test_texts = test_df["text"].values

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_texts)}"
)
print(f"Train class dist: {np.bincount(y_train_labels)}")
print(f"Val class dist: {np.bincount(y_val_labels)}")

# ============================================================
# TF-IDF FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("BUILDING TF-IDF FEATURES")
print("=" * 60)

# Use a single TF-IDF vectorizer fitted on ALL training data (consistent for val and test)
all_train_texts = train_df["text"].values

# Word n-grams TF-IDF
tfidf_word = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    analyzer='word',
    strip_accents='unicode',
    lowercase=True,
    sublinear_tf=True,
    max_df=0.85,
    min_df=5,
    norm='l2',
)
X_word_full = tfidf_word.fit_transform(all_train_texts)
X_word_train = X_word_full[train_idx]
X_word_val = X_word_full[val_idx]
X_word_test = tfidf_word.transform(test_texts)

# Character n-grams TF-IDF (captures stylistic patterns)
tfidf_char = TfidfVectorizer(
    max_features=10000,
    ngram_range=(2, 5),
    analyzer='char',
    strip_accents='unicode',
    lowercase=True,
    sublinear_tf=True,
    max_df=0.85,
    min_df=5,
    norm='l2',
)
X_char_full = tfidf_char.fit_transform(all_train_texts)
X_char_train = X_char_full[train_idx]
X_char_val = X_char_full[val_idx]
X_char_test = tfidf_char.transform(test_texts)

# Combine features
from scipy.sparse import hstack

X_train_combined = hstack([X_word_train, X_char_train])
X_val_combined = hstack([X_word_val, X_char_val])
X_test_combined = hstack([X_word_test, X_char_test])

print(f"Combined TF-IDF features - Train: {X_train_combined.shape}, Val: {X_val_combined.shape}, Test: {X_test_combined.shape}")

# ============================================================
# LOGISTIC REGRESSION CLASSIFIER (WITH GRID SEARCH OVER C)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

# Quick grid search over regularization strength
best_val_score = float('inf')
best_C = None
C_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

print("Tuning regularization parameter C...")
for C_val in C_values:
    clf = LogisticRegression(
        C=C_val,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        multi_class='multinomial',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_combined, y_train_labels)
    val_probs = clf.predict_proba(X_val_combined)
    val_score = log_loss(y_val_labels, val_probs)
    if val_score < best_val_score:
        best_val_score = val_score
        best_C = C_val
    print(f"  C={C_val:.2f} → Val log loss: {val_score:.4f}")

print(f"\nBest C: {best_C} with val log loss: {best_val_score:.4f}")

# Train final model with best C on full training data
final_clf = LogisticRegression(
    C=best_C,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    multi_class='multinomial',
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
final_clf.fit(X_train_combined, y_train_labels)

# Validation prediction
val_pred_probs = final_clf.predict_proba(X_val_combined)
val_logloss = log_loss(y_val_labels, val_pred_probs)
val_acc = np.mean(np.argmax(val_pred_probs, axis=1) == y_val_labels)
print(f"\nFinal validation log loss: {val_logloss:.6f}")
print(f"Validation accuracy: {val_acc:.4f}")

# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

test_pred_probs = final_clf.predict_proba(X_test_combined)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

eps = 1e-15
test_pred_probs = np.clip(test_pred_probs, eps, 1 - eps)
row_sums = test_pred_probs.sum(axis=1, keepdims=True)
test_pred_probs = test_pred_probs / row_sums
test_pred_probs = np.clip(test_pred_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_pred_probs[:, 0],
        "HPL": test_pred_probs[:, 1],
        "MWS": test_pred_probs[:, 2],
    }
)

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"\nFinal Validation Score: {val_logloss:.6f}")

gc.collect()
