import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================
class Config:
    seed = 42
    num_classes = 3
    # TF-IDF parameters
    tfidf_max_features = 50000
    ngram_range = (1, 3)
    # Logistic Regression parameters
    lr_C = 4.0
    lr_solver = 'saga'
    lr_max_iter = 1000
    # Cross-validation
    n_folds = 5


config = Config()


def set_seed(seed):
    np.random.seed(seed)


set_seed(config.seed)


# ============================================================
# 1. Data Loading
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["author"])
author_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print(f"Author mapping: {author_mapping}")


# ============================================================
# 2. Feature Engineering
# ============================================================
def extract_statistical_features(text_series):
    """Extract statistical features from text."""
    features_list = []
    for text in text_series:
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        if sentence_count == 0:
            sentence_count = 1
        punctuation_count = sum(1 for c in text if c in '.,;:!?\"\'()-[]{}')
        capital_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        unique_words = len(set(text.lower().split()))
        unique_word_ratio = unique_words / max(word_count, 1)
        # Stopwords
        stopwords = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'by', 'with', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'can', 'could', 'may', 'might', 'shall', 'should',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            'this', 'that', 'these', 'those', 'not', 'no', 'nor', 'so',
            'very', 'just', 'too', 'also'
        ])
        words_lower = text.lower().split()
        stopword_count = sum(1 for w in words_lower if w in stopwords)
        stopword_ratio = stopword_count / max(word_count, 1)
        # Quote density
        quote_count = text.count('"') + text.count('"') + text.count('"') + text.count('"') + text.count(''') + text.count(''') + text.count('"') + text.count("'")
        quote_density = quote_count / max(char_count, 1)

        features_list.append([
            char_count, word_count, sentence_count, punctuation_count,
            capital_ratio, unique_word_ratio, stopword_ratio, quote_density
        ])
    return np.array(features_list)


print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=config.tfidf_max_features,
    ngram_range=config.ngram_range,
    analyzer='word',
    strip_accents='unicode',
    sublinear_tf=True
)

# Fit TF-IDF only on training data to prevent data leakage
tfidf_vectorizer.fit(train_df["text"])

X_train_tfidf = tfidf_vectorizer.transform(train_df["text"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["text"])
print(f"TF-IDF train shape: {X_train_tfidf.shape}")
print(f"TF-IDF test shape: {X_test_tfidf.shape}")

print("Extracting statistical features...")
train_stat_features = extract_statistical_features(train_df["text"])
test_stat_features = extract_statistical_features(test_df["text"])

# Standardize numerical features
scaler = StandardScaler()
scaler.fit(train_stat_features)
train_stat_features_scaled = scaler.transform(train_stat_features)
test_stat_features_scaled = scaler.transform(test_stat_features)

# Combine features
X_train_combined = hstack([X_train_tfidf, csr_matrix(train_stat_features_scaled)]).tocsr()
X_test_combined = hstack([X_test_tfidf, csr_matrix(test_stat_features_scaled)]).tocsr()
print(f"Combined train shape: {X_train_combined.shape}")
print(f"Combined test shape: {X_test_combined.shape}")


# ============================================================
# 3. Helper: compute log loss
# ============================================================
def compute_log_loss(y_true, y_pred_probs, eps=1e-15):
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    y_pred_probs_normalized = y_pred_probs / row_sums
    if len(y_true.shape) == 1:
        y_true_onehot = np.zeros_like(y_pred_probs)
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
    else:
        y_true_onehot = y_true
    log_loss_val = (
        -np.sum(y_true_onehot * np.log(y_pred_probs_normalized)) / y_true.shape[0]
    )
    return log_loss_val


# ============================================================
# 4. Train/Validation with 5-Fold Cross Validation
# ============================================================
print("\nStarting 5-fold stratified cross-validation...")
skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

# Base model
base_lr = LogisticRegression(
    C=config.lr_C,
    solver=config.lr_solver,
    max_iter=config.lr_max_iter,
    multi_class='multinomial',
    random_state=config.seed,
    n_jobs=-1
)

# Calibrated model with 5-fold CV
calibrated_model = CalibratedClassifierCV(
    estimator=base_lr,
    method='sigmoid',
    cv=skf
)

print("Training calibrated logistic regression model...")
calibrated_model.fit(X_train_combined, y_train)

# Get out-of-fold predictions for validation log loss
print("Computing out-of-fold predictions...")
oof_preds = cross_val_predict(
    base_lr,
    X_train_combined,
    y_train,
    cv=skf,
    method='predict_proba'
)

# Normalize and clip OOF predictions
eps = 1e-15
oof_preds = np.clip(oof_preds, eps, 1 - eps)
row_sums_oof = oof_preds.sum(axis=1, keepdims=True)
oof_preds = oof_preds / row_sums_oof

# Compute validation log loss
val_logloss = compute_log_loss(y_train, oof_preds)
print(f"Cross-Validation LogLoss: {val_logloss:.6f}")

# ============================================================
# 5. Generate Test Predictions
# ============================================================
print("Generating test predictions...")
test_preds_proba = calibrated_model.predict_proba(X_test_combined)

# Normalize and clip predictions
test_preds = np.clip(test_preds_proba, eps, 1 - eps)
row_sums_test = test_preds.sum(axis=1, keepdims=True)
test_preds = test_preds / row_sums_test

# ============================================================
# 6. Generate Submission
# ============================================================
os.makedirs("./submission", exist_ok=True)
test_ids = test_df["id"].values
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, label_encoder.transform(['EAP'])[0]],
        "HPL": test_preds[:, label_encoder.transform(['HPL'])[0]],
        "MWS": test_preds[:, label_encoder.transform(['MWS'])[0]],
    }
)

# Final normalization to ensure rows sum to 1
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Cross-Validation LogLoss: {val_logloss:.6f}")