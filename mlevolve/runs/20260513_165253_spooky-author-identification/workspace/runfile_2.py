import os
os.sched_setaffinity(0, {6, 19, 20, 21, 22, 23, 24, 25, 26, 27})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import re
import os
import joblib
import scipy.sparse as sp

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


# --- Basic cleaning ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# --- Feature Engineering ---
def extract_stylometric_features(text):
    features = {}
    text = str(text)
    features["char_count"] = len(text)
    features["word_count"] = len(text.split())
    features["avg_word_len"] = features["char_count"] / max(features["word_count"], 1)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_sentence_len"] = features["word_count"] / features["sentence_count"]
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["dash_count"] = text.count("\u2014") + text.count("-")
    features["semicolon_count"] = text.count(";")
    features["colon_count"] = text.count(":")
    features["comma_count"] = text.count(",")
    features["quote_count"] = text.count('"') + text.count("'")
    features["paren_count"] = text.count("(") + text.count(")")
    features["punctuation_ratio"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["dash_count"]
        + features["semicolon_count"]
        + features["colon_count"]
        + features["comma_count"]
    ) / max(features["char_count"], 1)
    features["capital_word_ratio"] = sum(
        1 for w in text.split() if w and w[0].isupper()
    ) / max(features["word_count"], 1)
    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "whence",
        "thence",
        "wherefore",
        "therein",
        "thereof",
        "thereon",
        "hither",
        "thither",
    ]
    features["archaic_count"] = sum(text.lower().count(w) for w in archaic_words)
    stop_words = set(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "was",
            "were",
            "had",
            "has",
            "have",
            "been",
            "being",
            "is",
            "are",
            "be",
            "not",
            "no",
            "nor",
        ]
    )
    words = text.lower().split()
    features["stop_word_ratio"] = sum(1 for w in words if w in stop_words) / max(
        len(words), 1
    )
    features["unique_word_ratio"] = len(set(words)) / max(len(words), 1)
    features["ly_ending_count"] = len(re.findall(r"\w+ly\b", text.lower()))
    features["ing_ending_count"] = len(re.findall(r"\w+ing\b", text.lower()))
    features["ed_ending_count"] = len(re.findall(r"\w+ed\b", text.lower()))
    features["tion_ending_count"] = len(re.findall(r"\w+tion\b", text.lower()))
    features["number_count"] = len(re.findall(r"\d+", text))
    return features


train_features = train_df["text"].apply(extract_stylometric_features)
test_features = test_df["text"].apply(extract_stylometric_features)
train_feature_df = pd.DataFrame(train_features.tolist()).fillna(0)
test_feature_df = pd.DataFrame(test_features.tolist()).fillna(0)

scaler = StandardScaler()
numerical_cols = train_feature_df.columns
train_scaled = scaler.fit_transform(train_feature_df)
test_scaled = scaler.transform(test_feature_df)

# Text-based feature extraction
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=3000, lowercase=True
)
train_char = char_vectorizer.fit_transform(train_df["clean_text"])
test_char = char_vectorizer.transform(test_df["clean_text"])

word_vectorizer = CountVectorizer(
    analyzer="word", ngram_range=(1, 2), max_features=3000, stop_words="english"
)
train_word = word_vectorizer.fit_transform(train_df["clean_text"])
test_word = word_vectorizer.transform(test_df["clean_text"])

tfidf_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=4000,
    sublinear_tf=True,
    stop_words="english",
)
train_tfidf = tfidf_vectorizer.fit_transform(train_df["clean_text"])
test_tfidf = tfidf_vectorizer.transform(test_df["clean_text"])

# Combine all features
X_train_full = hstack(
    [csr_matrix(train_scaled), train_char, train_word, train_tfidf]
).tocsr()
X_test = hstack([csr_matrix(test_scaled), test_char, test_word, test_tfidf]).tocsr()

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(train_df["author"])
test_ids = test_df["id"].values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_full, test_size=0.15, random_state=42, stratify=y_full
)

# XGBoost parameters
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "tree_method": "hist",
    "seed": 42,
}

# 5-fold CV on combined train+val data for final predictions
X_full = sp.vstack([X_train, X_val])
y_full_combined = np.concatenate([y_train, y_val])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
fold_test_probs = []
best_score = float("inf")
best_model = None

print("Starting 5-fold cross-validation training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full_combined)):
    X_tr, X_val_fold = X_full[train_idx], X_full[val_idx]
    y_tr, y_val_fold = y_full_combined[train_idx], y_full_combined[val_idx]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    val_preds = model.predict(dval)
    val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
    row_sums = val_preds.sum(axis=1, keepdims=True)
    val_preds = val_preds / row_sums
    fold_score = log_loss(y_val_fold, val_preds)
    fold_scores.append(fold_score)
    print(f"Fold {fold+1}: Validation Log Loss = {fold_score:.6f}")
    dtest = xgb.DMatrix(X_test)
    fold_test_probs.append(model.predict(dtest))
    if fold_score < best_score:
        best_score = fold_score
        best_model = model

avg_cv_score = np.mean(fold_scores)
std_cv_score = np.std(fold_scores)
print(f"\nCross-Validation Log Loss: {avg_cv_score:.6f} (+/- {std_cv_score:.6f})")

# Average test predictions across folds
test_probs = np.mean(fold_test_probs, axis=0)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums

# Full training on combined data
dtrain_full = xgb.DMatrix(X_full, label=y_full_combined)
final_model = xgb.train(
    params,
    dtrain_full,
    num_boost_round=(
        best_model.best_iteration + 1 if hasattr(best_model, "best_iteration") else 200
    ),
    evals=[(dtrain_full, "train")],
    verbose_eval=False,
)
dtest_final = xgb.DMatrix(X_test)
test_preds_final = final_model.predict(dtest_final)
test_preds_final = np.clip(test_preds_final, 1e-15, 1 - 1e-15)
row_sums = test_preds_final.sum(axis=1, keepdims=True)
test_preds_final = test_preds_final / row_sums

# Create submission
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_final[:, label_encoder.transform(["EAP"])[0]],
        "HPL": test_preds_final[:, label_encoder.transform(["HPL"])[0]],
        "MWS": test_preds_final[:, label_encoder.transform(["MWS"])[0]],
    }
)
os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission_8ecd35a9d2394b888648f8355daff214.csv", index=False)
print(f"\nSubmission saved to ./submission/submission_8ecd35a9d2394b888648f8355daff214.csv")

score = avg_cv_score
print(f"Final Validation Score: {score}")
