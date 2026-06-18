import os
os.sched_setaffinity(0, {189})
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import re
import os
import pickle
import warnings
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
import xgboost as xgb
import lightgbm as lgb
import torch

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("./working/features", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)

with open("./working/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


def extract_linguistic_features(texts, is_train=True, scaler=None):
    features = []
    for text in texts:
        feat = []
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        word_lengths = [len(w) for w in words if w]
        feat.append(len(words))
        feat.append(np.mean(word_lengths) if word_lengths else 0)
        feat.append(np.std(word_lengths) if word_lengths else 0)
        feat.append(max(word_lengths) if word_lengths else 0)
        feat.append(len(sentences))
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            feat.append(np.mean(sent_lengths))
            feat.append(np.std(sent_lengths))
        else:
            feat.extend([0, 0])
        punct_counts = {
            "comma": text.count(","),
            "semicolon": text.count(";"),
            "colon": text.count(":"),
            "exclamation": text.count("!"),
            "question": text.count("?"),
            "period": text.count("."),
            "quote": text.count('"') + text.count("'"),
            "dash": text.count("-") + text.count("—"),
            "paren": text.count("(") + text.count(")"),
            "ellipsis": text.count("..."),
        }
        feat.extend(punct_counts.values())
        feat.append(sum(1 for w in words if w and w[0].isupper()))
        feat.append(sum(1 for c in text if c.isupper()))
        feat.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))
        feat.append(sum(1 for c in text if not c.isalnum() and not c.isspace()))
        feat.append(sum(1 for c in text if c.isspace()))
        feat.append(sum(1 for w in words if len(w) > 10) / max(len(words), 1))
        feat.append(sum(1 for w in words if len(w) <= 3) / max(len(words), 1))
        feat.append(feat[3] / max(feat[0], 1))
        features.append(feat)
    feature_names = [
        "word_count",
        "avg_word_len",
        "std_word_len",
        "max_word_len",
        "sentence_count",
        "avg_sent_len_words",
        "std_sent_len_words",
        "comma_count",
        "semicolon_count",
        "colon_count",
        "exclamation_count",
        "question_count",
        "period_count",
        "quote_count",
        "dash_count",
        "paren_count",
        "ellipsis_count",
        "capitalized_words",
        "uppercase_chars",
        "uppercase_ratio",
        "special_chars",
        "whitespace_count",
        "long_word_ratio",
        "short_word_ratio",
        "complexity_index",
    ]
    features = np.array(features)
    if is_train:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        return features_scaled, scaler, feature_names
    else:
        features_scaled = scaler.transform(features)
        return features_scaled, feature_names


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df["text"], train_df["author_encoded"]))

train_texts = train_df["text"].iloc[train_idx].values
val_texts = train_df["text"].iloc[val_idx].values
test_texts = test_df["text"].values

train_labels = train_df["author_encoded"].iloc[train_idx].values
val_labels = train_df["author_encoded"].iloc[val_idx].values

np.save("./working/train_idx.npy", train_idx)
np.save("./working/val_idx.npy", val_idx)

print(
    f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)
print(f"Train author distribution: {np.bincount(train_labels)}")
print(f"Val author distribution: {np.bincount(val_labels)}")

print("Extracting character-level n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 6),
    max_features=50000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=False,
    max_df=0.95,
    min_df=3,
)
train_char_tfidf = char_vectorizer.fit_transform(train_texts)
val_char_tfidf = char_vectorizer.transform(val_texts)
test_char_tfidf = char_vectorizer.transform(test_texts)
print(f"Character TF-IDF features: {train_char_tfidf.shape[1]}")

with open("./working/char_vectorizer.pkl", "wb") as f:
    pickle.dump(char_vectorizer, f)

print("Extracting word-level n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=30000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    max_df=0.90,
    min_df=3,
    token_pattern=r"(?u)\b\w+\b",
)
train_word_tfidf = word_vectorizer.fit_transform(train_texts)
val_word_tfidf = word_vectorizer.transform(val_texts)
test_word_tfidf = word_vectorizer.transform(test_texts)
print(f"Word TF-IDF features: {train_word_tfidf.shape[1]}")

with open("./working/word_vectorizer.pkl", "wb") as f:
    pickle.dump(word_vectorizer, f)

print("Extracting linguistic features...")
train_ling_features, ling_scaler, ling_feat_names = extract_linguistic_features(
    train_texts, is_train=True
)
val_ling_features, _ = extract_linguistic_features(
    val_texts, is_train=False, scaler=ling_scaler
)
test_ling_features, _ = extract_linguistic_features(
    test_texts, is_train=False, scaler=ling_scaler
)
print(f"Linguistic features: {train_ling_features.shape[1]}")

with open("./working/ling_scaler.pkl", "wb") as f:
    pickle.dump(ling_scaler, f)
with open("./working/ling_feat_names.pkl", "wb") as f:
    pickle.dump(ling_feat_names, f)

train_ling_sparse = csr_matrix(train_ling_features)
val_ling_sparse = csr_matrix(val_ling_features)
test_ling_sparse = csr_matrix(test_ling_features)

X_train = hstack([train_char_tfidf, train_word_tfidf, train_ling_sparse])
X_val = hstack([val_char_tfidf, val_word_tfidf, val_ling_sparse])
X_test = hstack([test_char_tfidf, test_word_tfidf, test_ling_sparse])

y_train = train_labels
y_val = val_labels

print(f"Final feature dimensions:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

save_npz("./working/X_train.npz", X_train)
save_npz("./working/X_val.npz", X_val)
save_npz("./working/X_test.npz", X_test)

np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)

test_ids = test_df["id"].values
np.save("./working/test_ids.npy", test_ids)

val_df_indices = train_df.iloc[val_idx].index.values
np.save("./working/val_df_indices.npy", val_df_indices)

print(f"Train features non-zero elements: {X_train.nnz:,}")
print(f"Val features non-zero elements: {X_val.nnz:,}")
print(f"Test features non-zero elements: {X_test.nnz:,}")
print(f"Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Val label distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
print("Data processing and feature engineering complete!")

# --- Training & Evaluation ---

X_train = X_train.toarray().astype(np.float32)
X_val = X_val.toarray().astype(np.float32)
X_test = X_test.toarray().astype(np.float32)

class_weights = {}
classes = np.unique(y_train)
for cls in classes:
    class_weights[cls] = len(y_train) / (len(classes) * np.sum(y_train == cls))
print(f"Class weights: {class_weights}")

dtrain = xgb.DMatrix(X_train, label=y_train, weight=[class_weights[c] for c in y_train])
dval = xgb.DMatrix(X_val, label=y_val, weight=[class_weights[c] for c in y_val])
dtest = xgb.DMatrix(X_test)

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "tree_method": "gpu_hist" if torch.cuda.is_available() else "hist",
    "predictor": "gpu_predictor" if torch.cuda.is_available() else "cpu_predictor",
    "random_state": 42,
    "verbosity": 0,
}

print("Training XGBoost model...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)
best_rounds = model.best_iteration
print(f"Best iteration: {best_rounds}")

print("Computing validation predictions...")
val_probs = model.predict(dval, iteration_range=(0, best_rounds))
val_logloss = log_loss(y_val, val_probs)
print(f"Validation Log Loss: {val_logloss:.6f}")

for i, class_name in enumerate(label_encoder.classes_):
    class_mask = y_val == i
    if class_mask.sum() > 0:
        class_loss = -np.mean(
            np.log(np.clip(val_probs[class_mask, i], 1e-15, 1 - 1e-15))
        )
        print(f"  {class_name} Log Loss: {class_loss:.6f}")

print("Saving best model...")
model.save_model("./working/best_xgboost_model.json")

print("Training meta-learner for calibration...")
train_probs = model.predict(dtrain, iteration_range=(0, best_rounds))
X_train_meta = np.hstack([X_train, train_probs])
X_val_meta = np.hstack([X_val, val_probs])

lgb_train = lgb.Dataset(X_train_meta, label=y_train)
lgb_val = lgb.Dataset(X_val_meta, label=y_val, reference=lgb_train)

lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_gain_to_split": 0.1,
    "min_data_in_leaf": 20,
}

print("Training LightGBM meta-learner...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

val_meta_probs = lgb_model.predict(X_val_meta)
val_final_logloss = log_loss(y_val, val_meta_probs)
print(f"Validation Log Loss (after calibration): {val_final_logloss:.6f}")

print("Generating test predictions...")
test_probs_base = model.predict(dtest, iteration_range=(0, best_rounds))
X_test_meta = np.hstack([X_test, test_probs_base])
test_probs = lgb_model.predict(X_test_meta)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
print(f"Test predictions shape: {test_probs.shape}")
print(f"Test predictions range: [{test_probs.min():.6f}, {test_probs.max():.6f}]")

print("Creating submission file...")
sample_submission = pd.read_csv("./input/sample_submission.csv")
label_order = sample_submission.columns[1:].tolist()
print(f"Label order: {label_order}")

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        label_order[0]: test_probs[:, 0],
        label_order[1]: test_probs[:, 1],
        label_order[2]: test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission_daaba2e58f394a48898953ec88aba93f.csv", index=False)
print(f"Submission saved to ./submission/submission_daaba2e58f394a48898953ec88aba93f.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

score = val_final_logloss
print(f"Final Validation Score: {score}")
