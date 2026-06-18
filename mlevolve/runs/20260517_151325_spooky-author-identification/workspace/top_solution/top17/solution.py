import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb
import torch

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
INFERENCE_DIR = "./input"
DATA_DIR = "./input"
TRAIN_CSV = "./input/train.csv"
TEST_CSV = "./input/test.csv"
OUTPUT_CSV = "./submission/submission.csv"
WORKING_DIR = "./working"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3

np.random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT
# ============================================================
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)

# ============================================================
# HANDCRAFTED FEATURES
# ============================================================

def extract_stylometric_features(texts):
    features = []

    archaic_words = set(
        [
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "dost",
            "art",
            "wast",
            "whence",
            "thence",
            "hither",
            "thither",
            "ere",
            "whilst",
            "anon",
            "methinks",
            "perchance",
            "forsooth",
            "betwixt",
            "wherefore",
        ]
    )

    emotional_words = set(
        [
            "dread",
            "horror",
            "terror",
            "fear",
            "awful",
            "hideous",
            "ghastly",
            "dismal",
            "gloomy",
            "mournful",
            "solemn",
            "melancholy",
            "agonizing",
            "torment",
            "anguish",
            "despair",
            "wretched",
            "appalling",
            "frightful",
        ]
    )

    lovecraft_words = set(
        [
            "eldritch",
            "cyclopean",
            "antediluvian",
            "non-euclidean",
            "gibbous",
            "squamous",
            "rugose",
            "ichor",
            "noisome",
            "foetid",
            "necronomicon",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "nyarlathotep",
            "azathoth",
            "shoggoth",
            "dimension",
            "immemorial",
            "unspeakable",
            "indescribable",
            "cryptic",
            "blasphemous",
            "cosmic",
            "gate",
            "yith",
            "mi-go",
        ]
    )

    function_words = set(
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
            "is",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "can",
            "could",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "him",
            "them",
            "who",
            "which",
            "what",
            "when",
            "where",
            "how",
            "why",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "than",
            "then",
            "very",
        ]
    )

    sub_conj = set(
        [
            "although",
            "because",
            "since",
            "unless",
            "while",
            "after",
            "before",
            "though",
            "until",
            "when",
            "where",
            "whether",
            "if",
            "as",
            "that",
            "which",
            "who",
            "whom",
        ]
    )

    for text in texts:
        if not isinstance(text, str) or pd.isna(text):
            text = ""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words) if words else 1
        char_count = len(text)
        sent_count = max(
            1, text.count(".") + text.count("!") + text.count("?") + text.count(";")
        )
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = word_count / sent_count
        upper_ratio = (
            sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0
        )
        lower_ratio = (
            sum(1 for c in text if c.islower()) / char_count if char_count > 0 else 0
        )
        digit_ratio = (
            sum(1 for c in text if c.isdigit()) / char_count if char_count > 0 else 0
        )
        whitespace_ratio = (
            sum(1 for c in text if c.isspace()) / char_count if char_count > 0 else 0
        )
        punct_counts = [
            text.count(p) / char_count if char_count > 0 else 0
            for p in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "—"]
        ]
        char_set = set(text.lower())
        char_diversity = len(char_set) / max(1, min(26, len(char_set)))
        long_words_ratio = sum(1 for w in words if len(w) > 6) / word_count
        capitalized_ratio = (
            sum(1 for w in text.split() if w and w[0].isupper()) / word_count
            if text.split()
            else 0
        )
        all_caps_ratio = (
            sum(1 for w in words if len(w) > 1 and w.isupper()) / word_count
        )
        sents = [s.split() for s in [text] if text]
        sent_lens = [len(s) for s in sents if len(s) > 0]
        sent_len_std = np.std(sent_lens) if len(sent_lens) > 1 else 0
        sent_len_var = np.var(sent_lens) if len(sent_lens) > 1 else 0
        function_word_ratio = sum(1 for w in words if w in function_words) / word_count
        archaic_ratio = sum(1 for w in words if w in archaic_words) / word_count
        emotional_ratio = sum(1 for w in words if w in emotional_words) / word_count
        lovecraft_ratio = sum(1 for w in words if w in lovecraft_words) / word_count
        sub_conj_ratio = sum(1 for w in words if w in sub_conj) / word_count
        features.append(
            [
                char_count,
                word_count,
                sent_count,
                avg_word_len,
                avg_sent_len,
                upper_ratio,
                lower_ratio,
                digit_ratio,
                whitespace_ratio,
            ]
            + punct_counts
            + [
                char_diversity,
                long_words_ratio,
                capitalized_ratio,
                all_caps_ratio,
                sent_len_std,
                sent_len_var,
                function_word_ratio,
                archaic_ratio,
                emotional_ratio,
                lovecraft_ratio,
                sub_conj_ratio,
            ]
        )
    return np.array(features)

def create_readability_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or pd.isna(text):
            text = ""
        words = text.split()
        word_count = len(words) if words else 1
        sent_count = max(
            1, text.count(".") + text.count("!") + text.count("?") + text.count(";")
        )
        char_count = len(text.replace(" ", ""))
        syllables = 0
        complex_words = 0
        for w in words:
            s = max(1, len(re.findall(r"[aeiouy]+", w.lower())))
            syllables += s
            if s >= 3:
                complex_words += 1
        avg_syllables = syllables / word_count
        complex_word_ratio = complex_words / word_count
        if sent_count > 0 and word_count > 1:
            flesch = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * avg_syllables
            ari = (
                4.71 * (char_count / word_count)
                + 0.5 * (word_count / sent_count)
                - 21.43
            )
        else:
            flesch = 0
            ari = 0
        features.append([flesch, ari, avg_syllables, complex_word_ratio])
    return np.array(features)

def create_pos_tag_approximation(texts):
    features = []
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ance",
        "ence",
        "ity",
        "dom",
        "ship",
        "ism",
    ]
    verb_suffixes = ["ed", "ing", "ate", "ize", "ify", "en", "ise", "fy"]
    adj_suffixes = [
        "ous",
        "ious",
        "al",
        "ial",
        "ic",
        "ical",
        "ful",
        "less",
        "able",
        "ible",
        "ive",
        "ish",
        "like",
        "y",
    ]
    adv_suffixes = ["ly", "ward", "wise"]
    for text in texts:
        if not isinstance(text, str) or pd.isna(text):
            text = ""
        words = text.lower().split()
        word_count = len(words) if words else 1
        noun_ratio = (
            sum(1 for w in words if any(w.endswith(s) for s in noun_suffixes))
            / word_count
        )
        verb_ratio = (
            sum(1 for w in words if any(w.endswith(s) for s in verb_suffixes))
            / word_count
        )
        adj_ratio = (
            sum(1 for w in words if any(w.endswith(s) for s in adj_suffixes))
            / word_count
        )
        adv_ratio = (
            sum(1 for w in words if any(w.endswith(s) for s in adv_suffixes))
            / word_count
        )
        content_words_ratio = 1 - sum(1 for w in words if len(w) <= 3) / word_count
        features.append(
            [noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_words_ratio]
        )
    return np.array(features)

print("Extracting handcrafted features...")
train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(test_df["text"].values)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)

train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# N-GRAM FEATURES
# ============================================================
print("Extracting n-gram features...")
char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vectorizer_short.fit_transform(X_train_texts)
val_char_short = char_vectorizer_short.transform(X_val_texts)
test_char_short = char_vectorizer_short.transform(test_df["text"].values)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_train_texts)
val_char_med = char_vectorizer_med.transform(X_val_texts)
test_char_med = char_vectorizer_med.transform(test_df["text"].values)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_train_texts)
val_char_long = char_vectorizer_long.transform(X_val_texts)
test_char_long = char_vectorizer_long.transform(test_df["text"].values)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vectorizer.fit_transform(X_train_texts)
val_word = word_vectorizer.transform(X_val_texts)
test_word = word_vectorizer.transform(test_df["text"].values)

def extract_punctuation_sequence(text):
    return "".join([c for c in text if c in string.punctuation]) if text else ""

# Fit punctuation vectorizer ONLY on training data to prevent leakage
train_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_train_texts]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
train_punct = punct_vectorizer.fit_transform(train_punct_sequences)

val_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_val_texts]
val_punct = punct_vectorizer.transform(val_punct_sequences)

test_punct_sequences = [extract_punctuation_sequence(str(t)) for t in test_df["text"].values]
test_punct = punct_vectorizer.transform(test_punct_sequences)

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Sparse train shape: {train_sparse.shape}")

# ============================================================
# COMPUTE LOG LOSS FUNCTION
# ============================================================
def compute_log_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(NUM_AUTHORS):
            if y_true[i] == j:
                loss -= np.log(y_pred_proba[i, j])
    return loss / n

# ============================================================
# TRAIN LIGHTGBM ON SPARSE FEATURES + HANDCRAFTED FEATURES
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM CLASSIFIER")
print("=" * 60)

import lightgbm as lgb

# Combine all features for LightGBM
lgb_train_features = np.hstack(
    [train_sparse.toarray(), train_stylo_filtered, train_read_scaled, train_pos_scaled]
)
lgb_val_features = np.hstack(
    [val_sparse.toarray(), val_stylo_filtered, val_read_scaled, val_pos_scaled]
)
lgb_test_features = np.hstack(
    [test_sparse.toarray(), test_stylo_filtered, test_read_scaled, test_pos_scaled]
)

print(f"LightGBM train features shape: {lgb_train_features.shape}")

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=-1,
    num_leaves=127,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective='multiclass',
    num_class=NUM_AUTHORS,
    metric='multi_logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    lgb_train_features,
    y_train_labels,
    eval_set=[(lgb_val_features, y_val_labels)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
)

lgb_val_probs = lgb_model.predict_proba(lgb_val_features)
lgb_test_probs = lgb_model.predict_proba(lgb_test_features)
lgb_val_loss = compute_log_loss(y_val_labels, lgb_val_probs)
print(f"LightGBM validation log loss: {lgb_val_loss:.4f}")

# ============================================================
# XGBOOST
# ============================================================
print("\nTraining XGBoost classifier...")
xgb_train_features = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled]
)
xgb_val_features = np.hstack(
    [val_stylo_filtered, val_read_scaled, val_pos_scaled]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled]
)
print(f"XGBoost train features: {xgb_train_features.shape}")

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    xgb_train_features,
    y_train_labels,
    eval_set=[(xgb_val_features, y_val_labels)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
print(
    f"XGBoost validation log loss: {compute_log_loss(y_val_labels, xgb_val_probs):.4f}"
)

# ============================================================
# TRAIN LOGISTIC REGRESSION ON SPARSE FEATURES
# ============================================================
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.5,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
lr_model.fit(train_sparse, y_train_labels)

lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)
lr_val_loss = compute_log_loss(y_val_labels, lr_val_probs)
print(
    f"Logistic Regression validation log loss: {lr_val_loss:.4f}"
)

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "lightgbm": lgb_val_probs,
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
}

best_ll = float("inf")
best_weights = None
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = (
            w1 * val_probas["lightgbm"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(y_val_labels, ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"lightgbm": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
test_probas = {
    "lightgbm": lgb_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
ensemble_test_probs = (
    best_weights["lightgbm"] * test_probas["lightgbm"]
    + best_weights["xgboost"] * test_probas["xgboost"]
    + best_weights["lr"] * test_probas["lr"]
)

eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSubmission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())
print(f"\nFinal Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()