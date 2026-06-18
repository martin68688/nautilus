import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
import xgboost as xgb
import lightgbm as lgb
import os
import warnings
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

# ============================================================
# Step 1: Data Processing and Feature Engineering
# ============================================================

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Create target encoding
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
num_classes = len(le.classes_)
print(f"Authors: {le.classes_}")


# Helper function for feature engineering
def create_linguistic_features(texts):
    """Extract rich linguistic features from text"""
    features = pd.DataFrame(index=range(len(texts)))

    # Basic features
    features["char_count"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["sentence_count"] = texts.str.count("[.!?]") + 1
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )

    # Punctuation features
    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    features["period_count"] = texts.str.count(r"\.")
    features["comma_count"] = texts.str.count(",")
    features["semicolon_count"] = texts.str.count(";")
    features["colon_count"] = texts.str.count(":")
    features["dash_count"] = texts.str.count("-")
    features["quote_count"] = texts.str.count('"')
    features["apostrophe_count"] = texts.str.count("'")
    features["ellipsis_count"] = texts.str.count(r"\.\.\.")
    features["punctuation_ratio"] = features[
        [c for c in features.columns if "count" in c]
    ].sum(axis=1) / (features["char_count"] + 1)

    # Capitalization features
    features["capital_letters"] = texts.str.findall(r"[A-Z]").str.len()
    features["capital_ratio"] = features["capital_letters"] / (
        features["char_count"] + 1
    )
    features["words_uppercase"] = texts.str.findall(r"\b[A-Z]+\b").str.len()
    features["words_capitalized"] = texts.str.findall(r"\b[A-Z][a-z]+\b").str.len()

    # Stop word features
    stop_words = {
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
        "by",
        "with",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "never",
        "nothing",
        "none",
        "very",
        "too",
        "so",
        "such",
        "more",
        "most",
        "less",
        "least",
        "all",
        "each",
        "every",
        "both",
        "few",
        "many",
        "much",
        "some",
        "any",
        "other",
        "another",
        "still",
        "already",
        "just",
        "only",
        "even",
        "though",
        "although",
        "while",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
    }

    def count_stop_words(text):
        words = text.lower().split()
        return sum(1 for w in words if w in stop_words)

    features["stop_word_count"] = texts.apply(count_stop_words)
    features["stop_word_ratio"] = features["stop_word_count"] / (
        features["word_count"] + 1
    )

    # Unique word features
    features["unique_words"] = texts.apply(lambda x: len(set(x.lower().split())))
    features["unique_word_ratio"] = features["unique_words"] / (
        features["word_count"] + 1
    )

    # Readability features (simplified)
    def count_syllables_approx(text):
        words = text.split()
        count = 0
        for word in words:
            word = word.lower().strip(string.punctuation)
            if len(word) == 0:
                continue
            vowels = "aeiouy"
            syllable_count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    syllable_count += 1
                prev_vowel = is_vowel
            if syllable_count == 0:
                syllable_count = 1
            count += syllable_count
        return count

    features["syllable_count"] = texts.apply(count_syllables_approx)
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["avg_sentence_len"]
        - 84.6 * (features["syllable_count"] / (features["word_count"] + 1))
    )

    # Part of speech patterns (simplified via word endings)
    features["ing_words"] = texts.str.findall(r"\b\w+ing\b").str.len()
    features["ed_words"] = texts.str.findall(r"\b\w+ed\b").str.len()
    features["ly_words"] = texts.str.findall(r"\b\w+ly\b").str.len()
    features["tion_words"] = texts.str.findall(r"\b\w+tion\b").str.len()
    features["ness_words"] = texts.str.findall(r"\b\w+ness\b").str.len()
    features["ment_words"] = texts.str.findall(r"\b\w+ment\b").str.len()

    # Thematic word features
    horror_words = [
        "dark",
        "shadow",
        "night",
        "fear",
        "terror",
        "horror",
        "ghost",
        "death",
        "dead",
        "soul",
        "spirit",
        "demon",
        "devil",
        "hell",
        "evil",
        "strange",
        "mystery",
        "mysterious",
        "dread",
        "awful",
        "hideous",
        "gloom",
        "gloomy",
        "pale",
        "cold",
        "silence",
        "alone",
        "lonely",
        "weird",
        "ancient",
        "monster",
        "creature",
        "phantom",
        "spectre",
        "wound",
        "blood",
        "corpse",
    ]

    lovecraft_words = [
        "eldritch",
        "cthulhu",
        "nyarlathotep",
        "yog",
        "sothoth",
        "r'lyeh",
        "kadath",
        "arkham",
        "innsmouth",
        "dunwich",
        "necronomicon",
        "unspeakable",
        "cyclopean",
        "antediluvian",
        "non",
        "euclidean",
        "cosmic",
        "gibbering",
        "blasphemous",
        "crawling",
        "nameless",
        "goatish",
        "squamous",
        "ichor",
        "lich",
        "predatory",
        "tentacle",
    ]

    poe_words = [
        "nevermore",
        "raven",
        "lenore",
        "chamber",
        "tapping",
        "rapping",
        "chilling",
        "dreary",
        "ghastly",
        "bust",
        "plutonian",
        "perched",
        "tinkle",
        "tintinnabulation",
        "ebony",
        "maudlin",
        "morose",
        "sculptured",
    ]

    shelley_words = [
        "frankenstein",
        "monster",
        "creature",
        "victor",
        "elizabeth",
        "geneva",
        "ingolstadt",
        "wretch",
        "memnon",
        "proserpine",
        "milton",
    ]

    def count_thematic_words(text, word_list):
        text_lower = text.lower()
        return sum(1 for w in word_list if w in text_lower)

    features["horror_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, horror_words)
    )
    features["lovecraft_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, lovecraft_words)
    )
    features["poe_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, poe_words)
    )
    features["shelley_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, shelley_words)
    )

    # Sentiment-like features
    positive_words = {
        "love",
        "beautiful",
        "happy",
        "joy",
        "wonderful",
        "sweet",
        "gentle",
        "kind",
        "pleasure",
        "delight",
        "hope",
        "bright",
        "light",
        "peace",
        "calm",
        "tender",
        "fair",
        "glad",
        "smile",
        "laugh",
    }
    negative_words = {
        "dark",
        "fear",
        "death",
        "pain",
        "sorrow",
        "dread",
        "horror",
        "terror",
        "gloom",
        "misery",
        "anguish",
        "agony",
        "suffering",
        "cruel",
        "hate",
        "wrath",
        "rage",
        "fury",
        "grief",
        "weep",
        "mourn",
        "despair",
    }

    features["positive_words"] = texts.apply(
        lambda x: count_thematic_words(x, positive_words)
    )
    features["negative_words"] = texts.apply(
        lambda x: count_thematic_words(x, negative_words)
    )
    features["sentiment_balance"] = (
        features["positive_words"] - features["negative_words"]
    ) / (features["word_count"] + 1)

    return features


print("Creating linguistic features...")
train_features = create_linguistic_features(train_df["text"])
test_features = create_linguistic_features(test_df["text"])

# Character n-gram features
print("Creating character n-gram features...")
char_vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    max_features=10000,
    lowercase=True,
    strip_accents="unicode",
)
char_features_train = char_vectorizer.fit_transform(train_df["text"])
char_features_test = char_vectorizer.transform(test_df["text"])
print(f"Character n-gram features shape: {char_features_train.shape}")

# Word n-gram TF-IDF features
print("Creating word n-gram TF-IDF features...")
word_tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=5000,
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    max_df=0.8,
    min_df=3,
)
word_features_train = word_tfidf.fit_transform(train_df["text"])
word_features_test = word_tfidf.transform(test_df["text"])
print(f"Word TF-IDF features shape: {word_features_train.shape}")

# Combine all sparse features
print("Combining features...")
train_features_sparse = csr_matrix(train_features.fillna(0).values)
test_features_sparse = csr_matrix(test_features.fillna(0).values)

X_train = hstack(
    [train_features_sparse, char_features_train, word_features_train]
).tocsr()
X_test = hstack([test_features_sparse, char_features_test, word_features_test]).tocsr()
print(f"Final feature matrix shape: {X_train.shape}")
print(f"Test feature matrix shape: {X_test.shape}")

# Create validation split using stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_indices = []
val_indices = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, train_df["author_encoded"])
):
    if fold == 0:
        val_fold_idx = val_idx
        train_fold_idx = train_idx
        break

print(f"Train size: {len(train_fold_idx)}, Validation size: {len(val_fold_idx)}")

# ============================================================
# Step 3: Training and Evaluation (Gradient-Boosted Trees)
# ============================================================

y_train = train_df["author_encoded"].values

# Split data
X_train_fold = X_train[train_fold_idx]
y_train_fold = y_train[train_fold_idx]
X_val_fold = X_train[val_fold_idx]
y_val_fold = y_train[val_fold_idx]

# Class weights for handling imbalance
class_counts = np.bincount(y_train_fold)
class_weights = {i: 1.0 / count for i, count in enumerate(class_counts)}
weight_sum = sum(class_weights.values())
class_weights = {
    i: w / weight_sum * len(class_weights) for i, w in class_weights.items()
}
sample_weights = np.array([class_weights[y] for y in y_train_fold])

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=sample_weights)
dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

# XGBoost parameters
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.3,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

print("\nTraining XGBoost...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, "eval")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

# LightGBM parameters
lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 8,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

print("\nTraining LightGBM...")
lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, weight=sample_weights)
lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# Ensemble predictions
print("\nGenerating ensemble predictions...")
xgb_val_preds = xgb_model.predict(dval)
xgb_test_preds = xgb_model.predict(xgb.DMatrix(X_test))

lgb_val_preds = lgb_model.predict(X_val_fold)
lgb_test_preds = lgb_model.predict(X_test)

# Weighted ensemble
ensemble_weight = 0.5
val_preds = ensemble_weight * xgb_val_preds + (1 - ensemble_weight) * lgb_val_preds
test_preds = ensemble_weight * xgb_test_preds + (1 - ensemble_weight) * lgb_test_preds

# Clip and normalize predictions
epsilon = 1e-15
val_preds = np.clip(val_preds, epsilon, 1 - epsilon)
test_preds = np.clip(test_preds, epsilon, 1 - epsilon)

val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Calculate validation log loss
score = log_loss(y_val_fold, val_preds)
print(f"\nFinal Validation Score: {score}")

# Generate submission file
print("\nGenerating submission file...")
os.makedirs("./submission", exist_ok=True)
test_df = pd.read_csv("./input/test.csv")
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
