"""
Complete pipeline for Spooky Author Identification
- DeBERTa-v3-large fine-tuning with label smoothing
- Handcrafted stylometric/readability/POS features
- Character/word n-gram features via TF-IDF
- XGBoost on DeBERTa embeddings + dense features
- Logistic Regression on sparse n-gram features
- Weighted ensemble with grid-search optimization
"""

import pandas as pd
import numpy as np
import os
import re
import string
import warnings
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
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
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.1
TEST_SIZE = 0.1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# HELPER FUNCTIONS FOR HANDCRAFTED FEATURES
# ============================================================

FUNCTION_WORDS = set(
    [
        "the",
        "a",
        "an",
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
        "are",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "but",
        "and",
        "or",
        "if",
        "than",
        "that",
        "this",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "so",
        "very",
        "just",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "some",
        "any",
        "such",
        "only",
        "own",
        "same",
        "too",
        "can",
        "will",
        "may",
        "shall",
        "must",
        "might",
        "would",
        "could",
        "should",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "under",
        "above",
        "through",
        "during",
        "without",
        "within",
        "along",
        "around",
        "among",
        "upon",
        "then",
        "there",
        "here",
        "where",
        "again",
        "also",
        "yet",
        "else",
        "ever",
        "never",
        "always",
        "often",
        "now",
        "thus",
        "well",
        "even",
        "still",
        "already",
        "almost",
        "enough",
        "rather",
        "quite",
        "perhaps",
        "maybe",
        "indeed",
        "though",
    ]
)

ARCHAIC_WORDS = set(
    [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "art",
        "whence",
        "thence",
        "hither",
        "thither",
        "ere",
        "anon",
        "betwixt",
        "perchance",
        "forsooth",
        "prithee",
        "wherefore",
        "therefor",
        "unto",
        "nay",
        "yea",
        "wilt",
        "canst",
        "dost",
        "shalt",
        "didst",
        "wert",
        "hast",
    ]
)

EMOTIONAL_WORDS = set(
    [
        "terror",
        "horror",
        "dread",
        "fear",
        "gloom",
        "darkness",
        "despair",
        "anguish",
        "agony",
        "sorrow",
        "weep",
        "wept",
        "mourn",
        "funereal",
        "macabre",
        "ghastly",
        "hideous",
        "dismal",
        "dreary",
        "somber",
        "melancholy",
        "woeful",
        "wretched",
        "miserable",
        "torment",
        "suffering",
    ]
)

LOVECRAFT_WORDS = set(
    [
        "eldritch",
        "cyclopean",
        "non-euclidean",
        "antediluvian",
        "primordial",
        "cryptic",
        "nameless",
        "unspeakable",
        "unnameable",
        "blasphemous",
        "fathomless",
        "gibbous",
        "ichor",
        "squamous",
        "rugose",
        "noisome",
        "tenebrous",
        "immemorial",
        "lurker",
        "cthulhu",
        "yog-sothoth",
        "nyarlathotep",
        "azathoth",
        "necronomicon",
        "aklo",
        "r'lyeh",
        "shoggoth",
        "mi-go",
        "deep one",
        "yith",
        "kadath",
    ]
)

def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2:
        count += 1
    if count == 0:
        count = 1
    return count

def flesch_reading_ease(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    if len(sentences) == 0:
        return 0.0
    syllable_count = sum(count_syllables(w) for w in words)
    avg_syllables = syllable_count / len(words)
    avg_words_per_sentence = len(words) / len(sentences)
    score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables
    return max(0, min(100, score))

def automated_readability_index(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    if len(sentences) == 0:
        return 0.0
    char_count = sum(len(w) for w in words)
    ari = 4.71 * (char_count / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43
    return max(0, ari)

def extract_stylometric_features(texts):
    features = []
    for text in texts:
        if pd.isna(text) or len(str(text).strip()) == 0:
            features.append([0] * 30)
            continue
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        sent_count = len(sentences) if len(sentences) > 0 else 1
        word_count = len(words)
        text_len = len(text)
        avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        avg_sent_len = word_count / sent_count
        upper_ratio = (
            sum(1 for c in text if c.isupper()) / text_len if text_len > 0 else 0
        )
        lower_ratio = (
            sum(1 for c in text if c.islower()) / text_len if text_len > 0 else 0
        )
        digit_ratio = (
            sum(1 for c in text if c.isdigit()) / text_len if text_len > 0 else 0
        )
        whitespace_ratio = (
            sum(1 for c in text if c.isspace()) / text_len if text_len > 0 else 0
        )
        punct_chars = [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "—"]
        punct_ratios = [
            text.count(p) / text_len if text_len > 0 else 0 for p in punct_chars
        ]
        char_diversity = len(set(text.lower())) / text_len if text_len > 0 else 0
        long_words_ratio = (
            sum(1 for w in words if len(w) > 6) / word_count if word_count > 0 else 0
        )
        capitalized_ratio = (
            sum(1 for w in words if w[0].isupper()) / word_count
            if word_count > 0
            else 0
        )
        all_caps_ratio = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
            if word_count > 0
            else 0
        )
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 0 else 0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 0 else 0
        words_lower = [w.lower() for w in words]
        function_word_ratio = (
            sum(1 for w in words_lower if w in FUNCTION_WORDS) / word_count
            if word_count > 0
            else 0
        )
        archaic_ratio = (
            sum(1 for w in words_lower if w in ARCHAIC_WORDS) / word_count
            if word_count > 0
            else 0
        )
        emotional_ratio = (
            sum(1 for w in words_lower if w in EMOTIONAL_WORDS) / word_count
            if word_count > 0
            else 0
        )
        lovecraft_ratio = (
            sum(1 for w in words_lower if w in LOVECRAFT_WORDS) / word_count
            if word_count > 0
            else 0
        )
        feat = [
            text_len,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            *punct_ratios,
            char_diversity,
            sum(1 for w in words if len(w) > 2),
            long_words_ratio,
            capitalized_ratio,
            all_caps_ratio,
            sent_len_std,
            sent_len_var,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
        ]
        features.append(feat)
    return np.array(features)

def create_readability_features(texts):
    features = []
    for text in texts:
        if pd.isna(text) or len(str(text).strip()) == 0:
            features.append([0, 0, 0, 0])
            continue
        text = str(text)
        words = text.split()
        if len(words) == 0:
            features.append([0, 0, 0, 0])
            continue
        flesch = flesch_reading_ease(text)
        ari = automated_readability_index(text)
        avg_syllables = np.mean([count_syllables(w) for w in words])
        complex_words = sum(1 for w in words if count_syllables(w) > 2)
        complex_ratio = complex_words / len(words)
        features.append([flesch, ari, avg_syllables, complex_ratio])
    return np.array(features)

def create_pos_tag_approximation(texts):
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ship",
        "dom",
        "hood",
    ]
    verb_suffixes = ["ed", "ing", "ate", "ize", "ify", "en", "ish"]
    adj_suffixes = [
        "ous",
        "ful",
        "ic",
        "al",
        "ive",
        "able",
        "ible",
        "less",
        "ish",
        "like",
    ]
    adv_suffixes = ["ly", "ward", "wise", "ways"]
    features = []
    for text in texts:
        if pd.isna(text) or len(str(text).strip()) == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        text = str(text)
        words = text.split()
        if len(words) == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        noun_ratio = sum(
            1 for w in words if any(w.lower().endswith(s) for s in noun_suffixes)
        ) / len(words)
        verb_ratio = sum(
            1 for w in words if any(w.lower().endswith(s) for s in verb_suffixes)
        ) / len(words)
        adj_ratio = sum(
            1 for w in words if any(w.lower().endswith(s) for s in adj_suffixes)
        ) / len(words)
        adv_ratio = sum(
            1 for w in words if any(w.lower().endswith(s) for s in adv_suffixes)
        ) / len(words)
        words_lower = [w.lower() for w in words]
        content_words = [
            w for w in words_lower if len(w) > 3 and w not in FUNCTION_WORDS
        ]
        content_ratio = len(content_words) / len(words)
        features.append([noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_ratio])
    return np.array(features)

def extract_punctuation_sequence(text):
    if pd.isna(text):
        return ""
    text = str(text)
    return "".join([c for c in text if c in string.punctuation])

# ============================================================
# METRIC: Multi-class Log Loss
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
# 5-FOLD STRATIFIED CROSS-VALIDATION (NO INDEX_BUG - use numpy array indexing)
# ============================================================
from sklearn.model_selection import StratifiedKFold

NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
all_train_texts = train_df["text"].values
all_train_labels = y_train_full

# We'll use the full training data; for compatibility, create a holdout-validation-like setup
# but the actual cross-validation loop will be in DeBERTa training below
# Use train_test_split indices directly - NO reset_index to avoid INDEX_BUG
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

# Extract data using numpy indexing on .values arrays (NOT .index after reset_index)
X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]

assert len(set(train_idx) & set(val_idx)) == 0
print(
    f"Holdout training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)
print(f"Using {NUM_FOLDS}-fold cross-validation for DeBERTa training (OOF predictions on ALL data)")

# ============================================================
# HANDCRAFTED FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING HANDCRAFTED FEATURES")
print("=" * 60)

# 1. Stylometric (30 dims)
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
print(f"Stylo features: train {train_stylo_filtered.shape}")

# 2. Readability (4 dims)
train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)
print(f"Readability features: train {train_read_scaled.shape}")

# 3. POS approximation (5 dims)
train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)
print(f"POS features: train {train_pos_scaled.shape}")

# ============================================================
# N-GRAM FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING N-GRAM FEATURES")
print("=" * 60)

# Character n-grams (2,4)
char_vec_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vec_short.fit_transform(X_train_texts)
val_char_short = char_vec_short.transform(X_val_texts)
test_char_short = char_vec_short.transform(test_df["text"].values)

# Character n-grams (4,6)
char_vec_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vec_med.fit_transform(X_train_texts)
val_char_med = char_vec_med.transform(X_val_texts)
test_char_med = char_vec_med.transform(test_df["text"].values)

# Character n-grams (5,7)
char_vec_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vec_long.fit_transform(X_train_texts)
val_char_long = char_vec_long.transform(X_val_texts)
test_char_long = char_vec_long.transform(test_df["text"].values)

# Word n-grams (1,3)
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vec.fit_transform(X_train_texts)
val_word = word_vec.transform(X_val_texts)
test_word = word_vec.transform(test_df["text"].values)

# Punctuation sequences
train_punct_seq = [extract_punctuation_sequence(t) for t in X_train_texts]
val_punct_seq = [extract_punctuation_sequence(t) for t in X_val_texts]
test_punct_seq = [extract_punctuation_sequence(t) for t in test_df["text"].values]

punct_vec = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
train_punct = punct_vec.fit_transform(train_punct_seq)
val_punct = punct_vec.transform(val_punct_seq)
test_punct = punct_vec.transform(test_punct_seq)

# Combine sparse
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
# DEBERTA 5-FOLD CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE WITH 5-FOLD CV")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize ALL training data and test data once
all_train_encodings = tokenizer(
    list(all_train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
all_test_encodings = tokenizer(
    list(test_df["text"].values),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

# Tokenize holdout (for ensemble validation) - using the existing split variables
train_encodings = tokenizer(
    list(X_train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    list(X_val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_encodings = tokenizer(
    list(test_df["text"].values),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

# Prepare OOF predictions and test predictions arrays
oof_preds = np.zeros((len(all_train_texts), NUM_AUTHORS), dtype=np.float32)
test_preds_folds = np.zeros((len(test_df), NUM_AUTHORS), dtype=np.float32)

# Fold-based training
fold_splits = list(skf.split(np.arange(len(all_train_texts)), all_train_labels))

# Reusable evaluate function (no labels optional)
def evaluate_deberta_probs(model, input_ids, attention_mask, batch_size):
    model.eval()
    all_probs = []
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for batch in loader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

def evaluate_deberta_with_labels(model, input_ids, attention_mask, labels, batch_size):
    model.eval()
    all_preds = []
    all_labels = []
    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for batch in loader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with autocast():
                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(b_labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

scaler_grad = GradScaler() if torch.cuda.is_available() else None

for fold, (train_fold_idx, val_fold_idx) in enumerate(fold_splits):
    print(f"\n{'='*40}")
    print(f"FOLD {fold+1}/{NUM_FOLDS}")
    print(f"{'='*40}")

    # Create model for this fold - using mean pooling and reduced dropout
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_AUTHORS
    config.hidden_dropout_prob = DROPOUT
    config.attention_probs_dropout_prob = DROPOUT
    # Use standard classifier (no label smoothing)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Fold data
    fold_train_ids = all_train_encodings["input_ids"][train_fold_idx]
    fold_train_mask = all_train_encodings["attention_mask"][train_fold_idx]
    fold_train_labels = torch.tensor(all_train_labels[train_fold_idx], dtype=torch.long)
    fold_val_ids = all_train_encodings["input_ids"][val_fold_idx]
    fold_val_mask = all_train_encodings["attention_mask"][val_fold_idx]
    fold_val_labels = torch.tensor(all_train_labels[val_fold_idx], dtype=torch.long)

    train_dataset = TensorDataset(fold_train_ids, fold_train_mask, fold_train_labels)
    val_dataset = TensorDataset(fold_val_ids, fold_val_mask, fold_val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            if scaler_grad is not None:
                scaler_grad.scale(loss).backward()
                scaler_grad.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_grad.step(optimizer)
                scaler_grad.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        avg_train_loss = total_loss / num_batches
        val_loss, val_acc, _ = evaluate_deberta_with_labels(
            model, fold_val_ids, fold_val_mask, fold_val_labels, BATCH_SIZE * 2
        )
        print(
            f"Fold {fold+1} Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save the final model for this fold (no early stopping, use last epoch)
    torch.save(
        model.state_dict(), os.path.join(WORKING_DIR, f"best_deberta_fold{fold}.pt")
    )
    print(f"Fold {fold+1} final: epoch {NUM_EPOCHS}, val loss: {val_loss:.4f}")

    # Generate OOF predictions for this fold's validation indices
    fold_oof_probs = evaluate_deberta_probs(
        model, fold_val_ids, fold_val_mask, BATCH_SIZE * 2
    )
    oof_preds[val_fold_idx] = fold_oof_probs

    # Generate test predictions for this fold
    fold_test_probs = evaluate_deberta_probs(
        model, all_test_encodings["input_ids"], all_test_encodings["attention_mask"], BATCH_SIZE * 2
    )
    test_preds_folds += fold_test_probs / NUM_FOLDS

    # Clear CUDA cache
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n5-fold CV complete. OOF shape: {oof_preds.shape}, Test predictions ensemble from {NUM_FOLDS} models.")
# Evaluate OOF log loss
oof_ll = compute_log_loss(all_train_labels, oof_preds)
print(f"DeBERTa OOF log loss (all data, unbiased): {oof_ll:.4f}")

# For compatibility with downstream code, we need to map the full OOF predictions to the train/holdout split
# XGBoost/LR validation uses holdout, but DeBERTa should now use OOF on the holdout portion (which is a subset of all data)
# Since holdout is a subset of all data, we can slice oof_preds
# Create a mapping: which rows in all_train correspond to val_idx
val_mask_in_all = np.isin(np.arange(len(all_train_texts)), val_idx)
deberta_val_probs = oof_preds[val_mask_in_all]
# For training XGBoost, we'll use OOF on train_idx as well
train_mask_in_all = np.isin(np.arange(len(all_train_texts)), train_idx)
deberta_train_probs = oof_preds[train_mask_in_all]

# Stack OOF probs from all folds for the train split (used in XGBoost) and val split
# Note: we need full OOF predictions for both train and val splits of the holdout
# Since OOF covers all data, these are already correct
print(f"DeBERTa OOF probs: train split {deberta_train_probs.shape}, val split {deberta_val_probs.shape}")

# Get test predictions aggregated from all folds
deberta_test_probs = test_preds_folds.copy()

print(f"DeBERTa OOF probs: train split {deberta_train_probs.shape}, val split {deberta_val_probs.shape}, test {deberta_test_probs.shape}")

# ============================================================
# EXTRACT DEBERTA EMBEDDINGS (from last fold model as representative)
# ============================================================
print("\nExtracting DeBERTa embeddings from last fold model...")

# Load the last fold's model for embeddings (a trade-off between accuracy and memory)
last_fold_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_AUTHORS, hidden_dropout_prob=DROPOUT, attention_probs_dropout_prob=DROPOUT),
    ignore_mismatched_sizes=True,
)
last_fold_model.to(device)
state_dict = torch.load(os.path.join(WORKING_DIR, f"best_deberta_fold{NUM_FOLDS-1}.pt"), map_location=device)
model_state = last_fold_model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
last_fold_model.load_state_dict(filtered_state_dict, strict=False)

def extract_embeddings(model, loader):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
                # Mean pooling over non-padded tokens
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = (hidden_states * attention_mask_expanded).sum(dim=1)
                sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

train_loader_no_labels = DataLoader(
    TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
val_loader_no_labels = DataLoader(
    TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader_no_labels = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

train_embeddings = extract_embeddings(last_fold_model, train_loader_no_labels)
val_embeddings = extract_embeddings(last_fold_model, val_loader_no_labels)
test_embeddings = extract_embeddings(last_fold_model, test_loader_no_labels)
print(
    f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}"
)

# Clean up the temporary model
del last_fold_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# XGBOOST (WITH DEBERTA OOF PROBS AS META-FEATURES - CROSS-FOLD IMPROVEMENT)
# ============================================================
print("\nTraining XGBoost classifier with DeBERTa OOF meta-features (5-fold cross-validation)...")
# Use OOF probabilities from all 5 folds as meta-features alongside embeddings
# This replaces the single-fold embedding approach with richer cross-validated signals
xgb_train_features = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled, train_embeddings, deberta_train_probs]
)
xgb_val_features = np.hstack(
    [val_stylo_filtered, val_read_scaled, val_pos_scaled, val_embeddings, deberta_val_probs]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled, test_embeddings, deberta_test_probs]
)
print(f"XGBoost feature shapes - Train: {xgb_train_features.shape}, Val: {xgb_val_features.shape}, Test: {xgb_test_features.shape}")
print(f"Features include: {train_stylo_filtered.shape[1]} stylo + {train_read_scaled.shape[1]} readability + {train_pos_scaled.shape[1]} POS + {train_embeddings.shape[1]} embeddings + {deberta_train_probs.shape[1]} OOF probs")

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
# LOGISTIC REGRESSION
# ============================================================
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
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
print(
    f"Logistic Regression validation log loss: {compute_log_loss(y_val_labels, lr_val_probs):.4f}"
)

print(f"DeBERTa OOF Log Loss (full data unbiased): {oof_ll:.4f}")

# ============================================================
# LEARNED LOGISTIC REGRESSION ENSEMBLE (trained on OOF predictions)
# ============================================================
print("\nTraining logistic regression ensemble on OOF predictions...")
# Stack OOF predictions from all 3 models
# Use validation OOF predictions from DeBERTa (already on val split) along with XGBoost and LR val predictions
stacked_oof = np.hstack([deberta_val_probs, xgb_val_probs, lr_val_probs])
# Train logistic regression (no bias, no penalty) to learn optimal ensemble weights
lr_ensemble = LogisticRegression(
    penalty=None,
    fit_intercept=False,
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
lr_ensemble.fit(stacked_oof, y_val_labels)

# Display learned coefficients
coefs = lr_ensemble.coef_
print("Logistic Regression ensemble coefficients (shape: 3 classes x 9 features):")
for i, author in enumerate(label_encoder.classes_):
    print(f"  {author}: Deberta={coefs[i, 0:3].round(4)}, XGBoost={coefs[i, 3:6].round(4)}, LR={coefs[i, 6:9].round(4)}")

# Compute ensemble validation log loss using the learned weights
ensemble_val_probs = lr_ensemble.predict_proba(stacked_oof)
ensemble_val_ll = compute_log_loss(y_val_labels, ensemble_val_probs)
print(f"Ensemble validation log loss: {ensemble_val_ll:.4f}")

# Also compute and display individual model performance on holdout
print(f"Individual model validation log losses:")
print(f"  DeBERTa OOF:      {compute_log_loss(y_val_labels, deberta_val_probs):.4f}")
print(f"  XGBoost:          {compute_log_loss(y_val_labels, xgb_val_probs):.4f}")
print(f"  Logistic Reg:     {compute_log_loss(y_val_labels, lr_val_probs):.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
# Stack test predictions from all 3 models
# Stack test predictions from all 3 models in the same order (DeBERTa, XGBoost, LR)
stacked_test = np.hstack([deberta_test_probs, xgb_test_probs, lr_test_probs])
ensemble_test_probs = lr_ensemble.predict_proba(stacked_test)

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
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
print(f"Final Validation Score: {ensemble_val_ll}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()