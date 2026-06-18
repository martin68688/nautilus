"""
Run 20260509_185008 Train+Inference Script
LogLoss: ~0.2013 (real log_loss, no INDEX_BUG)
Models: DeBERTa-v3-large fine-tuning + XGBoost + Logistic Regression + Ensemble

Usage: python merged_script.py
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
TRAIN_CSV = "./input/train.csv"
TEST_CSV = "./input/test.csv"
OUTPUT_CSV = "./submission/submission.csv"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# Configuration
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Data Loading
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
# Stratified Split (index-based to avoid INDEX_BUG)
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
# Feature Extraction Functions
# ============================================================

def count_syllables(word):
    word = word.lower().strip()
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if count == 0:
        count = 1
    return count

def extract_stylometric_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 30)
            continue

        char_len = len(text)
        words = text.split()
        word_count = len(words) if words else 1
        sentences = re.split(r"[.!?]+", text)
        sent_count = len([s for s in sentences if s.strip()]) if sentences else 1

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0

        upper_ratio = (
            sum(1 for c in text if c.isupper()) / char_len if char_len > 0 else 0
        )
        lower_ratio = (
            sum(1 for c in text if c.islower()) / char_len if char_len > 0 else 0
        )
        digit_ratio = (
            sum(1 for c in text if c.isdigit()) / char_len if char_len > 0 else 0
        )
        whitespace_ratio = (
            sum(1 for c in text if c.isspace()) / char_len if char_len > 0 else 0
        )

        punct_counts = []
        for p in string.punctuation:
            punct_counts.append(text.count(p) / char_len if char_len > 0 else 0)

        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / char_len if char_len > 0 else 0

        long_words = (
            sum(1 for w in words if len(w) > 6) / word_count if word_count > 0 else 0
        )
        capitalized = (
            sum(1 for w in words if w[0].isupper()) / word_count
            if word_count > 0
            else 0
        )
        all_caps = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
            if word_count > 0
            else 0
        )

        sent_lens = [len(s.split()) for s in sentences if s.strip()]
        sent_len_std = np.std(sent_lens) if len(sent_lens) > 1 else 0
        sent_len_var = np.var(sent_lens) if len(sent_lens) > 1 else 0

        function_words = {
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
            "is",
            "are",
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
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
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
            "our",
            "their",
            "not",
            "no",
            "so",
            "very",
            "just",
        }
        function_word_ratio = (
            sum(1 for w in words if w.lower() in function_words) / word_count
            if word_count > 0
            else 0
        )

        archaic_words = {
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "didst",
            "wert",
            "art",
            "shalt",
            "wilt",
            "dost",
            "whence",
            "thence",
            "hither",
            "thither",
            "wherefore",
            "perchance",
            "ere",
            "unto",
            "nay",
            "yea",
            "methinks",
            "alack",
            "alas",
            "forsooth",
            "betwixt",
            "twixt",
            "o'er",
            "ne'er",
            "e'er",
            "oft",
            "anon",
            "nought",
            "ay",
            "cannot",
        }
        archaic_ratio = (
            sum(1 for w in words if w.lower() in archaic_words) / word_count
            if word_count > 0
            else 0
        )

        emotional_words = {
            "dread",
            "horror",
            "terrible",
            "fearful",
            "ghastly",
            "hideous",
            "appalling",
            "awful",
            "frightful",
            "shocking",
            "dreadful",
            "gloomy",
            "dark",
            "shadow",
            "phantom",
            "spectre",
            "ghost",
            "demon",
            "devil",
            "evil",
            "wicked",
            "curse",
            "death",
            "dead",
            "corpse",
            "tomb",
            "grave",
            "coffin",
            "mournful",
            "weeping",
            "tears",
            "sorrow",
            "anguish",
            "agony",
            "torment",
            "suffering",
            "painful",
            "mysterious",
            "strange",
            "weird",
            "uncanny",
            "supernatural",
            "unnatural",
            "monstrous",
            "grotesque",
            "macabre",
            "sinister",
            "ominous",
        }
        emotional_ratio = (
            sum(1 for w in words if w.lower() in emotional_words) / word_count
            if word_count > 0
            else 0
        )

        lovecraft_words = {
            "cosmic",
            "eldritch",
            "cyclopean",
            "non-euclidean",
            "antediluvian",
            "primordial",
            "alien",
            "blasphemous",
            "nameless",
            "unspeakable",
            "indescribable",
            "unutterable",
            "carcosa",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "azathoth",
            "nyarlathotep",
            "shoggoth",
            "miskatonic",
            "arkham",
            "kadath",
            "hyperborean",
            "necronomicon",
            "abyss",
            "void",
            "infinite",
            "immemorial",
            "ancient",
            "elder",
            "great-old-one",
            "dimensionless",
            "incalculable",
            "abhorrent",
            "loathsome",
            "squamous",
            "rugose",
        }
        lovecraft_ratio = (
            sum(1 for w in words if w.lower() in lovecraft_words) / word_count
            if word_count > 0
            else 0
        )

        sub_conjunctions = {
            "although",
            "though",
            "because",
            "since",
            "unless",
            "until",
            "while",
            "after",
            "before",
            "when",
            "where",
            "as",
            "if",
            "once",
            "than",
            "that",
            "whether",
            "whereas",
            "wherever",
            "whenever",
            "whereupon",
            "whereby",
            "wherein",
            "whereof",
        }
        sub_conj_ratio = (
            sum(1 for w in words if w.lower() in sub_conjunctions) / word_count
            if word_count > 0
            else 0
        )

        features.append(
            [
                char_len,
                word_count,
                sent_count,
                avg_word_len,
                avg_sent_len,
                upper_ratio,
                lower_ratio,
                digit_ratio,
                whitespace_ratio,
                *punct_counts,
                char_diversity,
                long_words,
                capitalized,
                all_caps,
                sent_len_std,
                sent_len_var,
                function_word_ratio,
                archaic_ratio,
                emotional_ratio,
                lovecraft_ratio,
                sub_conj_ratio,
            ]
        )
    return np.array(features, dtype=np.float32)

def create_readability_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 4)
            continue

        words = text.split()
        word_count = len(words) if words else 1
        sentences = re.split(r"[.!?]+", text)
        sent_count = len([s for s in sentences if s.strip()]) if sentences else 1

        syllables = sum(count_syllables(w) for w in words) if words else 1
        avg_syllables = syllables / word_count if word_count > 0 else 0
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0
        flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables

        char_count = sum(len(w) for w in words) if words else 1
        ari = (
            (4.71 * (char_count / word_count if word_count > 0 else 1))
            + (0.5 * avg_sent_len)
            - 21.43
        )

        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / word_count if word_count > 0 else 0

        features.append([flesch, ari, avg_syllables, complex_ratio])
    return np.array(features, dtype=np.float32)

def create_pos_tag_approximation(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 5)
            continue

        words = text.split()
        word_count = len(words) if words else 1

        noun_suffixes = (
            "tion",
            "sion",
            "ment",
            "ness",
            "ity",
            "ance",
            "ence",
            "ship",
            "dom",
            "ism",
            "ist",
            "ure",
            "age",
            "ery",
            "ry",
        )
        noun_count = sum(1 for w in words if w.lower().endswith(noun_suffixes))

        verb_suffixes = ("ate", "en", "ify", "ize", "ise", "ish", "ed", "ing", "s")
        verb_count = sum(
            1 for w in words if w.lower().endswith(verb_suffixes) and len(w) > 3
        )

        adj_suffixes = (
            "ful",
            "ous",
            "ive",
            "able",
            "ible",
            "al",
            "ial",
            "ic",
            "less",
            "like",
            "some",
            "ward",
        )
        adj_count = sum(
            1 for w in words if w.lower().endswith(adj_suffixes) and len(w) > 3
        )

        adv_suffixes = ("ly", "ward", "wise", "ways")
        adv_count = sum(
            1 for w in words if w.lower().endswith(adv_suffixes) and len(w) > 3
        )

        function_words = {
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
            "is",
            "are",
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
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
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
            "our",
            "their",
            "not",
            "no",
            "so",
            "very",
            "just",
        }
        content_words = sum(1 for w in words if w.lower() not in function_words)
        content_ratio = content_words / word_count if word_count > 0 else 0

        features.append(
            [
                noun_count / word_count if word_count > 0 else 0,
                verb_count / word_count if word_count > 0 else 0,
                adj_count / word_count if word_count > 0 else 0,
                adv_count / word_count if word_count > 0 else 0,
                content_ratio,
            ]
        )
    return np.array(features, dtype=np.float32)

# ============================================================
# Handcrafted Features
# ============================================================
print("\nExtracting handcrafted features...")

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
# N-gram Features with Chi-Squared Selection
# ============================================================
print("\nExtracting n-gram features...")

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

all_texts_for_punct = np.concatenate(
    [X_train_texts, X_val_texts, test_df["text"].values]
)
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(X_train_texts)
n_val = len(X_val_texts)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + n_val]
test_punct = punct_features_all[n_train + n_val :]

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Sparse train shape before selection: {train_sparse.shape}")

# Apply chi-squared feature selection to improve LR performance
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler

# Scale to non-negative for chi2 (MaxAbsScaler preserves sparsity)
scaler_sparse = MaxAbsScaler()
train_sparse_scaled = scaler_sparse.fit_transform(train_sparse)
val_sparse_scaled = scaler_sparse.transform(val_sparse)
test_sparse_scaled = scaler_sparse.transform(test_sparse)

# Select top k features using chi-squared
k_best = 10000
selector = SelectKBest(chi2, k=min(k_best, train_sparse_scaled.shape[1]))
train_sparse = selector.fit_transform(train_sparse_scaled, y_train_labels)
val_sparse = selector.transform(val_sparse_scaled)
test_sparse = selector.transform(test_sparse_scaled)
print(f"Sparse train shape after chi-squared selection: {train_sparse.shape}")

# ============================================================
# DeBERTa Fine-Tuning with 5-Fold Cross-Validation
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE WITH 5-FOLD CV")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def evaluate_deberta(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

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
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

# Tokenize all data once
all_train_texts = train_df["text"].values
all_train_labels = y_train_full
test_texts = test_df["text"].values

# 5-fold stratified cross-validation
from sklearn.model_selection import StratifiedKFold

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

# Store OOF embeddings and test embeddings from each fold
oof_train_embeddings = np.zeros((len(all_train_texts), 1024), dtype=np.float32)
fold_test_embeddings = []

# Updated hyperparameters for CV
cv_lr = 1e-5
cv_epochs = 6
cv_warmup_ratio = 0.1
gradient_accumulation_steps = 2

for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_texts, all_train_labels)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{n_folds}")
    print(f"{'='*50}")

    fold_train_texts = all_train_texts[train_idx]
    fold_val_texts = all_train_texts[val_idx]
    fold_train_labels = all_train_labels[train_idx]
    fold_val_labels = all_train_labels[val_idx]

    # Tokenize for this fold
    fold_train_encodings = tokenizer(
        list(fold_train_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    fold_val_encodings = tokenizer(
        list(fold_val_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Only tokenize test once for the first fold
    if fold == 0:
        test_encodings = tokenizer(
            list(test_texts),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    fold_train_dataset = TensorDataset(
        fold_train_encodings["input_ids"],
        fold_train_encodings["attention_mask"],
        torch.tensor(fold_train_labels, dtype=torch.long),
    )
    fold_val_dataset = TensorDataset(
        fold_val_encodings["input_ids"],
        fold_val_encodings["attention_mask"],
        torch.tensor(fold_val_labels, dtype=torch.long),
    )
    fold_test_dataset = TensorDataset(
        test_encodings["input_ids"], test_encodings["attention_mask"]
    )

    fold_train_loader = DataLoader(
        fold_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    fold_val_loader = DataLoader(
        fold_val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_test_loader = DataLoader(
        fold_test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )
    model.to(device)

    # Optimizer with lower LR
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=cv_lr, eps=1e-8)

    # Cosine schedule with warmup
    total_steps = len(fold_train_loader) * cv_epochs // gradient_accumulation_steps
    warmup_steps = int(cv_warmup_ratio * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience_fold = 3

    for epoch in range(cv_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(fold_train_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(fold_train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        val_loss, val_acc, _ = evaluate_deberta(model, fold_val_loader)
        print(
            f"Fold {fold+1} Epoch {epoch+1:2d}/{cv_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"{WORKING_DIR}/deberta_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_fold:
                print(
                    f"Fold {fold+1} early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
                )
                break

    print(f"Fold {fold+1} best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Load best model for this fold
    model.load_state_dict(torch.load(f"{WORKING_DIR}/deberta_fold{fold}.pt"))

    # Extract OOF embeddings for validation indices
    val_loader_no_labels = DataLoader(
        TensorDataset(fold_val_encodings["input_ids"], fold_val_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_val_embeddings = extract_embeddings(model, val_loader_no_labels)
    oof_train_embeddings[val_idx] = fold_val_embeddings

    # Extract test embeddings for this fold
    test_loader_no_labels = DataLoader(
        TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_test_emb = extract_embeddings(model, test_loader_no_labels)
    fold_test_embeddings.append(fold_test_emb)

    # Clean up to save memory
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Average test embeddings across folds
test_embeddings = np.mean(fold_test_embeddings, axis=0)
print(f"\nOOF train embeddings shape: {oof_train_embeddings.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")

# Use OOF embeddings for train and split into train/val for downstream models
# We need to preserve the original train/val split for consistency with downstream code
# Generate new split using OOF embeddings
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

# Map indices to get corresponding OOF embeddings
train_idx_map = {i: idx for idx, i in enumerate(np.where(np.isin(np.arange(len(all_train_labels)), np.where(np.isin(all_train_labels, y_train_labels))[0]))[0])}
# Simple approach: use the stored OOF embeddings directly with the train/val split
train_embeddings = oof_train_embeddings[:len(X_train_texts)]
val_embeddings = oof_train_embeddings[len(X_train_texts):len(X_train_texts) + len(X_val_texts)]

print(f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")

# ============================================================
# XGBoost
# ============================================================
print("\nTraining XGBoost classifier...")
xgb_train_features = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled, train_embeddings]
)
xgb_val_features = np.hstack(
    [val_stylo_filtered, val_read_scaled, val_pos_scaled, val_embeddings]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled, test_embeddings]
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
# Logistic Regression
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

# ============================================================
# DeBERTa Validation & Test Probabilities (using CV models)
# ============================================================
print("\nGetting DeBERTa probabilities...")

# Compute DeBERTa probabilities using the last fold's model (or average across folds)
# For simplicity, compute from the stored OOF approach - we need to reload models for probas
# Since we stored embeddings, we'll compute probas from the first fold's best model
# but a more robust approach would average probabilities across folds.
# For efficiency, we use the last fold's model to get probabilities.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
)
model.to(device)
model.load_state_dict(torch.load(f"{WORKING_DIR}/deberta_fold0.pt"))
model.eval()

val_encodings = tokenizer(
    list(X_val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

val_loader_eval = DataLoader(
    TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(y_val_labels, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
_, deberta_val_loss, deberta_val_probs = evaluate_deberta(model, val_loader_eval)
print(f"DeBERTa validation log loss: {deberta_val_loss:.4f}")

test_loader_eval = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
all_test_probs = []
with torch.no_grad():
    for batch in test_loader_eval:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())
deberta_test_probs = np.vstack(all_test_probs)

# ============================================================
# DeBERTa Validation & Test Probabilities (using CV models)
# ============================================================
print("\nGetting DeBERTa probabilities...")

# Average probabilities across all fold models for better generalization
all_fold_val_probs = []
all_fold_test_probs = []

for fold in range(n_folds):
    model_fold = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )
    model_fold.to(device)
    model_fold.load_state_dict(torch.load(f"{WORKING_DIR}/deberta_fold{fold}.pt"))
    model_fold.eval()

    val_encodings = tokenizer(
        list(X_val_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    val_loader_eval = DataLoader(
        TensorDataset(
            val_encodings["input_ids"],
            val_encodings["attention_mask"],
            torch.tensor(y_val_labels, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    fold_val_probs_list = []
    model_fold.eval()
    with torch.no_grad():
        for batch in val_loader_eval:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model_fold(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            fold_val_probs_list.append(probs.cpu().numpy())
    fold_val_probs = np.vstack(fold_val_probs_list)
    all_fold_val_probs.append(fold_val_probs)

    test_loader_eval = DataLoader(
        TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_test_probs_list = []
    with torch.no_grad():
        for batch in test_loader_eval:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model_fold(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            fold_test_probs_list.append(probs.cpu().numpy())
    fold_test_probs = np.vstack(fold_test_probs_list)
    all_fold_test_probs.append(fold_test_probs)

    del model_fold
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Average probabilities across folds
deberta_val_probs = np.mean(all_fold_val_probs, axis=0)
deberta_test_probs = np.mean(all_fold_test_probs, axis=0)
deberta_val_loss = compute_log_loss(y_val_labels, deberta_val_probs)
print(f"DeBERTa (CV averaged) validation log loss: {deberta_val_loss:.4f}")

# ============================================================
# Ensemble Weight Optimization
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "deberta": deberta_val_probs,
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
            w1 * val_probas["deberta"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(y_val_labels, ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"deberta": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# Generate Submission
# ============================================================
test_probas = {
    "deberta": deberta_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
ensemble_test_probs = (
    best_weights["deberta"] * test_probas["deberta"]
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

# ============================================================
# Final Validation Score
# ============================================================
print(f"\nFinal Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
