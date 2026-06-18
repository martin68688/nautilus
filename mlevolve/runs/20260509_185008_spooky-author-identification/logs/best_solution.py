"""
Spooky Author Identification - Complete Pipeline
Combines data processing, DeBERTa fine-tuning, XGBoost, Logistic Regression, and ensemble
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
from scipy.sparse import hstack, save_npz
from collections import Counter
import string
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
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
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Authors distribution:\n{train_df['author'].value_counts()}")

# Encode labels
label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STRATIFIED SPLIT")
print("=" * 60)

X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print(f"Training samples: {len(X_train_texts)}")
print(f"Validation samples: {len(X_val_texts)}")
print(f"Test samples: {len(test_df)}")

# ============================================================
# FEATURE ENGINEERING - Stylometric Features
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING STYLOMETRIC FEATURES")
print("=" * 60)


def extract_stylometric_features(texts):
    features = []
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    uppercase_pattern = re.compile(r"[A-Z]")
    lowercase_pattern = re.compile(r"[a-z]")
    digit_pattern = re.compile(r"\d")
    whitespace_pattern = re.compile(r"\s+")

    for text in texts:
        if not text:
            features.append([0] * 30)
            continue
        text_len = len(text)
        word_count = len(text.split()) if text.strip() else 1
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sent_count = max(len(sentences), 1)
        avg_word_len = text_len / max(word_count, 1)
        avg_sent_len = word_count / sent_count
        upper_ratio = len(uppercase_pattern.findall(text)) / max(text_len, 1)
        lower_ratio = len(lowercase_pattern.findall(text)) / max(text_len, 1)
        digit_ratio = len(digit_pattern.findall(text)) / max(text_len, 1)
        whitespace_ratio = len(whitespace_pattern.findall(text)) / max(text_len, 1)
        comma_ratio = text.count(",") / max(text_len, 1)
        period_ratio = text.count(".") / max(text_len, 1)
        semicolon_ratio = text.count(";") / max(text_len, 1)
        colon_ratio = text.count(":") / max(text_len, 1)
        exclamation_ratio = text.count("!") / max(text_len, 1)
        question_ratio = text.count("?") / max(text_len, 1)
        dash_ratio = text.count("—") / max(text_len, 1)
        hyphen_ratio = text.count("-") / max(text_len, 1)
        quote_ratio = sum(1 for c in text if c in '""\'') / max(text_len, 1)
        paren_ratio = sum(1 for c in text if c in "()[]{}") / max(text_len, 1)
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / max(text_len, 1)
        words = whitespace_pattern.split(text)
        words = [w.strip(string.punctuation) for w in words if w.strip()]
        num_words = len(words)
        long_words = sum(1 for w in words if len(w) > 6) / max(num_words, 1)
        capitalized = sum(1 for w in words if w and w[0].isupper()) / max(num_words, 1)
        all_caps = sum(1 for w in words if len(w) > 1 and w.isupper()) / max(
            num_words, 1
        )
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sent_lengths:
            sent_len_std = np.std(sent_lengths) / max(np.mean(sent_lengths), 1)
            sent_len_var = np.var(sent_lengths) / max(np.mean(sent_lengths) ** 2, 1)
        else:
            sent_len_std = 0.0
            sent_len_var = 0.0
        function_words = {
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
            "and",
            "but",
            "or",
            "if",
            "as",
            "by",
            "from",
            "that",
            "this",
            "it",
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
            "not",
            "no",
            "nor",
            "so",
            "yet",
            "all",
            "each",
            "every",
            "some",
            "any",
            "many",
            "much",
            "few",
            "more",
            "most",
            "other",
            "such",
            "own",
            "same",
        }
        func_word_ratio = sum(1 for w in words if w.lower() in function_words) / max(
            num_words, 1
        )
        archaic_words = {
            "thou",
            "thee",
            "thy",
            "thine",
            "ye",
            "hath",
            "doth",
            "art",
            "wilt",
            "shalt",
            "dost",
            "ere",
            "whilst",
            "thence",
            "whence",
            "henceforth",
            "therein",
            "thereof",
            "hereunto",
            "perchance",
            "methinks",
            "forsooth",
            "wherefore",
            "hither",
            "thither",
            "unto",
            "nay",
            "betwixt",
            "prithee",
            "anon",
            "oft",
            "nigh",
        }
        archaic_ratio = sum(1 for w in words if w.lower() in archaic_words) / max(
            num_words, 1
        )
        emotional_words = {
            "horror",
            "terrible",
            "dreadful",
            "fearful",
            "awful",
            "hideous",
            "monstrous",
            "ghastly",
            "savage",
            "frightful",
            "appalling",
            "shocking",
            "terrifying",
            "horrible",
            "fearsome",
            "dread",
            "awe",
            "wonder",
            "sublime",
            "mysterious",
            "supernatural",
        }
        emotional_ratio = sum(1 for w in words if w.lower() in emotional_words) / max(
            num_words, 1
        )
        lovecraft_words = {
            "eldritch",
            "cyclopean",
            "antediluvian",
            "gibbous",
            "squamous",
            "ichor",
            "noisome",
            "cryptic",
            "foetid",
            "nacreous",
            "cosmic",
            "void",
            "nameless",
            "unspeakable",
            "unmentionable",
            "indescribable",
            "inconceivable",
            "abyss",
            "aeon",
            "non-euclidean",
        }
        lovecraft_ratio = sum(1 for w in words if w.lower() in lovecraft_words) / max(
            num_words, 1
        )
        sub_conj = {
            "although",
            "because",
            "since",
            "unless",
            "until",
            "while",
            "after",
            "before",
            "though",
            "whereas",
            "whenever",
            "wherever",
        }
        sub_conj_ratio = sum(1 for w in words if w.lower() in sub_conj) / max(
            num_words, 1
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
            comma_ratio,
            period_ratio,
            semicolon_ratio,
            colon_ratio,
            exclamation_ratio,
            question_ratio,
            dash_ratio,
            hyphen_ratio,
            quote_ratio,
            paren_ratio,
            char_diversity,
            long_words,
            capitalized,
            all_caps,
            sent_len_std,
            sent_len_var,
            func_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            sub_conj_ratio,
        ]
        features.append(feat)
    return np.array(features)


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

print(f"Stylometric features shape: {train_stylo_filtered.shape}")

# ============================================================
# FEATURE ENGINEERING - Readability Features
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING READABILITY FEATURES")
print("=" * 60)


def create_readability_features(texts):
    features = []
    vowels = "aeiouy"
    for text in texts:
        if not text or pd.isna(text):
            features.append([0, 0, 0, 0])
            continue
        text = str(text)
        words = text.split()
        num_words = len(words)
        num_sentences = max(len(re.split(r"[.!?]+", text)) - 1, 1)
        num_characters = len(text)
        syllable_count = 0
        for word in words:
            word = word.lower().strip(string.punctuation)
            if not word:
                continue
            prev_is_vowel = False
            word_syllables = 0
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    word_syllables += 1
                prev_is_vowel = is_vowel
            if word_syllables == 0:
                word_syllables = 1
            syllable_count += word_syllables
        if num_words > 0 and num_sentences > 0:
            fre = (
                206.835
                - 1.015 * (num_words / num_sentences)
                - 84.6 * (syllable_count / num_words)
            )
            fre = max(0, min(100, fre))
        else:
            fre = 0
        if num_words > 0 and num_sentences > 0:
            ari = (
                4.71 * (num_characters / num_words)
                + 0.5 * (num_words / num_sentences)
                - 21.43
            )
            ari = max(0, ari)
        else:
            ari = 0
        avg_syllables = syllable_count / max(num_words, 1)
        complex_words = 0
        for word in words:
            word = word.lower().strip(string.punctuation)
            if not word:
                continue
            prev_is_vowel = False
            word_syllables = 0
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    word_syllables += 1
                prev_is_vowel = is_vowel
            if word_syllables >= 3:
                complex_words += 1
        complex_ratio = complex_words / max(num_words, 1)
        features.append([fre, ari, avg_syllables, complex_ratio])
    return np.array(features)


train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

# ============================================================
# FEATURE ENGINEERING - POS Approximation
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING POS APPROXIMATION FEATURES")
print("=" * 60)


def create_pos_tag_approximation(texts):
    features = []
    noun_suffixes = (
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ence",
        "ance",
        "ism",
        "ist",
        "logy",
    )
    verb_suffixes = ("ed", "ing", "es", "ate", "ify", "ize", "ise", "en", "er", "est")
    adj_suffixes = (
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
        "ative",
    )
    adv_suffixes = ("ly", "wards", "wise", "ward")
    for text in texts:
        if not text or pd.isna(text):
            features.append([0, 0, 0, 0, 0])
            continue
        text = str(text)
        words = re.findall(r"\b\w+\b", text.lower())
        num_words = max(len(words), 1)
        noun_count = sum(1 for w in words if w.endswith(noun_suffixes))
        verb_count = sum(1 for w in words if w.endswith(verb_suffixes))
        adj_count = sum(1 for w in words if w.endswith(adj_suffixes))
        adv_count = sum(1 for w in words if w.endswith(adv_suffixes))
        content_words = noun_count + verb_count + adj_count + adv_count
        content_ratio = content_words / num_words
        features.append(
            [
                noun_count / num_words,
                verb_count / num_words,
                adj_count / num_words,
                adv_count / num_words,
                content_ratio,
            ]
        )
    return np.array(features)


train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# FEATURE ENGINEERING - Character N-grams
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING CHARACTER N-GRAM FEATURES")
print("=" * 60)

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

# ============================================================
# FEATURE ENGINEERING - Word N-grams
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING WORD N-GRAM FEATURES")
print("=" * 60)

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

# ============================================================
# FEATURE ENGINEERING - Punctuation Patterns
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING PUNCTUATION PATTERN FEATURES")
print("=" * 60)


def extract_punctuation_sequence(text):
    if not text:
        return ""
    return "".join([c for c in text if c in string.punctuation])


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

# ============================================================
# COMBINE SPARSE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("COMBINING SPARSE FEATURES")
print("=" * 60)

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
# LOAD DEBERTA MODEL
# ============================================================
print("\n" + "=" * 60)
print("LOADING DEBERTA MODEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
)
model.to(device)

# ============================================================
# TOKENIZE TEXT DATA
# ============================================================
print("\n" + "=" * 60)
print("TOKENIZING TEXT DATA")
print("=" * 60)

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

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(y_train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(y_val_labels, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

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
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# TRAIN DEBERTA
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE")
print("=" * 60)

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

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler() if torch.cuda.is_available() else None


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


best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

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
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, _ = evaluate_deberta(model, val_loader)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_deberta_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest DeBERTa model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
model.load_state_dict(
    torch.load("./working/best_deberta_model.pt", map_location=device)
)

# ============================================================
# EXTRACT DEBERTA EMBEDDINGS
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING DEBERTA EMBEDDINGS")
print("=" * 60)


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

train_embeddings = extract_embeddings(model, train_loader_no_labels)
val_embeddings = extract_embeddings(model, val_loader_no_labels)
test_embeddings = extract_embeddings(model, test_loader_no_labels)

print(f"Train embeddings: {train_embeddings.shape}")
print(f"Val embeddings: {val_embeddings.shape}")
print(f"Test embeddings: {test_embeddings.shape}")

# ============================================================
# TRAIN XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST CLASSIFIER")
print("=" * 60)

# Combine dense features with DeBERTa embeddings
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

xgb_train_probs = xgb_model.predict_proba(xgb_train_features)
xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)

xgb_val_loss = compute_log_loss(y_val_labels, xgb_val_probs)
print(f"XGBoost validation log loss: {xgb_val_loss:.4f}")

# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

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

lr_train_probs = lr_model.predict_proba(train_sparse)
lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)

lr_val_loss = compute_log_loss(y_val_labels, lr_val_probs)
print(f"Logistic Regression validation log loss: {lr_val_loss:.4f}")

# ============================================================
# GET DEBERTA VALIDATION PROBABILITIES
# ============================================================
print("\n" + "=" * 60)
print("GETTING DEBERTA VALIDATION PROBABILITIES")
print("=" * 60)

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

# Get DeBERTa test probabilities
test_loader_eval = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

model.eval()
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
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================
print("\n" + "=" * 60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("=" * 60)

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
# GENERATE TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING ENSEMBLE TEST PREDICTIONS")
print("=" * 60)

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

# ============================================================
# CREATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
print("\n" + "=" * 60)
print("FINAL VALIDATION SCORE")
print("=" * 60)

final_score = best_ll
print(f"Final Validation Score: {final_score}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
