#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merged solution for Spooky Author Identification
Combines data processing, model design, and training/evaluation.
"""

import pandas as pd
import numpy as np
import re
import os
import gc
import warnings
import math

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
submission_template = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# =============================================================================
# 2. TEXT CLEANING AND BASIC FEATURE ENGINEERING
# =============================================================================
print("Creating basic features...")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_stylistic_features(text):
    features = {}
    if not isinstance(text, str) or len(text) == 0:
        features["char_count"] = 0
        features["word_count"] = 0
        features["avg_word_length"] = 0
        features["sentence_count"] = 0
        features["exclamation_count"] = 0
        features["question_count"] = 0
        features["comma_count"] = 0
        features["period_count"] = 0
        features["semicolon_count"] = 0
        features["colon_count"] = 0
        features["dash_count"] = 0
        features["quote_count"] = 0
        features["paren_count"] = 0
        features["uppercase_word_ratio"] = 0
        features["stopword_ratio"] = 0
        features["punctuation_ratio"] = 0
        features["digit_ratio"] = 0
        features["capitalized_word_ratio"] = 0
        features["unique_word_ratio"] = 0
        features["hapax_legomena_ratio"] = 0
        features["char_per_word"] = 0
        features["ellipsis_count"] = 0
        features["interrobang_count"] = 0
        return features

    text_clean = clean_text(text)
    chars = len(text_clean)
    words = text_clean.split()
    num_words = len(words)

    features["char_count"] = chars
    features["word_count"] = num_words
    features["avg_word_length"] = (
        np.mean([len(w) for w in words]) if num_words > 0 else 0
    )
    features["exclamation_count"] = text_clean.count("!")
    features["question_count"] = text_clean.count("?")
    features["comma_count"] = text_clean.count(",")
    features["period_count"] = text_clean.count(".")
    features["semicolon_count"] = text_clean.count(";")
    features["colon_count"] = text_clean.count(":")
    features["dash_count"] = len(re.findall(r"[-–—]", text_clean))
    features["quote_count"] = (
        text_clean.count('"')
        + text_clean.count("'")
        + text_clean.count('"')
        + text_clean.count("""""")
    )
    features["paren_count"] = (
        text_clean.count("(")
        + text_clean.count(")")
        + text_clean.count("[")
        + text_clean.count("]")
    )
    features["ellipsis_count"] = text_clean.count("...") + text_clean.count("…")

    uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    capitalized_words = sum(1 for w in words if w[0].isupper() if len(w) > 0)
    features["uppercase_word_ratio"] = (
        uppercase_words / num_words if num_words > 0 else 0
    )
    features["capitalized_word_ratio"] = (
        capitalized_words / num_words if num_words > 0 else 0
    )

    unique_words = set(w.lower() for w in words)
    features["unique_word_ratio"] = (
        len(unique_words) / num_words if num_words > 0 else 0
    )

    from collections import Counter

    word_counts = Counter(w.lower() for w in words)
    hapax = sum(1 for count in word_counts.values() if count == 1)
    features["hapax_legomena_ratio"] = hapax / num_words if num_words > 0 else 0

    features["char_per_word"] = chars / num_words if num_words > 0 else 0
    punct_chars = sum(1 for c in text_clean if c in "!?.,;:-'\"()[]{}…—")
    features["punctuation_ratio"] = punct_chars / chars if chars > 0 else 0
    digit_count = sum(1 for c in text_clean if c.isdigit())
    features["digit_ratio"] = digit_count / chars if chars > 0 else 0

    stopwords = {
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
        "are",
        "has",
        "have",
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
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "am",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "all",
        "any",
        "each",
        "every",
        "some",
        "such",
    }
    stopword_count = sum(1 for w in words if w.lower() in stopwords)
    features["stopword_ratio"] = stopword_count / num_words if num_words > 0 else 0

    return features


print("Extracting stylistic features for training data...")
stylistic_features_train = train_df["text"].apply(extract_stylistic_features)
stylistic_df_train = pd.DataFrame(stylistic_features_train.tolist())

print("Extracting stylistic features for test data...")
stylistic_features_test = test_df["text"].apply(extract_stylistic_features)
stylistic_df_test = pd.DataFrame(stylistic_features_test.tolist())

# =============================================================================
# 3. TEXT FEATURES: N-GRAMS AND TF-IDF
# =============================================================================
print("Creating TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=15000,
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=15000,
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)

word_bigram_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(2, 2),
    max_features=5000,
    min_df=3,
    max_df=0.7,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)

train_text_clean = train_df["text"].apply(clean_text)
test_text_clean = test_df["text"].apply(clean_text)

print("Fitting word TF-IDF...")
word_tfidf_train = word_vectorizer.fit_transform(train_text_clean)
word_tfidf_test = word_vectorizer.transform(test_text_clean)

print("Fitting character TF-IDF...")
char_tfidf_train = char_vectorizer.fit_transform(train_text_clean)
char_tfidf_test = char_vectorizer.transform(test_text_clean)

print("Fitting word bigram TF-IDF...")
bigram_tfidf_train = word_bigram_vectorizer.fit_transform(train_text_clean)
bigram_tfidf_test = word_bigram_vectorizer.transform(test_text_clean)

# =============================================================================
# 4. ADDITIONAL ADVANCED FEATURES
# =============================================================================
print("Creating advanced features...")


def extract_advanced_features(text):
    features = {}
    if not isinstance(text, str) or len(text) == 0:
        features["avg_sentence_length"] = 0
        features["first_word_length"] = 0
        features["last_word_length"] = 0
        features["contraction_count"] = 0
        features["archaic_word_count"] = 0
        features["supernatural_word_count"] = 0
        return features

    text_clean = clean_text(text)
    words = text_clean.split()

    sentences = re.split(r"[.!?]+", text_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    features["avg_sentence_length"] = (
        np.mean([len(s.split()) for s in sentences]) if sentences else 0
    )

    if words:
        features["first_word_length"] = len(words[0])
        features["last_word_length"] = len(words[-1])
    else:
        features["first_word_length"] = 0
        features["last_word_length"] = 0

    contraction_pattern = r"\b(can't|don't|won't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|didn't|it's|i'm|i'll|i've|i'd|you're|you'll|you've|you'd|he's|he'll|he'd|she's|she'll|she'd|they're|they'll|they've|they'd|we're|we'll|we've|we'd|that's|who's|what's|where's|there's|let's)\b"
    features["contraction_count"] = len(
        re.findall(contraction_pattern, text_clean, re.IGNORECASE)
    )

    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "art",
        "wast",
        "wert",
        "whence",
        "thence",
        "hither",
        "thither",
        "unto",
        "betwixt",
        "ere",
        "whilst",
        "an",
        "perchance",
        "forsooth",
        "prithee",
        "pray",
        "wherefore",
        "oft",
        "albeit",
        "anon",
        "hark",
        "hie",
        "methinks",
        "nay",
        "ne'er",
        "o'er",
        "tis",
        "twas",
        "twixt",
        "yonder",
    ]
    features["archaic_word_count"] = sum(1 for w in words if w.lower() in archaic_words)

    supernatural_words = [
        "eldritch",
        "cyclopean",
        "cryptic",
        "primordial",
        "antediluvian",
        "nameless",
        "unnameable",
        "unspeakable",
        "cursed",
        "accursed",
        "blasphemous",
        "squamous",
        "ichor",
        "necronomicon",
        "cthulhu",
        "yog-sothoth",
        "nyarlathotep",
        "azathoth",
        "shoggoth",
        "moonbeast",
        "ghast",
        "ghoul",
        "ghastly",
        "hideous",
        "monstrous",
        "loathsome",
        "abysmal",
        "abyss",
        "chaos",
        "void",
        "cosmic",
        "daemon",
        "fiend",
        "hellish",
        "infernal",
        "malevolent",
        "diabolical",
        "sorcerous",
        "incantation",
        "occult",
        "arcane",
        "esoteric",
        "forbidden",
        "indescribable",
        "unutterable",
        "inconceivable",
    ]
    features["supernatural_word_count"] = sum(
        1 for w in words if w.lower() in supernatural_words
    )

    return features


print("Extracting advanced features for training data...")
advanced_train = train_df["text"].apply(extract_advanced_features)
advanced_df_train = pd.DataFrame(advanced_train.tolist())

print("Extracting advanced features for test data...")
advanced_test = test_df["text"].apply(extract_advanced_features)
advanced_df_test = pd.DataFrame(advanced_test.tolist())

# =============================================================================
# 5. COMBINE ALL FEATURES
# =============================================================================
print("Combining all features...")

scaler = StandardScaler()
numerical_features_train = pd.concat([stylistic_df_train, advanced_df_train], axis=1)
numerical_features_test = pd.concat([stylistic_df_test, advanced_df_test], axis=1)

numerical_features_train = numerical_features_train.fillna(0)
numerical_features_test = numerical_features_test.fillna(0)

numerical_scaled_train = scaler.fit_transform(numerical_features_train)
numerical_scaled_test = scaler.transform(numerical_features_test)

X_train = hstack(
    [
        csr_matrix(numerical_scaled_train),
        word_tfidf_train,
        char_tfidf_train,
        bigram_tfidf_train,
    ]
)

X_test = hstack(
    [
        csr_matrix(numerical_scaled_test),
        word_tfidf_test,
        char_tfidf_test,
        bigram_tfidf_test,
    ]
)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["author"])

# =============================================================================
# 6. CREATE TRAIN/VALIDATION SPLIT
# =============================================================================
print("Creating stratified train/validation split...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    if fold == 0:
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]
        train_indices = train_idx
        val_indices = val_idx
        break

# =============================================================================
# 7. SAVE PROCESSED DATA
# =============================================================================
print("Saving processed data...")
os.makedirs("./working", exist_ok=True)

import joblib

joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_bigram_vectorizer, "./working/word_bigram_vectorizer.pkl")
joblib.dump(scaler, "./working/scaler.pkl")
joblib.dump(label_encoder, "./working/label_encoder.pkl")
joblib.dump(X_train, "./working/X_train.pkl")
joblib.dump(X_test, "./working/X_test.pkl")
joblib.dump(y_train, "./working/y_train.pkl")
joblib.dump(X_train_fold, "./working/X_train_fold.pkl")
joblib.dump(X_val_fold, "./working/X_val_fold.pkl")
joblib.dump(y_train_fold, "./working/y_train_fold.pkl")
joblib.dump(y_val_fold, "./working/y_val_fold.pkl")
joblib.dump(test_df["id"].values, "./working/test_ids.pkl")
joblib.dump(train_df["id"].values, "./working/train_ids.pkl")
submission_template.to_csv("./working/submission_template.csv", index=False)

print(f"Feature matrix shapes:")
print(f"  X_train_fold: {X_train_fold.shape}")
print(f"  X_val_fold: {X_val_fold.shape}")
print(f"  X_test: {X_test.shape}")

del X_train, train_text_clean, test_text_clean
del stylistic_df_train, stylistic_df_test, advanced_df_train, advanced_df_test
del numerical_features_train, numerical_features_test
gc.collect()

# =============================================================================
# 8. MODEL SETUP
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "microsoft/deberta-v3-large"
NUM_LABELS = 3
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1

print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
model.to(device)

# Extract original text data for BERT tokenization
train_texts = train_df["text"].values
test_texts = test_df["text"].values
train_labels = label_encoder.transform(train_df["author"].values)

train_texts_fold = train_texts[train_indices]
val_texts_fold = train_texts[val_indices]
train_labels_fold = train_labels[train_indices]
val_labels_fold = train_labels[val_indices]

# =============================================================================
# 9. TOKENIZATION
# =============================================================================
print("Tokenizing training data...")
train_encodings = tokenizer(
    train_texts_fold.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors="pt",
)

print("Tokenizing validation data...")
val_encodings = tokenizer(
    val_texts_fold.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors="pt",
)

print("Tokenizing test data...")
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors="pt",
)

# =============================================================================
# 10. CREATE DATALOADERS
# =============================================================================
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels_fold, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels_fold, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# =============================================================================
# 11. OPTIMIZER, SCHEDULER, MIXED PRECISION
# =============================================================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
grad_scaler = torch.cuda.amp.GradScaler()

# =============================================================================
# 12. TRAINING LOOP
# =============================================================================
print("Starting training...")
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    train_steps = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        total_train_loss += loss.item()
        train_steps += 1

    avg_train_loss = total_train_loss / train_steps

    model.eval()
    val_preds = []
    val_true = []
    total_val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

            total_val_loss += loss.item()
            val_steps += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_preds.append(probs)
            val_true.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / val_steps
    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )
    val_log_loss = log_loss(val_true, val_preds_normalized)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Log Loss: {val_log_loss:.6f}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_epoch = epoch + 1
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), "./working/best_model.pt")
        print(f"  -> New best model saved (Log Loss: {val_log_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# =============================================================================
# 13. FINAL VALIDATION SCORE
# =============================================================================
print("Loading best model for final validation...")
model.load_state_dict(best_model_state)
model.eval()

val_final_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        val_final_preds.append(probs)

val_final_preds = np.concatenate(val_final_preds, axis=0)
val_final_preds_clipped = np.clip(val_final_preds, 1e-15, 1 - 1e-15)
val_final_preds_normalized = val_final_preds_clipped / val_final_preds_clipped.sum(
    axis=1, keepdims=True
)
final_val_score = log_loss(val_true, val_final_preds_normalized)

# =============================================================================
# 14. TEST INFERENCE
# =============================================================================
print("Running test inference...")
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        test_preds.append(probs)

test_preds = np.concatenate(test_preds, axis=0)

# =============================================================================
# 15. CREATE SUBMISSION FILE
# =============================================================================
print("Creating submission file...")
test_ids = joblib.load("./working/test_ids.pkl")
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission_df = submission_df[["id", "EAP", "HPL", "MWS"]]

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

print(f"Final Validation Score: {final_val_score}")