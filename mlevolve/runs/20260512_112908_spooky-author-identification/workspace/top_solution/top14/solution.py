import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from scipy.sparse import save_npz, hstack
import re
import pickle
import os
from collections import Counter
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import lightgbm as lgb
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# CONFIGURATION
# ============================================
MAX_LENGTH = 192
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_CLUSTERS = 5
TEMPERATURE = 2.0
MIXUP_ALPHA = 0.2

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================
# DATA LOADING
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# ============================================
# SPLIT DATA FIRST (must be before any fitting of feature extractors)
# ============================================
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_idx = list(skf.split(train_df, train_df["author_encoded"]))[0]
train_idx, val_idx = fold_idx

# Create train/val splits
# ============================================
# TEXT CLEANING FIRST (must be done before split)
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

train_df_split = train_df.iloc[train_idx].copy()
val_df_split = train_df.iloc[val_idx].copy()

train_texts = train_df_split["clean_text"].tolist()
val_texts = val_df_split["clean_text"].tolist()
test_texts = test_df["clean_text"].tolist()

train_labels = train_df_split["author_encoded"].values
val_labels = val_df_split["author_encoded"].values

np.save("./working/train_labels.npy", train_labels)
np.save("./working/val_labels.npy", val_labels)

print(
    f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_df)}"
)

# ============================================
# FEATURE ENGINEERING - ONLY ON TRAIN SPLIT
# ============================================


def extract_stylistic_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    features["word_count"] = len(words)
    features["char_count"] = len(text)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["word_length_std"] = np.std([len(w) for w in words]) if words else 0
    features["sent_count"] = len(sentences)
    features["avg_sent_length"] = features["word_count"] / max(
        features["sent_count"], 1
    )

    punct_counts = {
        "commas": text.count(","),
        "semicolons": text.count(";"),
        "colons": text.count(":"),
        "exclamation": text.count("!"),
        "question": text.count("?"),
        "quotes": text.count('"') + text.count("'"),
        "dashes": text.count("-") + text.count("—"),
        "parentheses": text.count("(") + text.count(")"),
        "periods": text.count("."),
    }
    for punct_name, count in punct_counts.items():
        features[f"punct_{punct_name}_density"] = count / max(features["word_count"], 1)

    caps_words = len(re.findall(r"\b[A-Z][a-z]*\b", str(text)))
    all_caps_words = len(re.findall(r"\b[A-Z]{2,}\b", str(text)))
    features["caps_word_ratio"] = caps_words / max(len(words), 1)
    features["all_caps_ratio"] = all_caps_words / max(len(words), 1)

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
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
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
    }
    stop_count = sum(1 for w in words if w in stop_words)
    features["stop_word_ratio"] = stop_count / max(len(words), 1)

    conjunctions = {"and", "but", "or", "nor", "for", "yet", "so"}
    conj_count = sum(1 for w in words if w in conjunctions)
    features["conjunction_density"] = conj_count / max(len(words), 1)

    prepositions = {
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
    }
    prep_count = sum(1 for w in words if w in prepositions)
    features["preposition_density"] = prep_count / max(len(words), 1)

    vowels = sum(1 for c in text if c in "aeiou")
    consonants = sum(1 for c in text if c.isalpha() and c not in "aeiou")
    features["vowel_ratio"] = vowels / max(consonants + vowels, 1)

    punct_chars = sum(1 for c in text if c in ".,;:!?\"'()-[]{}")
    features["punct_char_ratio"] = punct_chars / max(len(words), 1)

    word_lengths = [len(w) for w in words]
    if word_lengths:
        features["short_word_ratio"] = sum(1 for l in word_lengths if l <= 3) / len(
            word_lengths
        )
        features["medium_word_ratio"] = sum(
            1 for l in word_lengths if 4 <= l <= 7
        ) / len(word_lengths)
        features["long_word_ratio"] = sum(1 for l in word_lengths if l >= 8) / len(
            word_lengths
        )
    else:
        features["short_word_ratio"] = 0
        features["medium_word_ratio"] = 0
        features["long_word_ratio"] = 0

    return features


print("Extracting stylistic features...")
train_stylistic = train_df_split["clean_text"].apply(extract_stylistic_features)
val_stylistic = val_df_split["clean_text"].apply(extract_stylistic_features)
test_stylistic = test_df["clean_text"].apply(extract_stylistic_features)
train_stylistic_df = pd.DataFrame(train_stylistic.tolist())
val_stylistic_df = pd.DataFrame(val_stylistic.tolist())
test_stylistic_df = pd.DataFrame(test_stylistic.tolist())


def extract_authorial_patterns(text):
    features = {}
    archaic_words = {
        "thou",
        "thee",
        "thy",
        "thine",
        "doth",
        "hath",
        "art",
        "wast",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "perchance",
        "forsooth",
        "betwixt",
        "amongst",
        "whilst",
        "anon",
        "nay",
        "aye",
        "ne",
        "methinks",
        "prithee",
    }
    features["archaic_word_count"] = sum(
        1 for word in str(text).lower().split() if word in archaic_words
    )
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "cryptic",
        "lurking",
        "nameless",
        "unnameable",
        "antiquarian",
        "squamous",
        "gibbous",
        "blasphemous",
        "non-euclidean",
        "dimensional",
        "pandemoniac",
        "primordial",
        "antediluvian",
        "prehuman",
        "rugose",
        "ichor",
        "loathsome",
    }
    features["lovecraft_word_count"] = sum(
        1 for word in str(text).lower().split() if word in lovecraft_words
    )
    poe_words = {
        "nevermore",
        "chamber",
        "tintinnabulation",
        "sepulchre",
        "raven",
        "divan",
        "plutonian",
        "nightly",
        "dreaming",
        "stateliness",
        "outré",
        "phantasm",
        "mystification",
        "arabesque",
        "grotesque",
    }
    features["poe_word_count"] = sum(
        1 for word in str(text).lower().split() if word in poe_words
    )
    shelley_words = {
        "monster",
        "daemon",
        "creation",
        "creator",
        "wretch",
        "fiend",
        "vilest",
        "curse",
        "vengeance",
        "torment",
        "inarticulate",
        "agony",
        "despair",
        "benevolent",
        "ardour",
    }
    features["shelley_word_count"] = sum(
        1 for word in str(text).lower().split() if word in shelley_words
    )
    return features


print("Extracting authorial patterns...")
train_auth_patterns = train_df_split["clean_text"].apply(extract_authorial_patterns)
val_auth_patterns = val_df_split["clean_text"].apply(extract_authorial_patterns)
test_auth_patterns = test_df["clean_text"].apply(extract_authorial_patterns)
train_auth_patterns_df = pd.DataFrame(train_auth_patterns.tolist())
val_auth_patterns_df = pd.DataFrame(val_auth_patterns.tolist())
test_auth_patterns_df = pd.DataFrame(test_auth_patterns.tolist())


def extract_pos_patterns(text):
    features = {}
    words = str(text).lower().split()
    ing_words = sum(1 for w in words if w.endswith("ing") and len(w) > 4)
    features["ing_word_ratio"] = ing_words / max(len(words), 1)
    ed_words = sum(1 for w in words if w.endswith("ed") and len(w) > 4)
    features["ed_word_ratio"] = ed_words / max(len(words), 1)
    ly_words = sum(1 for w in words if w.endswith("ly") and len(w) > 4)
    features["ly_word_ratio"] = ly_words / max(len(words), 1)
    tion_words = sum(1 for w in words if w.endswith(("tion", "sion")))
    features["tion_word_ratio"] = tion_words / max(len(words), 1)
    adj_suffixes = sum(1 for w in words if w.endswith(("ive", "ous", "ful", "less")))
    features["adj_suffix_ratio"] = adj_suffixes / max(len(words), 1)
    return features


print("Extracting POS patterns...")
train_pos = train_df_split["clean_text"].apply(extract_pos_patterns)
val_pos = val_df_split["clean_text"].apply(extract_pos_patterns)
test_pos = test_df["clean_text"].apply(extract_pos_patterns)
train_pos_df = pd.DataFrame(train_pos.tolist())
val_pos_df = pd.DataFrame(val_pos.tolist())
test_pos_df = pd.DataFrame(test_pos.tolist())

print("Combining all features...")
print("Extracting character n-grams...")
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(1, 4), max_features=5000, min_df=3
)
train_char_ngrams = char_vectorizer.fit_transform(train_df_split["clean_text"])
val_char_ngrams = char_vectorizer.transform(val_df_split["clean_text"])
test_char_ngrams = char_vectorizer.transform(test_df["clean_text"])

print("Extracting word n-grams...")
word_vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 3), max_features=5000, min_df=3, sublinear_tf=True
)
train_word_ngrams = word_vectorizer.fit_transform(train_df_split["clean_text"])
val_word_ngrams = word_vectorizer.transform(val_df_split["clean_text"])
test_word_ngrams = word_vectorizer.transform(test_df["clean_text"])

train_dense = pd.concat(
    [train_stylistic_df, train_auth_patterns_df, train_pos_df], axis=1
)
val_dense = pd.concat(
    [val_stylistic_df, val_auth_patterns_df, val_pos_df], axis=1
)
test_dense = pd.concat([test_stylistic_df, test_auth_patterns_df, test_pos_df], axis=1)
train_dense = train_dense.fillna(0)
val_dense = val_dense.fillna(0)
test_dense = test_dense.fillna(0)

scaler = StandardScaler()
train_dense_scaled = scaler.fit_transform(train_dense)
val_dense_scaled = scaler.transform(val_dense)
test_dense_scaled = scaler.transform(test_dense)
train_dense_features = train_dense_scaled
val_dense_features = val_dense_scaled
test_dense_features = test_dense_scaled

train_char_grams_split = train_char_ngrams
val_char_grams_split = val_char_ngrams
test_char_grams_split = test_char_ngrams

train_word_grams_split = train_word_ngrams
val_word_grams_split = val_word_ngrams
test_word_grams_split = test_word_ngrams

# ============================================
# LOAD DEBERTA MODEL (Step 3 style)
# ============================================
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)
model.config.gradient_checkpointing = True

# ============================================
# CLUSTERING FOR AUGMENTATION
# ============================================
print("Creating style-based clusters for augmentation...")


def get_embeddings(texts, batch_size=64):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            with autocast():
                outputs = model.deberta(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                mask = attention_mask.unsqueeze(-1).float()
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


train_embeddings = get_embeddings(train_texts)
print(f"Train embeddings shape: {train_embeddings.shape}")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=5)
cluster_labels = kmeans.fit_predict(train_embeddings)
print(f"Cluster distribution: {np.bincount(cluster_labels)}")


# ============================================
# DATASET WITH MIXUP
# ============================================
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


train_dataset = SimpleDataset(
    train_texts, train_labels, tokenizer
)


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
)
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    list(zip(test_encodings["input_ids"], test_encodings["attention_mask"])),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================
# OPTIMIZER & SCHEDULER
# ============================================
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler_grad = GradScaler()


# ============================================
# TRAINING LOOP
# ============================================
def train_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    all_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)
    loss = log_loss(all_labels, all_preds)
    return loss, all_preds


print("\nStarting training...")
best_val_loss = float("inf")
best_epoch = 0
patience = 5
no_improve = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler_grad)
    val_loss, val_preds = evaluate(model, val_loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ============================================
# EXTRACT DEBERTA EMBEDDINGS FOR LIGHTGBM
# ============================================
print("\nExtracting embeddings for LightGBM...")
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()


def extract_embeddings(texts, batch_size=64):
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            with autocast():
                outputs = model.deberta(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                mask = attention_mask.unsqueeze(-1).float()
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


train_embs = extract_embeddings(train_texts)
val_embs = extract_embeddings(val_texts)
test_embs = extract_embeddings(test_texts)
print(
    f"Embeddings shape: train {train_embs.shape}, val {val_embs.shape}, test {test_embs.shape}"
)

# ============================================
# LIGHTGBM ON COMBINED FEATURES
# ============================================
print("\nTraining LightGBM ensemble...")
X_train_combined = np.hstack([train_embs, train_dense_features])
X_val_combined = np.hstack([val_embs, val_dense_features])
X_test_combined = np.hstack([test_embs, test_dense_features])

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=32,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train_combined,
    train_labels,
    eval_set=[(X_val_combined, val_labels)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
)

lgb_val_preds = lgb_model.predict_proba(X_val_combined)
lgb_val_preds = np.clip(lgb_val_preds, 1e-15, 1 - 1e-15)
lgb_val_loss = log_loss(val_labels, lgb_val_preds)
print(f"LightGBM Validation Log Loss: {lgb_val_loss:.4f}")

# ============================================
# FINAL VALIDATION SCORE (weighted average)
# ============================================
deberta_val_loss, _ = evaluate(model, val_loader)
print(f"DeBERTa Validation Log Loss: {deberta_val_loss:.4f}")

# Weighted ensemble
final_val_preds = 0.6 * val_preds + 0.4 * lgb_val_preds
final_val_preds = np.clip(final_val_preds, 1e-15, 1 - 1e-15)
final_val_loss = log_loss(val_labels, final_val_preds)
print(f"Ensemble Validation Log Loss: {final_val_loss:.4f}")

# ============================================
# TEST PREDICTIONS
# ============================================
print("\nGenerating test predictions...")

# DeBERTa predictions
model.eval()
deberta_test_preds = []
with torch.no_grad():
    for input_ids, attention_mask in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
        deberta_test_preds.append(probs.cpu().numpy())
deberta_test_preds = np.vstack(deberta_test_preds)

# LightGBM predictions
lgb_test_preds = lgb_model.predict_proba(X_test_combined)
lgb_test_preds = np.clip(lgb_test_preds, 1e-15, 1 - 1e-15)

# Weighted ensemble
final_test_preds = 0.6 * deberta_test_preds + 0.4 * lgb_test_preds
final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

# ============================================
# SUBMISSION
# ============================================
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": final_test_preds[:, 0],
        "HPL": final_test_preds[:, 1],
        "MWS": final_test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved with shape: {submission.shape}")

print(f"Final Validation Score: {final_val_loss}")