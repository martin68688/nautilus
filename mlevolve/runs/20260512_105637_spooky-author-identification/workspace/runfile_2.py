import os
os.sched_setaffinity(0, {19, 58, 59, 60, 61, 62, 63})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import re
import string
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from collections import Counter

# Create working directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================
# Configuration
# ============================================
MODEL_NAME = "microsoft/deberta-v3-large"
NUM_CLASSES = 3
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_PATIENCE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# 1. LOAD DATA
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors: {train_df['author'].value_counts().to_dict()}")


# ============================================
# 2. TEXT CLEANING
# ============================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# ============================================
# 3. FEATURE ENGINEERING (for potential use)
# ============================================
def extract_basic_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features["word_count"] = len(words)
    features["char_count"] = len(text)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["unique_word_ratio"] = len(set(w.lower() for w in words)) / max(
        len(words), 1
    )
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_sentence_length"] = (
        features["word_count"] / features["sentence_count"]
    )
    punct_counts = Counter(text)
    features["exclamation_count"] = punct_counts.get("!", 0)
    features["question_count"] = punct_counts.get("?", 0)
    features["comma_count"] = punct_counts.get(",", 0)
    features["semicolon_count"] = punct_counts.get(";", 0)
    features["colon_count"] = punct_counts.get(":", 0)
    features["dash_count"] = punct_counts.get("-", 0) + punct_counts.get("—", 0)
    features["quotes_count"] = punct_counts.get('"', 0) + punct_counts.get("'", 0)
    features["parentheses_count"] = punct_counts.get("(", 0) + punct_counts.get(")", 0)
    total_punct = sum(punct_counts.get(p, 0) for p in string.punctuation)
    features["punct_ratio"] = total_punct / max(len(text), 1)
    features["capital_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)
    features["ellipsis_count"] = text.count("...") + text.count("…")
    features["ampersand_count"] = text.count("&")
    return features


basic_train = train_df["clean_text"].apply(extract_basic_features)
basic_train_df = pd.DataFrame(basic_train.tolist())

# Character n-grams - will fit after train/val split
# Create train/validation split FIRST so we have X_train_texts
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["clean_text"].values,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train,
)

char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=500, lowercase=True
)
char_ngrams_train = char_vectorizer.fit_transform(X_train_texts)
char_ngram_train_df = pd.DataFrame(
    char_ngrams_train.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_ngrams_train.shape[1])],
)

# Word n-grams - will fit after train/val split
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=3000,
    lowercase=True,
    sublinear_tf=True,
    min_df=5,
    max_df=0.8,
    stop_words="english",
)
word_ngrams_train = word_vectorizer.fit_transform(X_train_texts)
word_ngram_train_df = pd.DataFrame(
    word_ngrams_train.toarray(),
    columns=[f"word_ngram_{i}" for i in range(word_ngrams_train.shape[1])],
)

# Function words
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
    "as",
    "was",
    "were",
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
    "is",
    "are",
    "be",
    "not",
    "no",
    "nor",
    "so",
    "if",
    "then",
    "than",
    "that",
    "this",
    "those",
    "these",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "their",
    "his",
    "her",
    "my",
    "your",
    "our",
    "who",
    "whom",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "many",
    "some",
    "any",
    "none",
    "one",
    "two",
    "other",
    "another",
    "more",
    "most",
    "such",
    "only",
    "own",
    "same",
    "into",
    "over",
    "under",
    "above",
    "below",
    "between",
    "through",
    "during",
    "before",
    "after",
    "until",
    "since",
    "about",
    "against",
    "without",
    "within",
    "along",
    "among",
    "upon",
    "across",
    "down",
    "up",
    "off",
    "out",
    "around",
    "very",
    "too",
    "quite",
    "rather",
    "almost",
    "nearly",
    "just",
    "even",
    "still",
    "already",
    "yet",
    "also",
    "again",
    "here",
    "there",
    "now",
    "then",
    "never",
    "always",
    "often",
    "sometimes",
    "us",
    "me",
    "i",
    "you",
}


def extract_function_word_features(text):
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return {"function_word_ratio": 0, "function_word_count": 0}
    func_count = sum(1 for w in words if w in function_words)
    return {
        "function_word_ratio": func_count / total,
        "function_word_count": func_count,
    }


func_word_train = train_df["clean_text"].apply(extract_function_word_features)
func_word_train_df = pd.DataFrame(func_word_train.tolist())

# Combine all features
train_features = pd.concat(
    [
        basic_train_df.reset_index(drop=True),
        func_word_train_df.reset_index(drop=True),
        char_ngram_train_df.reset_index(drop=True),
        word_ngram_train_df.reset_index(drop=True),
    ],
    axis=1,
)

# Scale numeric features - fit on training split only
scaler = StandardScaler()
numeric_cols = [
    c
    for c in train_features.columns
    if not c.startswith(("char_ngram_", "word_ngram_"))
]
boolean_cols = [c for c in numeric_cols if train_features[c].nunique() <= 2]
numeric_cols = [c for c in numeric_cols if c not in boolean_cols]
if numeric_cols:
    train_features_indexed = train_features.copy()
    train_features_indexed[numeric_cols] = scaler.fit_transform(train_features_indexed[numeric_cols])
    train_features = train_features_indexed

# Save preprocessors for consistency
pd.to_pickle(label_encoder, "./working/label_encoder.pkl")

print(
    f"Train samples: {len(X_train_texts)}, Val samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)


# ============================================
# 4. DATASET AND DATALOADER
# ============================================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = TextDataset(X_train_texts, y_train_labels)
val_dataset = TextDataset(X_val_texts, y_val_labels)
test_dataset = TextDataset(test_df["clean_text"].values)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================
# 5. MODEL DEFINITION
# ============================================
print("Loading DeBERTa-v3-large model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# ============================================
# 6. TRAINING LOOP
# ============================================
print("Starting training...")
best_val_loss = float("inf")
patience = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            val_loss += outputs.loss.item()
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_probs = np.concatenate(all_val_probs, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)

    # Clip probabilities
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    # Calculate log loss
    val_log_loss = -np.mean(
        np.sum(np.eye(NUM_CLASSES)[val_labels] * np.log(val_probs_clipped), axis=1)
    )

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Log Loss: {val_log_loss:.4f}"
    )

    # Save best model
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        patience = 0
        torch.save(model.state_dict(), "./working/best_model_d326d632f03e40ecb5822c39a483c5dd.pth")
        print(f"  -> Saved best model (val_log_loss: {best_val_loss:.4f})")
    else:
        patience += 1
        if patience >= MAX_PATIENCE:
            print(f"  -> Early stopping triggered")
            break

# ============================================
# 7. LOAD BEST MODEL AND EVALUATE
# ============================================
print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load("./working/best_model_d326d632f03e40ecb5822c39a483c5dd.pth"))
model.eval()

# Final validation
all_val_probs = []
all_val_true_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_true_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true_labels = np.concatenate(all_val_true_labels, axis=0)

eps = 1e-15
val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

final_val_log_loss = -np.mean(
    np.sum(np.eye(NUM_CLASSES)[val_true_labels] * np.log(val_probs_clipped), axis=1)
)
print(f"Final Validation Log Loss: {final_val_log_loss:.6f}")

# ============================================
# 8. TEST INFERENCE
# ============================================
print("Performing test inference...")
model.eval()
all_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)
print(f"Test predictions shape: {test_probs.shape}")

# ============================================
# 9. CREATE SUBMISSION
# ============================================
print("Creating submission file...")
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

# Normalize probabilities
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

# Clip to avoid extremes
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], eps, 1 - eps)
    submission[col] = submission[col] / submission[["EAP", "HPL", "MWS"]].sum(axis=1)

submission.to_csv("./submission/submission_d326d632f03e40ecb5822c39a483c5dd.csv", index=False)
print(f"Submission saved to ./submission/submission_d326d632f03e40ecb5822c39a483c5dd.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {final_val_log_loss}")