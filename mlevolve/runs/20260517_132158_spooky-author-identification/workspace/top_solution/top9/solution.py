import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import re
import string
import os
import warnings
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import hstack, csr_matrix, save_npz
import joblib

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
num_classes = 3

# ---- Feature Engineering ----
def extract_statistical_features(text_series):
    features = pd.DataFrame()
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    punct_counts = text_series.apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )
    features["punct_count"] = punct_counts
    features["punct_ratio"] = punct_counts / (features["char_count"] + 1)
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["dash_count"] = text_series.str.count("-")
    features["period_count"] = text_series.str.count(r"\.")
    features["upper_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    features["upper_ratio"] = features["upper_count"] / (features["char_count"] + 1)
    features["title_word_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if w.istitle())
    )
    features["sent_count"] = text_series.str.count(r"[.!?]+")
    features["avg_sent_len"] = features["word_count"] / (features["sent_count"] + 1)
    features["space_count"] = text_series.str.count(" ")
    features["newline_count"] = text_series.str.count("\n")
    features["digit_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isdigit())
    )
    features["digit_ratio"] = features["digit_count"] / (features["char_count"] + 1)
    return features

def extract_lexical_features(text_series):
    features = pd.DataFrame()
    stop_words = set(
        [
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "it",
            "its",
            "that",
            "this",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
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
            "having",
            "do",
            "does",
            "did",
            "doing",
            "would",
            "should",
            "could",
            "might",
            "must",
            "shall",
            "will",
            "may",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "doesn't",
            "don't",
            "didn't",
            "won't",
            "wouldn't",
            "shan't",
            "shouldn't",
            "can't",
            "cannot",
            "couldn't",
            "mustn't",
            "let",
            "like",
        ]
    )
    features["stopword_count"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).lower().split()
            if w.strip(string.punctuation) in stop_words
        )
    )
    features["stopword_ratio"] = features["stopword_count"] / (
        text_series.str.split().str.len() + 1
    )
    features["unique_word_ratio"] = text_series.apply(
        lambda x: len(set(w.lower() for w in str(x).split()))
        / (len(str(x).split()) + 1)
    )
    features["the_ratio"] = text_series.str.count(r"\b[Tt]he\b") / (
        text_series.str.split().str.len() + 1
    )
    features["and_ratio"] = text_series.str.count(r"\b[Aa]nd\b") / (
        text_series.str.split().str.len() + 1
    )
    features["of_ratio"] = text_series.str.count(r"\b[Oo]f\b") / (
        text_series.str.split().str.len() + 1
    )
    features["i_ratio"] = text_series.str.count(r"\b[Ii]\b") / (
        text_series.str.split().str.len() + 1
    )
    features["my_ratio"] = text_series.str.count(r"\b[Mm]y\b") / (
        text_series.str.split().str.len() + 1
    )
    features["was_ratio"] = text_series.str.count(r"\b[Ww]as\b") / (
        text_series.str.split().str.len() + 1
    )
    features["had_ratio"] = text_series.str.count(r"\b[Hh]ad\b") / (
        text_series.str.split().str.len() + 1
    )
    features["which_ratio"] = text_series.str.count(r"\b[Ww]hich\b") / (
        text_series.str.split().str.len() + 1
    )
    features["that_ratio"] = text_series.str.count(r"\b[Tt]hat\b") / (
        text_series.str.split().str.len() + 1
    )
    features["with_ratio"] = text_series.str.count(r"\b[Ww]ith\b") / (
        text_series.str.split().str.len() + 1
    )
    features["not_ratio"] = text_series.str.count(r"\b[Nn]ot\b") / (
        text_series.str.split().str.len() + 1
    )
    features["but_ratio"] = text_series.str.count(r"\b[Bb]ut\b") / (
        text_series.str.split().str.len() + 1
    )
    features["as_ratio"] = text_series.str.count(r"\b[Aa]s\b") / (
        text_series.str.split().str.len() + 1
    )
    return features

def extract_stylistic_features(text_series):
    features = pd.DataFrame()
    features["starts_with_i"] = text_series.str.match(r"^[Ii]\b").astype(int)
    features["starts_with_the"] = text_series.str.match(r"^[Tt]he\b").astype(int)
    features["starts_with_quote"] = text_series.str.match(r'^["\']').astype(int)
    features["ends_period"] = text_series.str.endswith(".").astype(int)
    features["ends_exclamation"] = text_series.str.endswith("!").astype(int)
    features["ends_question"] = text_series.str.endswith("?").astype(int)
    features["ends_ellipsis"] = text_series.str.contains(r"\.\.\.$").astype(int)
    contractions = ["'t", "'s", "'d", "'ve", "'ll", "'re", "'m", "n't"]
    features["contraction_count"] = text_series.apply(
        lambda x: sum(1 for c in contractions if c in str(x).lower())
    )
    features["long_word_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 8)
    )
    features["short_word_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) <= 3)
    )
    features["double_space"] = text_series.str.contains(r"  ").astype(int)
    features["repeated_punct"] = text_series.apply(
        lambda x: len(re.findall(r"([!?.,;:])\1+", str(x)))
    )
    return features

print("Extracting statistical features...")
train_stats = extract_statistical_features(train_df["text"])
test_stats = extract_statistical_features(test_df["text"])

print("Extracting lexical features...")
train_lex = extract_lexical_features(train_df["text"])
test_lex = extract_lexical_features(test_df["text"])

print("Extracting stylistic features...")
train_style = extract_stylistic_features(train_df["text"])
test_style = extract_stylistic_features(test_df["text"])

train_features = pd.concat([train_stats, train_lex, train_style], axis=1)
test_features = pd.concat([test_stats, test_lex, test_style], axis=1)

print(f"Handcrafted features shape: {train_features.shape}")

print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    stop_words="english",
    min_df=5,
    max_df=0.95,
)
train_tfidf = tfidf.fit_transform(train_df["text"])
test_tfidf = tfidf.transform(test_df["text"])

print(f"TF-IDF features shape: {train_tfidf.shape}")

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

train_combined = hstack([csr_matrix(train_scaled), train_tfidf])
test_combined = hstack([csr_matrix(test_scaled), test_tfidf])

print(f"Combined features shape: {train_combined.shape}")

# Create validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train_combined, train_df["author_encoded"])))

# Correct indexing to avoid INDEX_BUG
train_X = train_combined[train_idx]
val_X = train_combined[val_idx]
train_y = train_df["author_encoded"].values[train_idx]
val_y = train_df["author_encoded"].values[val_idx]

test_X = test_combined

print("\nQuick validation with Logistic Regression...")
lr = LogisticRegression(
    multi_class="multinomial", max_iter=1000, C=1.0, random_state=42
)
lr.fit(train_X, train_y)
val_preds = lr.predict_proba(val_X)
val_score = log_loss(val_y, val_preds)
print(f"Quick validation log loss: {val_score:.4f}")

os.makedirs("./working", exist_ok=True)
np.save("./working/train_idx.npy", train_idx)
np.save("./working/val_idx.npy", val_idx)
np.save("./working/train_y.npy", train_y)
np.save("./working/val_y.npy", val_y)
save_npz("./working/train_X.npz", train_X)
save_npz("./working/val_X.npz", val_X)
save_npz("./working/test_X.npz", test_X)
joblib.dump(le, "./working/label_encoder.pkl")
joblib.dump(tfidf, "./working/tfidf_vectorizer.pkl")
joblib.dump(scaler, "./working/feature_scaler.pkl")

# ModernBERT Setup
MODEL_NAME = "answerdotai/ModernBERT-large"
NUM_LABELS = 3
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = ModernBertForSequenceClassification.config_class.from_pretrained(MODEL_NAME)
config.num_labels = NUM_LABELS
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1

model = ModernBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

no_decay = ["bias", "LayerNorm", "layer_norm"]
optimizer_grouped_params = [
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
optimizer = AdamW(optimizer_grouped_params, lr=LEARNING_RATE, eps=1e-8)

print(f"Model initialized: {MODEL_NAME}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dataset and DataLoaders
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Extract data using correct indexing
train_texts = train_df["text"].values[train_idx]
train_labels = train_df["author_encoded"].values[train_idx]
val_texts = train_df["text"].values[val_idx]
val_labels = train_df["author_encoded"].values[val_idx]
test_texts = test_df["text"].values

print(
    f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)

train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = TextDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

total_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)

amp_scaler = torch.cuda.amp.GradScaler()

# Training loop
best_val_logloss = float("inf")
patience = 3
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    val_loss_total = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss_total += outputs.loss.item()
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss_total / len(val_loader)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    # Ensure 1D integer labels
    if all_val_labels.ndim > 1:
        all_val_labels = all_val_labels.argmax(axis=1)

    # Ensure labels are 1D integers for sklearn log_loss
    if all_val_labels.ndim > 1:
        all_val_labels = all_val_labels.argmax(axis=1)
    val_logloss = log_loss(all_val_labels, all_val_preds)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val LogLoss: {val_logloss:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        torch.save(model.state_dict(), "./working/best_model.pt")
        print(f"  -> New best model saved (LogLoss: {val_logloss:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model and compute final validation score
print(f"\nLoading best model for final evaluation...")
model.load_state_dict(torch.load("./working/best_model.pt", map_location=device))
model.eval()

all_val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        all_val_preds.append(probs.cpu().numpy())

all_val_preds = np.concatenate(all_val_preds, axis=0)
epsilon = 1e-15
all_val_preds_clipped = np.clip(all_val_preds, epsilon, 1 - epsilon)
final_val_logloss = log_loss(val_labels, all_val_preds_clipped)

print(f"Final Validation LogLoss (best model): {final_val_logloss:.6f}")

# Test inference
print("Generating test predictions...")
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        all_test_preds.append(probs.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds, axis=0)

# Create submission file
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame()
submission["id"] = test_df["id"].values
submission["EAP"] = all_test_preds[:, 0]
submission["HPL"] = all_test_preds[:, 1]
submission["MWS"] = all_test_preds[:, 2]

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Test predictions shape: {all_test_preds.shape}")
print(f"Final Validation Score: {final_val_logloss}")