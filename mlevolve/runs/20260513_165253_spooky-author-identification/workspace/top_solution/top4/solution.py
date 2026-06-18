import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import sparse
import re
import string
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

# Ensure directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


# Basic text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Advanced stylometric features
def extract_stylometric_features(text):
    features = {}
    text_str = str(text)
    words = text_str.split()
    sentences = re.split(r"[.!?]+", text_str)
    sentences = [s.strip() for s in sentences if s.strip()]

    features["word_count"] = len(words)
    features["char_count"] = len(text_str)
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_word_length"] = features["char_count"] / max(
        features["word_count"], 1
    )
    features["avg_sentence_length"] = (
        features["word_count"] / features["sentence_count"]
    )
    features["exclamation_count"] = text_str.count("!")
    features["question_count"] = text_str.count("?")
    features["comma_count"] = text_str.count(",")
    features["semicolon_count"] = text_str.count(";")
    features["colon_count"] = text_str.count(":")
    features["dash_count"] = text_str.count("-") + text_str.count("—")
    features["quote_count"] = text_str.count('"') + text_str.count("'")
    features["capital_word_ratio"] = sum(
        1 for w in words if w and w[0].isupper()
    ) / max(len(words), 1)
    features["all_caps_ratio"] = sum(
        1 for w in words if w.isupper() and len(w) > 1
    ) / max(len(words), 1)
    features["article_count"] = sum(1 for w in words if w.lower() in ["a", "an", "the"])
    features["pronoun_count"] = sum(
        1
        for w in words
        if w.lower()
        in [
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
        ]
    )
    features["preposition_count"] = sum(
        1
        for w in words
        if w.lower()
        in [
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "from",
            "of",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
        ]
    )
    features["conjunction_count"] = sum(
        1
        for w in words
        if w.lower()
        in [
            "and",
            "but",
            "or",
            "nor",
            "yet",
            "so",
            "because",
            "although",
            "while",
            "if",
            "when",
            "where",
            "how",
        ]
    )
    features["syllable_count"] = sum(
        1 for w in words for v in "aeiou" if v in w.lower()
    )
    features["complex_word_ratio"] = sum(1 for w in words if len(w) > 6) / max(
        len(words), 1
    )
    features["very_long_word_ratio"] = sum(1 for w in words if len(w) > 10) / max(
        len(words), 1
    )
    features["unique_chars_ratio"] = len(set(text_str.lower())) / max(len(text_str), 1)
    features["digit_ratio"] = sum(1 for c in text_str if c.isdigit()) / max(
        len(text_str), 1
    )
    features["type_token_ratio"] = len(set(w.lower() for w in words)) / max(
        len(words), 1
    )

    common_words = [
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "that",
        "was",
        "it",
        "with",
        "i",
        "had",
        "his",
        "my",
        "he",
        "not",
        "but",
        "me",
        "all",
        "this",
        "by",
        "were",
        "so",
        "no",
        "if",
        "up",
        "out",
        "as",
        "at",
        "for",
        "on",
        "be",
        "or",
        "from",
        "what",
        "which",
        "who",
        "more",
        "than",
        "their",
        "with",
        "when",
        "where",
        "there",
        "then",
        "upon",
        "into",
    ]
    for w in common_words:
        features[f"fw_{w}"] = sum(1 for tw in words if tw.lower() == w)

    return pd.Series(features)


print("Extracting stylometric features...")
train_stylometric = train_df["text"].apply(extract_stylometric_features)
test_stylometric = test_df["text"].apply(extract_stylometric_features)

# TF-IDF features with character n-grams
print("Creating TF-IDF features...")
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
)
train_char_features = tfidf_char.fit_transform(train_df["text"])
test_char_features = tfidf_char.transform(test_df["text"])

# Word n-grams up to 3
tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
    stop_words="english",
)
train_word_features = tfidf_word.fit_transform(train_df["text"])
test_word_features = tfidf_word.transform(test_df["text"])

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["author"])

# Create stratified splits first (needed for correct stylometric scaling)
print("Creating train/validation splits...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
split_idx = list(skf.split(train_char_features, train_labels))[0]
train_idx, val_idx = split_idx

# Now scale stylometric features using only training data to avoid leakage
stylometric_cols = train_stylometric.columns.tolist()
scaler = StandardScaler()
train_stylometric_train = train_stylometric.iloc[train_idx]
scaler.fit(train_stylometric_train)
train_stylometric_scaled = scaler.transform(train_stylometric)
test_stylometric_scaled = scaler.transform(test_stylometric)

# Combine all features
train_features_full = sparse.hstack(
    [
        train_char_features,
        train_word_features,
        sparse.csr_matrix(train_stylometric_scaled),
    ]
)
test_features = sparse.hstack(
    [test_char_features, test_word_features, sparse.csr_matrix(test_stylometric_scaled)]
)

# Create training and validation feature matrices after split
train_features = train_features_full[train_idx]
val_features = train_features_full[val_idx]

# Create training and validation sets
X_train = train_features.toarray()
y_train = train_labels[train_idx]
X_val = val_features.toarray()
y_val = train_labels[val_idx]
X_test = test_features.toarray()
test_ids = test_df["id"].values

print(
    f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}, Test samples: {test_df.shape[0]}"
)
print(f"Feature dimension: {train_features.shape[1]}")


# Define model
class FeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# Initialize model and data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
model = FeatureClassifier(input_dim=input_dim, num_classes=3, dropout_rate=0.3)
model.to(device)
print(
    f"Model initialized on {device} with {sum(p.numel() for p in model.parameters()):,} parameters"
)
print(f"Input dimension: {input_dim}")

batch_size = 16
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Optimizer with differentiated learning rates
optimizer = AdamW(
    [
        {"params": model.head[0].parameters(), "lr": 3e-4, "weight_decay": 0.01},
        {"params": model.head[2].parameters(), "lr": 3e-4, "weight_decay": 0.01},
        {"params": model.head[4].parameters(), "lr": 2e-4, "weight_decay": 0.01},
        {"params": model.head[6].parameters(), "lr": 2e-4, "weight_decay": 0.01},
        {"params": model.head[8].parameters(), "lr": 1e-4, "weight_decay": 0.01},
        {"params": model.head[10].parameters(), "lr": 1e-4, "weight_decay": 0.01},
    ]
)

# Label smoothing criterion
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

# Training hyperparameters
num_epochs = 50
patience = 7
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

# Cosine annealing scheduler with warm restarts
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
)
initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Warmup + Cosine scheduling
        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            for pg_idx, pg in enumerate(optimizer.param_groups):
                pg["lr"] = initial_lrs[pg_idx] * (current_step / max(1, warmup_steps))
        else:
            scheduler.step(current_step)

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(features)
                loss = criterion(logits, labels)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)

    # Clip probabilities and normalize
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )

    # Calculate log loss
    val_log_loss = log_loss(val_true, val_preds_normalized)

    # Print epoch summary
    print(
        f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}'
    )

    # Early stopping and model saving
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_log_loss": val_log_loss,
            },
            "./working/best_model.pt",
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

print(
    f"\nTraining completed. Best validation log loss: {best_val_loss:.6f} at epoch {best_epoch+1}"
)

# Load best model for validation and test inference
checkpoint = torch.load("./working/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Final validation prediction with best model
print("Performing final validation inference...")
val_preds_final = []
with torch.no_grad():
    for features, _ in val_loader:
        features = features.to(device)
        with autocast():
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
        val_preds_final.append(probs.cpu().numpy())

val_preds_final = np.concatenate(val_preds_final)
val_preds_final = np.clip(val_preds_final, 1e-15, 1 - 1e-15)
val_preds_final = val_preds_final / val_preds_final.sum(axis=1, keepdims=True)

# Calculate final validation score
score = log_loss(y_val, val_preds_final)

# Test inference
print("Performing test inference...")
test_preds = []
with torch.no_grad():
    for (features,) in test_loader:
        features = features.to(device)
        with autocast():
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds)
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission dataframe
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

# Save submission
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# Final validation score
print(f"Final Validation Score: {score}")