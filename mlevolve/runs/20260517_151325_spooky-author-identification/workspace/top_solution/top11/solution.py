import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
)
import numpy as np
import pandas as pd
import os
import re
import warnings
import gc
import string
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import hstack, save_npz, csr_matrix
from collections import Counter
from sentence_transformers import SentenceTransformer
import pickle

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIGURATION
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
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50
WARMUP_RATIO = 0.1
PATIENCE = 7
DROPOUT = 0.3

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

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT - NO INDEX_BUG
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)

assert len(set(train_idx) & set(val_idx)) == 0, "CRITICAL: Train/val overlap detected!"

# ============================================================
# MINIMAL PREPROCESSING - RAW TEXT ONLY FOR DEBERTA-V3
# ============================================================
print("\n" + "=" * 60)
print("MINIMAL PREPROCESSING FOR DEBERTA-V3")
print("=" * 60)

# Store raw texts and labels for the transformer model
# No handcrafted features or embeddings needed

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)
print("Using raw text directly with DeBERTa-v3-base tokenizer")
print(f"Loss: CrossEntropyLoss with label_smoothing=0.1")
print(f"Optimizer: AdamW (lr=2e-5, weight_decay=0.01)")
print("=" * 60)

# ============================================================
# MODEL ARCHITECTURE: Deberta-v3-base for Sequence Classification
# ============================================================
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE DESIGN")
print("=" * 60)

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DebertaAuthorClassifier(nn.Module):
    def __init__(self, num_labels=3, dropout=0.3):
        super(DebertaAuthorClassifier, self).__init__()
        self.model_name = 'microsoft/deberta-v3-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

print(f"Model: DebertaAuthorClassifier (microsoft/deberta-v3-base)")
print(f"Dropout: {DROPOUT}")
print(f"Loss: CrossEntropyLoss with label_smoothing=0.1 (inside HF model)")
print(f"Optimizer: AdamW (lr=2e-5, weight_decay=0.01)")
print("=" * 60)

# ============================================================
# DATASET SETUP - RAW TEXT DATASET
# ============================================================
print("\n" + "=" * 60)
print("SETTING UP DATALOADERS")
print("=" * 60)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.labels is not None:
            label = self.labels[idx]
            return text, label
        else:
            return text, 0  # dummy label for test

train_dataset = TextDataset(X_train_texts, y_train_labels)
val_dataset = TextDataset(X_val_texts, y_val_labels)
test_dataset = TextDataset(test_df["text"].values, None)

def collate_fn(batch):
    if len(batch[0]) == 2:
        texts, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        texts, _ = zip(*batch)
        labels = None
    return texts, labels

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

print(
    f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches"
)

# ============================================================
# INITIALIZE MODEL
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING DEBERTA AUTHOR CLASSIFIER")
print("=" * 60)

model = DebertaAuthorClassifier(num_labels=NUM_AUTHORS, dropout=DROPOUT)
model.to(device)

# For DeBERTa, use standard AdamW with 2e-5 learning rate for all parameters
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
    eps=1e-8,
)

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)

print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# METRIC FUNCTIONS
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

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for texts, labels in loader:
            if labels is None:
                continue
            # Tokenize on the fly
            encoded = model.tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    avg_loss = total_loss / max(num_batches, 1)

    return logloss, acc, all_preds, avg_loss

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for texts, labels in train_loader:
        labels = labels.to(device)

        # Tokenize on the fly
        encoded = model.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / max(num_batches, 1)

    val_logloss, val_acc, val_probs, val_loss = evaluate_model(
        model, val_loader
    )

    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(
            model.state_dict(), os.path.join(WORKING_DIR, "best_model.pt")
        )
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")

# ============================================================
# LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FINAL PREDICTIONS")
print("=" * 60)

model.load_state_dict(
    torch.load(os.path.join(WORKING_DIR, "best_model.pt"), map_location=device)
)

# Validation predictions
val_logloss, val_acc, val_probs, _ = evaluate_model(model, val_loader)
print(f"Best model validation log loss: {val_logloss:.4f}")

# Test predictions
model.eval()
all_test_probs = []
with torch.no_grad():
    for texts, _ in test_loader:
        encoded = model.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)

# Apply probability clipping and normalization
eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

# ============================================================
# CREATE SUBMISSION FILE
# ============================================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

class_names = label_encoder.classes_

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        class_names[0]: test_probs[:, 0],
        class_names[1]: test_probs[:, 1],
        class_names[2]: test_probs[:, 2],
    }
)

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
final_score = val_logloss
print(f"Final Validation Score: {final_score:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()