#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete solution for Spooky Author Identification
Uses DistilBERT with focal loss, gradient accumulation, and early stopping.
"""

import pandas as pd
import numpy as np
import re
import os
import gc
import warnings
import math
from collections import Counter

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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import joblib

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# =============================================================================
# 2. TEXT CLEANING FUNCTION
# =============================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (Stratified)
# =============================================================================
print("Creating stratified train/validation split...")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train_full)):
    if fold == 0:
        train_indices = train_idx
        val_indices = val_idx
        break

train_texts = train_df["text"].values
train_labels = train_labels_full = y_train_full

train_texts_fold = train_texts[train_indices]
val_texts_fold = train_texts[val_indices]
train_labels_fold = train_labels[train_indices]
val_labels_fold = train_labels[val_indices]
test_texts = test_df["text"].values

print(f"Train fold size: {len(train_texts_fold)}")
print(f"Val fold size: {len(val_texts_fold)}")

# =============================================================================
# 4. MODEL SETUP
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 2

print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
model.to(device)

# =============================================================================
# 5. TOKENIZATION
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
# 6. CREATE DATALOADERS
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
# 7. FOCAL LOSS DEFINITION
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


focal_loss_fn = FocalLoss(gamma=2.0)

# =============================================================================
# 8. OPTIMIZER, SCHEDULER, MIXED PRECISION
# =============================================================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# =============================================================================
# 9. TRAINING LOOP
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
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = focal_loss_fn(logits, labels) / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        train_steps += 1

    avg_train_loss = total_train_loss / max(train_steps, 1)

    # Validation
    model.eval()
    val_preds = []
    val_true = []
    total_val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = focal_loss_fn(logits, labels)

            total_val_loss += loss.item()
            val_steps += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_preds.append(probs)
            val_true.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / max(val_steps, 1)
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
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  -> New best model saved (Log Loss: {val_log_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# =============================================================================
# 10. FINAL VALIDATION SCORE
# =============================================================================
print("Loading best model for final validation...")
model.load_state_dict(best_model_state)
model.to(device)
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
# 11. TEST INFERENCE
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
# 12. CREATE SUBMISSION FILE
# =============================================================================
print("Creating submission file...")
test_ids = test_df["id"].values
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
