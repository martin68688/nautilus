import os
os.sched_setaffinity(0, {165, 114, 115, 54, 62})
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
# 3. FULL DATA PREPARATION (5-fold CV will be done in training loop)
# =============================================================================
print("Preparing full dataset for 5-fold cross-validation...")

# Extract text and label arrays BEFORE the cross-validation loop
train_texts = train_df["text"].values
y_train_full = train_df["author"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")

# =============================================================================
# 4. MODEL SETUP (shared configuration)
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
GRADIENT_ACCUMULATION_STEPS = 4

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =============================================================================
# 5. TOKENIZATION (test data only, train data will be tokenized per fold)
# =============================================================================
print("Tokenizing test data...")
test_encodings = tokenizer(
    test_df["text"].values.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors="pt",
)

# Create test dataset and dataloader (shared across folds)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# =============================================================================
# 6. CROSS-VALIDATION TRAINING LOOP
# =============================================================================
print("Starting 5-fold cross-validation training...")

# Store test predictions from each fold
all_test_preds = []
fold_val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, y_train_full)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*60}")

    # Prepare fold data - tokenize within the fold to prevent data leakage
    train_texts_fold = train_df["text"].values[train_idx]
    val_texts_fold = train_df["text"].values[val_idx]

    # Fit label encoder only on training fold
    label_encoder = LabelEncoder()
    train_labels_fold = label_encoder.fit_transform(train_df["author"].values[train_idx])
    val_labels_fold = label_encoder.transform(train_df["author"].values[val_idx])

    print(f"Train fold size: {len(train_texts_fold)}")
    print(f"Val fold size: {len(val_texts_fold)}")

    # Tokenize fold data separately
    train_encodings_fold = tokenizer(
        train_texts_fold.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    val_encodings_fold = tokenizer(
        val_texts_fold.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    # Create fold-specific datasets and dataloaders
    train_dataset_fold = TensorDataset(
        train_encodings_fold["input_ids"],
        train_encodings_fold["attention_mask"],
        torch.tensor(train_labels_fold, dtype=torch.long),
    )
    val_dataset_fold = TensorDataset(
        val_encodings_fold["input_ids"],
        val_encodings_fold["attention_mask"],
        torch.tensor(val_labels_fold, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize a new model for this fold
    print(f"Loading model for fold {fold+1}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(device)

        # Loss function
    class FocalLoss(nn.Module):
        """Focal Loss focusing on hard examples."""
        def __init__(self, gamma=2.0, alpha=None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.ce = nn.CrossEntropyLoss(reduction='none')

        def forward(self, logits, targets):
            ce_loss = self.ce(logits, targets)
            probs = torch.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.gamma
            if self.alpha is not None:
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            loss = focal_weight * ce_loss
            return loss.mean()

    loss_fn = FocalLoss(gamma=2.0, alpha=None)

    # Optimizer, scheduler for this fold
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop for this fold
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
                loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION_STEPS

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
                    loss = loss_fn(logits, labels)

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

    fold_val_scores.append(best_val_loss)
    print(f"\nFold {fold+1} best validation Log Loss: {best_val_loss:.6f}")

    # Load best model for this fold and run test inference
    print(f"Running test inference for fold {fold+1}...")
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            fold_test_preds.append(probs)

    fold_test_preds = np.concatenate(fold_test_preds, axis=0)
    all_test_preds.append(fold_test_preds)

    # Clean up to free memory
    del model, best_model_state, train_loader, val_loader, train_dataset_fold, val_dataset_fold
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"CROSS-VALIDATION COMPLETE")
print(f"{'='*60}")

# Average test predictions from all folds
test_preds = np.mean(all_test_preds, axis=0)

# Compute overall validation score (average of fold scores)
overall_val_score = np.mean(fold_val_scores)
print(f"\nIndividual fold validation scores: {fold_val_scores}")
print(f"Average validation Log Loss across folds: {overall_val_score:.6f}")

# Print final validation metric as required
print(f'Final Validation Score: {overall_val_score}')

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
submission_df.to_csv("./submission/submission_bbaff07309a843bbaff9067342fc0c39.csv", index=False)
print(f"Submission saved to ./submission/submission_bbaff07309a843bbaff9067342fc0c39.csv")
print(f"Submission shape: {submission_df.shape}")

print(f"Final ensemble prediction shape: {test_preds.shape}")