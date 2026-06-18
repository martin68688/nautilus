import os
os.sched_setaffinity(0, {200, 166})
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
# 2. TEXT CLEANING & AUGMENTATION FUNCTIONS
# =============================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

import random

def word_dropout(text, dropout_prob=0.1):
    """
    Randomly replace 10% of tokens with [MASK] token to force model to learn
    higher-level stylistic patterns.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    tokens = text.split()
    if len(tokens) == 0:
        return text
    masked_tokens = []
    for token in tokens:
        if random.random() < dropout_prob:
            masked_tokens.append("[MASK]")
        else:
            masked_tokens.append(token)
    return " ".join(masked_tokens)

# =============================================================================
# 2b. STYLOMETRIC FEATURE EXTRACTION
# =============================================================================
def extract_stylometric_features(text):
    """
    Extract advanced stylometric features from text:
    - archaic/supernatural word counts
    - punctuation densities
    - sentence length statistics
    - readability scores
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "archaic_word_count": 0,
            "supernatural_word_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "comma_density": 0.0,
            "semicolon_density": 0.0,
            "avg_sentence_length": 0.0,
            "std_sentence_length": 0.0,
            "char_count": 0,
            "word_count": 0,
        }

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words) if len(words) > 0 else 1
    char_count = len(text)

    # Archaic words commonly found in 19th century literature
    archaic_words = {
        "thou", "thee", "thy", "thine", "doth", "hath", "wert", "art", "dost",
        "hast", "shalt", "wilt", "whence", "thence", "hither", "thither",
        "whither", "ere", "betwixt", "twixt", "perchance", "anon", "hark",
        "forsooth", "prithee", "methinks", "methought", "alack", "alas",
        "oft", "ofttimes", "oftentimes", "wherefore", "therefor", "unto",
        "nay", "yea", "tis", "twas", "twill", "twere", "neer", "eer", "olde"
    }

    # Supernatural/horror related words
    supernatural_words = {
        "ghost", "phantom", "spectre", "specter", "apparition", "wraith",
        "demon", "devil", "satanic", "infernal", "hellish", "cursed",
        "haunted", "supernatural", "unnatural", "unearthly", "eerie",
        "dreadful", "horrible", "terrible", "terrifying", "frightful",
        "shadowy", "gloomy", "dismal", "dreary", "macabre", "grotesque",
        "hideous", "monstrous", "fiend", "fiendish", "ghastly", "ghoulish",
        "sepulchral", "funereal", "mournful", "wailing", "howling",
        "tempest", "abyss", "chaos", "void", "darkness", "oblivion",
        "corpse", "coffin", "tomb", "grave", "graveyard", "cemetery",
        "lurid", "grisly", "gruesome", "horrid", "horrific", "uncanny"
    }

    archaic_word_count = sum(1 for w in words if w in archaic_words)
    supernatural_word_count = sum(1 for w in words if w in supernatural_words)

    # Punctuation features
    exclamation_count = text.count("!")
    question_count = text.count("?")
    comma_density = text.count(",") / char_count if char_count > 0 else 0.0
    semicolon_density = text.count(";") / char_count if char_count > 0 else 0.0

    # Sentence-level features
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) > 0:
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        std_sentence_length = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    else:
        avg_sentence_length = 0.0
        std_sentence_length = 0.0

    return {
        "archaic_word_count": archaic_word_count,
        "supernatural_word_count": supernatural_word_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "comma_density": comma_density,
        "semicolon_density": semicolon_density,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length,
        "char_count": char_count,
        "word_count": word_count,
    }

def extract_stylometric_features_batch(texts):
    """Extract stylometric features for a batch of texts."""
    features_list = []
    for text in texts:
        features_list.append(extract_stylometric_features(text))
    return pd.DataFrame(features_list).values.astype(np.float32)


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
STYLOMETRIC_FEATURE_DIM = 10

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =============================================================================
# 4b. CUSTOM MODEL WITH BiLSTM HEAD
# =============================================================================
class DistilBERTBiLSTM(nn.Module):
    def __init__(self, model_name, num_labels, stylometric_dim, hidden_size=256):
        super().__init__()
        self.distilbert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        # Remove the original classification head
        self.distilbert.classifier = nn.Identity()

        # BiLSTM head (2-layer bidirectional)
        self.bilstm = nn.LSTM(
            input_size=self.distilbert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Layer normalization after BiLSTM
        self.layer_norm = nn.LayerNorm(hidden_size * 2 + stylometric_dim)

        # Final classifier
        self.classifier = nn.Linear(hidden_size * 2 + stylometric_dim, num_labels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, stylometric_features=None):
        # Get hidden states from DistilBERT
        outputs = self.distilbert.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use all hidden states for BiLSTM input (sequence of hidden states)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # Pass through BiLSTM
        lstm_out, (hidden, cell) = self.bilstm(hidden_states)

        # Concatenate the final forward and backward hidden states
        # hidden shape: [num_layers * 2, batch, hidden_size]
        forward_hidden = hidden[-2, :, :]  # Last forward layer
        backward_hidden = hidden[-1, :, :]  # Last backward layer
        bilstm_out = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch, hidden_size*2]

        bilstm_out = self.dropout(bilstm_out)

        # Concatenate with stylometric features if available
        if stylometric_features is not None:
            combined = torch.cat([bilstm_out, stylometric_features], dim=1)
        else:
            combined = bilstm_out

        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        logits = self.classifier(combined)

        return logits

# =============================================================================
# 4c. FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =============================================================================
# 4d. GET SCHEDULER (linear with warmup)
# =============================================================================
from transformers import get_linear_schedule_with_warmup

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

# Extract stylometric features for test data
test_stylometric_features = extract_stylometric_features_batch(test_df["text"].values)
test_stylometric_tensor = torch.tensor(test_stylometric_features, dtype=torch.float32)

# Create test dataset and dataloader (shared across folds)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], test_stylometric_tensor
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

    # Apply word dropout to training texts (not validation)
    train_texts_fold_augmented = [word_dropout(text, dropout_prob=0.1) for text in train_texts_fold]

    # Tokenize fold data separately (training uses augmented texts)
    train_encodings_fold = tokenizer(
        train_texts_fold_augmented,
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

    # Extract stylometric features for train and validation
    train_stylometric_features = extract_stylometric_features_batch(train_texts_fold)
    val_stylometric_features = extract_stylometric_features_batch(val_texts_fold)

    # Create fold-specific datasets and dataloaders
    train_dataset_fold = TensorDataset(
        train_encodings_fold["input_ids"],
        train_encodings_fold["attention_mask"],
        torch.tensor(train_stylometric_features, dtype=torch.float32),
        torch.tensor(train_labels_fold, dtype=torch.long),
    )
    val_dataset_fold = TensorDataset(
        val_encodings_fold["input_ids"],
        val_encodings_fold["attention_mask"],
        torch.tensor(val_stylometric_features, dtype=torch.float32),
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
    model = DistilBERTBiLSTM(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        stylometric_dim=STYLOMETRIC_FEATURE_DIM,
        hidden_size=256,
    )
    model.to(device)

    # Loss function - Focal Loss with gamma=2.0
    loss_fn = FocalLoss(gamma=2.0, alpha=None, reduction='mean')

    # Optimizer, scheduler for this fold
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
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
            input_ids, attention_mask, stylometric_features, labels = [b.to(device) for b in batch]

            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, stylometric_features=stylometric_features)
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
                input_ids, attention_mask, stylometric_features, labels = [b.to(device) for b in batch]
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, stylometric_features=stylometric_features)
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
            input_ids, attention_mask, stylometric_features = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, stylometric_features=stylometric_features)
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
submission_df.to_csv("./submission/submission_5656e1370ea74295a73accb3afc83502.csv", index=False)
print(f"Submission saved to ./submission/submission_5656e1370ea74295a73accb3afc83502.csv")
print(f"Submission shape: {submission_df.shape}")

print(f"Final ensemble prediction shape: {test_preds.shape}")