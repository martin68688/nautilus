import os
import sys
import gc
import math
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)


# Fix seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SAMPLE_SUB_PATH = "./input/sample_submission.csv"
SUBMISSION_PATH = "./submission/submission.csv"

BATCH_SIZE = 32
MAX_LEN = 256
EPOCHS = 10
N_FOLDS = 5
SEED = 42
MODEL_NAME = "distilroberta-base"

set_seed(SEED)

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode authors
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_mapping)
num_classes = len(author_mapping)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Create full train dataset for training (we do cross-validation)
train_texts = train_df["text"].values
train_labels = train_df["label"].values
test_texts = test_df["text"].values

# Stratified K-Fold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_models = []
fold_val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*50}")

    # Split data
    X_train_fold = train_texts[train_idx]
    y_train_fold = train_labels[train_idx]
    X_val_fold = train_texts[val_idx]
    y_val_fold = train_labels[val_idx]

    # Create datasets
    train_dataset = SpookyDataset(X_train_fold, y_train_fold, tokenizer, MAX_LEN)
    val_dataset = SpookyDataset(X_val_fold, y_val_fold, tokenizer, MAX_LEN)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_classes)
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config
    ).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training
    scaler = GradScaler()
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches

        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                total_val_loss += loss.item()
                val_batches += 1

                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / val_batches

        val_preds = np.concatenate(all_val_preds, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)

        # Clamp and normalize
        val_preds_clamped = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds_clamped = val_preds_clamped / val_preds_clamped.sum(
            axis=1, keepdims=True
        )

        val_log_loss = -np.mean(
            np.log(val_preds_clamped[np.arange(len(val_labels)), val_labels])
        )

        print(
            f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    fold_models.append(model)
    fold_val_scores.append(best_val_loss)

    # Clear some memory
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*50}")
print(f"Cross-validation scores: {fold_val_scores}")
print(f"Mean CV score: {np.mean(fold_val_scores):.4f}")

# Create test dataset and loader
test_dataset = SpookyDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# Ensemble predictions
all_test_probs = []

for model in fold_models:
    model.eval()
    fold_test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast():
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

            probs = F.softmax(logits, dim=1).cpu().numpy()
            fold_test_probs.append(probs)

    fold_test_probs = np.concatenate(fold_test_probs, axis=0)
    all_test_probs.append(fold_test_probs)
    torch.cuda.empty_cache()

# Average predictions across folds
test_probs = np.mean(all_test_probs, axis=0)

# Clamp and normalize
test_probs_final = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# Create submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

# Compute final validation score (average across folds)
# Use the last model's validation predictions as final validation score
final_val_score = np.mean(fold_val_scores)
print(f"\nFinal Validation Score: {final_val_score}")
