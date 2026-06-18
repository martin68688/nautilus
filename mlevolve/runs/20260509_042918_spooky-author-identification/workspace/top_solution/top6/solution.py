import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import os
import gc
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
NUM_EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 256
NUM_LABELS = 3
EARLY_STOPPING_PATIENCE = 3
N_SPLITS = 5
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Load data
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# ============================================================
# Label encoding
# ============================================================
author_to_label = {"EAP": 0, "HPL": 1, "MWS": 2}
label_to_author = {v: k for k, v in author_to_label.items()}
train_df["label"] = train_df["author"].map(author_to_label)

# ============================================================
# Load ModernBERT tokenizer and model
# ============================================================
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# ============================================================
# Custom Dataset
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts.values if hasattr(texts, "values") else texts
        self.labels = labels
        self.is_train = labels is not None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }
        if self.is_train:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# Stratified K-Fold Cross Validation
# ============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []
test_predictions_list = []

# ============================================================
# Training Loop for each fold
# ============================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")

    train_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = SpookyDataset(train_fold["text"], train_fold["label"].values)
    val_dataset = SpookyDataset(val_fold["text"], val_fold["label"].values)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    model = ModernBertForSequenceClassification.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
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
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8
    )

    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_val_logloss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

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
                    logits = outputs.logits

                val_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)
        all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)
        val_logloss = log_loss(all_labels, all_preds)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_logloss:.4f}"
        )

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    fold_scores.append(best_val_logloss)
    print(f"Fold {fold+1} Best Validation LogLoss: {best_val_logloss:.6f}")

    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    test_dataset = SpookyDataset(test_df["text"], labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            fold_test_preds.append(probs.cpu().numpy())

    fold_test_preds = np.concatenate(fold_test_preds, axis=0)
    test_predictions_list.append(fold_test_preds)

    del (
        model,
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Average folds for final prediction
# ============================================================
final_test_preds = np.mean(test_predictions_list, axis=0)
final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)
final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

# ============================================================
# Create submission file
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_preds[:, 0],
        "HPL": final_test_preds[:, 1],
        "MWS": final_test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")

# ============================================================
# Final validation score
# ============================================================
final_score = np.mean(fold_scores)
print(f"Final Validation Score: {final_score}")
