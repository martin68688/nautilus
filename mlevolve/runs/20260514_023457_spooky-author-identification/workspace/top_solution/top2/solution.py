import pandas as pd
import numpy as np
import re
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from accelerate import Accelerator

print("Starting data processing and feature engineering...")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
author_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
num_classes = len(le.classes_)
print(f"Author mapping: {author_mapping}")

# --- Tokenization with DistilRoBERTa ---
print("Loading tokenizer and tokenizing text...")
MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_texts(texts, tokenizer, max_length=256):
    return tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

train_encodings = tokenize_texts(train_df["text"], tokenizer)
test_encodings = tokenize_texts(test_df["text"], tokenizer)

# Save tokenized data for later use (avoiding .npy issues with object arrays)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

torch.save(train_encodings["input_ids"], "./working/train_input_ids.pt")
torch.save(train_encodings["attention_mask"], "./working/train_attention_mask.pt")
torch.save(torch.tensor(train_df["author_encoded"].values, dtype=torch.long), "./working/train_labels.pt")
torch.save(test_encodings["input_ids"], "./working/test_input_ids.pt")
torch.save(test_encodings["attention_mask"], "./working/test_attention_mask.pt")

# Save test_ids as a list (not numpy array to avoid object loading issues)
with open("./working/test_ids.pkl", "wb") as f:
    pickle.dump(test_df["id"].values.tolist(), f)

# Save metadata
metadata = {
    "author_mapping": author_mapping,
    "classes": list(le.classes_),
    "num_classes": num_classes,
}
with open("./working/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Save raw text for potential debugging
train_df.to_pickle("./working/train_raw.pkl")
test_df.to_pickle("./working/test_raw.pkl")

# Save StratifiedKFold splits (applied to indices of training data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_splits = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_df)), train_df["author_encoded"])):
    fold_splits.append({"train": train_idx, "val": val_idx})
    print(f"Fold {fold_idx}: Train {len(train_idx)}, Val {len(val_idx)}")

with open("./working/fold_splits.pkl", "wb") as f:
    pickle.dump(fold_splits, f)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print("Data processing and feature engineering complete!")

# --- Model Design ---
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

def create_model(num_classes, dropout=0.3):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    # Freeze pretrained layers initially (will unfreeze during fine-tuning)
    for param in model.roberta.parameters():
        param.requires_grad = False
    return model

# --- Training & Evaluation ---
print("\nStarting training and evaluation...")

# Load tokenized data
train_input_ids = torch.load("./working/train_input_ids.pt")
train_attention_mask = torch.load("./working/train_attention_mask.pt")
train_labels = torch.load("./working/train_labels.pt")
test_input_ids = torch.load("./working/test_input_ids.pt")
test_attention_mask = torch.load("./working/test_attention_mask.pt")

with open("./working/test_ids.pkl", "rb") as f:
    test_ids = pickle.load(f)
with open("./working/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("./working/fold_splits.pkl", "rb") as f:
    fold_splits = pickle.load(f)

num_classes = metadata["num_classes"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training hyperparameters
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1

# 5-fold cross-validation
all_test_preds = np.zeros((len(test_ids), num_classes))

for fold_idx, fold in enumerate(fold_splits):
    print(f"\nFold {fold_idx + 1}/5")

    train_idx = fold["train"]
    val_idx = fold["val"]

    # Create datasets
    train_dataset = TextDataset(
        train_input_ids[train_idx],
        train_attention_mask[train_idx],
        train_labels[train_idx],
    )
    val_dataset = TextDataset(
        train_input_ids[val_idx],
        train_attention_mask[val_idx],
        train_labels[val_idx],
    )
    test_dataset = TextDataset(test_input_ids, test_attention_mask)

    # Create model for this fold
    model = create_model(num_classes)
    model.to(device)

    # Unfreeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Accelerator for mixed precision and gradient accumulation
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
    )
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model,
        optimizer,
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
    )

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_all = []
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                val_preds.append(probs.cpu().numpy())
                val_labels_all.append(batch["labels"].cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels_all = np.concatenate(val_labels_all, axis=0)
        val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
        fold_score = log_loss(val_labels_all, val_preds)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss/len(train_dataloader):.4f} - Val Loss: {val_loss/len(val_dataloader):.4f} - Val Log Loss: {fold_score:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = accelerator.unwrap_model(model).state_dict()

    # Load best model for this fold
    model = create_model(num_classes)
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # Predict on test set
    test_preds = []
    with torch.no_grad():
        for batch in DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)
    all_test_preds += test_preds / 5

    print(f"Fold {fold_idx + 1} Best Val Loss: {best_val_loss:.4f}")

# Generate submission
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Final Test Predictions Ready")