import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
import os
import pickle
import re

# ================================================================
# DATA LOADING AND SPLIT
# ================================================================

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

os.makedirs("./working", exist_ok=True)

# Encode target
le = LabelEncoder()
le.fit(train_df["author"])
train_df["author_encoded"] = le.transform(train_df["author"])

# Create stratified train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df["text"], train_df["author_encoded"]))

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print(
    f"Train size: {len(train_split)}, Val size: {len(val_split)}, Test size: {len(test_df)}"
)

# Save indices and splits
np.save("./working/train_indices.npy", train_idx)
np.save("./working/val_indices.npy", val_idx)
train_split.to_csv("./working/train_preprocessed.csv", index=False)
val_split.to_csv("./working/val_preprocessed.csv", index=False)

# ================================================================
# MODEL DESIGN - ModernBERT
# ================================================================

MODEL_ID = "answerdotai/ModernBERT-large"
NUM_LABELS = 3
MAX_LENGTH = 512

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading ModernBERT-large for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)

model.config.hidden_dropout_prob = 0.3
model.config.attention_probs_dropout_prob = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Freeze all backbone parameters initially — only classifier trainable
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
weight_decay = 0.01
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay) and "classifier" not in n
        ],
        "weight_decay": weight_decay,
        "lr": 2e-5,
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay) and "classifier" not in n
        ],
        "weight_decay": 0.0,
        "lr": 2e-5,
    },
    {
        "params": [p for n, p in model.named_parameters() if "classifier" in n],
        "weight_decay": weight_decay,
        "lr": 2e-5 * 5,
    },
]

learning_rate = 2e-5
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# ================================================================
# DATASET CLASS
# ================================================================


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


# ================================================================
# CREATE DATALOADERS
# ================================================================

train_labels = train_split["author_encoded"].values
val_labels = val_split["author_encoded"].values

train_dataset = TextDataset(
    train_split["text"].values, train_labels, tokenizer, MAX_LENGTH
)
val_dataset = TextDataset(val_split["text"].values, val_labels, tokenizer, MAX_LENGTH)
test_dataset = TextDataset(
    test_df["text"].values, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)

batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ================================================================
# TRAINING SETUP
# ================================================================

num_epochs = 5
gradient_accumulation_steps = 2
total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
warmup_steps = int(total_steps * 0.1)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
best_val_loss = float("inf")
best_model_path = "./working/best_model.pt"

print(f"Starting training for {num_epochs} epochs...")

# ================================================================
# TRAINING LOOP
# ================================================================

for epoch in range(num_epochs):
    # Progressive unfreezing: unfreeze last k layers at each epoch
    if epoch == 1:
        # Unfreeze last 8 transformer layers
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            elif "transformer.layer" in name:
                # Extract layer number
                parts = name.split(".layer.")
                if len(parts) > 1:
                    layer_num_str = parts[1].split(".")[0]
                    if layer_num_str.isdigit():
                        if int(layer_num_str) >= model.config.num_hidden_layers - 8:
                            param.requires_grad = True
            # Always keep embeddings frozen to avoid catastrophic overfitting
            if "embedding" in name or "embeddings" in name:
                param.requires_grad = False

    if epoch == 3:
        # Unfreeze all remaining layers except embeddings
        for name, param in model.named_parameters():
            if "embedding" not in name and "embeddings" not in name:
                param.requires_grad = True

    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Gradient centralization: subtract row-wise mean from gradients of weight tensors
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name and param.dim() >= 2:
                    param.grad.data = param.grad.data - param.grad.data.mean(dim=tuple(range(1, param.dim())), keepdim=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            total_val_loss += loss.item()
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert labels to one-hot for log loss computation
    all_labels_onehot = np.zeros((len(all_labels), NUM_LABELS))
    all_labels_onehot[np.arange(len(all_labels)), all_labels] = 1

    eps_clip = 1e-15
    all_preds = np.clip(all_preds, eps_clip, 1 - eps_clip)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_loss_val = -np.mean(np.sum(all_labels_onehot * np.log(all_preds), axis=1))
    if np.isnan(log_loss_val) or np.isinf(log_loss_val):
        log_loss_val = 10.0  # Placeholder to avoid breaking, will not be used for final

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {log_loss_val:.4f} [SAVED]"
        )
    else:
        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {log_loss_val:.4f}"
        )

# ================================================================
# FINAL EVALUATION ON BEST MODEL
# ================================================================

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds_val = []
all_labels_val = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        all_preds_val.append(probs.cpu().numpy())
        all_labels_val.append(labels.cpu().numpy())

all_preds_val = np.concatenate(all_preds_val, axis=0)
all_labels_val = np.concatenate(all_labels_val, axis=0)
all_labels_onehot = np.zeros((len(all_labels_val), NUM_LABELS))
all_labels_onehot[np.arange(len(all_labels_val)), all_labels_val] = 1
eps_clip = 1e-15
all_preds_val = np.clip(all_preds_val, eps_clip, 1 - eps_clip)
val_log_loss = -np.mean(np.sum(all_labels_onehot * np.log(all_preds_val), axis=1))

# ================================================================
# TEST INFERENCE
# ================================================================

all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        all_test_preds.append(probs.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds, axis=0)
all_test_preds = np.clip(all_test_preds, 1e-15, 1 - 1e-15)
all_test_preds = all_test_preds / all_test_preds.sum(axis=1, keepdims=True)

# ================================================================
# SUBMISSION
# ================================================================

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {val_log_loss}")