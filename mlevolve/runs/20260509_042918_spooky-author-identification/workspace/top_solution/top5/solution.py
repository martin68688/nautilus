import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc
import os
import re
import math
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import warnings

warnings.filterwarnings("ignore")

# Load data
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

X_train_raw = train_df["text"].values
y_train_raw = train_df["author"].values
X_test_raw = test_df["text"].values
test_ids = test_df["id"].values

# Encode labels
label_enc = LabelEncoder()
y_train_encoded = label_enc.fit_transform(y_train_raw)
num_classes = len(label_enc.classes_)

print(f"Total train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
print(f"Classes: {label_enc.classes_} -> {num_classes}")

# =============================================
# Helper: Text normalization
# =============================================
def normalize_text(text):
    text = str(text).lower().strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text


# =============================================
# Dataset class for fine-tuning
# =============================================
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = normalize_text(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =============================================
# Fine-tuning with DeBERTa-v3-base (Single 90/10 split)
# =============================================
print("Loading DeBERTa-v3-base tokenizer and model...")
MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 16  # adjust based on GPU memory
GRAD_ACCUM_STEPS = 1  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 10
PATIENCE = 3
CLIP_GRAD_NORM = 0.5
LABEL_SMOOTHING = 0.1

print("Using single 90/10 stratified split (random_state=42)")

from sklearn.model_selection import train_test_split

# Single stratified split: 90% train, 10% validation
train_idx, val_idx = train_test_split(
    np.arange(len(X_train_raw)),
    test_size=0.1,
    stratify=y_train_encoded,
    random_state=42,
)

train_texts = X_train_raw[train_idx]
val_texts = X_train_raw[val_idx]
train_labels = y_train_encoded[train_idx]
val_labels = y_train_encoded[val_idx]

# Normalize texts
train_texts = [normalize_text(t) for t in train_texts]
val_texts = [normalize_text(t) for t in val_texts]
test_texts_norm = [normalize_text(t) for t in X_test_raw]

# Create datasets
train_dataset = AuthorDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = AuthorDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = AuthorDataset(test_texts_norm, None, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
model = model.cuda()

# Optimizer with weight decay (no decay on bias and layer norms)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': WEIGHT_DECAY,
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

# Scheduler: linear warmup + cosine decay
num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Loss function with label smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Mixed precision
scaler = GradScaler()

# Training loop
best_val_loss = float('inf')
best_val_logloss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    total_train_loss = 0.0
    train_steps = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        optimizer.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()
        train_steps += 1

    avg_train_loss = total_train_loss / max(train_steps, 1)

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            total_val_loss += loss.item()
            val_steps += 1

            probs = torch.softmax(logits, dim=-1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / max(val_steps, 1)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    val_logloss = log_loss(all_val_labels, all_val_preds)

    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f}")

    # Early stopping
    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
        print(f"  * New best model (logloss: {best_val_logloss:.4f})")
    else:
        patience_counter += 1
        print(f"  Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("  Early stopping triggered!")
            break

# Restore best model
model.load_state_dict(best_model_state)

# Predict on test set
model.eval()
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_test_preds.append(probs.cpu().numpy())
test_probs = np.concatenate(all_test_preds, axis=0)

# Free memory
del model, optimizer, scheduler, scaler, train_dataset, val_dataset, test_dataset
gc.collect()
torch.cuda.empty_cache()

print(f"Best validation logloss: {best_val_logloss:.4f}")

# Clip probabilities to avoid log(0)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)

# Normalize row-wise
row_sums = test_probs.sum(axis=1)
test_probs = test_probs / row_sums[:, np.newaxis]

# Create submission
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print("Done!")