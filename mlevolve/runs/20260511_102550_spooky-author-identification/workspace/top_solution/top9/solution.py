import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
import lightgbm as lgb
import joblib
import os
import gc
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# ============================================
# 1. Data Loading
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Clean text
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# ============================================
# No engineered features - using only raw text for transformer
# ============================================


# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])

# Create stratified split first
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["author_encoded"])
):
    if fold == 0:
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        train_fold_indices = train_idx
        val_fold_indices = val_idx
        break

# ============================================
# Save processed data without engineered features
# ============================================
os.makedirs("./working", exist_ok=True)
train_df[["id", "author", "author_encoded"]].to_pickle("./working/train_labels.pkl")
test_df[["id"]].to_pickle("./working/test_ids.pkl")
train_fold.to_pickle("./working/train_fold.pkl")
val_fold.to_pickle("./working/val_fold.pkl")
joblib.dump(le, "./working/label_encoder.pkl")


# ============================================
# Model Design
# ============================================
class AuthorshipDataset(Dataset):
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


MODEL_NAME = "microsoft/deberta-v3-large"
NUM_AUTHORS = 3
MAX_LENGTH = 512
BATCH_SIZE = 8
NUM_WORKERS = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_indices = train_fold.index.tolist()
val_indices = val_fold.index.tolist()

train_texts = train_fold["clean_text"].tolist()
train_labels_list = train_fold["author_encoded"].tolist()
val_texts = val_fold["clean_text"].tolist()
val_labels_list = val_fold["author_encoded"].tolist()
test_texts = test_df["clean_text"].tolist()

train_dataset = AuthorshipDataset(train_texts, train_labels_list, tokenizer, MAX_LENGTH)
val_dataset = AuthorshipDataset(val_texts, val_labels_list, tokenizer, MAX_LENGTH)
test_dataset = AuthorshipDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


class AuthorshipModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.deberta_hidden_size = self.deberta.config.hidden_size
        # Initially freeze all layers - will unfreeze progressively
        for param in self.deberta.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.deberta_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.deberta_hidden_size, self.deberta_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.deberta_hidden_size // 2, self.deberta_hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.deberta_hidden_size // 4, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


model = AuthorshipModel(
    model_name=MODEL_NAME,
    num_labels=NUM_AUTHORS,
    dropout=0.3,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Weighted loss
class_counts = train_fold["author"].value_counts()
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
label_encoder = joblib.load("./working/label_encoder.pkl")
weight_tensor = torch.tensor(
    [
        class_weights[label_encoder.transform([name])[0]]
        for name in label_encoder.classes_
    ],
    dtype=torch.float,
).to(device)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)

optimizer = AdamW(
    [
        {"params": model.classifier.parameters(), "lr": 2e-5},
        {"params": model.deberta.parameters(), "lr": 1e-5},
    ],
    weight_decay=0.01,
)

total_epochs_first_stage = 2  # classifier only
total_epochs_second_stage = 3  # unfreeze last 6 layers
total_epochs_third_stage = 3   # unfreeze last 12 layers
total_epochs_fourth_stage = 12 # fully unfreeze
total_steps = len(train_loader) * (total_epochs_first_stage + total_epochs_second_stage + total_epochs_third_stage + total_epochs_fourth_stage)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)

# ============================================
# Training and Evaluation with Progressive Unfreezing, Cosine Annealing, SWA
# ============================================
NUM_EPOCHS_FROZEN = 2  # train classifier only
NUM_EPOCHS_UNFREEZE_6 = 3  # unfreeze last 6 layers
NUM_EPOCHS_UNFREEZE_12 = 3  # unfreeze last 12 layers
NUM_EPOCHS_FULL = 12  # fully unfreeze
SWA_EPOCHS = 3  # additional epochs for SWA averaging
EARLY_STOPPING_PATIENCE = 6
GRADIENT_ACCUMULATION_STEPS = 2
USE_MIXED_PRECISION = True

scaler_gpu = GradScaler(enabled=USE_MIXED_PRECISION)
best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

def unfreeze_layers(model, num_layers_unfrozen):
    for i, param in enumerate(model.deberta.parameters()):
        if i < len(list(model.deberta.parameters())) - num_layers_unfrozen:
            param.requires_grad = False
        else:
            param.requires_grad = True

# Phase 1: Train only classifier (all DeBERTa frozen)
print("Phase 1: Training classifier only (all DeBERTa frozen)")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val LogLoss':<12} {'Best?'}")
for epoch in range(NUM_EPOCHS_FROZEN):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler_gpu.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    val_probs = np.concatenate(all_preds, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_labels_onehot = np.zeros((len(val_labels), 3))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    val_logloss = log_loss(val_labels_onehot, val_probs_clipped)
    val_loss_tensor = criterion(
        torch.tensor(val_probs_clipped).float().log().to(device),
        torch.tensor(val_labels).long().to(device),
    ).item()
    is_best = val_logloss < best_val_loss
    if is_best:
        best_val_loss = val_logloss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    print(f"{epoch+1:<8} {train_loss:<12.6f} {val_loss_tensor:<12.6f} {val_logloss:<12.6f} {'*' if is_best else ''}")
    scheduler.step()

# Phase 2: Unfreeze last 6 layers
print("\nPhase 2: Unfreezing last 6 layers")
unfreeze_layers(model, 6)
for epoch in range(NUM_EPOCHS_UNFREEZE_6):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler_gpu.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
    train_loss = total_loss / len(train_loader)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    val_probs = np.concatenate(all_preds, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_labels_onehot = np.zeros((len(val_labels), 3))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    val_logloss = log_loss(val_labels_onehot, val_probs_clipped)
    val_loss_tensor = criterion(
        torch.tensor(val_probs_clipped).float().log().to(device),
        torch.tensor(val_labels).long().to(device),
    ).item()
    is_best = val_logloss < best_val_loss
    if is_best:
        best_val_loss = val_logloss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    print(f"{epoch+1+NUM_EPOCHS_FROZEN:<8} {train_loss:<12.6f} {val_loss_tensor:<12.6f} {val_logloss:<12.6f} {'*' if is_best else ''}")
    scheduler.step()

# Phase 3: Unfreeze last 12 layers
print("\nPhase 3: Unfreezing last 12 layers")
unfreeze_layers(model, 12)
for epoch in range(NUM_EPOCHS_UNFREEZE_12):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler_gpu.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
    train_loss = total_loss / len(train_loader)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    val_probs = np.concatenate(all_preds, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_labels_onehot = np.zeros((len(val_labels), 3))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    val_logloss = log_loss(val_labels_onehot, val_probs_clipped)
    val_loss_tensor = criterion(
        torch.tensor(val_probs_clipped).float().log().to(device),
        torch.tensor(val_labels).long().to(device),
    ).item()
    is_best = val_logloss < best_val_loss
    if is_best:
        best_val_loss = val_logloss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    print(f"{epoch+1+NUM_EPOCHS_FROZEN+NUM_EPOCHS_UNFREEZE_6:<8} {train_loss:<12.6f} {val_loss_tensor:<12.6f} {val_logloss:<12.6f} {'*' if is_best else ''}")
    scheduler.step()

# Phase 4: Fully unfreeze
print("\nPhase 4: Fully unfreezing DeBERTa")
unfreeze_layers(model, 24)  # all layers
for epoch in range(NUM_EPOCHS_FULL):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler_gpu.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
    train_loss = total_loss / len(train_loader)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    val_probs = np.concatenate(all_preds, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_labels_onehot = np.zeros((len(val_labels), 3))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    val_logloss = log_loss(val_labels_onehot, val_probs_clipped)
    val_loss_tensor = criterion(
        torch.tensor(val_probs_clipped).float().log().to(device),
        torch.tensor(val_labels).long().to(device),
    ).item()
    is_best = val_logloss < best_val_loss
    if is_best:
        best_val_loss = val_logloss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    print(f"{epoch+1+NUM_EPOCHS_FROZEN+NUM_EPOCHS_UNFREEZE_6+NUM_EPOCHS_UNFREEZE_12:<8} {train_loss:<12.6f} {val_loss_tensor:<12.6f} {val_logloss:<12.6f} {'*' if is_best else ''}")
    scheduler.step()
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs in phase 4")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation log loss: {best_val_loss:.6f}")

# SWA: Additional training at constant high LR with weight averaging
print("\nSWA Phase: Training at constant high LR for 3 epochs with weight averaging")
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
for epoch in range(SWA_EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler_gpu.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        swa_model.update_parameters(model)
    train_loss = total_loss / len(train_loader)
    swa_scheduler.step()
    print(f"SWA Epoch {epoch+1}/{SWA_EPOCHS}, Train Loss: {train_loss:.6f}")

# Use SWA model for final predictions
model = swa_model
model.eval()

# Validation with SWA
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
val_probs = np.concatenate(all_preds, axis=0)
val_labels = np.concatenate(all_labels, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
val_labels_onehot = np.zeros((len(val_labels), 3))
val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
final_val_logloss = log_loss(val_labels_onehot, val_probs_clipped)
print(f"\nFinal Validation Log Loss (with SWA): {final_val_logloss:.6f}")

# Test inference
print("Generating test predictions...")
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission
test_ids_loaded = pd.read_pickle("./working/test_ids.pkl")
# Ensure label order matches encoder classes
label_classes = le.classes_.tolist()
submission = pd.DataFrame(
    {
        "id": test_ids_loaded["id"].values,
        label_classes[0]: test_preds[:, 0],
        label_classes[1]: test_preds[:, 1],
        label_classes[2]: test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:")
print(submission.head())

score = final_val_logloss
print(f"Final Validation Score: {score}")