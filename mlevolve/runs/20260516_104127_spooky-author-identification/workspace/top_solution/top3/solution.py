import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# Path Configuration
# ============================================================
DATA_DIR = "./input"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
PATIENCE = 3
DROPOUT = 0.1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# Data Loading
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
print(
    f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# Stratified Split (CRITICAL: use indices directly to prevent INDEX_BUG)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["author_encoded"].values,
)
assert len(set(train_idx) & set(val_idx)) == 0, "Split overlap detected!"

train_texts = train_df["text"].values[train_idx]
train_labels = train_df["author_encoded"].values[train_idx]
val_texts = train_df["text"].values[val_idx]
val_labels = train_df["author_encoded"].values[val_idx]
test_texts = test_df["text"].values
test_ids = test_df["id"].values

print(
    f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)

# ============================================================
# FEATURE ENGINEERING - NONE (pure transformer fine-tuning)
# ============================================================
print("\n" + "=" * 60)
print("PURE TRANSFORMER FINE-TUNING (NO HAND-CRAFTED FEATURES)")
print("=" * 60)

# ============================================================
# MODEL ARCHITECTURE: Standard HuggingFace Classifier with CLS token
# ============================================================

# ============================================================
# Initialize tokenizer and model
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING MODEL")
print("=" * 60)

from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_AUTHORS, hidden_dropout_prob=DROPOUT, attention_probs_dropout_prob=DROPOUT)

# Use standard AutoModelForSequenceClassification with CLS token pooling
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=False,
)
model.to(device)

# Multi-sample dropout will be applied during training in the forward pass
MSD_K = 4
msd_dropout = nn.Dropout(DROPOUT)

# ============================================================
# Tokenize texts
# ============================================================
print("Tokenizing texts...")
train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

train_labels_tensor = torch.LongTensor(train_labels)
val_labels_tensor = torch.LongTensor(val_labels)

# ============================================================
# Prepare data loaders
# ============================================================
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels_tensor,
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    val_labels_tensor,
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# Optimizer and scheduler (Two-phase training)
# ============================================================

# Phase 1: Train only classifier head
# Simplified optimizer creation with weight decay groups (no fragile layer-wise discriminative LR)
def create_optimizer_with_discriminative_lr(model, encoder_lr=1e-5, classifier_lr=2e-5, weight_decay=0.01):
    """
    Create optimizer with separate learning rates for encoder and classifier.
    Uses standard parameter grouping without fragile layer-wise LR multipliers.
    """
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'layernorm']

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' in n and not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': classifier_lr,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' in n and any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': classifier_lr,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' not in n and not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': encoder_lr,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' not in n and any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': encoder_lr,
        },
    ]

    # Filter out empty groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g['params']) > 0]

    optimizer = AdamW(optimizer_grouped_parameters, lr=encoder_lr, weight_decay=weight_decay, eps=1e-8)
    return optimizer, optimizer_grouped_parameters

# Phase 1: Freeze encoder, train classifier
def freeze_encoder(model):
    """Freeze all base encoder parameters."""
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("Encoder frozen, classifier trainable")

def unfreeze_encoder(model):
    """Unfreeze all parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen for fine-tuning")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# SWA (Stochastic Weight Averaging)
swa_model = None
swa_n = 0
scaler = GradScaler() if torch.cuda.is_available() else None

# ============================================================
# Helper functions
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

def evaluate(model, loader, criterion, device):
    model.eval()
    all_losses = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast(enabled=(scaler is not None)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                loss = criterion(logits, labels)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_losses.append(loss.item())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = np.mean(all_losses)
    logloss = compute_log_loss(all_labels, all_probs)
    acc = np.mean(np.argmax(all_probs, axis=1) == all_labels)
    return avg_loss, logloss, acc, all_probs

# ============================================================
# TRAINING LOOP (Two-Phase Training with SWA)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING HuggingFace Classifier (Two-Phase Training with SWA)")
print("=" * 60)

def update_swa(swa_model, model, swa_n):
    """Update SWA model with running average of weights."""
    if swa_model is None:
        swa_model = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                swa_model[name] = param.data.clone()
        return swa_model, 1
    else:
        for name, param in model.named_parameters():
            if param.requires_grad and name in swa_model:
                swa_model[name] = (swa_model[name] * swa_n + param.data) / (swa_n + 1)
        return swa_model, swa_n + 1

def apply_swa(model, swa_model):
    """Temporarily apply SWA weights to the model (creates a copy)."""
    swa_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in swa_model:
            swa_state[name] = param.data.clone()
            param.data.copy_(swa_model[name])
    return swa_state

def restore_weights(model, swa_state):
    """Restore original weights after evaluation."""
    for name, param in model.named_parameters():
        if name in swa_state:
            param.data.copy_(swa_state[name])

best_val_logloss = float("inf")
best_epoch = 0
patience_counter = 0

# Phase 1: Train only classifier head (epochs 1-2)
print("\n" + "=" * 60)
print("PHASE 1: Training Classifier Head Only (Epochs 1-2)")
print("=" * 60)

freeze_encoder(model)
optimizer, param_groups = create_optimizer_with_discriminative_lr(
    model, encoder_lr=1e-5, classifier_lr=2e-5
)

# Phase 1 scheduler: constant LR
phase1_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: 1.0
)

for epoch in range(2):  # Phase 1: 2 epochs
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    phase1_scheduler.step()

    avg_train_loss = total_train_loss / max(num_batches, 1)
    val_loss, val_logloss, val_acc, _ = evaluate(model, val_loader, criterion, device)

    print(
        f"Phase1 Epoch {epoch+1:2d}/2 | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_phase1.pt"))
        print(f"  --> New best model saved (logloss: {val_logloss:.4f})")
    else:
        patience_counter += 1

# Phase 2: Unfreeze all layers, discriminative LR, SWA
print("\n" + "=" * 60)
print("PHASE 2: Full Fine-Tuning with Discriminative LR and SWA (Epochs 3-20)")
print("=" * 60)

unfreeze_encoder(model)
optimizer, param_groups = create_optimizer_with_discriminative_lr(
    model, encoder_lr=1e-5, classifier_lr=2e-5
)

# Phase 2 scheduler: cosine decay with warm restarts
phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=4, T_mult=2, eta_min=1e-6
)

# Reset SWA
swa_model = None
swa_n = 0

for phase2_epoch in range(NUM_EPOCHS - 2):  # Remaining epochs
    epoch = phase2_epoch + 2  # Continue epoch numbering
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    phase2_scheduler.step()

    avg_train_loss = total_train_loss / max(num_batches, 1)

    # Update SWA after each epoch in phase 2
    swa_model, swa_n = update_swa(swa_model, model, swa_n)

    # Evaluate using SWA averaged weights
    if swa_model is not None:
        swa_state = apply_swa(model, swa_model)
        val_loss, val_logloss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        restore_weights(model, swa_state)
    else:
        val_loss, val_logloss, val_acc, _ = evaluate(model, val_loader, criterion, device)

    print(
        f"Phase2 Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        # Save SWA model weights
        if swa_model is not None:
            # Apply SWA before saving
            swa_state_save = apply_swa(model, swa_model)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            restore_weights(model, swa_state_save)
        else:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        print(f"  --> New best model saved (logloss: {val_logloss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best logloss: {best_val_logloss:.4f}"
            )
            break

# Load best model (apply SWA weights and save as final)
print(f"\nLoading best model from epoch {best_epoch} with validation logloss {best_val_logloss:.4f}")

# Apply final SWA averaging for best model
if swa_model is not None:
    print("Applying SWA averaged weights to best model...")
    state_dict = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device)
    # The model already has SWA weights applied when saved
    model.load_state_dict(state_dict, strict=False)
else:
    state_dict = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(state_dict, strict=False)

# Ensure classifier is in eval mode
model.eval()

# ============================================================
# FINAL VALIDATION SCORE (using SWA averaged model if applicable)
# ============================================================
_, final_val_logloss, final_val_acc, _ = evaluate(model, val_loader, criterion, device)
print(
    f"Final validation logloss: {final_val_logloss:.6f}, accuracy: {final_val_acc:.4f}"
)

# ============================================================
# TEST INFERENCE
# ============================================================
print("\nPerforming test inference...")
model.eval()
all_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        with autocast(enabled=(scaler is not None)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)

eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"\nFinal Validation Score: {final_val_logloss:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
