import os
os.sched_setaffinity(0, {37, 41, 42, 43, 44})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""
Merged Script: Spooky Author Identification
Combines data processing, feature engineering, bi-encoder model design, and training/evaluation.
"""

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
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import copy
import math

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & CONFIGURATION
# ============================================================
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "./working"
SUBMISSION_PATH = "./submission/submission_bb378f9f93734c49ae057de727ea8528.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
HIDDEN_SIZE = 1024
EMBEDDING_DIM = 256
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2
NUM_EXPERTS = 6
TOP_K_EXPERTS = 2
TEMPERATURE = 0.05

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# No hand-crafted features - using only transformer embeddings

# ============================================================
# MODEL DEFINITION: AutoModelForSequenceClassification with Multi-Sample Dropout
# ============================================================
from transformers import AutoModelForSequenceClassification

class DebertaMultiSampleDropout(nn.Module):
    """Multi-sample dropout wrapper for DeBERTa classifier head"""
    def __init__(self, model, num_dropouts=4):
        super().__init__()
        self.model = model
        self.num_dropouts = num_dropouts
        self.config = model.config
        self.num_labels = model.config.num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get hidden states from DeBERTa backbone
        outputs = self.model.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state

        # Pool using mean pooling (more robust than [CLS] for this task)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1)

        # Apply multi-sample dropout during training
        if self.training:
            logits_list = []
            for _ in range(self.num_dropouts):
                logits = self.model.classifier(self.model.dropout(pooled))
                logits_list.append(logits)
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            logits = self.model.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.15)
            loss = loss_fct(logits, labels)

        return type('Output', (), {'loss': loss, 'logits': logits})()
from torch.optim.lr_scheduler import OneCycleLR
import torch.cuda.amp as amp
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import copy
import math

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# TOKENIZATION
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize all data once
all_train_texts = train_df["text"].values
test_texts = test_df["text"].values

all_train_encodings = tokenizer(
    list(all_train_texts),
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

print(f"Training samples: {len(all_train_texts)}, Test samples: {len(test_df)}")

# No hand-crafted features - removed all feature extraction

# ============================================================
# TRAINING WITH STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("TRAINING WITH STRATIFIED 5-FOLD CROSS VALIDATION")
print("=" * 60)

def compute_log_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    y_true_onehot = np.eye(NUM_AUTHORS)[y_true]
    loss = -np.sum(y_true_onehot * np.log(y_pred_proba)) / n
    return loss

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_oof_preds = []
fold_test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_texts, y_train_full)):
    print(f"\n{'='*40}")
    print(f"FOLD {fold + 1}/5")
    print(f"{'='*40}")

    # Get fold data
    train_texts_fold = all_train_texts[train_idx]
    val_texts_fold = all_train_texts[val_idx]
    y_train_fold = y_train_full[train_idx]
    y_val_fold = y_train_full[val_idx]

    # Tokenize for this fold
    train_encodings_fold = tokenizer(
        list(train_texts_fold),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    val_encodings_fold = tokenizer(
        list(val_texts_fold),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Create datasets
    train_dataset_fold = TensorDataset(
        train_encodings_fold["input_ids"],
        train_encodings_fold["attention_mask"],
        torch.tensor(y_train_fold, dtype=torch.long),
    )
    val_dataset_fold = TensorDataset(
        val_encodings_fold["input_ids"],
        val_encodings_fold["attention_mask"],
        torch.tensor(y_val_fold, dtype=torch.long),
    )

    train_loader_fold = DataLoader(
        train_dataset_fold,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader_fold = DataLoader(
        val_dataset_fold,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = DebertaMultiSampleDropout(model, num_dropouts=4)
    model.to(device)

    # Optimizer with weight decay
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # OneCycleLR scheduler
    total_steps = len(train_loader_fold) * NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-5,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4,
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    gradient_accumulation_steps = 4

    # Training loop with SWA
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    swa_model = None
    swa_start_epoch = 15

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader_fold):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.cuda.amp.autocast():
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

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        val_loss, val_acc, _ = evaluate(model, val_loader_fold)
        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # SWA update
        if epoch + 1 >= swa_start_epoch:
            if swa_model is None:
                swa_model = copy.deepcopy(model.state_dict())
            else:
                # Update running average
                n_swa = epoch + 1 - swa_start_epoch + 1
                for key in swa_model.keys():
                    swa_model[key] = (swa_model[key] * (n_swa - 1) + model.state_dict()[key]) / n_swa

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
                )
                break

    # Load best model or SWA model if available
    if swa_model is not None:
        model.load_state_dict(swa_model)
        print(f"Using SWA model (started at epoch {swa_start_epoch})")
    else:
        model.load_state_dict(best_model_state)

    # Get OOF predictions
    _, _, oof_probs = evaluate(model, val_loader_fold)
    fold_oof_preds.append((oof_probs, val_idx))
    print(f"Fold {fold + 1} OOF log loss: {compute_log_loss(y_val_fold, oof_probs):.4f}")

    # Get test predictions
    model.eval()
    test_loader = DataLoader(
        TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    all_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu().numpy())
    test_probs_fold = np.vstack(all_test_probs)
    fold_test_preds.append(test_probs_fold)

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================
# CONSOLIDATE OOF AND TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("CONSOLIDATING PREDICTIONS")
print("=" * 60)

# Reconstruct OOF predictions
oof_probs_full = np.zeros((len(all_train_texts), NUM_AUTHORS))
for oof_probs, val_idx in fold_oof_preds:
    oof_probs_full[val_idx] = oof_probs

oof_logloss = compute_log_loss(y_train_full, oof_probs_full)
print(f"Overall OOF log loss: {oof_logloss:.4f}")

# Average test predictions across folds
test_probs_ensemble = np.mean(fold_test_preds, axis=0)

# Apply temperature scaling
T = 0.8
test_probs_ensemble = np.power(test_probs_ensemble, 1.0 / T)
test_probs_ensemble = test_probs_ensemble / test_probs_ensemble.sum(axis=1, keepdims=True)

eps = 1e-15
test_probs_ensemble = np.clip(test_probs_ensemble, eps, 1 - eps)
test_probs_ensemble = test_probs_ensemble / test_probs_ensemble.sum(axis=1, keepdims=True)
test_probs_ensemble = np.clip(test_probs_ensemble, eps, 1 - eps)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs_ensemble[:, 0],
        "HPL": test_probs_ensemble[:, 1],
        "MWS": test_probs_ensemble[:, 2],
    }
)

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nFinal OOF Validation Score: {oof_logloss:.6f}")
