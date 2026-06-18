import os
os.sched_setaffinity(0, {110})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
"""
Merged Script: Spooky Author Identification
Simplified version: pure Transformer fine-tuning with EMA, no handcrafted features.
"""

import pandas as pd
import numpy as np
import os
import warnings
import gc
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & CONFIGURATION
# ============================================================
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "./working"
SUBMISSION_PATH = "./submission/submission_80ea67b4f3a74c94b4c5477e4c4d99e0.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
PATIENCE = 3
DROPOUT = 0.1
N_FOLDS = 5
PSEUDO_LABEL_THRESHOLD = 0.95
PSEUDO_EPOCHS = 2
NUM_DROPOUT_SAMPLES = 4

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# CUSTOM DATASET
# ============================================================
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.texts[idx], self.labels[idx]
        return self.texts[idx], 0

def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return encodings["input_ids"], encodings["attention_mask"], labels

# ============================================================
# MULTI-SAMPLE DROPOUT MODEL
# ============================================================
class MultiSampleDropoutModel(nn.Module):
    def __init__(self, base_model, num_dropout_samples=NUM_DROPOUT_SAMPLES, dropout_prob=DROPOUT):
        super().__init__()
        self.base_model = base_model
        self.num_dropout_samples = num_dropout_samples
        # We will store the pooled output for multi-sample dropout in the forward pass
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get the base model's sequence output
        outputs = self.base_model.deberta.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        # Apply our own pooling: use the [CLS] token or mean pooling
        # We use mean pooling over non-padded tokens (more robust)
        expanded_mask = attention_mask.unsqueeze(-1).float()
        pooled = (sequence_output * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1)
        # Now apply multi-sample dropout before the classifier
        logits_list = []
        for _ in range(self.num_dropout_samples):
            dropped = self.dropout(pooled)
            logits = self.base_model.classifier(dropped)
            logits_list.append(logits)
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(avg_logits, labels)

        return type('Output', (object,), {'logits': avg_logits, 'loss': loss})()

# ============================================================
# EMA MODEL
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# ============================================================
# DATA LOADING & TOKENIZATION
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("\nTokenizing test texts...")
test_dataset = AuthorDataset(list(test_df["text"].values))
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
)

# ============================================================
# MODEL DEFINITION (helper function)
# ============================================================
def create_model():
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL (DeBERTa-v3-large) for a fold")
    print("=" * 60)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.hidden_dropout_prob = DROPOUT
    config.attention_probs_dropout_prob = DROPOUT
    config.num_labels = NUM_AUTHORS

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    )

    # Wrap with MultiSampleDropoutModel for multi-sample dropout
    model = MultiSampleDropoutModel(base_model, num_dropout_samples=NUM_DROPOUT_SAMPLES, dropout_prob=DROPOUT)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    return model

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

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
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

def train_fold(train_idx, val_idx, fold, pseudo_texts=None, pseudo_labels=None):
    """Train a single fold with optional pseudo-labeling."""
    X_fold_texts = train_df["text"].values[train_idx]
    y_fold_labels = y_train_full[train_idx]
    X_val_texts_fold = train_df["text"].values[val_idx]
    y_val_labels_fold = y_train_full[val_idx]

    # Optionally add pseudo-labeled data
    if pseudo_texts is not None and pseudo_labels is not None:
        X_fold_texts = np.concatenate([X_fold_texts, pseudo_texts])
        y_fold_labels = np.concatenate([y_fold_labels, pseudo_labels])
        print(f"  Fold {fold}: Added {len(pseudo_texts)} pseudo-labeled samples")

    train_dataset_fold = AuthorDataset(list(X_fold_texts), list(y_fold_labels))
    val_dataset_fold = AuthorDataset(list(X_val_texts_fold), list(y_val_labels_fold))

    train_loader_fold = DataLoader(
        train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader_fold = DataLoader(
        val_dataset_fold, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    model = create_model()
    total_steps = len(train_loader_fold) * (NUM_EPOCHS + PSEUDO_EPOCHS if pseudo_texts is None else NUM_EPOCHS)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4,
    )

    scaler = GradScaler() if torch.cuda.is_available() else None
    ema = EMA(model, decay=0.999)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader_fold:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            ema.update()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_train_loss = total_loss / num_batches
        ema.apply_shadow()
        val_loss, val_acc, _ = evaluate(model, val_loader_fold)
        ema.restore()

        print(
            f"  Fold {fold}, Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            ema.apply_shadow()
            best_state = copy.deepcopy(model.state_dict())
            ema.restore()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Fold {fold}: Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
                break

    print(f"  Fold {fold}: Best model epoch {best_epoch}, val loss: {best_val_loss:.4f}")

    # Load best EMA weights
    model.load_state_dict(best_state)
    return model

def get_test_predictions(model):
    """Generate test predictions using a trained model."""
    model.eval()
    all_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu().numpy())
    return np.vstack(all_test_probs)

def pseudo_label(model):
    """Generate pseudo-labels for test set with high confidence."""
    model.eval()
    pseudo_texts = []
    pseudo_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.vstack(all_probs)
    max_probs = np.max(all_probs, axis=1)
    confident_idx = np.where(max_probs > PSEUDO_LABEL_THRESHOLD)[0]
    pseudo_texts = test_df["text"].values[confident_idx]
    pseudo_labels = np.argmax(all_probs[confident_idx], axis=1)
    print(f"  Pseudo-labels: {len(confident_idx)} samples with confidence > {PSEUDO_LABEL_THRESHOLD}")
    return pseudo_texts, pseudo_labels

# ============================================================
# 5-FOLD STRATIFIED K-FOLD + PSEUDO-LABELING
# ============================================================
print("\n" + "=" * 60)
print("5-FOLD STRATIFIED K-FOLD TRAINING WITH PSEUDO-LABELING")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df["text"].values, y_train_full)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    # Stage 1: Train without pseudo-labels
    model = train_fold(train_idx, val_idx, fold+1)

    # Stage 2: Pseudo-labeling - generate pseudo-labels with current model
    pseudo_texts, pseudo_labels = pseudo_label(model)

    # Stage 3: Train again with pseudo-labels added for a few epochs
    if len(pseudo_texts) > 0:
        print(f"  Fine-tuning with pseudo-labels for {PSEUDO_EPOCHS} epochs...")
        # Create a new model for pseudo-label fine-tuning and load trained weights
        model_pl = create_model()
        model_pl.load_state_dict(copy.deepcopy(model.state_dict()))
        # We need to do a short training with pseudo-labels
        X_fold_texts = train_df["text"].values[train_idx]
        y_fold_labels = y_train_full[train_idx]
        X_val_texts_fold = train_df["text"].values[val_idx]
        y_val_labels_fold = y_train_full[val_idx]

        # Combine original fold data with pseudo-labels
        combined_texts = np.concatenate([X_fold_texts, pseudo_texts])
        combined_labels = np.concatenate([y_fold_labels, pseudo_labels])
        train_pl = AuthorDataset(list(combined_texts), list(combined_labels))
        val_pl = AuthorDataset(list(X_val_texts_fold), list(y_val_labels_fold))
        train_loader_pl = DataLoader(train_pl, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_loader_pl = DataLoader(val_pl, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

        total_steps_pl = len(train_loader_pl) * PSEUDO_EPOCHS
        optimizer_pl = AdamW(model_pl.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY, eps=1e-8)
        scheduler_pl = OneCycleLR(optimizer_pl, max_lr=LEARNING_RATE * 0.1, total_steps=total_steps_pl, pct_start=0.1, div_factor=25, final_div_factor=1e4)
        scaler_pl = GradScaler() if torch.cuda.is_available() else None
        ema_pl = EMA(model_pl, decay=0.999)

        for epoch in range(PSEUDO_EPOCHS):
            model_pl.train()
            total_loss = 0.0
            num_batches = 0
            for batch in train_loader_pl:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer_pl.zero_grad()
                with autocast():
                    outputs = model_pl(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                if scaler_pl is not None:
                    scaler_pl.scale(loss).backward()
                    scaler_pl.unscale_(optimizer_pl)
                    torch.nn.utils.clip_grad_norm_(model_pl.parameters(), max_norm=1.0)
                    scaler_pl.step(optimizer_pl)
                    scaler_pl.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_pl.parameters(), max_norm=1.0)
                    optimizer_pl.step()
                scheduler_pl.step()
                ema_pl.update()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            ema_pl.apply_shadow()
            val_loss_pl, val_acc_pl, _ = evaluate(model_pl, val_loader_pl)
            ema_pl.restore()
            print(f"  Pseudo Epoch {epoch+1}/{PSEUDO_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss_pl:.4f} | Val Acc: {val_acc_pl:.4f}")

        # Use pseudo-label fine-tuned model for test predictions
        ema_pl.apply_shadow()
        test_preds = get_test_predictions(model_pl)
        ema_pl.restore()
        fold_test_preds.append(test_preds)

        # Clean up
        del model_pl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Use the original model if no pseudo-labels
        test_preds = get_test_predictions(model)
        fold_test_preds.append(test_preds)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# ENSEMBLE: Average predictions from all folds
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLING: AVERAGING PREDICTIONS FROM ALL FOLDS")
print("=" * 60)
test_probs = np.mean(fold_test_preds, axis=0)
print(f"Averaged predictions shape: {test_probs.shape}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\nGenerating submission...")

eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Compute final log-loss on a held-out set for reporting (use last fold's val set)
print(f"\nFinal Ensemble Validation Score (last fold): {val_loss_pl:.6f}")