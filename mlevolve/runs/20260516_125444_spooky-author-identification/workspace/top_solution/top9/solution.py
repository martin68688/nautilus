"""
Merged script: Spooky Author Identification using Multi-Level Neural Network
Combines character CNNs, word BiLSTM, and stylometric features
"""

import pandas as pd
import numpy as np
import os
import re
import gc
import warnings
import string
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import save_npz
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
NUM_AUTHORS = 3
MAX_LENGTH = 384
DROPOUT_RATE = 0.2
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
PATIENCE = 3
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 1.0
RANDOM_STATE = 42
PRETRAINED_MODEL_NAME = "distilroberta-base"
WARMUP_EPOCHS = 2
T_0 = 8
T_MULT = 1
ETA_MIN = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# PATH CONFIGURATION
# ============================================================
DATA_DIR = "./input"
WORKING_DIR = "./working"
OUTPUT_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(train_df["author"])
author_classes = label_encoder.classes_
author_mapping = dict(zip(author_classes, range(len(author_classes))))
print(f"Author mapping: {author_mapping}")

# ============================================================
# STRATIFIED SPLIT
# ============================================================
print("Creating stratified train/validation split...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_val_losses = []
all_oof_preds = np.zeros((len(train_df), NUM_AUTHORS))
all_test_preds = np.zeros((len(test_df), NUM_AUTHORS))

# Store best model validation metrics for reporting
best_avg_val_loss = float("inf")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_full)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*60}")

    train_texts = train_df["text"].values[train_idx]
    train_labels = y_full[train_idx]
    val_texts = train_df["text"].values[val_idx]
    val_labels = y_full[val_idx]
    test_texts = test_df["text"].values

    assert (
        len(set(train_idx) & set(val_idx)) == 0
    ), f"INDEX BUG: Train and validation overlap in fold {fold+1}!"
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# ============================================================
# TOKENIZATION WITH DISTILROBERTA TOKENIZER
# ============================================================
print("Loading DistilRoBERTa tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def tokenize_texts(texts, max_length=MAX_LENGTH):
    encodings = tokenizer(
        texts.tolist() if isinstance(texts, np.ndarray) else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encodings["input_ids"], encodings["attention_mask"]

print("Tokenizing texts...")
train_input_ids, train_attention_mask = tokenize_texts(train_texts)
val_input_ids, val_attention_mask = tokenize_texts(val_texts)
test_input_ids, test_attention_mask = tokenize_texts(test_texts)

# ============================================================
# MODEL ARCHITECTURE (DistilRoBERTa with Custom Deep Head)
# ============================================================
class DistilRoBERTaCustom(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_rate=0.2):
        super().__init__()
        # Use AutoModel (no classification head) to get hidden states
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.hidden_size = self.backbone.config.hidden_size
        self.num_labels = num_labels

        # Weights for the last 4 hidden states (layers 7-10)
        self.layer_weights = nn.Parameter(torch.ones(4) / 4.0)

        # Deeper classification head with LayerNorm
        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple of 13 tensors (embedding + 12 layers)

        # Take last 4 hidden states (layers 7-10 in 0-indexed, plus embedding layer is index 0)
        # DistilRoBERTa has 6 layers, so last 4 layers are layers 3-6 (indices 4-7 in tuple)
        # Actually DistilRoBERTa has 6 hidden layers, so indices: 0=embed, 1..6=hidden layers
        # Last 4 hidden layers: indices 3,4,5,6
        last_four = [hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]]

        # Apply learned weights to the last 4 hidden states
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_hidden = sum(w * h for w, h in zip(weights, last_four))

        # Mean pooling over sequence dimension, masked
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_hidden.size()).float()
        sum_embeddings = torch.sum(weighted_hidden * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Deeper classification head
        x = self.pre_classifier(pooled_output)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.classifier(x)
        x = self.dropout2(x)  # Additional dropout before output (helps regularization)

        return x

print(f"Loading pre-trained DistilRoBERTa model with custom deep head: {PRETRAINED_MODEL_NAME}")
model = DistilRoBERTaCustom(
    pretrained_model_name=PRETRAINED_MODEL_NAME,
    num_labels=NUM_AUTHORS,
    dropout_rate=DROPOUT_RATE,
)
model.to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# LOSS FUNCTION: CrossEntropyLoss with Label Smoothing
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
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
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

# ============================================================
# CREATE DATALOADERS
# ============================================================
train_dataset = TensorDataset(
    train_input_ids,
    train_attention_mask,
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_input_ids,
    val_attention_mask,
    torch.tensor(val_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MULTI-LEVEL CLASSIFIER")
print("=" * 60)

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
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

scaler = GradScaler()
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

# Cosine annealing with warm restarts after linear warmup
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = WARMUP_EPOCHS * len(train_loader)

# Cosine annealing with warm restarts (no separate warmup needed)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN
)
warmup_scheduler = None  # Not used

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        # Cosine annealing with warm restarts - step each batch
        # To keep it simple, step after each epoch (or use more complex stepping)
        # We'll step after each batch to stay close to original
        cosine_scheduler.step(epoch + num_batches / len(train_loader))
        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, _ = evaluate(model, val_loader)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_model.pt")
        print(f"  -> New best model saved (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(
    f"\nTraining complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
)

# ============================================================
# LOAD BEST MODEL AND COMPUTE FINAL METRICS
# ============================================================
print("Loading best model...")
model.load_state_dict(torch.load(f"{WORKING_DIR}/best_model.pt", map_location=device))
model.eval()

val_logloss, val_accuracy, val_probs = evaluate(model, val_loader)
print(f"Final Validation - Log Loss: {val_logloss:.6f}, Accuracy: {val_accuracy:.4f}")

# ============================================================
# TEST INFERENCE
# ============================================================
print("Performing test inference...")
test_dataset = TensorDataset(
    test_input_ids,
    test_attention_mask,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
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
print("Generating submission file...")
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

submission_df.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
print(f"Submission saved to {OUTPUT_DIR}/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# CLEANUP
# ============================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {val_logloss:.6f}")