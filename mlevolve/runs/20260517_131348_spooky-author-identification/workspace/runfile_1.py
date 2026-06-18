import os
os.sched_setaffinity(0, {32, 33, 34})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler
import os
import re
import warnings
import joblib
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# Create stratified split - CORRECT approach (no INDEX_BUG)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(skf.split(train_df, train_df["author_encoded"]))

# Direct numpy indexing to avoid INDEX_BUG
train_texts_orig = train_df["clean_text"].values
train_labels_orig = train_df["author_encoded"].values

train_texts = train_texts_orig[train_idx]
train_labels = train_labels_orig[train_idx]
val_texts = train_texts_orig[val_idx]
val_labels = train_labels_orig[val_idx]

# Verify no overlap
assert len(set(train_idx) & set(val_idx)) == 0, "Data leakage detected!"
print(
    f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_df)}"
)

test_texts = test_df["clean_text"].values
test_ids = test_df["id"].values


# ============================================================
# DATASET CLASS
# ============================================================
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# DATALOADERS
# ============================================================
train_dataset = AuthorDataset(train_texts, train_labels)
val_dataset = AuthorDataset(val_texts, val_labels)
test_dataset = AuthorDataset(test_texts)

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
# MODEL, OPTIMIZER, SCHEDULER, SCALER
# ============================================================
model = ModernBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)
model.to(DEVICE)
model.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)

total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

scaler = GradScaler()

# ============================================================
# TRAINING LOOP
# ============================================================
best_val_score = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                all_preds.append(probs.cpu().numpy())
                all_true.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    all_preds_clipped = np.clip(all_preds, 1e-15, 1 - 1e-15)
    all_preds_clipped = all_preds_clipped / all_preds_clipped.sum(axis=1, keepdims=True)
    val_score = log_loss(all_true, all_preds_clipped)

    print(
        f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_score:.4f}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model_7319a34e40b64b6f8b2d9af51c77e083.pth")
        print(f"  -> New best model saved (Val LogLoss: {val_score:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  -> Early stopping triggered after epoch {epoch+1}")
            break

# ============================================================
# LOAD BEST MODEL AND FINAL VALIDATION
# ============================================================
model.load_state_dict(torch.load("./working/best_model_7319a34e40b64b6f8b2d9af51c77e083.pth", map_location=DEVICE))
model.eval()

all_val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_val_preds.append(probs.cpu().numpy())

all_val_preds = np.concatenate(all_val_preds, axis=0)
all_val_preds_clipped = np.clip(all_val_preds, 1e-15, 1 - 1e-15)
all_val_preds_clipped = all_val_preds_clipped / all_val_preds_clipped.sum(
    axis=1, keepdims=True
)
final_val_score = log_loss(val_labels, all_val_preds_clipped)

# ============================================================
# TEST INFERENCE AND SUBMISSION
# ============================================================
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_test_preds.append(probs.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds, axis=0)
all_test_preds = np.clip(all_test_preds, 1e-15, 1 - 1e-15)
all_test_preds = all_test_preds / all_test_preds.sum(axis=1, keepdims=True)

# Create submission with exact column names and order from sample_submission.csv
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)
submission_df = submission_df[["id", "EAP", "HPL", "MWS"]]
submission_df.to_csv("./submission/submission_7319a34e40b64b6f8b2d9af51c77e083.csv", index=False)

print(f"Submission saved to ./submission/submission_7319a34e40b64b6f8b2d9af51c77e083.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_score}")
