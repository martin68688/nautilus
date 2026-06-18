import os
os.sched_setaffinity(0, {19, 58, 59, 60, 61, 62, 63})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import gc
import re
import string
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

# ============================================================
# CONFIGURATION
# ============================================================
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 2e-5
SEED = 42
NUM_FOLDS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# DATASET CLASS
# ============================================================
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
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
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


# ============================================================
# CUSTOM CLASSIFIER HEAD
# ============================================================
class CustomClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)
        self.ln = nn.LayerNorm(hidden_size // 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Label encode authors
author_labels = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_labels)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    return total_loss / len(dataloader)


# ============================================================
# VALIDATION FUNCTION
# ============================================================
def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)
    all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)

    score = log_loss(all_labels, all_preds)
    accuracy = (all_preds.argmax(axis=1) == all_labels).mean()

    return score, accuracy, all_preds


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict(model, dataloader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds


# ============================================================
# STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
fold_scores = []
test_predictions = np.zeros((len(test_df), NUM_AUTHORS))
best_val_score = float("inf")

print(f"Starting {NUM_FOLDS}-fold cross validation...")

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df["text"], train_df["label"])
):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*50}")

    train_texts = train_df["text"].iloc[train_idx].values
    train_labels = train_df["label"].iloc[train_idx].values
    val_texts = train_df["text"].iloc[val_idx].values
    val_labels = train_df["label"].iloc[val_idx].values

    train_dataset = AuthorDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = AuthorDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = AuthorDataset(test_df["text"].values, None, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15,
    )

    hidden_size = model.config.hidden_size
    model.classifier = CustomClassifier(hidden_size, NUM_AUTHORS)
    model.to(device)

    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (NUM_AUTHORS * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = FocalLoss(gamma=2.0, alpha=class_weights_tensor)

    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": LEARNING_RATE,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": LEARNING_RATE,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    best_fold_score = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_score, val_accuracy, _ = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val LogLoss: {val_score:.4f} - Val Acc: {val_accuracy:.4f}"
        )

        if val_score < best_fold_score:
            best_fold_score = val_score
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    val_score, val_accuracy, _ = validate(model, val_loader, device)
    fold_scores.append(val_score)
    print(
        f"Fold {fold + 1} Best Validation LogLoss: {val_score:.4f}, Accuracy: {val_accuracy:.4f}"
    )

    fold_test_preds = predict(model, test_loader, device)
    test_predictions += fold_test_preds

    del model, train_dataset, val_dataset, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

test_predictions /= NUM_FOLDS

test_predictions = np.clip(test_predictions, 1e-15, 1 - 1e-15)
test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)

final_val_score = np.mean(fold_scores)
print(f"\n{'='*50}")
print(f"Cross-validation results:")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"Mean CV LogLoss: {final_val_score:.4f}")
print(f"Std CV LogLoss: {np.std(fold_scores):.4f}")

# ============================================================
# CREATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_predictions[:, 0],
        "HPL": test_predictions[:, 1],
        "MWS": test_predictions[:, 2],
    }
)
submission.to_csv("./submission/submission_8e4f3fe94d5a4285ab5c39412fa70683.csv", index=False)
print(f"Submission saved to ./submission/submission_8e4f3fe94d5a4285ab5c39412fa70683.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")
