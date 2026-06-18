import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================
class Config:
    seed = 42
    num_classes = 3
    model_name = "distilbert-base-uncased"
    max_length = 128
    batch_size = 16
    epochs = 10
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    val_split = 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(config.seed)


# ============================================================
# 1. Data Loading
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["author"])
author_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print(f"Author mapping: {author_mapping}")


# ============================================================
# 2. Tokenizer and Dataset
# ============================================================
tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
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
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 3. Helper: compute log loss
# ============================================================
def compute_log_loss(y_true, y_pred_probs, eps=1e-15):
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    y_pred_probs_normalized = y_pred_probs / row_sums
    if len(y_true.shape) == 1:
        y_true_onehot = np.zeros_like(y_pred_probs)
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
    else:
        y_true_onehot = y_true
    log_loss_val = (
        -np.sum(y_true_onehot * np.log(y_pred_probs_normalized)) / y_true.shape[0]
    )
    return log_loss_val


# ============================================================
# 4. Train/Val Split
# ============================================================
skf = StratifiedKFold(n_splits=int(1.0 / config.val_split), shuffle=True, random_state=config.seed)
train_idx, val_idx = next(skf.split(train_df["text"], y_train))

train_texts = train_df["text"].iloc[train_idx].values
val_texts = train_df["text"].iloc[val_idx].values
y_train_split = y_train[train_idx]
y_val_split = y_train[val_idx]

print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

train_dataset = TextDataset(train_texts, y_train_split, tokenizer, config.max_length)
val_dataset = TextDataset(val_texts, y_val_split, tokenizer, config.max_length)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)


# ============================================================
# 5. Model Definition
# ============================================================
model = DistilBertForSequenceClassification.from_pretrained(
    config.model_name,
    num_labels=config.num_classes
)
model.to(config.device)

# Optimizer and scheduler
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": config.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

total_steps = len(train_loader) * config.epochs
warmup_steps = int(total_steps * config.warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = GradScaler() if config.device.type == "cuda" else None

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. Training Loop
# ============================================================
os.makedirs("./submission", exist_ok=True)
best_val_loss = float("inf")

for epoch in range(config.epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["label"].to(config.device)

        if scaler:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)

            if scaler:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss

            total_val_loss += loss.item()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            val_preds.append(probs)
            val_labels_list.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels_concat = np.concatenate(val_labels_list, axis=0)
    val_logloss = compute_log_loss(val_labels_concat, val_preds)

    print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - Val LogLoss: {val_logloss:.6f}")

    # Save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "./submission/best_model.pt")
        print(f"  Best model saved (Val Loss: {best_val_loss:.6f})")

# Load best model
model.load_state_dict(torch.load("./submission/best_model.pt"))
model.eval()

# ============================================================
# 7. Generate Test Predictions
# ============================================================
test_dataset = TextDataset(test_df["text"].values, labels=None, tokenizer=tokenizer, max_length=config.max_length)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        if scaler:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        test_preds.append(probs)

test_preds = np.concatenate(test_preds, axis=0)

# ============================================================
# 8. Generate Submission
# ============================================================
test_ids = test_df["id"].values
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

# Normalize rows to sum to 1
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

# Clip to avoid extremes
eps = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col].clip(eps, 1 - eps)

# Re-normalize after clipping
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Best Validation LogLoss: {best_val_loss:.6f}")