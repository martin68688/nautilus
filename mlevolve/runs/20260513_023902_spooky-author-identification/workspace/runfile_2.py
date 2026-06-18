import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9, 11, 12, 13, 14, 15})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
import warnings

warnings.filterwarnings("ignore")

# ============== DATA LOADING AND PREPROCESSING ==============
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return ""


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text_clean"].values,
    train_df["author_encoded"].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

test_texts = test_df["text_clean"].values
test_ids = test_df["id"].values


# ============== DATASET CLASS ==============
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


# ============== MODEL CONFIGURATION ==============
model_name = "microsoft/deberta-v3-large"
num_authors = 3
max_length = 512
batch_size = 8
learning_rate = 2e-5
weight_decay = 0.01
num_epochs = 30
patience = 5
focal_gamma = 2.0

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_authors,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============== DATALOADERS ==============
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = TextDataset(test_texts, None, tokenizer, max_length)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# ============== LOSS FUNCTION ==============
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        probs = F.softmax(logits, dim=1)
        pt = probs[torch.arange(probs.size(0)), targets]
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


class_counts = np.bincount(train_labels)
class_weights = torch.tensor(
    [1.0 / count for count in class_counts], dtype=torch.float32
).to(device)
class_weights = class_weights / class_weights.sum()
criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)

# ============== OPTIMIZER AND SCHEDULER ==============
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
)

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# ============== TRAINING LOOP ==============
scaler = GradScaler()
best_val_score = float("inf")
no_improve = 0

os.makedirs("./working", exist_ok=True)

print(f"Training started on {device}")
print(
    f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_probs = []
    val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = criterion(outputs.logits, labels)

            val_loss += loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    val_loss_avg = val_loss / len(val_loader)
    val_probs = np.concatenate(val_probs, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_log_loss = -np.mean(
        np.sum(np.eye(num_authors)[val_true] * np.log(val_probs_clipped), axis=1)
    )

    print(
        f"Epoch {epoch+1:2d} | Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss_avg:.4f} LogLoss: {val_log_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if val_log_loss < best_val_score:
        best_val_score = val_log_loss
        torch.save(model.state_dict(), "./working/best_model_d1c50a062bf64890acfa1fb50913f1e4.pth")
        no_improve = 0
        print(f"  → Saved best model (LogLoss: {val_log_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping triggered after {epoch+1} epochs")
            break

# ============== LOAD BEST MODEL AND EVALUATE ==============
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load("./working/best_model_d1c50a062bf64890acfa1fb50913f1e4.pth"))
model.eval()

val_probs_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        val_probs_final.append(probs.cpu().numpy())

val_probs_final = np.concatenate(val_probs_final, axis=0)
val_probs_final_clipped = np.clip(val_probs_final, eps, 1 - eps)
val_probs_final_clipped = val_probs_final_clipped / val_probs_final_clipped.sum(
    axis=1, keepdims=True
)

val_true_final = np.concatenate(
    [batch["labels"].numpy() for batch in val_loader], axis=0
)
val_log_loss_final = -np.mean(
    np.sum(
        np.eye(num_authors)[val_true_final] * np.log(val_probs_final_clipped), axis=1
    )
)

print(f"Final Validation Log Loss: {val_log_loss_final:.6f}")

# ============== TEST INFERENCE ==============
print("\nPerforming test inference...")
test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs, axis=0)
test_probs_clipped = np.clip(test_probs, eps, 1 - eps)
test_probs_clipped = test_probs_clipped / test_probs_clipped.sum(axis=1, keepdims=True)

# ============== SAVE SUBMISSION ==============
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_clipped[:, 0],
        "HPL": test_probs_clipped[:, 1],
        "MWS": test_probs_clipped[:, 2],
    }
)
submission_df.to_csv("./submission/submission_d1c50a062bf64890acfa1fb50913f1e4.csv", index=False)
print(f"Submission saved to ./submission/submission_d1c50a062bf64890acfa1fb50913f1e4.csv")

score = val_log_loss_final
print(f"Final Validation Score: {score}")
