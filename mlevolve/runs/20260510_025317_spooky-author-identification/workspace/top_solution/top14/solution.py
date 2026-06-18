import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import os
import gc
import re
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ===== DATA LOADING AND SPLIT =====
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_encoded"] = train_df["author"].map(author_map)

# Stratified train/val split
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    train_df["text"].values,
    train_df["author_encoded"].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

# Save split indices for reference using iloc positions
train_idx = list(range(len(X_train_texts)))
val_idx = list(range(len(X_train_texts), len(X_train_texts) + len(X_val_texts)))
split_info = pd.DataFrame(
    {
        "train_idx": pd.Series(train_idx),
        "val_idx": pd.Series(val_idx + [None] * (len(train_idx) - len(val_idx))),
    }
)
split_info.to_parquet("./working/train_val_split.parquet")
train_df.to_parquet("./working/train_original.parquet")
test_df.to_parquet("./working/test_original.parquet")

print(f"Training samples: {len(X_train_texts)}")
print(f"Validation samples: {len(X_val_texts)}")
print(f"Test samples: {len(test_df)}")


# ===== DATASET CLASS =====
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===== MODEL DEFINITION =====
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ModernBertForSequenceClassification.from_pretrained(model_id, num_labels=3)
model.config.hidden_dropout_prob = 0.1
model.config.attention_probs_dropout_prob = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
model = model.to(device)

# Optimizer with weight decay
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler() if device.type == "cuda" else None

# ===== DATA LOADERS =====
max_length = 512
batch_size = 8

train_dataset = AuthorDataset(X_train_texts, y_train, tokenizer, max_length)
val_dataset = AuthorDataset(X_val_texts, y_val, tokenizer, max_length)
test_dataset = AuthorDataset(
    test_df["text"].values, labels=None, tokenizer=tokenizer, max_length=max_length
)

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

# ===== TRAINING =====
max_epochs = 5
patience = 2
best_val_logloss = float("inf")
patience_counter = 0
best_model_state = None

total_steps = (len(train_dataset) // batch_size + 1) * max_epochs
warmup_steps = int(0.1 * total_steps)


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("Starting training...")
for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    # Validation
    model.eval()
    val_losses = []
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logits = outputs.logits
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

            val_losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = np.mean(val_losses)
    val_probs = np.concatenate(all_val_probs, axis=0)
    val_labels_concat = np.concatenate(all_val_labels, axis=0)

    eps = 1e-15
    val_probs = val_probs.astype(np.float64)
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_probs_normalized = np.clip(val_probs_normalized, eps, 1 - eps)

    val_one_hot = np.zeros((len(val_labels_concat), 3), dtype=np.float64)
    val_one_hot[np.arange(len(val_labels_concat)), val_labels_concat] = 1.0
    log_loss_val = -np.mean(np.sum(val_one_hot * np.log(val_probs_normalized), axis=1))

    print(
        f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Log Loss: {log_loss_val:.4f}"
    )

    if log_loss_val < best_val_logloss:
        best_val_logloss = log_loss_val
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "./working/best_model.pt")
        print(f"  -> New best model saved with log loss: {best_val_logloss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# ===== FINAL EVALUATION =====
print("\nLoading best model for final evaluation...")
# Force load from disk to ensure clean state
del model
gc.collect()
torch.cuda.empty_cache()

model = ModernBertForSequenceClassification.from_pretrained(model_id, num_labels=3)
model.config.hidden_dropout_prob = 0.1
model.config.attention_probs_dropout_prob = 0.1
model.load_state_dict(torch.load("./working/best_model.pt", map_location="cpu"))
model = model.to(device)
model.eval()

# Recreate validation dataset and loader to ensure clean state
val_dataset_final = AuthorDataset(X_val_texts, y_val, tokenizer, max_length)
val_loader_final = DataLoader(
    val_dataset_final,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

all_val_probs = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader_final:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs_final = np.concatenate(all_val_probs, axis=0).astype(np.float64)
val_labels_final = np.concatenate(all_val_labels, axis=0)
eps = 1e-15
val_probs_clipped = np.clip(val_probs_final, eps, 1 - eps)
val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
val_probs_normalized = np.clip(val_probs_normalized, eps, 1 - eps)
val_one_hot = np.zeros((len(val_labels_final), 3), dtype=np.float64)
val_one_hot[np.arange(len(val_labels_final)), val_labels_final] = 1.0
final_log_loss = -np.mean(np.sum(val_one_hot * np.log(val_probs_normalized), axis=1))
print(f"\nFinal Validation Log Loss: {final_log_loss:.6f}")

# ===== TEST PREDICTIONS =====
print("\nGenerating test predictions...")

# Recreate test dataset and loader for consistency
test_dataset_final = AuthorDataset(
    test_df["text"].values, labels=None, tokenizer=tokenizer, max_length=max_length
)
test_loader_final = DataLoader(
    test_dataset_final,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader_final:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0).astype(np.float64)
test_probs_clipped = np.clip(test_probs, eps, 1 - eps)
test_probs_normalized = test_probs_clipped / test_probs_clipped.sum(
    axis=1, keepdims=True
)
test_probs_normalized = np.clip(test_probs_normalized, eps, 1 - eps)

# ===== SUBMISSION =====
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs_normalized[:, 0],
        "HPL": test_probs_normalized[:, 1],
        "MWS": test_probs_normalized[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

score = final_log_loss
print(f"Final Validation Score: {score}")