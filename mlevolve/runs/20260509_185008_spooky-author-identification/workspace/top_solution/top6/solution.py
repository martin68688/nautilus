import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import os
import warnings
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA LOADING AND STRATIFIED SPLIT
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_label"] = train_df["author"].map(author_map)
train_texts = train_df["text"].values
train_labels = train_df["author_label"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Stratified split (70-15-15 split; we use val for early stopping, rest for train)
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_texts, train_labels, test_size=0.15, stratify=train_labels, random_state=42
)


# ============================================================
# 2. FEATURE ENGINEERING: DistilBERT tokenization
# ============================================================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_texts(texts, max_length=128):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, 'tolist') else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encodings

train_encodings = tokenize_texts(X_train_texts)
val_encodings = tokenize_texts(X_val_texts)
test_encodings = tokenize_texts(test_texts)


# ============================================================
# 3. DATASET FOR DISTILBERT
# ============================================================
class DistilBertDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = DistilBertDataset(train_encodings, y_train_labels)
val_dataset = DistilBertDataset(val_encodings, y_val_labels)
test_dataset = DistilBertDataset(test_encodings, None)

batch_size = 16
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)

# ============================================================
# 4. MODEL, LOSS, OPTIMIZER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False,
).to(device)

# Freeze first 4 of 6 transformer layers
for i, layer in enumerate(model.distilbert.transformer.layer):
    if i < 4:
        for param in layer.parameters():
            param.requires_grad = False

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Linear warmup + cosine decay
num_epochs = 30
num_training_steps = num_epochs * len(train_loader) // 2  # gradient accumulation steps=2
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
scaler = GradScaler()
gradient_accumulation_steps = 2

# ============================================================
# 5. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
epochs = num_epochs
best_val_logloss = float("inf")
patience_counter = 0
patience = 7

for epoch in range(epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                probs = F.softmax(outputs.logits, dim=1)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
    val_logloss = -np.mean(
        np.sum(np.eye(3)[val_targets] * np.log(val_preds_clipped), axis=1)
    )

    print(
        f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}'
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print(f"Training complete. Best validation log loss: {best_val_logloss:.6f}")

# ============================================================
# 6. FINAL VALIDATION SCORE AND TEST PREDICTIONS
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

# Quick validation re-evaluation for final score
val_preds = []
val_targets = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            probs = F.softmax(outputs.logits, dim=1)

        val_preds.append(probs.cpu().numpy())
        val_targets.append(labels.cpu().numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
score = -np.mean(np.sum(np.eye(3)[val_targets] * np.log(val_preds_clipped), axis=1))

print(f"Final Validation Score: {score}")

# ============================================================
# 7. GENERATE SUBMISSION
# ============================================================
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probs = F.softmax(outputs.logits, dim=1)

        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)
test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)
submission = submission[["id", "EAP", "HPL", "MWS"]]
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")