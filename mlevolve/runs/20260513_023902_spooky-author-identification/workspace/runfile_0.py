import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
import re

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode target
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_authors = len(label_encoder.classes_)

# Create train/val split
train_indices, val_indices = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=train_df["author_encoded"],
)

# Prepare data splits
train_texts = train_df.iloc[train_indices]["text"].values
val_texts = train_df.iloc[val_indices]["text"].values
train_labels = train_df.iloc[train_indices]["author_encoded"].values
val_labels = train_df.iloc[val_indices]["author_encoded"].values
test_texts = test_df["text"].values


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


# Model definition
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class DebertaAuthorClassifier(nn.Module):
    def __init__(self, num_labels, dropout_prob=0.2):
        super().__init__()
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
        )
        hidden_size = self.deberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_prob * 0.5),
            nn.Linear(hidden_size // 4, num_labels),
        )
        self.deberta.classifier = self.classifier

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaAuthorClassifier(num_labels=num_authors).to(device)

# Class weights
class_weights = compute_class_weight(
    "balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2,
    eta_min=1e-6,
)

scaler = torch.cuda.amp.GradScaler()

# Create datasets and dataloaders
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_texts, labels=None, tokenizer=tokenizer)

batch_size = 16
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

# Training
num_epochs = 30
best_val_loss = float("inf")
best_model_state = None
patience = 5
patience_counter = 0
gradient_accumulation_steps = 2

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_steps = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps
        train_steps += 1

    avg_train_loss = total_train_loss / train_steps

    model.eval()
    total_val_loss = 0
    val_steps = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            total_val_loss += loss.item()
            val_steps += 1
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_val_probs.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / val_steps
    val_probs = np.vstack(all_val_probs)
    val_labels_np = np.concatenate(all_val_labels)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_score = log_loss(val_labels_np, val_probs_clipped)

    scheduler.step()
    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_score:.4f}"
    )

    if val_score < best_val_loss:
        best_val_loss = val_score
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Load best model
print(f"Loading best model with validation log loss: {best_val_loss:.4f}")
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Final validation score
all_val_probs = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.vstack(all_val_probs)
val_labels_np = np.concatenate(all_val_labels)
val_probs_final = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_final = val_probs_final / val_probs_final.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_labels_np, val_probs_final)

# Test inference
model.eval()
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

test_probs = np.vstack(all_test_probs)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

author_classes = label_encoder.classes_
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        author_classes[0]: test_probs[:, 0],
        author_classes[1]: test_probs[:, 1],
        author_classes[2]: test_probs[:, 2],
    }
)
submission_df = submission_df[["id", "EAP", "HPL", "MWS"]]
submission_df.to_csv("./submission/submission_75128a61d6ce4d17bbe4cc30fc09b0d4.csv", index=False)

print(f"Final Validation Score: {final_val_score}")