import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
import os
import gc
from sklearn.metrics import log_loss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode target
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = train_df["author"].map(author_mapping).values

# Load tokenizer and model - DeBERTa-v3-base (86M parameters)
model_id = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize model with 3 output classes (EAP, HPL, MWS)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
model.to(device)

# Freeze first 8 out of 12 transformer layers
num_layers = 12
layers_to_freeze = 8
for i in range(layers_to_freeze):
    for param in model.deberta.encoder.layer[i].parameters():
        param.requires_grad = False

print("Frozen layers 0-7 (first 8 layers), training layers 8-11 (last 4 layers)")

# Add custom dropout layer in the classification head
# Replace the classifier with one that has dropout
original_classifier = model.classifier
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.config.hidden_size, 3)
)
# Move new classifier to device
model.classifier.to(device)

# Label Smoothing CrossEntropyLoss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.15, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n_classes = preds.size(1)
        log_probs = torch.log_softmax(preds, dim=1)
        one_hot = torch.zeros_like(preds).scatter(1, target.unsqueeze(1), 1)
        smooth_target = (1 - self.epsilon) * one_hot + self.epsilon / n_classes
        loss = -(smooth_target * log_probs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

criterion = LabelSmoothingCrossEntropy(epsilon=0.15)

# Entropy penalty loss component
class EntropyPenalty(nn.Module):
    def __init__(self, lambda_entropy=0.01):
        super().__init__()
        self.lambda_entropy = lambda_entropy

    def forward(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return self.lambda_entropy * entropy

entropy_penalty_fn = EntropyPenalty(lambda_entropy=0.01)

# Define separate parameter groups for differential learning rates
unfreezed_backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'classifier' in name:
            head_params.append(param)
        else:
            unfreezed_backbone_params.append(param)

optimizer = AdamW([
    {'params': unfreezed_backbone_params, 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': head_params, 'lr': 5e-4, 'weight_decay': 0.01}
], betas=(0.9, 0.999), eps=1e-8)

# Define learning rate scheduler - Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=4,
    T_mult=2,
    eta_min=1e-6,
)

# Check if CUDA is available for mixed precision
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

# Maximum sequence length
max_length = 256

print(f"Model loaded: {model_id}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(f"Max sequence length: {max_length}")
print(f"Label smoothing epsilon: 0.15, Entropy penalty lambda: 0.01")
print(f"Backbone LR: 1e-5, Head LR: 5e-4")


# Custom Dataset for ModernBERT
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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Create train/validation split (stratified)
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.15, random_state=42, stratify=train_labels
)

train_texts = train_df.iloc[train_idx]["text"].values
train_labels_split = train_labels[train_idx]
val_texts = train_df.iloc[val_idx]["text"].values
val_labels_split = train_labels[val_idx]
test_texts = test_df["text"].values

# Create DataLoaders
batch_size = 32

train_dataset = AuthorDataset(
    texts=train_texts,
    labels=train_labels_split,
    tokenizer=tokenizer,
    max_length=max_length,
)
val_dataset = AuthorDataset(
    texts=val_texts,
    labels=val_labels_split,
    tokenizer=tokenizer,
    max_length=max_length,
)
test_dataset = AuthorDataset(
    texts=test_texts, labels=None, tokenizer=tokenizer, max_length=max_length
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(
    f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
)

# Training Setup
num_epochs = 8
accumulation_steps = 2
best_val_loss = float("inf")
best_epoch = -1
patience = 3
no_improve_count = 0

os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Training Loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss += entropy_penalty_fn(logits)
            loss = loss / accumulation_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_train_loss += loss.item() * accumulation_steps
        train_batches += 1

        del input_ids, attention_mask, labels, outputs, loss
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / train_batches

    # Validation phase
    model.eval()
    val_preds = []
    val_true = []
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())
            total_val_loss += loss.item()
            val_batches += 1

            del input_ids, attention_mask, labels, outputs, logits, probs, loss

    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)
    avg_val_loss = total_val_loss / val_batches

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

    val_log_loss = log_loss(val_true, val_preds_clipped)

    scheduler.step(epoch)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_epoch = epoch
        no_improve_count = 0
        torch.save(model.state_dict(), "./working/models/best_model.pt")
        print(f"  → Saved best model (Val LogLoss: {val_log_loss:.6f})")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"  → Early stopping triggered after {epoch+1} epochs")
            break

    gc.collect()
    torch.cuda.empty_cache()

print(
    f"\nTraining complete. Best epoch: {best_epoch+1}, Best Val LogLoss: {best_val_loss:.6f}"
)

# Load best model and compute final validation metric
model.load_state_dict(torch.load("./working/models/best_model.pt"))
model.eval()

val_preds_final = []
val_true_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        val_preds_final.append(probs.cpu().numpy())
        val_true_final.append(labels.cpu().numpy())

        del input_ids, attention_mask, labels, outputs, logits, probs

val_preds_final = np.concatenate(val_preds_final, axis=0)
val_true_final = np.concatenate(val_true_final, axis=0)

eps = 1e-15
val_preds_clipped = np.clip(val_preds_final, eps, 1 - eps)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

final_val_score = log_loss(val_true_final, val_preds_clipped)

print(f"Final Validation Score: {final_val_score}")

# Test Inference and Submission
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

        del input_ids, attention_mask, outputs, logits, probs

test_preds = np.concatenate(test_preds, axis=0)

test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with {len(submission_df)} rows")
print(submission_df.head())
