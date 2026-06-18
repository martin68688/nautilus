import os
os.sched_setaffinity(0, {6, 19, 20, 21, 22})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import warnings

warnings.filterwarnings("ignore")

# Ensure directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_df["text"].values,
    train_df["label"].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"].values,
)

test_texts = test_df["text"].values
test_ids = test_df["id"].values

print(
    f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(test_texts)}"
)


# Dataset class for transformer inputs
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Only lowercase, no other cleaning
        text = text.lower()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Create datasets
train_dataset = TextDataset(X_train, y_train, tokenizer, max_len=256)
val_dataset = TextDataset(X_val, y_val, tokenizer, max_len=256)
test_dataset = TextDataset(test_texts, labels=None, tokenizer=tokenizer, max_len=256)

# Create data loaders
batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
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


# Define DistilBERT classifier with simple CLS pooling
class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        # Simple classification head using CLS token
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get CLS token representation
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertClassifier(num_classes=num_classes, dropout_rate=0.2)
model.to(device)
print(
    f"Model initialized on {device} with {sum(p.numel() for p in model.parameters()):,} parameters"
)

# Optimizer with different learning rates for transformer and head
optimizer = AdamW(
    [
        {
            "params": model.bert.parameters(),
            "lr": 2e-5,
            "weight_decay": 0.01,
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-3,
            "weight_decay": 0.01,
        },
    ]
)

# Loss function with label smoothing (0.2) to reduce overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
scaler = GradScaler()

# Training hyperparameters
num_epochs = 10
patience = 5
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

# Learning rate scheduler: CosineAnnealingLR per epoch
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

print(f"Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)

    # Clip probabilities and normalize
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )

    # Calculate log loss
    val_log_loss = log_loss(val_true, val_preds_normalized)

    # Print epoch summary
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.6f} | LR: {current_lr:.2e}'
    )

    # Step scheduler after each epoch
    scheduler.step()

    # Early stopping and model saving
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save(
            model.state_dict(),
            "./working/best_model_1e2b2d2689a4424d98e8ff0875e6454b.pt",
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

print(
    f"\nTraining completed. Best validation log loss: {best_val_loss:.6f} at epoch {best_epoch+1}"
)

# Load best model for final inference
model.load_state_dict(torch.load("./working/best_model_1e2b2d2689a4424d98e8ff0875e6454b.pt", map_location=device))
model.eval()

# Final validation prediction with best model
print("Performing final validation inference...")
val_preds_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        val_preds_final.append(probs.cpu().numpy())

val_preds_final = np.concatenate(val_preds_final)
val_preds_final = np.clip(val_preds_final, 1e-15, 1 - 1e-15)
val_preds_final = val_preds_final / val_preds_final.sum(axis=1, keepdims=True)

# Calculate final validation score
score = log_loss(y_val, val_preds_final)

# Test inference
print("Performing test inference...")
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds)
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission dataframe using label encoder classes to ensure correct column order
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        label_encoder.classes_[0]: test_preds[:, 0],
        label_encoder.classes_[1]: test_preds[:, 1],
        label_encoder.classes_[2]: test_preds[:, 2],
    }
)

# Save submission
submission_df.to_csv("./submission/submission_1e2b2d2689a4424d98e8ff0875e6454b.csv", index=False)
print(f"Submission saved to ./submission/submission_1e2b2d2689a4424d98e8ff0875e6454b.csv")
print(f"Submission shape: {submission_df.shape}")

# Final validation score
print(f"Final Validation Score: {score}")