import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================
class Config:
    seed = 42
    num_classes = 3
    # Model parameters
    model_name = "microsoft/deberta-v3-base"
    max_length = 256
    dropout = 0.2
    # Training parameters
    n_folds = 3  # Reduced from 5 to ensure time constraint
    epochs = 4
    batch_size = 32
    learning_rate = 3e-5
    weight_decay = 0.01
    max_grad_norm = 1.0
    warmup_ratio = 0.1
    val_split = 0.05  # 5% validation split for early stopping within each fold
    patience = 2
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


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
# 2. Dataset and Model Definition
# ============================================================
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
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DebertaClassifier(nn.Module):
    def __init__(
        self, model_name="microsoft/deberta-v3-base", num_classes=3, dropout=0.2
    ):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_epoch(
    model, dataloader, optimizer, scheduler, criterion, device, max_grad_norm=1.0
):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    logloss = log_loss(all_labels, all_preds)
    return avg_loss, logloss


# ============================================================
# 3. Prepare Tokenizer
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# ============================================================
# 4. 3-Fold Cross Validation Training
# ============================================================
print(f"\nStarting {config.n_folds}-fold stratified cross-validation...")
print(f"Device: {config.device}")

skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

# Store test predictions from each fold
test_preds_folds = np.zeros((len(test_df), config.num_classes))
all_val_loglosses = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{config.n_folds}")
    print(f"{'='*50}")

    # Split data
    train_texts = train_df.iloc[train_idx]["text"].values
    train_labels = y_train[train_idx]
    val_texts = train_df.iloc[val_idx]["text"].values
    val_labels = y_train[val_idx]

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_length)
    test_dataset = TextDataset(
        test_df["text"].values, None, tokenizer, config.max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Initialize model
    model = DebertaClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    model.to(config.device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_logloss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            config.device,
            config.max_grad_norm,
        )
        val_loss, val_logloss = validate_epoch(
            model, val_loader, criterion, config.device
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f}"
        )

        # Save best model
        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    all_val_loglosses.append(best_val_logloss)

    # Load best model for this fold
    model.load_state_dict(best_model_state)

    # Generate predictions on test set
    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            fold_test_preds.append(probs.cpu().numpy())

    fold_test_preds = np.concatenate(fold_test_preds, axis=0)
    test_preds_folds += fold_test_preds

    print(f"Fold {fold + 1} completed. Best Val LogLoss: {best_val_logloss:.4f}")

# Average predictions across folds
test_preds = test_preds_folds / config.n_folds

# Clip and normalize predictions
eps = 1e-15
test_preds = np.clip(test_preds, eps, 1 - eps)
row_sums = test_preds.sum(axis=1, keepdims=True)
test_preds = test_preds / row_sums

# Compute final validation score (average across folds)
final_val_score = np.mean(all_val_loglosses)

# ============================================================
# 5. Generate Submission
# ============================================================
os.makedirs("./submission", exist_ok=True)
test_ids = test_df["id"].values
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, label_encoder.transform(["EAP"])[0]],
        "HPL": test_preds[:, label_encoder.transform(["HPL"])[0]],
        "MWS": test_preds[:, label_encoder.transform(["MWS"])[0]],
    }
)

# Final normalization to ensure rows sum to 1
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Final Validation Score: {final_val_score}")
