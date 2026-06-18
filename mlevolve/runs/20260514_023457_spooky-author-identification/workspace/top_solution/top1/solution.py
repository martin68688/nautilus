import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from transformers import DataCollatorWithPadding
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

print("Starting data processing and feature engineering...")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
author_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
num_classes = len(le.classes_)
print(f"Author mapping: {author_mapping}")

# Save raw text and labels as pickle for the next stages
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Save processed DataFrames
train_df[["id", "text", "author_encoded"]].to_pickle("./working/train_data.pkl")
test_df[["id", "text"]].to_pickle("./working/test_data.pkl")

# Save metadata
metadata = {
    "classes": list(le.classes_),
    "author_mapping": author_mapping,
    "num_classes": num_classes,
}
with open("./working/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print("Data processing and feature engineering complete! (raw text only)")

# --- Model Design & Training/Evaluation ---
print("\nStarting training and evaluation...")

# Load preprocessed data
train_df = pd.read_pickle("./working/train_data.pkl")
test_df = pd.read_pickle("./working/test_data.pkl")

with open("./working/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

num_classes = metadata["num_classes"]
print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

# Tokenizer
MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom Dataset
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 5-fold cross-validation with DistilRoBERTa
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train = train_df["author_encoded"].values
all_oof_preds = np.zeros((len(train_df), num_classes))
all_test_preds = np.zeros((len(test_df), num_classes))
completed_folds = 0

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 3

completed_folds = 0
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_df)), y_train)):
    print(f"\n{'='*40}")
    print(f"Fold {fold_idx + 1}/5")
    print(f"{'='*40}")

    train_texts = train_df.iloc[train_idx]["text"].values
    train_labels = y_train[train_idx]
    val_texts = train_df.iloc[val_idx]["text"].values
    val_labels = y_train[val_idx]

    train_dataset = AuthorDataset(train_texts, train_labels)
    val_dataset = AuthorDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    # Model and optimizer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                total_val_loss += loss.item()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                val_preds.append(probs)
                val_labels_list.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_preds = np.concatenate(val_preds, axis=0)
        val_labels_concat = np.concatenate(val_labels_list, axis=0)

        # Clip and normalize predictions
        val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)

        val_score = log_loss(val_labels_concat, val_preds)
        print(f"Val Loss: {avg_val_loss:.6f} | Val Log Loss: {val_score:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save fold predictions
            all_oof_preds[val_idx] = val_preds

            # Generate test predictions for this fold
            test_dataset = AuthorDataset(test_df["text"].values)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

            model.eval()
            fold_test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    fold_test_preds.append(probs)
            fold_test_preds = np.concatenate(fold_test_preds, axis=0)
            fold_test_preds = np.clip(fold_test_preds, 1e-15, 1 - 1e-15)
            fold_test_preds = fold_test_preds / fold_test_preds.sum(axis=1, keepdims=True)
            all_test_preds += fold_test_preds
            completed_folds += 1
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    # Fold final score using best model
    fold_val_preds = all_oof_preds[val_idx]
    fold_val_score = log_loss(val_labels_concat, fold_val_preds)
    print(f"Fold {fold_idx + 1} Best Log Loss: {fold_val_score:.6f}")

# Overall validation score using all out-of-fold predictions
val_score = log_loss(y_train, all_oof_preds)
print(f"\n{'='*40}")
print(f"Overall Validation Log Loss: {val_score:.6f}")
print(f"{'='*40}")

# Generate final submission
test_preds_final = all_test_preds / completed_folds
test_preds_final = np.clip(test_preds_final, 1e-15, 1 - 1e-15)
test_preds_final = test_preds_final / test_preds_final.sum(axis=1, keepdims=True)

class_names = metadata["classes"]  # ['EAP', 'HPL', 'MWS']
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        class_names[0]: test_preds_final[:, 0],
        class_names[1]: test_preds_final[:, 1],
        class_names[2]: test_preds_final[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {val_score}")