import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Create stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(train_df["text"], train_df["author"]))

train_texts = train_df.iloc[train_idx]["text"].reset_index(drop=True)
val_texts = train_df.iloc[val_idx]["text"].reset_index(drop=True)
train_authors = train_df.iloc[train_idx]["author"].reset_index(drop=True)
val_authors = train_df.iloc[val_idx]["author"].reset_index(drop=True)
test_texts = test_df["text"]
test_ids = test_df["id"]

# Encode authors
author_to_label = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = np.array([author_to_label[a] for a in train_authors])
val_labels = np.array([author_to_label[a] for a in val_authors])

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# Model configuration
MODEL_NAME = "answerdotai/ModernBERT-large"
NUM_CLASSES = 3
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 1.5e-5
NUM_EPOCHS = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ModernBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
)
model.config.hidden_dropout_prob = 0.1
model.config.attention_probs_dropout_prob = 0.1
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
)


# Tokenize all texts
def tokenize_texts(texts, max_length=MAX_LENGTH):
    all_encodings = []
    chunk_size = 500
    for i in range(0, len(texts), chunk_size):
        chunk = (
            texts.iloc[i : i + chunk_size].tolist()
            if hasattr(texts, "iloc")
            else texts[i : i + chunk_size]
        )
        encodings = tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        all_encodings.append(encodings)
    final_encodings = {
        "input_ids": torch.cat([e["input_ids"] for e in all_encodings], dim=0),
        "attention_mask": torch.cat(
            [e["attention_mask"] for e in all_encodings], dim=0
        ),
    }
    return final_encodings


print("Tokenizing data...")
train_encodings = tokenize_texts(train_texts)
val_encodings = tokenize_texts(val_texts)
test_encodings = tokenize_texts(test_texts)

# Create datasets
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Scheduler
total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = GradScaler()

# Training loop
best_val_loss = float("inf")
best_epoch = -1
no_improve_epochs = 0
early_stop_patience = 3

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds = []
    val_true = []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            total_val_loss += loss.item()
            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_log_loss = log_loss(val_true, val_preds_clipped)
    val_accuracy = (np.argmax(val_preds, axis=1) == val_true).mean()

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f} - Val Acc: {val_accuracy:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        no_improve_epochs = 0
        os.makedirs("./working", exist_ok=True)
        model.save_pretrained("./working/best_model")
        tokenizer.save_pretrained("./working/best_tokenizer")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

print(f"Best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

# Load best model and compute final validation score
best_model = ModernBertForSequenceClassification.from_pretrained("./working/best_model")
best_model.to(device)
best_model.eval()

val_preds_best = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        with autocast():
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        val_preds_best.append(probs.cpu().numpy())

val_preds_best = np.concatenate(val_preds_best, axis=0)
eps = 1e-15
val_preds_clipped = np.clip(val_preds_best, eps, 1 - eps)
val_log_loss_final = log_loss(val_labels, val_preds_clipped)
print(f"Final Validation Log Loss: {val_log_loss_final:.6f}")

# Test inference
print("Running test inference...")
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_ids.values,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

score = val_log_loss_final
print(f"Final Validation Score: {score}")
