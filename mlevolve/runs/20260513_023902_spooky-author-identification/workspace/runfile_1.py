import os
os.sched_setaffinity(0, {11, 12, 13, 14, 15})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import os
import pickle
import re
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# Create output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
print(f"Classes: {train_df['author'].unique()}")

# Encode labels
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)

# Save label encoder for later use
np.save("./working/label_classes.npy", label_encoder.classes_)

# Create stratified train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_df["text"], train_df["author_encoded"]))[0]

train_data = train_df.iloc[train_idx].reset_index(drop=True)
val_data = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

# Prepare raw text for transformer models
train_texts = train_data["text"].tolist()
val_texts = val_data["text"].tolist()
test_texts = test_df["text"].tolist()
train_labels = train_data["author_encoded"].values
val_labels = val_data["author_encoded"].values

# Define model architecture
model_name = "microsoft/deberta-v3-large"
num_authors = 3  # EAP, HPL, MWS

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_authors,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Define optimizer with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

# Tokenize texts
print(f"Tokenizing {len(train_texts)} training texts...")
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

print(f"Tokenizing {len(val_texts)} validation texts...")
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

print(f"Tokenizing {len(test_texts)} test texts...")
test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

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
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

# Create dataloaders
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

# Training configuration
num_epochs = 20
accumulation_steps = 2
num_training_steps = num_epochs * (len(train_loader) // accumulation_steps + 1)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

best_val_metric = float("inf")
early_stopping_counter = 0
early_stopping_patience = 5
scaler = GradScaler()

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {model_name}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Device: {device}")

print(f"Starting training for {num_epochs} epochs...")
print(f"Batch size: {batch_size}, Accumulation steps: {accumulation_steps}")
print(f"Effective batch size: {batch_size * accumulation_steps}")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    all_val_probs = []
    all_val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            total_val_loss += loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_true.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)

    # Compute validation log-loss
    all_val_probs = np.concatenate(all_val_probs, axis=0)
    all_val_true = np.concatenate(all_val_true, axis=0)

    eps = 1e-15
    all_val_probs_clipped = np.clip(all_val_probs, eps, 1 - eps)
    all_val_probs_clipped = all_val_probs_clipped / all_val_probs_clipped.sum(
        axis=1, keepdims=True
    )

    val_log_loss = 0
    n = len(all_val_true)
    for i in range(n):
        for j in range(num_authors):
            if all_val_true[i] == j:
                val_log_loss -= np.log(all_val_probs_clipped[i, j])
    val_log_loss /= n

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LogLoss: {val_log_loss:.4f} | LR: {current_lr:.2e}"
    )

    # Save best model
    if val_log_loss < best_val_metric:
        best_val_metric = val_log_loss
        torch.save(model.state_dict(), "./working/best_model_48c5de1d59c54471be8570048c2663cd.pt")
        print(f"  -> New best model saved! Val LogLoss: {val_log_loss:.6f}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print(f"\nTraining complete. Best Val LogLoss: {best_val_metric:.6f}")

# Load best model for inference
print("Loading best model for inference...")
model.load_state_dict(torch.load("./working/best_model_48c5de1d59c54471be8570048c2663cd.pt"))
model.eval()

# Validation inference (for final metric)
print("Running validation inference...")
all_val_probs_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        all_val_probs_final.append(probs.cpu().numpy())

all_val_probs_final = np.concatenate(all_val_probs_final, axis=0)

# Compute final validation log-loss
eps = 1e-15
all_val_probs_clipped = np.clip(all_val_probs_final, eps, 1 - eps)
all_val_probs_clipped = all_val_probs_clipped / all_val_probs_clipped.sum(
    axis=1, keepdims=True
)

val_log_loss_final = 0
n = len(val_labels)
for i in range(n):
    for j in range(num_authors):
        if val_labels[i] == j:
            val_log_loss_final -= np.log(all_val_probs_clipped[i, j])
val_log_loss_final /= n

# Test inference
print("Running test inference...")
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

all_test_probs = np.concatenate(all_test_probs, axis=0)

# Clip and normalize test probabilities
all_test_probs = np.clip(all_test_probs, eps, 1 - eps)
all_test_probs = all_test_probs / all_test_probs.sum(axis=1, keepdims=True)

# Create submission dataframe
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": all_test_probs[:, 0],
        "HPL": all_test_probs[:, 1],
        "MWS": all_test_probs[:, 2],
    }
)

# Ensure columns match sample submission exactly
submission = submission[["id", "EAP", "HPL", "MWS"]]

# Save submission
submission.to_csv("./submission/submission_48c5de1d59c54471be8570048c2663cd.csv", index=False)
print(f"Submission saved to ./submission/submission_48c5de1d59c54471be8570048c2663cd.csv")
print(f"Submission shape: {submission.shape}")
print(f"Submission preview:")
print(submission.head())

# Print final validation score (REQUIRED)
score = val_log_loss_final
print(f"Final Validation Score: {score}")
