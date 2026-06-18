import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# DATA LOADING & PROCESSING
# ============================================================

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors distribution:\n{train_df['author'].value_counts()}")

# Author encoding
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_label"] = train_df["author"].map(author_map)

# Split data first
train_texts = train_df["text"].tolist()
y_train_all = train_df["author_label"].values

# Create train/validation split (stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_texts, y_train_all))[0]

train_texts_split = [train_texts[i] for i in train_idx]
val_texts_split = [train_texts[i] for i in val_idx]
y_train = y_train_all[train_idx]
y_val = y_train_all[val_idx]
test_texts = test_df["text"].tolist()

# Tokenize texts (only fit tokenizer on training data)
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize all texts with consistent padding
max_len = 384  # Shorter sequence length for faster training and less overfitting

train_encodings = tokenizer(
    train_texts_split, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
)
val_encodings = tokenizer(
    val_texts_split, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
)
test_encodings = tokenizer(
    test_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
)

train_input_ids = train_encodings["input_ids"].numpy()
train_attention_mask = train_encodings["attention_mask"].numpy()
val_input_ids = val_encodings["input_ids"].numpy()
val_attention_mask = val_encodings["attention_mask"].numpy()
test_input_ids = test_encodings["input_ids"].numpy()
test_attention_mask = test_encodings["attention_mask"].numpy()

print(
    f"Train samples: {len(train_texts_split)}, Val samples: {len(val_texts_split)}, Test samples: {len(test_texts)}"
)

# ============================================================
# MODEL DEFINITION
# ============================================================

NUM_AUTHORS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3,
)


class TemperatureScaledModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            output_attentions=False,
        )
        logits = outputs.logits / self.T
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, NUM_AUTHORS), labels.view(-1))
            return loss, logits
        return None, logits


model = TemperatureScaledModel(model)
model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# TRAINING & EVALUATION
# ============================================================

batch_size = 16

train_dataset = TensorDataset(
    torch.tensor(train_input_ids, dtype=torch.long),
    torch.tensor(train_attention_mask, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(val_input_ids, dtype=torch.long),
    torch.tensor(val_attention_mask, dtype=torch.long),
    torch.tensor(y_val, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# Optimizer with layer-wise learning rate decay
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.1,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8, betas=(0.9, 0.999))

# Scheduler - shorter training with more aggressive decay
num_epochs = 12
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# Training loop - track best log loss on validation
best_val_logloss = float("inf")
best_model_state = None
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_batches = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, _ = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()
        train_batches += 1

    avg_train_loss = total_train_loss / train_batches

    # Validation
    model.eval()
    val_loss = 0
    val_batches = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            with torch.cuda.amp.autocast():
                loss, logits = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            val_loss += loss.item()
            val_batches += 1

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_val_probs.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    all_val_probs = np.concatenate(all_val_probs)
    all_val_labels = np.concatenate(all_val_labels)

    eps = 1e-15
    all_val_probs = np.clip(all_val_probs, eps, 1 - eps)
    all_val_probs = all_val_probs / all_val_probs.sum(axis=1, keepdims=True)

    val_log_loss = log_loss(all_val_labels, all_val_probs)

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_log_loss:.4f}"
    )

    if val_log_loss < best_val_logloss:
        best_val_logloss = val_log_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model (based on log loss)
model.load_state_dict(best_model_state)
print(f"Best validation log loss: {best_val_logloss:.4f}")

# Final validation with best model
model.eval()
all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with torch.cuda.amp.autocast():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)
        all_val_labels.append(labels.cpu().numpy())

all_val_probs = np.concatenate(all_val_probs)
all_val_labels = np.concatenate(all_val_labels)

eps = 1e-15
all_val_probs = np.clip(all_val_probs, eps, 1 - eps)
all_val_probs = all_val_probs / all_val_probs.sum(axis=1, keepdims=True)

val_log_loss_final = log_loss(all_val_labels, all_val_probs)

# Test inference
test_dataset = TensorDataset(
    torch.tensor(test_input_ids, dtype=torch.long),
    torch.tensor(test_attention_mask, dtype=torch.long),
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]

        with torch.cuda.amp.autocast():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

all_test_probs = np.concatenate(all_test_probs)

all_test_probs = np.clip(all_test_probs, eps, 1 - eps)
all_test_probs = all_test_probs / all_test_probs.sum(axis=1, keepdims=True)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": all_test_probs[:, 0],
        "HPL": all_test_probs[:, 1],
        "MWS": all_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {val_log_loss_final}")