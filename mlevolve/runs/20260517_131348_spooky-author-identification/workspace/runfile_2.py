import os
os.sched_setaffinity(0, {35, 36, 37})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from sklearn.model_selection import train_test_split
import os
import joblib

# =============================================
# DATA PROCESSING AND FEATURE ENGINEERING (Step 1)
# =============================================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Basic text cleaning function
def clean_text(text):
    text = text.strip()
    text = " ".join(text.split())
    return text

# Apply cleaning to training and test texts
train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# Feature engineering: handcrafted stylistic features
def get_text_features(text):
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    num_sentences = text.count(".") + text.count("!") + text.count("?")
    num_commas = text.count(",")
    num_exclamations = text.count("!")
    num_questions = text.count("?")
    num_quotes = text.count('"') + text.count("'")
    num_colons = text.count(":")
    num_semicolons = text.count(";")
    num_dashes = text.count("-") + text.count("—")
    num_parentheses = text.count("(") + text.count(")")
    avg_word_len = num_chars / max(num_words, 1)
    return pd.Series(
        {
            "num_words": num_words,
            "num_chars": num_chars,
            "num_sentences": num_sentences,
            "num_commas": num_commas,
            "num_exclamations": num_exclamations,
            "num_questions": num_questions,
            "num_quotes": num_quotes,
            "num_colons": num_colons,
            "num_semicolons": num_semicolons,
            "num_dashes": num_dashes,
            "num_parentheses": num_parentheses,
            "avg_word_len": avg_word_len,
        }
    )

# Extract features for training and test
train_features = train_df["text_clean"].apply(get_text_features)
test_features = test_df["text_clean"].apply(get_text_features)

# Concatenate features with original dataframes
train_df = pd.concat([train_df, train_features], axis=1)
test_df = pd.concat([test_df, test_features], axis=1)

# Encode author labels
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_encoded"] = train_df["author"].map(author_mapping)

# Stratified split with correct indexing (no index reset bug)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=train_df["author_encoded"],
)

# Create train/val splits using direct numpy indexing
train_set = train_df.iloc[train_idx].copy()
val_set = train_df.iloc[val_idx].copy()

# Prepare arrays for later use
train_texts = train_set["text_clean"].values
train_labels = train_set["author_encoded"].values
val_texts = val_set["text_clean"].values
val_labels = val_set["author_encoded"].values
test_texts = test_df["text_clean"].values

# Keep original IDs for submission
test_ids = test_df["id"].values

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")
print(f"Classes: {author_mapping}")

# =============================================
# MODEL DESIGN (Step 2)
# =============================================

MODEL_NAME = "answerdotai/ModernBERT-large"
NUM_LABELS = 3
MAX_LENGTH = 512

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ModernBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

# Freeze lower layers to prevent catastrophic forgetting
for name, param in model.named_parameters():
    if "encoder.layer" in name:
        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
        if layer_num < 16:
            param.requires_grad = False

# Loss function with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = AdamW(
    params=[
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": 2e-5,
        }
    ],
    lr=2e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Learning rate schedule (will be updated with actual steps)
total_steps = 1000
warmup_steps = int(total_steps * 0.1)
warmup_scheduler = LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =============================================
# TRAINING AND EVALUATION (Step 3)
# =============================================

# Tokenization function
def tokenize_texts(texts, tokenizer, max_length=512):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

# Tokenize all data
print("Tokenizing training data...")
train_encodings = tokenize_texts(train_texts, tokenizer)
val_encodings = tokenize_texts(val_texts, tokenizer)
test_encodings = tokenize_texts(test_texts, tokenizer)

# Create DataLoaders
batch_size = 16

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

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# Training loop with mixed precision
scaler = GradScaler()
num_epochs = 5
gradient_accumulation_steps = 2
best_val_loss = float("inf")
patience = 3
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs.loss.item()

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    # Compute multi-class log loss
    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped /= val_preds_clipped.sum(axis=1, keepdims=True)
    val_log_loss = -np.mean(
        np.sum(np.eye(3)[val_true] * np.log(val_preds_clipped), axis=1)
    )

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
    )

    # Early stopping and model saving
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_model_state)
model.to(device)

# Final validation score
model.eval()
all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        all_val_preds.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_preds = np.concatenate(all_val_preds, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)

eps = 1e-15
val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_clipped /= val_preds_clipped.sum(axis=1, keepdims=True)
final_val_score = -np.mean(
    np.sum(np.eye(3)[val_true] * np.log(val_preds_clipped), axis=1)
)

# Test inference
model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        all_test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)

# Save submission
os.makedirs("./submission", exist_ok=True)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

submission_df.to_csv("./submission/submission_f5b4e284157a487aa71450cf71a1cb32.csv", index=False)
print(f"Submission saved to ./submission/submission_f5b4e284157a487aa71450cf71a1cb32.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_score}")
