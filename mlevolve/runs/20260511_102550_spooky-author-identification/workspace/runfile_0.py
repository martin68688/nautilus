import os
os.sched_setaffinity(0, {64, 59, 60, 62, 63})
import pandas as pd
import numpy as np
import re
import string
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])

# Split data into train/val
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    train_df["text"].values,
    train_df["author_encoded"].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

# ========== MODEL DESIGN ==========

NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 30
WARMUP_STEPS = 100

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)


class BiLSTMBlock(nn.Module):
    def __init__(self, hidden_size, lstm_hidden_size=256, dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(hidden_states)  # (batch, seq_len, lstm_hidden_size*2)
        # Mean pooling over sequence
        pooled = lstm_out.mean(dim=1)  # (batch, lstm_hidden_size*2)
        return self.dropout(pooled)


class AuthorClassifier(nn.Module):
    def __init__(self, base_model, num_classes=NUM_AUTHORS, dropout_rate=0.2):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.bilstm = BiLSTMBlock(hidden_size, lstm_hidden_size=256, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        lstm_output_size = 256 * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        # BiLSTM features with mean pooling
        lstm_features = self.bilstm(hidden_states)  # (batch, lstm_hidden_size*2)
        combined = self.dropout(lstm_features)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}


# Initialize model: free up base model's raw model for parameter grouping
# Unwrap if necessary
# Use the full base_model (with its pooling/classification head discarded automatically by AuthorClassifier)
model = AuthorClassifier(
    base_model,
    num_classes=NUM_AUTHORS,
    dropout_rate=0.2,
)

# Freeze backbone initially for warm-up
for param in model.base_model.parameters():
    param.requires_grad = False

# Single learning rate for all parameters (backbone will be frozen initially, then all params get same LR)
all_params = model.parameters()
optimizer = AdamW(
    all_params,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
)

total_training_steps = len(X_train_texts) // BATCH_SIZE * NUM_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_training_steps
)

scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters (initial): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(f"Device: {device}")

# ========== TRAINING & EVALUATION ==========


class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if self.labels is not None:
            return input_ids, attention_mask, self.labels[idx]
        return input_ids, attention_mask


train_dataset = AuthorshipDataset(X_train_texts, y_train)
val_dataset = AuthorshipDataset(X_val_texts, y_val)
test_dataset = AuthorshipDataset(test_df["text"].values)

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

model.train()
best_val_loss = float("inf")
best_val_score = float("inf")
patience = 8
patience_counter = 0
best_model_state = None
warmup_epochs = 3

for epoch in range(NUM_EPOCHS):
        # Unfreeze backbone after warm-up epochs
    if epoch == warmup_epochs:
        for param in model.base_model.parameters():
            param.requires_grad = True
        # Recreate optimizer with single LR for all parameters
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_training_steps
        )
        print("Backbone unfrozen - single learning rate applied")
    # Skip scheduler.step() during frozen epochs to avoid double-stepping
    if epoch < warmup_epochs:
        continue

    model.train()
    total_train_loss = 0
    train_batches = 0

    for batch in train_loader:
        input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_train_loss += loss.item()
        train_batches += 1

    avg_train_loss = total_train_loss / train_batches

    model.eval()
    val_loss = 0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            if scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=batch_labels,
                    )
                    loss = outputs["loss"]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs["loss"]

            val_loss += loss.item()
            val_batches += 1

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(batch_labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    eps = 1e-15
    val_preds = np.clip(val_preds, eps, 1 - eps)

    N = len(val_true)
    val_log_loss = 0
    for i in range(N):
        for j in range(NUM_AUTHORS):
            if val_true[i] == j:
                val_log_loss += np.log(val_preds[i, j])
    val_log_loss = -val_log_loss / N

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_log_loss:.4f}"
    )

    if val_log_loss < best_val_score:
        best_val_score = val_log_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)
model.eval()

# Final validation evaluation
all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

        if scaler is not None:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())
        all_val_labels.append(batch_labels.cpu().numpy())

val_preds = np.concatenate(all_val_preds, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)

eps = 1e-15
val_preds = np.clip(val_preds, eps, 1 - eps)

N = len(val_true)
final_val_log_loss = 0
for i in range(N):
    for j in range(NUM_AUTHORS):
        if val_true[i] == j:
            final_val_log_loss += np.log(val_preds[i, j])
final_val_log_loss = -final_val_log_loss / N

print(f"Final Transformer Validation Log-Loss: {final_val_log_loss:.6f}")

# --- Stacking: Train XGBoost meta-classifier on concatenated transformer probs + manual features ---
# We keep minimal manual features for stacking: basic counts and POS
from xgboost import XGBClassifier

def extract_minimal_features(text_series):
    """Minimal manual features for stacking"""
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)

    features["char_count"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["sentence_count"] = texts.str.count("[.!?]") + 1
    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    features["comma_count"] = texts.str.count(",")
    features["quote_count"] = texts.str.count('"') + texts.str.count("'")

    first_person = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"]
    third_person = ["he", "him", "his", "she", "her", "it", "its", "they", "them", "their"]
    features["first_person_count"] = texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(string.punctuation) in first_person)
    )
    features["third_person_count"] = texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(string.punctuation) in third_person)
    )
    features["all_caps_words"] = texts.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
    )
    return features

train_minimal_feats = extract_minimal_features(pd.Series(X_train_texts))
val_minimal_feats = extract_minimal_features(pd.Series(X_val_texts))
test_minimal_feats = extract_minimal_features(test_df["text"])

# Combine transformer probabilities with manual features
X_meta_train = np.concatenate([val_preds, val_minimal_feats.values], axis=1)
y_meta_train = val_true

# Train XGBoost meta-classifier
print("Training XGBoost meta-classifier...")
meta_clf = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=NUM_AUTHORS,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
)
meta_clf.fit(X_meta_train, y_meta_train)

# Generate test predictions for stacking
# First, get transformer test predictions again (already computed)
# We use the existing test_preds from earlier test inference (must be computed before stacking)
# Actually we haven't computed test_preds yet, so let's do it now
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        if scaler is not None:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())
test_preds_transformer = np.concatenate(all_test_preds, axis=0)

X_meta_test = np.concatenate([test_preds_transformer, test_minimal_feats.values], axis=1)
test_preds = meta_clf.predict_proba(X_meta_test)

# Ensure EAP->0, HPL->1, MWS->2 ordering from label encoder
# The label_encoder classes_ give us the order: we need to map to column names
# Let's check the order
print(f"Label encoder classes: {label_encoder.classes_}")
# In sample_submission, order is EAP, HPL, MWS
# We need to map properly
# Typically classes_ order may be ['EAP', 'HPL', 'MWS'] but let's handle generically
class_order = label_encoder.classes_  # e.g., ['EAP', 'HPL', 'MWS']
submission_cols = ['EAP', 'HPL', 'MWS']
# Create mapping dictionary
col_to_idx = {col: list(class_order).index(col) for col in submission_cols if col in class_order}
if len(col_to_idx) == 3:
    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "EAP": test_preds[:, col_to_idx['EAP']],
        "HPL": test_preds[:, col_to_idx['HPL']],
        "MWS": test_preds[:, col_to_idx['MWS']],
    })
else:
    # Fallback: assume order matches
    submission = pd.DataFrame(
        {
            "id": test_df["id"].values,
            "EAP": test_preds[:, 0],
            "HPL": test_preds[:, 1],
            "MWS": test_preds[:, 2],
        }
    )

# Normalize probabilities to sum to 1
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

eps = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], eps, 1 - eps)

submission.to_csv("./submission/submission_5077ac4839fa4ef8b70347b7b455d473.csv", index=False)

print(f"Final Validation Score: {final_val_log_loss}")
print("Submission saved to ./submission/submission_5077ac4839fa4ef8b70347b7b455d473.csv")