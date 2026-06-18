import os
os.sched_setaffinity(0, {18, 19, 20, 21, 22})
"""
Merged solution: Spooky Author Identification using ModernBERT
Combines data processing, model design, and training/evaluation into a single script.
"""

import os
import re
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import string
import joblib

# ========================
# CONFIGURATION
# ========================
MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
PATIENCE = 7
NUM_FOLDS = 5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# ========================
# 1. LOAD AND PREPARE DATA
# ========================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode labels
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)
author_mapping = dict(zip(label_encoder.classes_, range(num_classes)))
print(f"Author mapping: {author_mapping}")


# ========================
# 2. TEXT CLEANING
# ========================
def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\r", "")
    return text.strip()


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# ========================
# 3. STYLOMETRIC FEATURES (for potential use, but ModernBERT will use raw text)
# ========================
def extract_stylometric_features(df, text_col="clean_text"):
    """Extract handcrafted features. These can be used as auxiliary features."""
    features = pd.DataFrame(index=df.index)

    features["char_count"] = df[text_col].str.len()
    features["word_count"] = df[text_col].str.split().str.len()
    features["sentence_count"] = df[text_col].str.split(r"[.!?]+").str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)

    punct_counts = df[text_col].apply(
        lambda x: Counter(c for c in x if c in string.punctuation)
    )
    features["exclamation_count"] = punct_counts.apply(lambda c: c.get("!", 0))
    features["question_count"] = punct_counts.apply(lambda c: c.get("?", 0))
    features["period_count"] = punct_counts.apply(lambda c: c.get(".", 0))
    features["comma_count"] = punct_counts.apply(lambda c: c.get(",", 0))
    features["total_punct"] = features[
        [c for c in features.columns if "count" in c]
    ].sum(axis=1)
    features["punct_ratio"] = features["total_punct"] / (features["char_count"] + 1)

    features["vowel_count"] = df[text_col].apply(
        lambda x: sum(1 for c in x if c in "aeiou")
    )
    features["consonant_ratio"] = 0

    return features


# We'll keep the stylometric features as potential auxiliary inputs
train_stylo = extract_stylometric_features(train_df)
test_stylo = extract_stylometric_features(test_df)

# Handle NaNs/Infs in stylometric features
train_stylo = train_stylo.replace([np.inf, -np.inf], np.nan).fillna(0)
test_stylo = test_stylo.replace([np.inf, -np.inf], np.nan).fillna(0)

# Standardize stylometric features
scaler = StandardScaler()
train_stylo_scaled = scaler.fit_transform(train_stylo)
test_stylo_scaled = scaler.transform(test_stylo)

# ========================
# 4. TOKENIZER AND DATASET
# ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, stylo_features=None):
        self.texts = texts
        self.labels = labels
        self.stylo_features = stylo_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.stylo_features is not None:
            item["stylo_features"] = torch.FloatTensor(self.stylo_features[idx])

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ========================
# 5. MODEL DEFINITION
# ========================
class CrossAttentionFusion(nn.Module):
    """Gated cross-attention fusion between BERT output and stylometric features."""

    def __init__(self, hidden_size, stylo_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.stylo_projection = nn.Linear(stylo_dim, hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_sigmoid = nn.Sigmoid()
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bert_pooled, stylo_features):
        # Project stylo features to BERT hidden size
        stylo_proj = self.stylo_projection(stylo_features)  # [batch, hidden]
        # Reshape for cross-attention: need seq_len dimension (add dummy seq dim)
        stylo_proj = stylo_proj.unsqueeze(1)  # [batch, 1, hidden]
        bert_pooled_seq = bert_pooled.unsqueeze(1)  # [batch, 1, hidden]

        # Cross-attention: BERT pooled output as query, stylo features as key/value
        attended, _ = self.cross_attention(
            query=bert_pooled_seq,
            key=stylo_proj,
            value=stylo_proj,
        )
        attended = attended.squeeze(1)  # [batch, hidden]

        # Gated residual connection
        gate_input = torch.cat([bert_pooled, attended], dim=-1)
        gate = self.gate_sigmoid(self.gate(gate_input))
        fused = gate * bert_pooled + (1 - gate) * attended
        fused = self.output_norm(fused)
        fused = self.dropout(fused)
        return fused


class ModernBERTClassifier(nn.Module):
    """ModernBERT with classification head and gated cross-attention fusion."""

    def __init__(self, num_labels=3, stylo_dim=0):
        super().__init__()
        self.bert = ModernBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels
        )
        self.num_labels = num_labels
        self.stylo_dim = stylo_dim

        # If we have stylometric features, add gated cross-attention fusion
        if stylo_dim > 0:
            hidden_size = self.bert.config.hidden_size
            self.cross_attention_fusion = CrossAttentionFusion(
                hidden_size=hidden_size, stylo_dim=stylo_dim, num_heads=4, dropout=0.1
            )

    def forward(self, input_ids, attention_mask=None, stylo_features=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get the pooled output from ModernBERT
        logits = outputs.logits

        if stylo_features is not None and self.stylo_dim > 0:
            # Get pooled representation
            pooled = (
                outputs.pooler_output
                if hasattr(outputs, "pooler_output")
                else outputs.hidden_states[-1][:, 0, :]
            )
            # Apply gated cross-attention fusion
            fused = self.cross_attention_fusion(pooled, stylo_features)
            logits = self.bert.classifier(fused)

        return logits


# ========================
# 6. TRAINING FUNCTION
# ========================
def train_model(model, train_loader, val_loader, epochs=NUM_EPOCHS):
    """Train the model with differential learning rates, warmup, and early stopping."""

    # Separate parameters into two groups for differential learning rates
    # Group 1: BERT backbone (lower lr)
    # Group 2: Fusion layers and classifier (higher lr)
    bert_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bert" in name:
                bert_params.append(param)
            else:
                fusion_params.append(param)

    optimizer = AdamW([
        {"params": bert_params, "lr": 1e-5, "weight_decay": 0.01},
        {"params": fusion_params, "lr": 2e-5, "weight_decay": 0.01},
    ])

    total_steps = len(train_loader) * epochs

    # Linear warmup for first 5 epochs, then cosine decay
    warmup_epochs = 5
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    global_step = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            stylo = batch.get("stylo_features", None)
            if stylo is not None:
                stylo = stylo.to(device)

            optimizer.zero_grad()

            with autocast():
                logits = model(
                    input_ids, attention_mask=attention_mask, stylo_features=stylo
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                stylo = batch.get("stylo_features", None)
                if stylo is not None:
                    stylo = stylo.to(device)

                logits = model(
                    input_ids, attention_mask=attention_mask, stylo_features=stylo
                )
                loss = criterion(logits, labels)

                val_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_val_preds.append(probs)
                all_val_labels.append(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Compute log loss
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)

        all_val_preds_clipped = np.clip(all_val_preds, 1e-15, 1 - 1e-15)
        all_val_preds_clipped = all_val_preds_clipped / all_val_preds_clipped.sum(
            axis=1, keepdims=True
        )

        val_logloss = -np.mean(
            np.sum(
                np.eye(num_classes)[all_val_labels] * np.log(all_val_preds_clipped),
                axis=1,
            )
        )

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping with increased patience to 7
        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            torch.save(best_model_state, "./working/best_model_3b99ab91734c4358b00217288b583f13.pt")
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load("./working/best_model_3b99ab91734c4358b00217288b583f13.pt"))
    model.to(device)

    return model, best_val_loss


# ========================
# 7. MAIN EXECUTION
# ========================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Create stratified folds
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
train_texts = train_df["clean_text"].values
train_labels = train_df["author_encoded"].values

# Use first fold for validation
for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    if fold == 0:
        X_train_texts, X_val_texts = train_texts[train_idx], train_texts[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        stylo_train = train_stylo_scaled[train_idx]
        stylo_val = train_stylo_scaled[val_idx]
        break

print(f"Train samples: {len(X_train_texts)}, Val samples: {len(X_val_texts)}")

# Create datasets and dataloaders
train_dataset = TextDataset(X_train_texts.tolist(), y_train.tolist(), stylo_train)
val_dataset = TextDataset(X_val_texts.tolist(), y_val.tolist(), stylo_val)
test_dataset = TextDataset(
    test_df["clean_text"].tolist(), stylo_features=test_stylo_scaled
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# Initialize model
model = ModernBERTClassifier(
    num_labels=num_classes, stylo_dim=train_stylo_scaled.shape[1]
)
model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train model
model, best_val_logloss = train_model(
    model, train_loader, val_loader, epochs=NUM_EPOCHS
)

# ========================
# 8. TEST INFERENCE
# ========================
model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo = batch.get("stylo_features", None)
        if stylo is not None:
            stylo = stylo.to(device)

        logits = model(input_ids, attention_mask=attention_mask, stylo_features=stylo)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_preds.append(probs)

all_test_preds = np.concatenate(all_test_preds, axis=0)

# Clip and normalize as per competition rules
all_test_preds = np.clip(all_test_preds, 1e-15, 1 - 1e-15)
all_test_preds = all_test_preds / all_test_preds.sum(axis=1, keepdims=True)

# ========================
# 9. SAVE SUBMISSION
# ========================
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)

submission.to_csv("./submission/submission_3b99ab91734c4358b00217288b583f13.csv", index=False)
print(f"Submission saved to ./submission/submission_3b99ab91734c4358b00217288b583f13.csv")
print(f"Submission shape: {submission.shape}")

# ========================
# 10. CLEANUP
# ========================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {best_val_logloss:.6f}")