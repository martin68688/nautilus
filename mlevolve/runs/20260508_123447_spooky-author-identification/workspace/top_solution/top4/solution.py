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
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, ModernBertModel, AutoConfig
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import string

# ========================
# CONFIGURATION
# ========================
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
PATIENCE = 3
NUM_FOLDS = 5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')

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


# Character substitution mapping for augmentation
CHAR_SUBSTITUTION_MAP = {
    'a': ['ä', 'à', 'á', 'â', 'ã', 'å'],
    'e': ['é', 'è', 'ê', 'ë', 'ē', 'ė', 'ę'],
    'i': ['ï', 'î', 'ì', 'í', 'ī'],
    'o': ['ö', 'ô', 'ò', 'ó', 'õ', 'ø'],
    'u': ['ü', 'û', 'ù', 'ú', 'ū'],
    'c': ['ç', 'ć', 'č'],
    'n': ['ñ', 'ń'],
    's': ['ß', 'ś', 'š'],
    'y': ['ý', 'ÿ'],
    'A': ['Ä', 'À', 'Á', 'Â', 'Ã', 'Å'],
    'E': ['É', 'È', 'Ê', 'Ë', 'Ē'],
    'I': ['Ï', 'Î', 'Ì', 'Í', 'Ī'],
    'O': ['Ö', 'Ô', 'Ò', 'Ó', 'Õ', 'Ø'],
    'U': ['Ü', 'Û', 'Ù', 'Ú', 'Ū'],
    'C': ['Ç', 'Ć', 'Č'],
    'N': ['Ñ', 'Ń'],
    'S': ['Ś', 'Š'],
    'Y': ['Ý', 'Ÿ'],
}

def apply_char_augmentation(text, prob=0.5, replace_ratio=0.05, rng=None):
    """Apply character-level substitution augmentation."""
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() >= prob:
        return text

    chars = list(text)
    n_chars = len(chars)
    n_replace = max(1, int(n_chars * replace_ratio))

    # Find indices of characters that have substitutions
    replaceable_indices = []
    for i, c in enumerate(chars):
        if c in CHAR_SUBSTITUTION_MAP:
            replaceable_indices.append(i)

    if len(replaceable_indices) == 0:
        return text

    # Randomly select indices to replace
    n_replace = min(n_replace, len(replaceable_indices))
    indices_to_replace = rng.choice(replaceable_indices, size=n_replace, replace=False)

    for idx in indices_to_replace:
        original_char = chars[idx]
        substitutions = CHAR_SUBSTITUTION_MAP[original_char]
        chars[idx] = rng.choice(substitutions)

    return ''.join(chars)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, stylo_features=None, augment=False):
        self.texts = texts
        self.labels = labels
        self.stylo_features = stylo_features
        self.augment = augment and labels is not None  # Only augment during training
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Apply character-level augmentation during training
        if self.augment:
            text = apply_char_augmentation(text, rng=self.rng)

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
class ModernBERTClassifier(nn.Module):
    """ModernBERT with simple pooling and classification head."""

    def __init__(self, num_labels=3, stylo_dim=0):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.output_hidden_states = False
        config.output_attentions = False
        self.bert = ModernBertModel.from_pretrained(MODEL_NAME, config=config)
        self.num_labels = num_labels
        self.stylo_dim = stylo_dim
        self.hidden_size = self.bert.config.hidden_size

        # Simple classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, stylo_features=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ========================
# 6. TRAINING FUNCTION
# ========================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_model(model, train_loader, val_loader, epochs=NUM_EPOCHS):
    """Train the model with early stopping and label smoothing."""

    # Discriminative learning rates: backbone (low lr), classifier head (high lr)
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': classifier_params, 'lr': 1e-4}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda")

    criterion = FocalLoss(gamma=2.0)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda"):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

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
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                logits = model(input_ids, attention_mask=attention_mask)
                loss = nn.functional.cross_entropy(logits, labels)

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
        all_val_preds_clipped = all_val_preds_clipped / all_val_preds_clipped.sum(axis=1, keepdims=True)

        val_logloss = -np.mean(
            np.sum(
                np.eye(num_classes)[all_val_labels] * np.log(all_val_preds_clipped),
                axis=1,
            )
        )

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping
        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            torch.save(best_model_state, "./working/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load("./working/best_model.pt", weights_only=True))
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
train_dataset = TextDataset(X_train_texts.tolist(), y_train.tolist(), None, augment=True)
val_dataset = TextDataset(X_val_texts.tolist(), y_val.tolist(), None, augment=False)
test_dataset = TextDataset(
    test_df["clean_text"].tolist(), stylo_features=None, augment=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# Initialize model
model = ModernBERTClassifier(num_labels=num_classes, stylo_dim=0)
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

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

# ========================
# 10. CLEANUP
# ========================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {best_val_logloss:.6f}")