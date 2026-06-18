import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import (
    AutoTokenizer,
    ModernBertModel,
    BertTokenizer,
)
import os
import pickle
import re
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ================================================================
# DATA LOADING AND SPLIT
# ================================================================

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

os.makedirs("./working", exist_ok=True)

# Encode target
le = LabelEncoder()
le.fit(train_df["author"])
train_df["author_encoded"] = le.transform(train_df["author"])

# Create stratified train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df["text"], train_df["author_encoded"]))

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print(
    f"Train size: {len(train_split)}, Val size: {len(val_split)}, Test size: {len(test_df)}"
)

# Save indices and splits
np.save("./working/train_indices.npy", train_idx)
np.save("./working/val_indices.npy", val_idx)
train_split.to_csv("./working/train_preprocessed.csv", index=False)
val_split.to_csv("./working/val_preprocessed.csv", index=False)

# ================================================================
# PRECOMPUTE TF-IDF FEATURES
# ================================================================

print("Fitting TF-IDF vectorizer on training data...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
train_tfidf = tfidf_vectorizer.fit_transform(train_split["text"].values)
val_tfidf = tfidf_vectorizer.transform(val_split["text"].values)
test_tfidf = tfidf_vectorizer.transform(test_df["text"].values)

print("Reducing TF-IDF dimensions with TruncatedSVD...")
svd = TruncatedSVD(n_components=256, random_state=42)
train_tfidf_svd = svd.fit_transform(train_tfidf)
val_tfidf_svd = svd.transform(val_tfidf)
test_tfidf_svd = svd.transform(test_tfidf)

print(f"TF-IDF SVD shapes - Train: {train_tfidf_svd.shape}, Val: {val_tfidf_svd.shape}, Test: {test_tfidf_svd.shape}")

# Store as dict keyed by id for easy retrieval
def build_tfidf_dict(ids, features):
    return {id_val: feat for id_val, feat in zip(ids, features)}

train_tfidf_dict = build_tfidf_dict(train_split["id"].values, train_tfidf_svd)
val_tfidf_dict = build_tfidf_dict(val_split["id"].values, val_tfidf_svd)
test_tfidf_dict = build_tfidf_dict(test_df["id"].values, test_tfidf_svd)

# Save for reuse
np.save("./working/train_tfidf_features.npy", train_tfidf_svd)
np.save("./working/val_tfidf_features.npy", val_tfidf_svd)
np.save("./working/test_tfidf_features.npy", test_tfidf_svd)

del train_tfidf, val_tfidf, test_tfidf

# ================================================================
# MODEL DESIGN - ModernBERT with TF-IDF Stylometric Fusion
# ================================================================

MODEL_ID = "answerdotai/ModernBERT-large"
NUM_LABELS = 3
MAX_LENGTH = 512  # first-window-only, use full 512 context

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading ModernBERT-large backbone...")
bert_backbone = ModernBertModel.from_pretrained(MODEL_ID)
bert_backbone.config.hidden_dropout_prob = 0.3
bert_backbone.config.attention_probs_dropout_prob = 0.2

# Freeze backbone initially for progressive unfreezing
for param in bert_backbone.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# TEXT AUGMENTATION FUNCTION
# ================================================================

def augment_text_segment(text):
    """Apply a single random augmentation to a text segment with probability 0.5."""
    if random.random() > 0.5:
        return text
    aug_type = random.choice(['word_swap', 'punctuation_insert', 'vowel_noise', 'synonym_replace'])
    words = text.split()
    if not words:
        return text
    if aug_type == 'word_swap' and len(words) >= 2:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    elif aug_type == 'punctuation_insert':
        punct = random.choice(['.', ',', '!', '?', ';', ':'])
        pos = random.randint(0, len(text))
        return text[:pos] + punct + text[pos:]
    elif aug_type == 'vowel_noise':
        vowels = 'aeiouAEIOU'
        chars = list(text)
        for i in range(len(chars)):
            if chars[i] in vowels and random.random() < 0.1:
                chars[i] = random.choice(vowels)
        return ''.join(chars)
    elif aug_type == 'synonym_replace' and len(words) >= 1:
        # Simple synonym mapping (very basic, just swaps some common words)
        synonym_map = {
            'good': 'fine', 'bad': 'poor', 'big': 'large', 'small': 'tiny',
            'happy': 'glad', 'sad': 'unhappy', 'said': 'spoke', 'went': 'walked',
            'very': 'quite', 'much': 'many', 'could': 'might', 'would': 'shall',
        }
        word_idx = random.randint(0, len(words)-1)
        word_lower = words[word_idx].lower().strip('.,!?;:')
        if word_lower in synonym_map:
            old_word = words[word_idx]
            # Preserve case
            if old_word[0].isupper():
                words[word_idx] = synonym_map[word_lower].capitalize()
            else:
                words[word_idx] = synonym_map[word_lower]
        return ' '.join(words)
    return text

# ================================================================
# MODEL CLASS - TextFeatureEncoder
# ================================================================

class TextFeatureEncoder(nn.Module):
    def __init__(self, backbone, num_labels=3):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = backbone.config.hidden_size  # 768

        # Stylometric MLP encoder for TF-IDF features (256-d input)
        self.stylometric_encoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # MLP classifier: 768 (BERT [CLS]) + 16 (stylometric) -> 512 -> 3
        total_features = self.hidden_size + 16
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.stylometric_encoder, self.classifier]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def forward(self, input_ids, attention_mask, tfidf_features):
        # input_ids: (batch, seq_len)
        # attention_mask: (batch, seq_len)
        # tfidf_features: (batch, 256)

        # Get [CLS] embedding from backbone (first-window-only)
        with autocast('cuda'):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Stylometric features
        stylo = self.stylometric_encoder(tfidf_features)  # (batch, 16)

        # Concatenate and classify
        combined = torch.cat([pooled, stylo], dim=1)  # (batch, 784)
        logits = self.classifier(combined)

        return logits

from transformers import get_linear_schedule_with_warmup

bert_backbone.to(device)
model = TextFeatureEncoder(bert_backbone, num_labels=NUM_LABELS)
model.to(device)
print(f"Model loaded on {device}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer with two param groups: backbone (frozen initially) and new layers
backbone_params = []
new_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'backbone' not in name:
            new_params.append(param)
    else:
        if 'backbone' in name:
            backbone_params.append(param)

optimizer = AdamW([
    {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.01},
    {'params': new_params, 'lr': 1e-4, 'weight_decay': 0.05},
], lr=2e-5, eps=1e-8)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters initially (new layers only): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Backbone lr: 2e-5, New layers lr: 1e-4, weight_decay: backbone=0.01, new=0.05")

# ================================================================
# DATASET CLASS
# ================================================================


class TextDataset(Dataset):
    def __init__(self, texts, ids, labels=None, tokenizer=None,
                 max_length=512, is_training=False, tfidf_dict=None):
        self.texts = texts
        self.ids = ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.tfidf_dict = tfidf_dict if tfidf_dict is not None else {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text_id = self.ids[idx]

        # Apply augmentation for training
        if self.is_training:
            text = augment_text_segment(text)

        # Tokenize with first-window-only (truncation to max_length)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        # Get precomputed TF-IDF features
        tfidf_feat = self.tfidf_dict.get(text_id, np.zeros(256, dtype=np.float32))
        tfidf_tensor = torch.tensor(tfidf_feat, dtype=torch.float32)

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "tfidf_features": tfidf_tensor,
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ================================================================
# CREATE DATALOADERS
# ================================================================

train_labels = train_split["author_encoded"].values
val_labels = val_split["author_encoded"].values

train_dataset = TextDataset(
    train_split["text"].values, train_split["id"].values, train_labels, tokenizer,
    max_length=MAX_LENGTH, is_training=True, tfidf_dict=train_tfidf_dict
)
val_dataset = TextDataset(
    val_split["text"].values, val_split["id"].values, val_labels, tokenizer,
    max_length=MAX_LENGTH, is_training=False, tfidf_dict=val_tfidf_dict
)
test_dataset = TextDataset(
    test_df["text"].values, test_df["id"].values, labels=None, tokenizer=tokenizer,
    max_length=MAX_LENGTH, is_training=False, tfidf_dict=test_tfidf_dict
)

batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ================================================================
# TRAINING SETUP
# ================================================================

num_epochs = 10
gradient_accumulation_steps = 2

# Linear scheduler with warmup
total_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
warmup_steps = 100
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)

scaler = GradScaler('cuda')
best_val_loss = float("inf")
best_model_path = "./working/best_model.pt"
patience = 3
patience_counter = 0

def unfreeze_backbone():
    """Unfreeze backbone parameters for fine-tuning."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    # Re-add backbone params to optimizer if not already there
    print("Backbone unfrozen for fine-tuning.")

print(f"Starting training for up to {num_epochs} epochs with early stopping (patience={patience})...")
print("Epoch 1: Backbone frozen, training stylometric encoder and classifier only.")
print("After epoch 1: Backbone unfrozen for full fine-tuning.")

# ================================================================
# TRAINING LOOP
# ================================================================

for epoch in range(num_epochs):
    # Progressive unfreezing: unfreeze backbone after first epoch
    if epoch == 1:
        unfreeze_backbone()
        # Reinitialize optimizer with both groups (backbone now requires_grad)
        backbone_params = []
        new_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    new_params.append(param)
        optimizer = AdamW([
            {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.01},
            {'params': new_params, 'lr': 1e-4, 'weight_decay': 0.05},
        ], lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        print(f"Optimizer reinitialized with backbone unfrozen. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        tfidf_features = batch["tfidf_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast('cuda'):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tfidf_features=tfidf_features,
            )
            loss = criterion(logits, labels) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            tfidf_features = batch["tfidf_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tfidf_features=tfidf_features,
                )
                loss = criterion(logits, labels)

            total_val_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert labels to one-hot for log loss computation
    all_labels_onehot = np.zeros((len(all_labels), NUM_LABELS))
    all_labels_onehot[np.arange(len(all_labels)), all_labels] = 1

    eps_clip = 1e-15
    all_preds = np.clip(all_preds, eps_clip, 1 - eps_clip)
    log_loss_val = -np.mean(np.sum(all_labels_onehot * np.log(all_preds), axis=1))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {log_loss_val:.4f} [SAVED]"
        )
    else:
        patience_counter += 1
        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {log_loss_val:.4f} (patience {patience_counter}/{patience})"
        )

    if patience_counter >= patience:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# ================================================================
# FINAL EVALUATION ON BEST MODEL
# ================================================================

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds_val = []
all_labels_val = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        tfidf_features = batch["tfidf_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tfidf_features=tfidf_features,
            )
            probs = torch.softmax(logits, dim=-1)
        all_preds_val.append(probs.cpu().numpy())
        all_labels_val.append(labels.cpu().numpy())

all_preds_val = np.concatenate(all_preds_val, axis=0)
all_labels_val = np.concatenate(all_labels_val, axis=0)
all_labels_onehot = np.zeros((len(all_labels_val), NUM_LABELS))
all_labels_onehot[np.arange(len(all_labels_val)), all_labels_val] = 1
eps_clip = 1e-15
all_preds_val = np.clip(all_preds_val, eps_clip, 1 - eps_clip)
val_log_loss = -np.mean(np.sum(all_labels_onehot * np.log(all_preds_val), axis=1))

# ================================================================
# TEST INFERENCE
# ================================================================

all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        tfidf_features = batch["tfidf_features"].to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tfidf_features=tfidf_features,
            )
            probs = torch.softmax(logits, dim=-1)
        all_test_preds.append(probs.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds, axis=0)
# Clip test predictions before normalization
eps_clip = 1e-15
all_test_preds = np.clip(all_test_preds, eps_clip, 1 - eps_clip)
all_test_preds = all_test_preds / all_test_preds.sum(axis=1, keepdims=True)

# ================================================================
# SUBMISSION
# ================================================================

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {val_log_loss}")