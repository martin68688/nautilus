import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
import os
import gc
from sklearn.metrics import log_loss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode target
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = train_df["author"].map(author_mapping).values

# Load tokenizer and model - ModernBERT-large
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize model with 3 output classes (EAP, HPL, MWS)
model = ModernBertForSequenceClassification.from_pretrained(model_id, num_labels=3)
model.to(device)

# Define loss function - CrossEntropyLoss with label smoothing to mitigate overconfidence
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixed precision scaler for efficient training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Maximum sequence length
max_length = 256

print(f"Model loaded: {model_id}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(f"Max sequence length: {max_length}")


# Custom Dataset for ModernBERT
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Download required NLTK data (will be cached after first run)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def get_wordnet_synonyms(word):
    """Get synonyms for a word using WordNet. Returns empty list if WordNet is unavailable."""
    try:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)
    except Exception:
        return []

class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Apply synonym replacement augmentation during training with probability 0.3
        if self.augment and np.random.random() < 0.3:
            tokens = text.split()
            augmented_tokens = []
            for token in tokens:
                # Skip stopwords and non-alpha tokens
                if token.lower() in stop_words or not token.isalpha():
                    augmented_tokens.append(token)
                elif np.random.random() < 0.15:
                    # Try to replace with a synonym
                    synonyms = get_wordnet_synonyms(token)
                    if synonyms:
                        # Use original capitalization style
                        if token[0].isupper():
                            synonym = np.random.choice(synonyms).capitalize()
                        else:
                            synonym = np.random.choice(synonyms).lower()
                        augmented_tokens.append(synonym)
                    else:
                        augmented_tokens.append(token)
                else:
                    augmented_tokens.append(token)
            text = ' '.join(augmented_tokens)

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


# Create train/validation split (stratified)
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.15, random_state=42, stratify=train_labels
)

train_texts = train_df.iloc[train_idx]["text"].values
train_labels_split = train_labels[train_idx]
val_texts = train_df.iloc[val_idx]["text"].values
val_labels_split = train_labels[val_idx]
test_texts = test_df["text"].values

# Create DataLoaders
batch_size = 32

train_dataset = AuthorDataset(
    texts=train_texts,
    labels=train_labels_split,
    tokenizer=tokenizer,
    max_length=max_length,
    augment=True,
)
val_dataset = AuthorDataset(
    texts=val_texts,
    labels=val_labels_split,
    tokenizer=tokenizer,
    max_length=max_length,
)
test_dataset = AuthorDataset(
    texts=test_texts, labels=None, tokenizer=tokenizer, max_length=max_length
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(
    f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
)

# Training Setup
num_epochs = 12
accumulation_steps = 2

# Define optimizer - AdamW with weight decay for regularization
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

# Define learning rate scheduler - Linear warmup then cosine decay per iteration
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.2 * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        # Linear warmup from 0 to 1
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # Cosine decay from 1 to eta_min/lr
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6 / 3e-5, 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

best_val_loss = float("inf")
best_epoch = -1
patience = 2
no_improve_count = 0

os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Training Loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / accumulation_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

        total_train_loss += loss.item() * accumulation_steps
        train_batches += 1

        # Step scheduler per iteration (not per epoch)
        if (batch_idx + 1) % accumulation_steps == 0:
            scheduler.step()

        del input_ids, attention_mask, labels, outputs, loss
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / train_batches

    # Validation phase
    model.eval()
    val_preds = []
    val_true = []
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())
            total_val_loss += loss.item()
            val_batches += 1

            del input_ids, attention_mask, labels, outputs, logits, probs, loss

    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)
    avg_val_loss = total_val_loss / val_batches

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

    val_log_loss = log_loss(val_true, val_preds_clipped)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_epoch = epoch
        no_improve_count = 0
        torch.save(model.state_dict(), "./working/models/best_model.pt")
        print(f"  → Saved best model (Val LogLoss: {val_log_loss:.6f})")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"  → Early stopping triggered after {epoch+1} epochs")
            break

    gc.collect()
    torch.cuda.empty_cache()

print(
    f"\nTraining complete. Best epoch: {best_epoch+1}, Best Val LogLoss: {best_val_loss:.6f}"
)

# Load best model and compute final validation metric
model.load_state_dict(torch.load("./working/models/best_model.pt"))
model.eval()

val_preds_final = []
val_true_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        val_preds_final.append(probs.cpu().numpy())
        val_true_final.append(labels.cpu().numpy())

        del input_ids, attention_mask, labels, outputs, logits, probs

val_preds_final = np.concatenate(val_preds_final, axis=0)
val_true_final = np.concatenate(val_true_final, axis=0)

eps = 1e-15
val_preds_clipped = np.clip(val_preds_final, eps, 1 - eps)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

final_val_score = log_loss(val_true_final, val_preds_clipped)

print(f"Final Validation Score: {final_val_score}")

# Test Inference and Submission
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

        del input_ids, attention_mask, outputs, logits, probs

test_preds = np.concatenate(test_preds, axis=0)

test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with {len(submission_df)} rows")
print(submission_df.head())