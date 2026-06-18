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

import random
import re

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

# Enhanced text augmentation
def augment_text(text: str) -> str:
    """Apply exactly one random augmentation per training sample."""
    words = text.split()
    aug_type = random.random()

    # Random word swaps within 3-word windows at 10% probability
    if aug_type < 0.10 and len(words) >= 4:
        window_start = random.randint(0, len(words) - 4)
        idx1 = window_start + random.randint(0, 2)
        idx2 = idx1 + 1
        if idx2 < len(words):
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    # Random punctuation insertion at sentence boundaries at 15% probability
    elif aug_type < 0.25:
        punct_choices = ['--', ';', ':', '...']
        punct = random.choice(punct_choices)
        # Find sentence boundaries (period, exclamation, question mark)
        sentence_ends = [i for i, c in enumerate(text) if c in '.!?']
        if sentence_ends:
            insert_pos = random.choice(sentence_ends)
            # Insert after the sentence end
            if insert_pos + 1 < len(text):
                text = text[:insert_pos+1] + ' ' + punct + text[insert_pos+1:]
            else:
                text = text + ' ' + punct
        return text

    # Character-level vowel noise: 5% of vowels replaced
    else:
        vowels = 'aeiouAEIOU'
        chars = list(text)
        vowel_indices = [i for i, c in enumerate(chars) if c in vowels]
        if vowel_indices:
            num_to_replace = max(1, int(len(vowel_indices) * 0.05))
            replace_indices = random.sample(vowel_indices, min(num_to_replace, len(vowel_indices)))
            for idx in replace_indices:
                if chars[idx].islower():
                    chars[idx] = random.choice('aeiou')
                else:
                    chars[idx] = random.choice('AEIOU')
        return ''.join(chars)

# Apply augmentation to training texts
print("Applying text augmentation...")
train_texts_aug = train_texts.apply(augment_text)
# Use augmented texts for training
train_texts = train_texts_aug

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

# Multi-head attention pooling module
class CustomModernBertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = ModernBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.bert.config.hidden_dropout_prob = 0.1
        self.bert.config.attention_probs_dropout_prob = 0.1

        hidden_size = self.bert.config.hidden_size
        # Use a simple mean pooling + MLP head for stability
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]

        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels)

        return type('Output', (), {'loss': loss, 'logits': logits})()

    def freeze_bert_layers(self):
        """Freeze all BERT layers."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_next_layers(self, num_layers=4):
        """Unfreeze the top num_layers of BERT."""
        for name, param in self.bert.named_parameters():
            if 'classifier' in name or 'pooler' in name:
                param.requires_grad = True
        if hasattr(self.bert, 'bert') and hasattr(self.bert.bert, 'encoder'):
            layers = self.bert.bert.encoder.layer
            num_layers_total = len(layers)
            start_layer = max(0, num_layers_total - num_layers)
            for i in range(start_layer, num_layers_total):
                for param in layers[i].parameters():
                    param.requires_grad = True

# Initialize tokenizer and custom model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = CustomModernBertForSequenceClassification(MODEL_NAME, NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Initially freeze BERT layers
model.freeze_bert_layers()
# Ensure new layers are trainable - classifier is the only new module
for param in model.classifier.parameters():
    param.requires_grad = True

# Separate param groups for differential learning rates
classifier_params = list(model.classifier.parameters())
bert_params = [p for n, p in model.bert.named_parameters() if p.requires_grad]

optimizer = AdamW(
    [
        {'params': classifier_params, 'lr': 1e-4, 'weight_decay': WEIGHT_DECAY},
        {'params': bert_params, 'lr': 5e-5, 'weight_decay': WEIGHT_DECAY},
    ],
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

# Scheduler - Cosine annealing warm restarts
total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2, T_mult=2, eta_min=1e-6
)

scaler = GradScaler()

# Training loop
best_val_loss = float("inf")
best_epoch = -1
no_improve_epochs = 0
early_stop_patience = 3

for epoch in range(NUM_EPOCHS):
    # Progressive unfreezing
    if epoch == 2:
        model.unfreeze_next_layers(num_layers=4)
        # Rebuild optimizer with updated trainable parameters
        all_trainable = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamW(
            all_trainable,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=2, eta_min=1e-6
        )

    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with torch.amp.autocast('cuda'):
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
            with torch.amp.autocast('cuda'):
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

    # Check for NaN and handle it
    if np.isnan(val_preds).any():
        print(f"WARNING: NaN detected in predictions at epoch {epoch+1}, replacing with zeros")
        val_preds = np.nan_to_num(val_preds, nan=0.0)
        # Add small constant to avoid zero probabilities
        val_preds = val_preds + 1e-15
        row_sums = val_preds.sum(axis=1, keepdims=True)
        val_preds = val_preds / row_sums

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
        torch.save(model.state_dict(), "./working/best_model_custom.pt")
        model.bert.save_pretrained("./working/best_model")
        tokenizer.save_pretrained("./working/best_tokenizer")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

print(f"Best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

# Load best model and compute final validation score
best_model = CustomModernBertForSequenceClassification(MODEL_NAME, NUM_CLASSES)
if os.path.exists("./working/best_model_custom.pt"):
    best_model.load_state_dict(torch.load("./working/best_model_custom.pt", map_location=device))
else:
    best_model.load_state_dict(model.state_dict())
best_model.to(device)
best_model.eval()

val_preds_best = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        with torch.amp.autocast('cuda'):
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        val_preds_best.append(probs.cpu().numpy())

val_preds_best = np.concatenate(val_preds_best, axis=0)
# Handle any potential NaN in best model predictions
if np.isnan(val_preds_best).any():
    val_preds_best = np.nan_to_num(val_preds_best, nan=0.0)
    val_preds_best = val_preds_best + 1e-15
    row_sums = val_preds_best.sum(axis=1, keepdims=True)
    val_preds_best = val_preds_best / row_sums

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
        with torch.amp.autocast('cuda'):
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)

# Ensure test predictions are valid
if np.isnan(test_preds).any():
    test_preds = np.nan_to_num(test_preds, nan=0.0)
    test_preds = test_preds + 1e-15
    row_sums = test_preds.sum(axis=1, keepdims=True)
    test_preds = test_preds / row_sums

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