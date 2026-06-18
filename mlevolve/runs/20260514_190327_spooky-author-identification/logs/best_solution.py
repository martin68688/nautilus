import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size + 2, num_authors)  # +2 for extra features
        # Multi-head attention pooling: 8 heads, each head dimension = hidden_size/8
        self.num_heads = 8
        self.head_dim = hidden_size // self.num_heads
        assert hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        # Learnable query tensor
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        # MultiheadAttention: key and value come from mean-pooled layers (4 layers)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.num_heads,
            batch_first=False,
        )
        # LayerNorm for residual connection
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask, extra_features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract all hidden states from all layers
        all_hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)
        # Take the last 4 layers (excluding the embedding layer which is the first element)
        last_4_layers = all_hidden_states[-4:]  # list of 4 tensors

        # Apply mean pooling (masked) to each of the last 4 layers
        masked_pooled = []
        for layer_hidden in last_4_layers:
            # Expand attention_mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            # Zero out padding tokens and sum
            masked_sum = (layer_hidden * mask_expanded).sum(dim=1)  # (batch, hidden_size)
            # Count non-padding tokens
            token_counts = mask_expanded.sum(dim=1) + 1e-10  # (batch, 1)
            # Compute mean
            mean_pooled = masked_sum / token_counts  # (batch, hidden_size)
            masked_pooled.append(mean_pooled)

        # Stack pooled representations: (batch, 4, hidden_size)
        stacked_pooled = torch.stack(masked_pooled, dim=1)  # (batch, 4, hidden_size)

        # Multi-head attention pooling
        # Permute to (seq_len=4, batch, hidden_size) for batch_first=False
        stacked_pooled_t = stacked_pooled.permute(1, 0, 2)  # (4, batch, hidden_size)
        # Expand query to match batch size: (1, batch, hidden_size)
        query_t = self.query.expand(-1, stacked_pooled_t.size(1), -1)  # (1, batch, hidden_size)
        # Apply multihead attention: query = learned query, key=value=mean-pooled layers
        attn_output, _ = self.multihead_attn(
            query=query_t,
            key=stacked_pooled_t,
            value=stacked_pooled_t,
        )  # attn_output: (1, batch, hidden_size)
        # Squeeze sequence dimension: (batch, hidden_size)
        attn_output = attn_output.squeeze(0)  # (batch, hidden_size)

        # Global mean pool of last layer
        last_layer_hidden = all_hidden_states[-1]  # (batch, seq_len, hidden_size)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_sum = (last_layer_hidden * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1) + 1e-10
        global_mean_pool = masked_sum / token_counts  # (batch, hidden_size)

        # Residual connection: add global mean pool to attention output
        pooled_output = self.layer_norm(attn_output + global_mean_pool)  # (batch, hidden_size)

        # Concatenate extra features if provided
        if extra_features is not None:
            pooled_output = torch.cat([pooled_output, extra_features], dim=1)  # (batch, hidden_size+2)

        logits = self.head(pooled_output)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

# Focal Loss with class weights
class_weights = torch.tensor([0.42, 0.31, 0.27], device=device)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2.0, weight=class_weights)

# Collect all backbone params initially frozen (progressive unfreezing)
# All backbone params start frozen, head is always trainable
head_params = list(model.head.parameters())

# Define unfreeze milestones: after 2 epochs unfreeze last 8, after 4 last 12, after 6 last 16
NUM_LAYERS = len(model.backbone.deberta.encoder.layer)

def get_unfrozen_params(model, num_unfrozen_layers):
    """Return parameters from the last num_unfrozen_layers of the backbone (excluding bias/LayerNorm)"""
    params = []
    if num_unfrozen_layers > 0:
        layers_to_unfreeze = model.backbone.deberta.encoder.layer[-num_unfrozen_layers:]
        for layer in layers_to_unfreeze:
            for name, param in layer.named_parameters():
                if "bias" not in name and "LayerNorm" not in name:
                    params.append(param)
    return params

# Initially unfreeze last 8 layers (same as before)
initial_unfrozen = get_unfrozen_params(model, 8)
for param in initial_unfrozen:
    param.requires_grad = True

optimizer = AdamW(
    [
        {
            "params": initial_unfrozen,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Ensure optimizer param groups are correctly ordered
# Group 0: backbone unfrozen layers (lr=2e-5)
# Group 1: head (lr=5e-5)

print(f"Backbone unfrozen params (initial): {sum(p.numel() for p in initial_unfrozen):,}")
print(f"Head params: {sum(p.numel() for p in head_params):,}")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Compute text length (characters) and punctuation count
        text_length = len(text)
        punctuation_count = sum(1 for ch in text if ch in '. , ! ? ; :')
        extra_features = torch.tensor([text_length / 1000.0, punctuation_count / 100.0], dtype=torch.float)
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
            "extra_features": extra_features,
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Get original texts for training
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Use previously computed indices for train/validation split
train_indices = train_idx
val_indices = val_idx

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length
)
val_dataset = SpookyDataset(val_texts_final, val_labels_final, tokenizer, max_length)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
num_epochs = 20
patience = 7
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()
os.makedirs("./submission", exist_ok=True)

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math

# Linear warmup, then cosine annealing (no restarts) per batch
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

# Use SequentialLR: LinearLR (start_factor=0.1) then CosineAnnealingLR (eta_min=1e-6)
# total_iters for LinearLR = warmup_steps, then CosineAnnealingLR T_max = total_steps - warmup_steps
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=warmup_steps,
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=1e-6,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)

# Note: scheduler will .step() once per batch after optimizer.step()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask, extra_features=batch["extra_features"].to(device))
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        # Step scheduler once per batch after optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, extra_features=batch["extra_features"].to(device))
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

            total_val_loss += loss.item()
            num_val_batches += 1
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_score = log_loss(val_true, val_probs_clipped)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if avg_val_loss < best_val_score:
        best_val_score = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Progressive unfreezing: after epoch 2, 4, 6 unfreeze more layers
    # epoch is 0-indexed inside loop (epoch starts at 0)
    if epoch + 1 == 2:
        # Unfreeze last 8 layers (already done) -> no change needed for first milestone
        print("Progressive unfreeze: already 8 layers unfrozen.")
    elif epoch + 1 == 4:
        # Unfreeze last 12 layers
        new_unfrozen = get_unfrozen_params(model, 12)
        for param in new_unfrozen:
            param.requires_grad = True
        # Recreate optimizer with updated params
        optimizer = AdamW(
            [
                {
                    "params": get_unfrozen_params(model, 12),
                    "lr": 2e-5,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                },
                {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
            ],
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        # Recreate scheduler with new optimizer
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        print("Progressive unfreeze: unfroze last 12 layers.")
    elif epoch + 1 == 6:
        # Unfreeze last 16 layers
        new_unfrozen = get_unfrozen_params(model, 16)
        for param in new_unfrozen:
            param.requires_grad = True
        optimizer = AdamW(
            [
                {
                    "params": get_unfrozen_params(model, 16),
                    "lr": 2e-5,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                },
                {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
            ],
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        print("Progressive unfreeze: unfroze last 16 layers.")

# Load best model (no full-data retraining)
model.load_state_dict(torch.load("./working/best_model.pt"))
print("Loaded best checkpoint.")

# ============================================================
# TEST INFERENCE
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask, extra_features=batch["extra_features"].to(device))
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {best_val_score}")