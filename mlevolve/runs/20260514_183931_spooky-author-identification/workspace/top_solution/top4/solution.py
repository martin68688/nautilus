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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3, chunk_size=256, chunk_stride=128):
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
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride

        # Attention pooling over all token hidden states (global view)
        self.attention_query = nn.Linear(hidden_size, 1, bias=False)
        # Two-layer feed-forward network for global representation
        self.ffn_global = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, hidden_size),
            nn.Dropout(dropout_rate),
        )

        # Chunked local representation: project each chunk's pooled hidden states
        self.chunk_projection = nn.Linear(hidden_size, 256)
        # Multi-perspective fusion head: global and local projections
        self.global_projection = nn.Linear(hidden_size, 256)
        # Small feed-forward networks for each view
        self.global_ffn = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        self.local_ffn = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        # Learned weighted sum scalar per view
        self.view_weight_global = nn.Parameter(torch.tensor(0.5))
        self.view_weight_local = nn.Parameter(torch.tensor(0.5))
        self.final_head = nn.Linear(256, num_authors)

    def forward(self, input_ids, attention_mask):
        # Full sequence forward pass for global representation
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # --- Global representation (attention pooling) ---
        attn_scores = self.attention_query(hidden_states)  # (batch, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch, seq_len)
        attn_mask = attention_mask.bool()
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        global_pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)
        ffn_global_out = self.ffn_global(global_pooled)
        global_repr = global_pooled + ffn_global_out  # (batch, hidden_size)

        # --- Local chunked representation ---
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # Create overlapping chunks
        # We'll process each sample independently and average chunk representations
        local_reprs = []
        for i in range(batch_size):
            # Get valid token positions (non-padding)
            valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            valid_len = valid_indices.shape[0]
            # Extract valid hidden states
            valid_hidden = hidden_states[i, :valid_len, :]  # (valid_len, hidden_dim)
            # Generate chunk start indices
            chunk_starts = list(range(0, valid_len - self.chunk_size + 1, self.chunk_stride))
            if len(chunk_starts) == 0:
                # If sequence too short, just use the whole sequence as one chunk
                chunk_starts = [0]
            chunk_reprs = []
            for start in chunk_starts:
                end = start + self.chunk_size
                chunk = valid_hidden[start:end, :]  # (chunk_size, hidden_dim)
                # Mean pool over chunk tokens
                chunk_pooled = chunk.mean(dim=0)  # (hidden_dim)
                chunk_reprs.append(chunk_pooled)
            # Average all chunk representations for this sample
            if chunk_reprs:
                sample_local = torch.stack(chunk_reprs, dim=0).mean(dim=0)  # (hidden_dim)
            else:
                sample_local = valid_hidden.mean(dim=0)
            local_reprs.append(sample_local)
        local_repr = torch.stack(local_reprs, dim=0)  # (batch, hidden_dim)

        # --- Multi-perspective fusion ---
        global_proj = self.global_projection(global_repr)  # (batch, 256)
        local_proj = self.chunk_projection(local_repr)     # (batch, 256)

        # Apply separate small FFNs
        global_transformed = self.global_ffn(global_proj)  # (batch, 256)
        local_transformed = self.local_ffn(local_proj)     # (batch, 256)

        # Learned weighted sum
        combined = torch.sigmoid(self.view_weight_global) * global_transformed + \
                   torch.sigmoid(self.view_weight_local) * local_transformed

        logits = self.final_head(combined)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Collect backbone unfrozen params (last 8 layers)
backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

# Collect all head/extra params (attention pooling, FFN, chunk projection, local FFN, fusion, final linear layer)
head_params = (
    list(model.attention_query.parameters())
    + list(model.ffn_global.parameters())
    + list(model.chunk_projection.parameters())
    + list(model.global_projection.parameters())
    + list(model.global_ffn.parameters())
    + list(model.local_ffn.parameters())
    + list(model.final_head.parameters())
    + [model.view_weight_global, model.view_weight_local]
)

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_unfrozen_params):,}")
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
train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

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
num_epochs = 40
patience = 4
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()
os.makedirs("./submission", exist_ok=True)

# Linear warmup then cosine decay without restarts
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)
cosine_steps = total_steps - warmup_steps

# Initialize cosine annealing scheduler for after warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cosine_steps, eta_min=1e-6
)

# Save initial LR for warmup scaling
initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

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
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        # Apply scheduler step per batch: linear warmup then cosine decay
        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = initial_lrs[i] * warmup_factor
        else:
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
                logits = model(input_ids, attention_mask)
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

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_true, val_probs_clipped)

# ============================================================
# TEST INFERENCE
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
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

print(f"Final Validation Score: {final_val_score}")