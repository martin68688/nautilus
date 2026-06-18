import os
os.sched_setaffinity(0, {128, 4, 5, 6, 7})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


# ============================================================
# TEXT CLEANING (from Step 1)
# ============================================================
def clean_text(text):
    """Clean text while preserving author-specific punctuation patterns"""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)


# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

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
        # Remove unused head to prevent accidental use
        del self.backbone.classifier
        del self.backbone.pooler
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.attention_mlp_residual = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, hidden_size),
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        attn_logits = self.attention_mlp(hidden_states).squeeze(-1)  # [batch, seq_len]
        # Use large negative value instead of -inf for stability
        attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_logits, dim=1)  # [batch, seq_len]
        attended = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]
        # Mean pooling of hidden_states
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        mean_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [batch, hidden]
        # Residual: attended + mean_pooled, then pass through MLP
        combined = attended + mean_pooled
        combined = self.attention_layer_norm(combined)
        residual_out = self.attention_mlp_residual(combined)
        pooled = combined + residual_out
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
# Collect backbone unfrozen params (last 8 layers)
backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

# Collect head params (attention MLP, residual MLP, and classifier)
head_params = list(model.attention_mlp.parameters()) + list(model.attention_mlp_residual.parameters()) + list(model.classifier.parameters()) + list(model.dropout.parameters())

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 1e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.999)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Ensure optimizer param groups are correctly ordered
# Group 0: backbone unfrozen layers (lr=2e-5)
# Group 1: head (lr=5e-5)

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
train_texts_orig = train_df["text_clean"].values  # Use cleaned text
train_labels_orig = le.transform(train_df["author"].values)
test_texts = test_df["text_clean"].values  # Use cleaned text
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
patience = 3
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

from torch.optim.lr_scheduler import CosineAnnealingLR
import math

# Linear warmup, then cosine annealing
num_warmup_epochs = 2

def get_lr_scheduler(optimizer, num_epochs, num_warmup_epochs):
    """Create a scheduler with linear warmup followed by cosine annealing."""
    def lambda_lr(epoch):
        if epoch < num_warmup_epochs:
            # Linear warmup from 0 to 1
            return (epoch + 1) / num_warmup_epochs
        else:
            # Cosine annealing from 1 to 0
            progress = (epoch - num_warmup_epochs) / (num_epochs - num_warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

scheduler = get_lr_scheduler(optimizer, num_epochs, num_warmup_epochs)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0
    accumulation_steps = 2

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()
            optimizer.zero_grad()

        total_train_loss += loss.item()
        num_train_batches += 1

    # Handle remaining gradients if batches not divisible by accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()
        optimizer.zero_grad()

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

    # Re-check for any NaN after clipping/normalization
    if np.any(np.isnan(val_probs_clipped)):
        val_score = 1.0  # Default high loss if NaN encountered
    else:
        val_score = log_loss(val_true, val_probs_clipped)

    current_lr = optimizer.param_groups[0]["lr"]
    # Step scheduler at end of epoch
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model_adf9569dd769444faa83f179712d06f9.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model_adf9569dd769444faa83f179712d06f9.pt"))
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
        probs_np = probs.cpu().numpy()
        if not np.any(np.isnan(probs_np)):
            all_val_probs.append(probs_np)
            all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
if np.any(np.isnan(val_probs_clipped)):
    final_val_score = 1.0
else:
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

submission.to_csv("./submission/submission_adf9569dd769444faa83f179712d06f9.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")