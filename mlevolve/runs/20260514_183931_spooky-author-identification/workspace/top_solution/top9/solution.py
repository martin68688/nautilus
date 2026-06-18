import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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

# ============================================================
# TF-IDF FEATURE ENGINEERING (fitted only on training data)
# ============================================================
# Character n-grams (2-5) and word n-grams (1-2)

# Get original texts for feature engineering
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

# Fit TF-IDF vectorizers only on training fold
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=50000,
    sublinear_tf=True,
    dtype=np.float32,
)
tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    dtype=np.float32,
)

char_features_train = tfidf_char.fit_transform(train_texts_final)
word_features_train = tfidf_word.fit_transform(train_texts_final)

# Transform validation and test
char_features_val = tfidf_char.transform(val_texts_final)
word_features_val = tfidf_word.transform(val_texts_final)
char_features_test = tfidf_char.transform(test_texts)
word_features_test = tfidf_word.transform(test_texts)

# Concatenate and reduce dimensionality
from scipy.sparse import hstack
combined_train = hstack([char_features_train, word_features_train], format="csr")
combined_val = hstack([char_features_val, word_features_val], format="csr")
combined_test = hstack([char_features_test, word_features_test], format="csr")

svd = TruncatedSVD(n_components=200, random_state=42)
svd_features_train = svd.fit_transform(combined_train).astype(np.float32)
svd_features_val = svd.transform(combined_val).astype(np.float32)
svd_features_test = svd.transform(combined_test).astype(np.float32)

# Normalize to zero mean unit variance
scaler = StandardScaler()
tfidf_features = scaler.fit_transform(svd_features_train).astype(np.float32)
tfidf_features_val = scaler.transform(svd_features_val).astype(np.float32)
tfidf_features_test = scaler.transform(svd_features_test).astype(np.float32)

print(f"TF-IDF + SVD features shape (train): {tfidf_features.shape}")
print(f"TF-IDF + SVD features shape (val): {tfidf_features_val.shape}")
print(f"TF-IDF + SVD features shape (test): {tfidf_features_test.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3, tfidf_dim=200):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # Initially freeze all backbone layers
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        # Project TF-IDF features to hidden size and combine with BERT pooled representation
        self.tfidf_proj = nn.Linear(tfidf_dim, hidden_size)
        # Attention pooling over all token hidden states
        self.attention_query = nn.Linear(hidden_size, 1, bias=False)
        # Two-layer feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.final_head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, tfidf=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        # Attention pooling: compute attention weights over sequence length
        attn_scores = self.attention_query(hidden_states)  # (batch, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch, seq_len)
        attn_mask = attention_mask.bool()  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)
        # Incorporate TF-IDF features if provided
        if tfidf is not None:
            tfidf_proj = self.tfidf_proj(tfidf)  # (batch, hidden_size)
            pooled = pooled + tfidf_proj
        # Apply feed-forward network with residual connection
        ffn_out = self.ffn(pooled)
        combined = pooled + ffn_out
        logits = self.final_head(combined)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3, tfidf_dim=200)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Initially all backbone frozen, only head params + tfidf_proj are trainable
head_params = (
    list(model.tfidf_proj.parameters())
    + list(model.attention_query.parameters())
    + list(model.ffn.parameters())
    + list(model.final_head.parameters())
)

optimizer = AdamW(
    [
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Tracking for gradual unfreezing
current_unfrozen_layers = 0  # number of top layers currently unfrozen (from the top)
unfreeze_every_n_epochs = 2
layers_to_unfreeze_per_step = 2

print(f"Head params (trainable initially): {sum(p.numel() for p in head_params):,}")
print(f"Model parameters (total): {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, tfidf_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tfidf_features = tfidf_features

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
        if self.tfidf_features is not None:
            item["tfidf"] = torch.tensor(
                self.tfidf_features[idx], dtype=torch.float32
            )
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

# Compute tfidf_features for train/val/test splits
train_tfidf = tfidf_features[train_indices]
val_tfidf = tfidf_features[val_indices]
test_tfidf = tfidf_features_test

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length, tfidf_features=train_tfidf
)
val_dataset = SpookyDataset(
    val_texts_final, val_labels_final, tokenizer, max_length, tfidf_features=val_tfidf
)
test_dataset = SpookyDataset(
    test_texts, None, tokenizer, max_length, tfidf_features=test_tfidf
)

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

# Save initial LR for warmup scaling - ensure this handles dynamic param group additions
initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
# Keep a shared reference that can be updated when new param groups are added
def _refresh_initial_lrs():
    global initial_lrs
    initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
_refresh_initial_lrs()

for epoch in range(num_epochs):
    # Gradual unfreezing: unfreeze more layers as training progresses
    if epoch > 0 and epoch % unfreeze_every_n_epochs == 0:
        if current_unfrozen_layers < 8:  # max 8 layers unfrozen
            additional_layers = layers_to_unfreeze_per_step
            if current_unfrozen_layers + additional_layers > 8:
                additional_layers = 8 - current_unfrozen_layers
            # Unfreeze the next set of layers from the top
            for i in range(current_unfrozen_layers, current_unfrozen_layers + additional_layers):
                layer_idx = len(model.backbone.deberta.encoder.layer) - 1 - i
                layer = model.backbone.deberta.encoder.layer[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
            current_unfrozen_layers += additional_layers
            print(f"Epoch {epoch+1}: Unfreezing {additional_layers} layer(s) (total unfrozen: {current_unfrozen_layers})")
            # Collect newly unfrozen params and add to optimizer
            newly_unfrozen_params = []
            for i in range(current_unfrozen_layers - additional_layers, current_unfrozen_layers):
                layer_idx = len(model.backbone.deberta.encoder.layer) - 1 - i
                layer = model.backbone.deberta.encoder.layer[layer_idx]
                for name, param in layer.named_parameters():
                    if "bias" not in name and "LayerNorm" not in name:
                        newly_unfrozen_params.append(param)
            if newly_unfrozen_params:
                new_lr = 2e-5
                optimizer.add_param_group({
                    "params": newly_unfrozen_params,
                    "lr": new_lr,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                })
                # Update initial_lrs to include this new group's LR for warmup scaling
                initial_lrs.append(new_lr)

    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        tfidf = batch["tfidf"].to(device) if "tfidf" in batch else None

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask, tfidf=tfidf)
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
            # Use index-wrapping to handle dynamic param groups: each group uses its own initial_lr
            for i, param_group in enumerate(optimizer.param_groups):
                if i < len(initial_lrs):
                    param_group["lr"] = initial_lrs[i] * warmup_factor
                else:
                    # Fallback for any unexpected extra groups (should not happen)
                    param_group["lr"] = param_group.get("lr", 5e-5) * warmup_factor
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
            tfidf = batch["tfidf"].to(device) if "tfidf" in batch else None

            with autocast():
                logits = model(input_ids, attention_mask, tfidf=tfidf)
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
        tfidf = batch["tfidf"].to(device) if "tfidf" in batch else None
        with autocast():
            logits = model(input_ids, attention_mask, tfidf=tfidf)
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
        tfidf = batch["tfidf"].to(device) if "tfidf" in batch else None
        with autocast():
            logits = model(input_ids, attention_mask, tfidf=tfidf)
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