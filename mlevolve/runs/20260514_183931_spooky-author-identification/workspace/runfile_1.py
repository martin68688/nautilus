import os
os.sched_setaffinity(0, {132, 133, 134, 135})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

# ============================================================
# TF-IDF FEATURE PREPARATION (character + word n-grams)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# First split, then fit TF-IDF only on training data
train_texts_only = train_df["text"].values
test_texts_only = test_df["text"].values

tfidf_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=50000,
    sublinear_tf=True,
)
char_tfidf_train = tfidf_vectorizer.fit_transform(train_texts_only)
char_tfidf_test = tfidf_vectorizer.transform(test_texts_only)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    stop_words="english",
)
word_tfidf_train = word_vectorizer.fit_transform(train_texts_only)
word_tfidf_test = word_vectorizer.transform(test_texts_only)

# Combine character and word n-gram features
from scipy.sparse import hstack
combined_tfidf_train = hstack([char_tfidf_train, word_tfidf_train])
combined_tfidf_test = hstack([char_tfidf_test, word_tfidf_test])

# Reduce dimensionality - fit only on training data
svd = TruncatedSVD(n_components=200, random_state=42)
train_tfidf = svd.fit_transform(combined_tfidf_train)
test_tfidf = svd.transform(combined_tfidf_test)
print(f"TF-IDF features - Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")

# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

# Now fit TF-IDF vectorizers only on training split (after split, to avoid data leakage)
train_texts_split = train_set["text"].values
val_texts_split = val_set["text"].values

char_tfidf_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=50000,
    sublinear_tf=True,
)
char_tfidf_train_split = char_tfidf_vectorizer.fit_transform(train_texts_split)
char_tfidf_val_split = char_tfidf_vectorizer.transform(val_texts_split)
char_tfidf_test_split = char_tfidf_vectorizer.transform(test_texts_only)

word_vectorizer_split = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    stop_words="english",
)
word_tfidf_train_split = word_vectorizer_split.fit_transform(train_texts_split)
word_tfidf_val_split = word_vectorizer_split.transform(val_texts_split)
word_tfidf_test_split = word_vectorizer_split.transform(test_texts_only)

# Combine character and word n-gram features
combined_tfidf_train_split = hstack([char_tfidf_train_split, word_tfidf_train_split])
combined_tfidf_val_split = hstack([char_tfidf_val_split, word_tfidf_val_split])
combined_tfidf_test_split = hstack([char_tfidf_test_split, word_tfidf_test_split])

# Reduce dimensionality - fit only on training split
svd_split = TruncatedSVD(n_components=200, random_state=42)
train_tfidf_split = svd_split.fit_transform(combined_tfidf_train_split)
val_tfidf_split = svd_split.transform(combined_tfidf_val_split)
test_tfidf_split = svd_split.transform(combined_tfidf_test_split)

# Overwrite the old tfidf variables with the split-correct versions
train_tfidf = train_tfidf_split
test_tfidf = test_tfidf_split

val_tfidf = val_tfidf_split
print(f"TF-IDF features - Train: {train_tfidf.shape}, Val: {val_tfidf.shape}, Test: {test_tfidf.shape}")

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3, tfidf_dim=200, tfidf_proj_dim=128):
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
        # Track which layers are unfrozen for gradual unfreezing
        self.unfrozen_layers = 8
        self.total_encoder_layers = len(self.backbone.deberta.encoder.layer)

        hidden_size = self.backbone.config.hidden_size
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
        # TF-IDF fusion: project TF-IDF features to a smaller dim, apply LayerNorm + GELU
        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, tfidf_proj_dim),
            nn.LayerNorm(tfidf_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        # Final classification head: backbone pooled output (hidden_size) + TF-IDF projected features
        self.final_head = nn.Linear(hidden_size + tfidf_proj_dim, num_authors)

    def forward(self, input_ids, attention_mask, tfidf=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        # Attention pooling: compute attention weights over sequence length
        # Use attention_mask to ignore padding tokens
        attn_scores = self.attention_query(hidden_states)  # (batch, seq_len, 1)
        # Mask out padding tokens by setting their scores to a large negative number
        attn_scores = attn_scores.squeeze(-1)  # (batch, seq_len)
        attn_mask = attention_mask.bool()  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        # Weighted sum of hidden states
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)
        # Apply feed-forward network with residual connection
        ffn_out = self.ffn(pooled)
        combined = pooled + ffn_out

        # If TF-IDF features are present, fuse them with the backbone output
        if tfidf is not None:
            tfidf_features = self.tfidf_proj(tfidf)  # (batch, tfidf_proj_dim)
            combined = torch.cat([combined, tfidf_features], dim=1)  # (batch, hidden_size + tfidf_proj_dim)

        logits = self.final_head(combined)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3, tfidf_dim=200, tfidf_proj_dim=128)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Collect ALL trainable parameters initially (last 8 backbone layers + head)
def get_trainable_backbone_params():
    """Get current trainable backbone parameters."""
    params = []
    for i, layer in enumerate(model.backbone.deberta.encoder.layer):
        # Only include last layers that are currently unfrozen
        if i >= len(model.backbone.deberta.encoder.layer) - model.unfrozen_layers:
            for name, param in layer.named_parameters():
                if "bias" not in name and "LayerNorm" not in name:
                    params.append(param)
    return params

def get_all_head_params():
    return (
        list(model.attention_query.parameters())
        + list(model.ffn.parameters())
        + list(model.tfidf_proj.parameters())
        + list(model.final_head.parameters())
    )

# Initial optimizer setup
backbone_unfrozen_params = get_trainable_backbone_params()
head_params = get_all_head_params()

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
            item["tfidf"] = torch.tensor(self.tfidf_features[idx], dtype=torch.float)
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
    train_texts_final, train_labels_final, tokenizer, max_length, tfidf_features=train_tfidf[train_indices]
)
val_dataset = SpookyDataset(
    val_texts_final, val_labels_final, tokenizer, max_length, tfidf_features=train_tfidf[val_indices]
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

# OneCycleLR scheduler: handles warmup + annealing automatically
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps) + 1

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[pg["lr"] for pg in optimizer.param_groups],
    total_steps=total_steps,
    pct_start=warmup_steps / total_steps,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4,
)

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
            tfidf_tensor = batch.get("tfidf", None)
            if tfidf_tensor is not None:
                tfidf_tensor = tfidf_tensor.to(device)
            logits = model(input_ids, attention_mask, tfidf=tfidf_tensor)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        # Apply scheduler step per batch (OneCycleLR handles warmup + annealing)
        scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    # Gradual unfreezing: unfreeze 2 layers every 2 epochs after epoch 4
    # CRITICAL: Rebuild optimizer to include newly unfrozen parameters
    if (epoch + 1) % 2 == 0 and (epoch + 1) >= 4:
        target_unfrozen = min(8 + ((epoch + 1 - 4) // 2) * 2, model.total_encoder_layers)
        if target_unfrozen > model.unfrozen_layers:
            current_frozen_start = model.total_encoder_layers - model.unfrozen_layers
            new_frozen_start = model.total_encoder_layers - target_unfrozen
            for i in range(new_frozen_start, current_frozen_start):
                for param in model.backbone.deberta.encoder.layer[i].parameters():
                    param.requires_grad = True
            model.unfrozen_layers = target_unfrozen

            # REBUILD OPTIMIZER to include newly unfrozen parameters
            new_backbone_params = get_trainable_backbone_params()
            new_head_params = get_all_head_params()

            optimizer = AdamW(
                [
                    {
                        "params": new_backbone_params,
                        "lr": 2e-5,
                        "weight_decay": 0.01,
                        "betas": (0.9, 0.999),
                    },
                    {"params": new_head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
                ],
                weight_decay=0.01,
                betas=(0.9, 0.999),
            )

            # Rebuild scheduler with new total steps
            total_steps = len(train_loader) * (num_epochs - epoch - 1)
            warmup_steps = int(0.1 * total_steps) + 1
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[pg["lr"] for pg in optimizer.param_groups],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4,
            )

            # Recompute total unfrozen for the print message
            total_unfrozen = sum(
                p.numel()
                for layer in model.backbone.deberta.encoder.layer[-model.unfrozen_layers:]
                for name, p in layer.named_parameters()
                if "bias" not in name and "LayerNorm" not in name
            )
            print(f"Gradual unfreezing: now {model.unfrozen_layers} layers unfrozen ({total_unfrozen:,} backbone params)")

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
            tfidf_tensor = batch.get("tfidf", None)
            if tfidf_tensor is not None:
                tfidf_tensor = tfidf_tensor.to(device)

            with autocast():
                logits = model(input_ids, attention_mask, tfidf=tfidf_tensor)
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
        torch.save(model.state_dict(), "./working/best_model_13fa216857694a6e9782fdc91805d2a1.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model_13fa216857694a6e9782fdc91805d2a1.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        tfidf_tensor = batch.get("tfidf", None)
        if tfidf_tensor is not None:
            tfidf_tensor = tfidf_tensor.to(device)
        with autocast():
            logits = model(input_ids, attention_mask, tfidf=tfidf_tensor)
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
        tfidf_tensor = batch.get("tfidf", None)
        if tfidf_tensor is not None:
            tfidf_tensor = tfidf_tensor.to(device)
        with autocast():
            logits = model(input_ids, attention_mask, tfidf=tfidf_tensor)
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

submission.to_csv("./submission/submission_13fa216857694a6e9782fdc91805d2a1.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")