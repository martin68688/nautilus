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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR, CosineAnnealingLR
import warnings
import math

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
# BASIC TEXT FEATURES (for reference/ensemble later)
# ============================================================
def extract_basic_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        char_count = len(text)
        word_count = len(text.split())
        sent_count = max(len(re.split(r"[.!?]+", text)) - 1, 1)
        words = text.split()
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        unique_words = len(set(words)) / max(word_count, 1)
        exclamations = text.count("!")
        questions = text.count("?")
        periods = text.count(".")
        commas = text.count(",")
        semicolons = text.count(";")
        colons = text.count(":")
        dashes = text.count("-") + text.count("—")
        quotes = text.count('"') + text.count("'") + text.count('"') + text.count('"')
        parentheses = text.count("(") + text.count(")")
        ellipsis = text.count("...") + text.count("…")
        capital_words = sum(1 for w in words if w[0].isupper()) if words else 0
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
        has_dialogue = 1 if re.search(r'["""]', text) else 0
        dialogue_marks = len(re.findall(r'["""]', text))
        avg_words_per_sent = word_count / max(sent_count, 1)
        char_diversity = len(set(text.lower())) / max(char_count, 1)
        digit_count = sum(c.isdigit() for c in text)
        space_ratio = text.count(" ") / max(char_count, 1)
        features.append(
            [
                char_count,
                word_count,
                sent_count,
                avg_word_len,
                unique_words,
                exclamations,
                questions,
                periods,
                commas,
                semicolons,
                colons,
                dashes,
                quotes,
                parentheses,
                ellipsis,
                capital_words,
                all_caps,
                has_dialogue,
                dialogue_marks,
                avg_words_per_sent,
                char_diversity,
                digit_count,
                space_ratio,
            ]
        )
    return np.array(features)


# ============================================================
# TRAIN/VALIDATION SPLIT
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)

# These features are not used in the current training pipeline
# but we want to keep them for potential future use
train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()
basic_train = extract_basic_features(train_df.loc[train_indices, "text"].values) if len(train_indices) > 0 else None
basic_val = extract_basic_features(train_df.loc[val_indices, "text"].values) if len(val_indices) > 0 else None
basic_test = extract_basic_features(test_df["text"].values)

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
        self.head = nn.Linear(hidden_size, num_authors)
        self.layer_weights = nn.Parameter(torch.ones(4) * 0.25)  # learnable weights for 4 layers

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple of all layers
        # Take last 4 layers: indices -4, -3, -2, -1 (1-indexed layers, after embedding)
        # hidden_states[0] is embeddings, hidden_states[1..24] are transformer layers
        # So last 4 hidden layers are hidden_states[-4], hidden_states[-3], hidden_states[-2], hidden_states[-1]
        last_four_layers = hidden_states[-4:]  # list of (batch, seq_len, hidden_size)
        # Masked mean pooling for each layer
        layer_pooled = []
        for layer_hidden in last_four_layers:
            # Expand attention_mask to match hidden dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
            sum_hidden = (layer_hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
            layer_pooled.append(pooled)
        # Stack and compute weighted sum with learnable weights
        stacked = torch.stack(layer_pooled, dim=1)  # (batch, 4, hidden_size)
        # Softmax over the 4 weights
        weights = torch.softmax(self.layer_weights, dim=0)  # (4,)
        weighted_pool = (stacked * weights.view(1, -1, 1)).sum(dim=1)  # (batch, hidden_size)
        logits = self.head(weighted_pool)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

head_params = list(model.head.parameters())

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


author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

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
num_epochs = 30
patience = 5
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()
os.makedirs("./submission", exist_ok=True)

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

# Clean scheduler: LinearLR warmup for first 10%, then CosineAnnealingLR for the rest
warmup_scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
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

        scheduler.step()  # scheduler handles warmup and cosine annealing automatically

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    model.eval()
    total_val_loss = 0.0
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

            # Guard against empty batches (num_workers can produce empty tensors)
            if probs.numel() > 0 and labels.numel() > 0:
                total_val_loss += loss.item()
                num_val_batches += 1
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

    # Ensure we have valid data before concatenating
    if num_val_batches == 0 or len(all_val_probs) == 0:
        continue  # skip the rest of the epoch if no valid data

    avg_val_loss = total_val_loss / num_val_batches
    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    # Check for NaN before log_loss
    if np.any(np.isnan(val_probs_clipped)):
        # Fallback: print warning and use raw probs
        val_score = log_loss(val_true, val_probs_clipped)
    else:
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