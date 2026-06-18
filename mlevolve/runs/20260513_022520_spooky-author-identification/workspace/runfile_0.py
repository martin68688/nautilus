import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import os
import json
import re
import string

# ============ DATA LOADING ============
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_ids = train_df["id"].values
test_ids = test_df["id"].values

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_encoded"] = train_df["author"].map(author_map)


# ============ TEXT CLEANING ============
def clean_text_for_pretrained(text_series):
    cleaned = []
    for text in text_series:
        text = re.sub(r"\s+", " ", text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        cleaned.append(text.strip())
    return cleaned


train_text_clean = clean_text_for_pretrained(train_df["text"])
test_text_clean = clean_text_for_pretrained(test_df["text"])

# ============ STRATIFIED SPLIT ============
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(train_text_clean, train_df["author_encoded"]):
    train_texts = [train_text_clean[i] for i in train_idx]
    val_texts = [train_text_clean[i] for i in val_idx]
    train_labels = train_df["author_encoded"].iloc[train_idx].values
    val_labels = train_df["author_encoded"].iloc[val_idx].values
    break

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ============ MODEL ARCHITECTURE ============
class AuthorshipAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask_expanded
            query = masked_hidden.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            query = query.unsqueeze(1)
            key_padding_mask = ~attention_mask.bool()
        else:
            query = hidden_states.mean(dim=1, keepdim=True)
            key_padding_mask = None

        attn_output, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        attn_output = query + self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)
        pooled_output = attn_output.squeeze(1)
        return pooled_output


class DebertaAuthorshipModel(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        model_name: str = "microsoft/deberta-v3-large",
        dropout: float = 0.15,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.deberta = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size

        self.attention_pooling = AuthorshipAttentionPooling(
            hidden_size=self.hidden_size, num_heads=num_attention_heads, dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.hidden_size // 4, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.attention_pooling(
            hidden_states=sequence_output, attention_mask=attention_mask
        )
        logits = self.classifier(pooled_output)
        result = {"logits": logits}

        if labels is not None:
            loss = self._compute_label_smoothed_loss(logits, labels, epsilon=0.1)
            result["loss"] = loss

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            result["probs"] = probs
        return result

    def _compute_label_smoothed_loss(
        self, logits, labels, epsilon=0.1, reduction="mean"
    ):
        num_classes = logits.size(-1)
        with torch.no_grad():
            confidence = 1.0 - epsilon
            smooth_targets = torch.full_like(
                logits, fill_value=epsilon / (num_classes - 1)
            )
            smooth_targets.scatter_(1, labels.unsqueeze(1), confidence)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


# ============ DATASET ============
class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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


# ============ INITIALIZATION ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DebertaAuthorshipModel(
    num_labels=3,
    model_name=model_name,
    dropout=0.15,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_attention_heads=8,
)
model.to(device)

batch_size = 16
accumulation_steps = 2
num_epochs = 40
max_norm = 1.0

train_dataset = AuthorshipDataset(train_texts, train_labels, tokenizer, max_length=512)
val_dataset = AuthorshipDataset(val_texts, val_labels, tokenizer, max_length=512)
test_dataset = AuthorshipDataset(test_text_clean, None, tokenizer, max_length=512)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

optimizer = AdamW(
    [
        {"params": model.deberta.parameters(), "lr": 2e-5, "weight_decay": 0.01},
        {
            "params": model.attention_pooling.parameters(),
            "lr": 5e-5,
            "weight_decay": 0.01,
        },
        {"params": model.classifier.parameters(), "lr": 1e-4, "weight_decay": 0.01},
    ],
    betas=(0.9, 0.999),
    eps=1e-8,
)

total_steps = len(train_loader) * num_epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler()

# ============ TRAINING ============
best_val_loss = float("inf")
best_epoch = -1
patience = 5
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"] / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_train_loss += loss.item() * accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs["loss"].item()

            all_val_probs.append(outputs["probs"].cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_val_probs = np.concatenate(all_val_probs)
    all_val_labels = np.concatenate(all_val_labels)
    val_log_loss = log_loss(all_val_labels, all_val_probs)

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_log_loss:.4f}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_log_loss": val_log_loss,
            },
            "./working/best_model_98b7459b4cb440548a4874c5abd0d318.pt",
        )
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# ============ LOAD BEST MODEL & INFERENCE ============
print(
    f"Loading best model from epoch {best_epoch} with validation log-loss: {best_val_loss:.6f}"
)
checkpoint = torch.load("./working/best_model_98b7459b4cb440548a4874c5abd0d318.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_probs = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        val_probs.append(outputs["probs"].cpu().numpy())

val_probs = np.concatenate(val_probs)
val_log_loss = log_loss(val_labels, val_probs)

test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        test_probs.append(outputs["probs"].cpu().numpy())

test_probs = np.concatenate(test_probs)

# ============ SUBMISSION ============
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

epsilon = 1e-15
submission[["EAP", "HPL", "MWS"]] = submission[["EAP", "HPL", "MWS"]].clip(
    lower=epsilon, upper=1 - epsilon
)
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission_98b7459b4cb440548a4874c5abd0d318.csv", index=False)

print(f"Final Validation Score: {val_log_loss:.6f}")
