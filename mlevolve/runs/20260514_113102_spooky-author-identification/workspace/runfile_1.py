import os
os.sched_setaffinity(0, {9, 12, 13, 14})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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

le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large with MHA + Mean-Max Concatenation Pooling
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.0
        )

    def forward(self, last_hidden, attention_mask):
        # Create padding mask (True for padded positions)
        padding_mask = attention_mask == 0
        # Apply MHA: using token embeddings as query, key, value
        attended, _ = self.mha(
            query=last_hidden,
            key=last_hidden,
            value=last_hidden,
            key_padding_mask=padding_mask,
        )
        return attended


class SpookyAuthorClassifier(nn.Module):
    """DeBERTa-v3-large with MHA + Mean-Max Concatenation pooling and multi-sample dropout"""

    def __init__(self, num_authors=3, dropout_rate=0.2, hidden_dim=256, n_dropouts=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            output_hidden_states=True,
            output_attentions=False,
        )

        # Freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 8 layers for fine-tuning
        for layer in self.backbone.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.hidden_size = self.backbone.config.hidden_size

        # Multi-Head Attention Pooling layer
        self.mha_pooling = MultiHeadAttentionPooling(self.hidden_size, num_heads=8)

        # Projection layer: 3 * hidden_size (MHA-pooled, mean-pooled, max-pooled) -> hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Multi-sample dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_dropouts)]
        )
        self.classifier = nn.Linear(hidden_dim, num_authors)

        self._init_weights()

    def _init_weights(self):
        for module in [self.projector, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, return_dropout_samples=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # 1. Multi-Head Attention pooling
        attended_hidden = self.mha_pooling(last_hidden, attention_mask)
        mha_pooled = (attended_hidden * mask_expanded).sum(dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

        # 2. Mean pooling
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # 3. Max pooling
        last_hidden_masked = last_hidden * mask_expanded
        # Set padded positions to -inf for max pooling
        last_hidden_masked = last_hidden_masked.masked_fill(
            (1 - mask_expanded).bool(), -1e9
        )
        max_pooled, _ = torch.max(last_hidden_masked, dim=1)

        # Concatenate all three representations
        pooled = torch.cat([mha_pooled, mean_pooled, max_pooled], dim=1)

        # Project to hidden_dim
        features = self.projector(pooled)

        if return_dropout_samples:
            logits_list = []
            for dropout in self.dropouts:
                dropped = dropout(features)
                logits = self.classifier(dropped)
                logits_list.append(logits)
            return torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            logits = self.classifier(features)
            return logits


model = SpookyAuthorClassifier(
    num_authors=3, dropout_rate=0.2, hidden_dim=256, n_dropouts=4
)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Differential learning rates (will be moved inside fold training)
backbone_params = []
head_params = []

def get_optimizer(model):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 5e-5, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer, backbone_params, head_params


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


# Prepare data arrays
train_texts_orig = train_df["text"].values
train_labels_orig = train_df["author_encoded"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

batch_size = 16
max_length = 512

test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working/fold_models", exist_ok=True)

# ============================================================
# TRAINING LOOP - 5-Fold Stratified Cross Validation
# ============================================================
num_epochs = 40
patience = 6

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_test_probs = []  # to store each fold's test probabilities
all_val_scores = []  # to store each fold's best validation score

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts_orig, train_labels_orig)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*50}")

    # Prepare data for this fold
    train_texts_fold = train_texts_orig[train_idx]
    train_labels_fold = train_labels_orig[train_idx]
    val_texts_fold = train_texts_orig[val_idx]
    val_labels_fold = train_labels_orig[val_idx]

    train_dataset = SpookyDataset(
        train_texts_fold, train_labels_fold, tokenizer, max_length
    )
    val_dataset = SpookyDataset(
        val_texts_fold, val_labels_fold, tokenizer, max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model for this fold
    model_fold = SpookyAuthorClassifier(
        num_authors=3, dropout_rate=0.2, hidden_dim=256, n_dropouts=4
    )
    model_fold.to(device)
    model_fold.train()

    # Get optimizer
    optimizer_fold, backbone_params, head_params = get_optimizer(model_fold)

    if fold == 0:
        print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_params):,}")
        print(f"Head params: {sum(p.numel() for p in head_params):,}")
        total_trainable = sum(p.numel() for p in model_fold.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_trainable:,}")

    # Linear warmup + Cosine decay scheduler (per-step)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_fold = torch.optim.lr_scheduler.LambdaLR(optimizer_fold, lr_lambda)

    # Training loop for this fold
    best_val_score = float("inf")
    epochs_no_improve = 0
    scaler_grad = GradScaler()

    for epoch in range(num_epochs):
        model_fold.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer_fold.zero_grad()
            with autocast():
                logits = model_fold(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler_grad.scale(loss).backward()
            scaler_grad.unscale_(optimizer_fold)
            torch.nn.utils.clip_grad_norm_(model_fold.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer_fold)
            scaler_grad.update()
            scheduler_fold.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # Validation
        model_fold.eval()
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
                    logits = model_fold(input_ids, attention_mask)
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

        current_lr = optimizer_fold.param_groups[0]["lr"]
        print(
            f"Fold {fold+1} | Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save(model_fold.state_dict(), f"./working/fold_models/best_model_fold{fold+1}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered for fold {fold+1} after {epoch+1} epochs")
                break

    all_val_scores.append(best_val_score)

    # Test inference for this fold
    model_fold.load_state_dict(torch.load(f"./working/fold_models/best_model_fold{fold+1}.pt"))
    model_fold.eval()

    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model_fold(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(fold_test_probs, axis=0)
    all_test_probs.append(fold_test_probs)

    # Clean up to free memory
    del model_fold, optimizer_fold, scheduler_fold, train_loader, val_loader
    torch.cuda.empty_cache()

# ============================================================
# AVERAGE FOLD PREDICTIONS AND GENERATE SUBMISSION
# ============================================================
print(f"\n{'='*50}")
print("ENSEMBLE: Averaging predictions from 5 folds")
print(f"{'='*50}")
print(f"Validation scores per fold: {[f'{s:.4f}' for s in all_val_scores]}")
print(f"Mean validation score: {np.mean(all_val_scores):.4f}")

# Average test probabilities across folds
test_probs = np.mean(all_test_probs, axis=0)

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

submission.to_csv("./submission/submission_4f969338b4324ac08469eab20bcd5424.csv", index=False)
print(f"Submission saved: {submission.shape}")
