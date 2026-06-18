import os
os.sched_setaffinity(0, {23, 24, 25, 26, 27})
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import warnings
from transformers import AutoTokenizer, ModernBertModel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Extract labels
y = train_df["author"].values
label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
y_encoded = np.array([label_map[a] for a in y])

# Calculate class weights (inversely proportional to class frequencies)
class_counts = np.bincount(y_encoded)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# ============================================================
# 2. SETUP MODERNBERT MODEL WITH CUSTOM POOLING HEAD
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

class ModernBertWithCustomPooling(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.backbone = ModernBertModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Pooling: concatenate CLS, mean-pooled, and max-pooled
        self.pooling_dim = hidden_size * 3

        # Classification head
        self.layer_norm = nn.LayerNorm(self.pooling_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.pooling_dim, num_labels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        last_hidden = outputs.last_hidden_state

        # CLS token (first token)
        cls_vec = last_hidden[:, 0, :]

        # Mean pooling (masked)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_vec = sum_embeddings / sum_mask

        # Max pooling (masked)
        last_hidden_masked = last_hidden * input_mask_expanded + (-1e9) * (1 - input_mask_expanded)
        max_vec = torch.max(last_hidden_masked, dim=1).values

        # Concatenate all three
        pooled = torch.cat([cls_vec, mean_vec, max_vec], dim=-1)

        # Classification head
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits

model = ModernBertWithCustomPooling(model_id, num_labels=3, dropout=0.1).to(device)

# All backbone parameters are trainable
for param in model.backbone.parameters():
    param.requires_grad = True

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# 3. DEFINE DATASET
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    ret = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "labels" in batch[0]:
        ret["labels"] = torch.stack([item["labels"] for item in batch])
    return ret


# ============================================================
# 4. CREATE TRAIN/VALIDATION SPLIT (STRATIFIED 85/15)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=y_encoded,
)

X_train_texts = train_df["text"].iloc[train_idx].tolist()
y_train_labels = y_encoded[train_idx]
X_val_texts = train_df["text"].iloc[val_idx].tolist()
y_val_labels = y_encoded[val_idx]

print(f"Train samples: {len(X_train_texts)}")
print(f"Validation samples: {len(X_val_texts)}")

train_dataset = TextDataset(X_train_texts, y_train_labels, tokenizer, max_length=512)
val_dataset = TextDataset(X_val_texts, y_val_labels, tokenizer, max_length=512)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)

# ============================================================
# 5. SETUP TRAINING COMPONENTS
# ============================================================
num_epochs = 5
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Linear warmup for 10% of steps, then cosine decay
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)

# Loss function with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

# Mixed precision
scaler = GradScaler()

# Early stopping
patience = 3
best_val_loss = float("inf")
patience_counter = 0
best_model_state_dict = None

# ============================================================
# 6. TRAINING LOOP
# ============================================================
print("\nStarting fine-tuning...")
global_step = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_batches = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate scheduler
        if global_step < warmup_steps:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        global_step += 1
        train_loss += loss.item()
        train_batches += 1

        if train_batches % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{num_epochs}, Step {train_batches}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

    avg_train_loss = train_loss / train_batches

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            val_loss += loss.item()
            val_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_val_preds.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    all_val_preds = np.vstack(all_val_preds)
    all_val_labels = np.concatenate(all_val_labels)

    # Clip and normalize predictions for log loss calculation
    eps = 1e-15
    all_val_preds = np.clip(all_val_preds, eps, 1 - eps)
    all_val_preds = all_val_preds / all_val_preds.sum(axis=1, keepdims=True)

    val_ll = log_loss(all_val_labels, all_val_preds)

    print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {val_ll:.6f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"New best model! Val Log Loss: {val_ll:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Load best model
print("\nLoading best model...")
model.load_state_dict(best_model_state_dict)
model.to(device)

print(f"Best Validation Log Loss: {best_val_loss:.6f}")

# ============================================================
# 7. TEST INFERENCE
# ============================================================
print("\nPerforming test inference...")
test_dataset = TextDataset(test_df["text"].tolist(), labels=None, tokenizer=tokenizer, max_length=512)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)

model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_test_preds.append(probs)

test_pred_proba = np.vstack(all_test_preds)

# Clip and normalize
eps = 1e-15
test_pred_proba = np.clip(test_pred_proba, eps, 1 - eps)
test_pred_proba = test_pred_proba / test_pred_proba.sum(axis=1, keepdims=True)

# ============================================================
# 8. CREATE SUBMISSION FILE
# ============================================================
print("Creating submission file...")
submission_dir = "./submission"
os.makedirs(submission_dir, exist_ok=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_pred_proba[:, 0],
        "HPL": test_pred_proba[:, 1],
        "MWS": test_pred_proba[:, 2],
    }
)

submission.to_csv(os.path.join(submission_dir, "submission_4bebd31edfc44c30a009b82b315c2e3e.csv"), index=False)
print(f"Submission saved to {submission_dir}/submission_4bebd31edfc44c30a009b82b315c2e3e.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# 9. PRINT FINAL VALIDATION SCORE
# ============================================================
print(f"Final Validation Score: {best_val_loss:.6f}")
