import os
os.sched_setaffinity(0, {0, 1, 2, 3, 12})
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
from transformers import AutoTokenizer, ModernBertModel

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

# ============================================================
# 2. DEFINE CUSTOM MODEL: ModernBERT with CLS+Mean Pooling
# ============================================================
class ModernBertForAuthorClassification(nn.Module):
    """ModernBERT with multi-perspective aggregation (CLS + statistical pooling + char CNN) and gradient checkpointing."""
    def __init__(self, model_id, num_labels=3, dropout_prob=0.3):
        super().__init__()
        self.backbone = ModernBertModel.from_pretrained(model_id)
        hidden_size = self.backbone.config.hidden_size

        # Character-level CNN for additional stylistic features
        self.char_embedding = nn.Embedding(128, 32, padding_idx=0)  # 128 char vocab, 32-dim embeddings
        self.char_cnn = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.char_cnn_pool = nn.AdaptiveMaxPool1d(1)
        self.char_cnn_out = nn.Linear(64, hidden_size // 4)  # project to smaller dimension

        # Dropout and classifier
        self.dropout = nn.Dropout(dropout_prob)
        # Input: CLS (hidden) + mean (hidden) + max (hidden) + std (hidden) + char_cnn (hidden//4)
        classifier_input_size = hidden_size * 4 + (hidden_size // 4)
        self.classifier = nn.Linear(classifier_input_size, num_labels)

        # Register gradient checkpointing
        self.use_checkpointing = True

    def _get_char_cnn_features(self, input_ids):
        """Extract character-level features using 1D CNN."""
        # Use input_ids as proxy for character-level features (token IDs)
        # Convert to char embeddings
        char_ids = torch.clamp(input_ids, 0, 127)  # Limit to vocab size
        char_embeds = self.char_embedding(char_ids)  # (batch, seq_len, 32)
        # Permute for Conv1d: (batch, channels, seq_len)
        char_embeds = char_embeds.permute(0, 2, 1)  # (batch, 32, seq_len)
        cnn_out = self.char_cnn(char_embeds)  # (batch, 64, seq_len)
        cnn_pooled = self.char_cnn_pool(cnn_out).squeeze(-1)  # (batch, 64)
        char_features = self.char_cnn_out(cnn_pooled)  # (batch, hidden//4)
        return char_features

    def forward(self, input_ids, attention_mask):
        # Gradient checkpointing for backbone
        if self.use_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                self._backbone_forward,
                input_ids, attention_mask,
                use_reentrant=False
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # CLS token: first token
        cls_token = hidden_states[:, 0, :]  # (batch, hidden)

        # Mask for pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * mask_expanded

        # Mean pooling
        sum_embeddings = torch.sum(masked_hidden, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # (batch, hidden)

        # Max pooling
        max_pooled = torch.max(masked_hidden + (1.0 - mask_expanded) * -1e9, dim=1)[0]  # (batch, hidden)

        # Standard deviation pooling
        variance = torch.sum(((masked_hidden - mean_pooled.unsqueeze(1)) ** 2) * mask_expanded, dim=1) / sum_mask
        std_pooled = torch.sqrt(variance + 1e-9)  # (batch, hidden)

        # Character-level CNN features
        char_features = self._get_char_cnn_features(input_ids)  # (batch, hidden//4)

        # Concatenate all features
        pooled = torch.cat([cls_token, mean_pooled, max_pooled, std_pooled, char_features], dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def _backbone_forward(self, input_ids, attention_mask):
        """Separate method for gradient checkpointing."""
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )


# ============================================================
# 3. SETUP DEVICE, TOKENIZER, AND MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ModernBertForAuthorClassification(model_id, num_labels=3, dropout_prob=0.3)
model.to(device)
# All backbone parameters are trainable (fine-tuning)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} total parameters")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# Enable gradient checkpointing for memory efficiency
model.use_checkpointing = True


# ============================================================
# 4. DEFINE DATASET FOR FINE-TUNING
# ============================================================
class AuthorDataset(Dataset):
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
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 5. STRATIFIED SPLIT (85/15)
# ============================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(np.arange(len(train_df)), y_encoded))
print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# Compute class weights (inverse frequency)
class_counts = np.bincount(y_encoded[train_idx])
class_weights = 1.0 / class_counts.astype(np.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize
class_weights = torch.tensor(class_weights, device=device)
print(f"Class weights: {class_weights.cpu().numpy()}")

train_dataset = AuthorDataset(
    train_df["text"].iloc[train_idx].tolist(),
    labels=y_encoded[train_idx],
    tokenizer=tokenizer,
    max_length=512
)
val_dataset = AuthorDataset(
    train_df["text"].iloc[val_idx].tolist(),
    labels=y_encoded[val_idx],
    tokenizer=tokenizer,
    max_length=512
)
test_dataset = AuthorDataset(
    test_df["text"].tolist(),
    labels=None,
    tokenizer=tokenizer,
    max_length=512
)

BATCH_SIZE = 8  # Effective batch size = BATCH_SIZE * gradient_accumulation_steps = 8 * 2 = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# 6. TRAINING SETUP
# ============================================================
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
WARMUP_STEPS_RATIO = 0.1  # 10% of training steps for warmup

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(optimizer, max_lr=2e-5, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1, final_div_factor=1e3)
loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
scaler = GradScaler()


# ============================================================
# 7. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
print("\nStarting fine-tuning...")
best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    # Calculate total training steps for warmup
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_STEPS_RATIO)
    # Track which step we are at globally
    global_step = epoch * len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        current_step = global_step + batch_idx

        # Linear warmup: scale LR from 0 to LEARNING_RATE during warmup steps
        if current_step < warmup_steps:
            warmup_factor = float(current_step) / float(max(1, warmup_steps - 1))
            # Adjust LR for each parameter group
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * warmup_factor

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    all_val_preds = []
    all_val_labels = []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)

            total_val_loss += loss.item()

            # Clipped softmax for log loss
            probs = torch.softmax(logits, dim=1)
            eps = 1e-15
            probs = torch.clamp(probs, eps, 1 - eps)
            probs = probs / probs.sum(dim=1, keepdim=True)

            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_preds_concat = np.vstack(all_val_preds)
    val_labels_concat = np.concatenate(all_val_labels)
    val_log_loss = log_loss(val_labels_concat, val_preds_concat)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - Val Log Loss: {val_log_loss:.6f}")

    # Update scheduler (warmup is handled per-step before cosine annealing)
    scheduler.step()

    # Early stopping check
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"  -> New best model (Val Log Loss: {best_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print(f"\nBest Validation Log Loss: {best_val_loss:.6f}")

# ============================================================
# 8. LOAD BEST MODEL FOR TEST INFERENCE
# ============================================================
print("Loading best model for test inference...")
model.load_state_dict(best_model_state)
model.to(device)
model.eval()


# ============================================================
# 9. TEST INFERENCE
# ============================================================
print("Performing test inference...")
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with autocast():
            logits = model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=1)
        eps = 1e-15
        probs = torch.clamp(probs, eps, 1 - eps)
        probs = probs / probs.sum(dim=1, keepdim=True)

        all_test_preds.append(probs.cpu().numpy())

test_pred_proba = np.vstack(all_test_preds)
print(f"Test predictions shape: {test_pred_proba.shape}")


# ============================================================
# 10. CREATE SUBMISSION FILE
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

submission.to_csv(os.path.join(submission_dir, "submission_f691d78c16fd45f9b6627c0fe07e2178.csv"), index=False)
print(f"Submission saved to {submission_dir}/submission_f691d78c16fd45f9b6627c0fe07e2178.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# 11. PRINT FINAL VALIDATION SCORE
# ============================================================
print(f"Final Validation Log Loss: {best_val_loss:.6f}")
