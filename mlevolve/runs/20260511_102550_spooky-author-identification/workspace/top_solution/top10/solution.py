import pandas as pd
import numpy as np
import re
import os
import warnings
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# ============================================================
# DATA PROCESSING - No engineered features, only raw text for transformer
# ============================================================
X = train_df["text"].values
y = train_df["author"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(X, y))[0]

train_texts = X[train_idx]
val_texts = X[val_idx]
train_labels = y[train_idx]
val_labels = y[val_idx]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

test_texts = test_df["text"].values

os.makedirs("./working", exist_ok=True)
np.save("./working/train_labels.npy", train_labels_encoded)
np.save("./working/val_labels.npy", val_labels_encoded)
np.save("./working/test_ids.npy", test_df["id"].values)
np.save("./working/label_classes.npy", label_encoder.classes_)

print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")

# ============================================================
# MODEL DESIGN - SpookyAuthorClassifier
# ============================================================
NUM_CLASSES = 3
PRETRAINED_MODEL = "microsoft/deberta-v3-small"
MAX_SEQ_LENGTH = 256
HIDDEN_SIZE = 768


class MultiScaleConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2
        )
        self.conv4 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=7, padding=3
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        c1 = self.relu(self.conv1(x_perm))
        c2 = self.relu(self.conv2(x_perm))
        c3 = self.relu(self.conv3(x_perm))
        c4 = self.relu(self.conv4(x_perm))
        pooled1 = F.adaptive_max_pool1d(c1, 1).squeeze(-1)
        pooled2 = F.adaptive_max_pool1d(c2, 1).squeeze(-1)
        pooled3 = F.adaptive_max_pool1d(c3, 1).squeeze(-1)
        pooled4 = F.adaptive_max_pool1d(c4, 1).squeeze(-1)
        combined = torch.cat([pooled1, pooled2, pooled3, pooled4], dim=-1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        return combined


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        scores = self.attention_weights(hidden_states).squeeze(-1)
        if attention_mask is not None:
            # Use -10000.0 instead of -1e9 to avoid float16 overflow in mixed precision
            scores = scores.masked_fill(attention_mask == 0, -10000.0)
        attention_weights = F.softmax(scores, dim=-1)
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(
            1
        )
        return weighted_sum


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_bert=True, dropout=0.3):
        super().__init__()
        self.deberta = None
        self.freeze_bert = freeze_bert
        self.bert_dim = HIDDEN_SIZE
        self.multi_scale_conv = MultiScaleConvBlock(
            input_dim=self.bert_dim, hidden_dim=256, dropout=dropout
        )
        self.attention_pool = AttentionPooling(self.bert_dim)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 + self.bert_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_backbone(self, model_name=PRETRAINED_MODEL):
        self.deberta = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        conv_features = self.multi_scale_conv(sequence_output)
        attended_features = self.attention_pool(sequence_output, attention_mask)
        cls_features = sequence_output[:, 0, :]
        combined_features = torch.cat([conv_features, attended_features], dim=-1)
        logits = self.classifier(combined_features)
        if return_embeddings:
            return logits, combined_features
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0)
            true_dist = true_dist * (1.0 - self.smoothing) + self.smoothing / n_classes
            if self.weight is not None:
                weights = self.weight[target].unsqueeze(1)
                true_dist = true_dist * weights
        log_probs = F.log_softmax(pred, dim=-1)
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        label_smoothing=0.1,
        focal_gamma=2.0,
        smoothing_weight=0.7,
        focal_weight=0.3,
        class_weights=None,
    ):
        super().__init__()
        self.smoothing_loss = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=class_weights
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.smoothing_weight = smoothing_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        loss1 = self.smoothing_loss(pred, target)
        loss2 = self.focal_loss(pred, target)
        return self.smoothing_weight * loss1 + self.focal_weight * loss2


def compute_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        weights[cls] = total / (num_classes * count)
    weights = weights / weights.mean()
    return weights.float()


# ============================================================
# TRAINING AND EVALUATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
max_length = MAX_SEQ_LENGTH


def tokenize_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


print("Tokenizing training data...")
train_input_ids, train_attention_mask = tokenize_texts(
    train_texts, tokenizer, max_length
)
print("Tokenizing validation data...")
val_input_ids, val_attention_mask = tokenize_texts(val_texts, tokenizer, max_length)
print("Tokenizing test data...")
test_input_ids, test_attention_mask = tokenize_texts(test_texts, tokenizer, max_length)

train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)
val_labels_tensor = torch.tensor(val_labels_encoded, dtype=torch.long)

batch_size = 8
train_dataset = TensorDataset(
    train_input_ids, train_attention_mask, train_labels_tensor
)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels_tensor)
test_dataset = TensorDataset(test_input_ids, test_attention_mask)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

print(
    f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
)

# Initialize model
model = SpookyAuthorClassifier(num_classes=NUM_CLASSES, freeze_bert=True)
model.initialize_backbone()
model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,}, Total params: {total_params:,}")

class_weights = compute_class_weights(train_labels_encoded).to(device)
criterion = CombinedLoss(
    label_smoothing=0.1,
    focal_gamma=2.0,
    smoothing_weight=0.7,
    focal_weight=0.3,
    class_weights=class_weights,
)

# Stage 1: Freeze backbone, only train head
for param in model.deberta.parameters():
    param.requires_grad = False

head_params = [p for n, p in model.named_parameters() if p.requires_grad and "deberta" not in n]
optimizer = AdamW([{"params": head_params, "lr": 2e-5, "weight_decay": 0.01}], lr=2e-5)

scaler = GradScaler()
num_epochs = 20
best_val_loss = float("inf")
best_model_state = None
patience = 6
patience_counter = 0

# Note: gradient checkpointing disabled to avoid computation graph conflicts with gradient accumulation

stage_1_epochs = 5
stage_2_activated = False
gradient_accumulation_steps = 4

print("\n===== Starting Training =====")
for epoch in range(num_epochs):
    # --- Stage 2 activation: unfreeze backbone with differential LR and cosine schedule ---
    if epoch >= stage_1_epochs and not stage_2_activated:
        stage_2_activated = True
        print(f"\n===== Stage 2: Unfreezing backbone (epoch {epoch+1}) =====")
        for param in model.deberta.parameters():
            param.requires_grad = True
        # Create two parameter groups: backbone (low LR) and head (higher LR)
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "deberta" in n:
                backbone_params.append(p)
            else:
                head_params.append(p)
        optimizer = AdamW([
            {"params": backbone_params, "lr": 1e-6, "weight_decay": 0.01},
            {"params": head_params, "lr": 2e-5, "weight_decay": 0.01},
        ], lr=2e-5)
        # Cosine scheduler with 10% warmup of remaining steps
        remaining_epochs = num_epochs - epoch
        total_steps = len(train_loader) * remaining_epochs // gradient_accumulation_steps
        warmup_steps = int(0.1 * total_steps)
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print(f"Stage 2: remaining_epochs={remaining_epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    model.train()
    total_train_loss = 0.0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        loss = None  # Reset loss for accumulation

        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        # Scale loss to account for gradient accumulation
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if stage_2_activated:
                scheduler.step()

        total_train_loss += loss.item() * gradient_accumulation_steps
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            num_val_batches += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_val_preds.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_preds_concat = np.concatenate(all_val_preds, axis=0)
    val_labels_concat = np.concatenate(all_val_labels, axis=0)

    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds_concat, epsilon, 1 - epsilon)
    row_sums = val_preds_clipped.sum(axis=1, keepdims=True)
    val_preds_normalized = val_preds_clipped / row_sums
    val_log_loss = log_loss(val_labels_concat, val_preds_normalized)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f} | LR: {current_lr:.2e}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

print(f"\n===== Training Complete =====")
print(f"Best validation log-loss: {best_val_loss:.6f}")

# Load best model
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Final validation
print("\nComputing final validation metrics...")
all_val_probs = []
all_val_labels_final = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)
        all_val_labels_final.append(labels.cpu().numpy())

val_probs_concat = np.concatenate(all_val_probs, axis=0)
val_labels_concat = np.concatenate(all_val_labels_final, axis=0)

epsilon = 1e-15
val_probs_clipped = np.clip(val_probs_concat, epsilon, 1 - epsilon)
row_sums = val_probs_clipped.sum(axis=1, keepdims=True)
val_probs_normalized = val_probs_clipped / row_sums
final_val_log_loss = log_loss(val_labels_concat, val_probs_normalized)
print(f"Final Validation Log-Loss: {final_val_log_loss:.6f}")

# Test inference
print("\nGenerating test predictions...")
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

test_probs_concat = np.concatenate(all_test_probs, axis=0)
test_probs_clipped = np.clip(test_probs_concat, epsilon, 1 - epsilon)
row_sums_test = test_probs_clipped.sum(axis=1, keepdims=True)
test_probs_normalized = test_probs_clipped / row_sums_test

test_ids = np.load("./working/test_ids.npy", allow_pickle=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_normalized[:, 0],
        "HPL": test_probs_normalized[:, 1],
        "MWS": test_probs_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_log_loss}")