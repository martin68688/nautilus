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
# DATA PROCESSING - Raw text only, no feature engineering
# ============================================================
# Use all training data for pseudo-labeling and final training
train_texts = train_df["text"].values
train_labels = train_df["author"].values
test_texts = test_df["text"].values

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Create a mini validation set (500 samples) for early stopping during initial training
# Use stratified split to maintain class distribution
skf_mini = StratifiedKFold(n_splits=len(train_texts) // 500, shuffle=True, random_state=42)
mini_train_idx, mini_val_idx = next(iter(skf_mini.split(train_texts, train_labels_encoded)))

# Ensure mini_val is exactly 500 samples
mini_val_idx = mini_val_idx[:500]
mini_train_idx = np.setdiff1d(np.arange(len(train_texts)), mini_val_idx)

# For the final training, we use all data (full train + pseudo-labeled test)
# These are the initial splits for the first model
train_texts_initial = train_texts[mini_train_idx]
val_texts_initial = train_texts[mini_val_idx]
train_labels_initial = train_labels_encoded[mini_train_idx]
val_labels_initial = train_labels_encoded[mini_val_idx]

os.makedirs("./working", exist_ok=True)
np.save("./working/test_ids.npy", test_df["id"].values)
np.save("./working/label_classes.npy", label_encoder.classes_)

print(f"Initial train samples: {len(train_texts_initial)}")
print(f"Initial val samples: {len(val_texts_initial)}")
print(f"Test samples: {len(test_texts)}")
print(f"Total available training samples: {len(train_texts)}")

# ============================================================
# MODEL DESIGN - SpookyAuthorClassifier
# ============================================================
NUM_CLASSES = 3
PRETRAINED_MODEL = "microsoft/deberta-v3-small"
MAX_SEQ_LENGTH = 128
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
# TRAINING AND EVALUATION with Pseudo-Labeling and Warm-Up
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


def train_model(
    train_texts_input,
    val_texts_input,
    train_labels_input,
    val_labels_input,
    test_texts_input=None,
    is_pseudo_round=False,
):
    """Train a model with warm-up unfreezing and differential learning rates."""
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts_input, tokenizer, max_length
    )
    val_input_ids, val_attention_mask = tokenize_texts(
        val_texts_input, tokenizer, max_length
    )

    train_labels_tensor = torch.tensor(train_labels_input, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels_input, dtype=torch.long)

    batch_size = 8
    train_dataset = TensorDataset(
        train_input_ids, train_attention_mask, train_labels_tensor
    )
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model with frozen backbone
    model = SpookyAuthorClassifier(num_classes=NUM_CLASSES, freeze_bert=True)
    model.initialize_backbone()
    for param in model.deberta.parameters():
        param.requires_grad = False
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,}, Total params: {total_params:,}")

    class_weights = compute_class_weights(train_labels_input).to(device)
    criterion = CombinedLoss(
        label_smoothing=0.1,
        focal_gamma=2.0,
        smoothing_weight=0.7,
        focal_weight=0.3,
        class_weights=class_weights,
    )

    scaler = GradScaler()
    num_epochs = 20
    warmup_epochs = 5
    best_val_loss = float("inf")
    best_model_state = None
    patience = 6
    patience_counter = 0

    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs

    # Build initial optimizer (only head params, backbone frozen)
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
            ],
            "lr": 2e-5,
            "weight_decay": 0.01,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n===== Starting Training (Pseudo round: {is_pseudo_round}) =====")
    for epoch in range(num_epochs):
        # Unfreeze backbone after warm-up epochs
        if epoch == warmup_epochs:
            print("Unfreezing backbone for fine-tuning with differential LR...")
            for param in model.deberta.parameters():
                param.requires_grad = True
            # Rebuild optimizer with differential LR for backbone vs head
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and "deberta" in n
                    ],
                    "lr": 1e-6,
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and "deberta" not in n
                    ],
                    "lr": 2e-5,
                    "weight_decay": 0.01,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters)
            # Recalculate scheduler with remaining steps
            remaining_steps = total_steps - (epoch * len(train_loader))
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=remaining_steps,
            )

        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
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
                with autocast():
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

    print(f"Best validation log-loss: {best_val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # Generate test predictions if needed
    test_probs_normalized = None
    if test_texts_input is not None:
        test_input_ids, test_attention_mask = tokenize_texts(
            test_texts_input, tokenizer, max_length
        )
        test_dataset = TensorDataset(test_input_ids, test_attention_mask)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        print("Generating test predictions...")
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

    return best_val_loss, best_model_state, test_probs_normalized


# === Phase 1: Train initial model on full training data (with mini val set for early stopping) ===
print("\n========== PHASE 1: Initial Training on Full Training Data ==========")
best_val_loss_initial, best_model_state_initial, test_probs_initial = train_model(
    train_texts_initial,
    val_texts_initial,
    train_labels_initial,
    val_labels_initial,
    test_texts_input=test_texts,
    is_pseudo_round=False,
)

# === Phase 2: Final training on full training data (no pseudo-labeling to avoid data leakage) ===
print("\n========== PHASE 2: Final Training on Full Training Data ==========")
# Use the full training data (no test data contamination)
# Create a proper stratified split from training data only
skf_final = StratifiedKFold(n_splits=len(train_texts) // 500, shuffle=True, random_state=42)
final_train_idx, final_val_idx = next(iter(skf_final.split(train_texts, train_labels_encoded)))
final_val_idx = final_val_idx[:500]
final_train_idx = np.setdiff1d(np.arange(len(train_texts)), final_val_idx)

final_train_texts = train_texts[final_train_idx]
final_val_texts = train_texts[final_val_idx]
final_train_labels = train_labels_encoded[final_train_idx]
final_val_labels = train_labels_encoded[final_val_idx]

best_val_loss_final, best_model_state_final, test_probs_final = train_model(
    final_train_texts,
    final_val_texts,
    final_train_labels,
    final_val_labels,
    test_texts_input=test_texts,
    is_pseudo_round=False,
)

# === Generate Final Submission ===
print("\n========== GENERATING FINAL SUBMISSION ==========")
test_ids = np.load("./working/test_ids.npy", allow_pickle=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final best validation log-loss: {best_val_loss_final:.6f}")