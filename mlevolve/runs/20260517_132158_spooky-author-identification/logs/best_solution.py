import numpy as np
import pandas as pd
import re
import os
import gc
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import log_loss
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# DATA PROCESSING AND FEATURE ENGINEERING
# =====================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors distribution:\n{train_df['author'].value_counts()}")

# =====================
# TEXT PREPROCESSING
# =====================

def clean_text(text):
    """Clean text while preserving author-specific patterns"""
    if not isinstance(text, str):
        return ""
    text = str(text)
    # Fix common abbreviations and contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'twas", " it was", text)
    # Preserve em-dashes and semicolons (stylistic markers)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Apply cleaning (only basic text cleaning is safe before split)
train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# =====================
# ENCODE TARGET (safe before split - label encoding is not data leakage)
# =====================

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
author_mapping = dict(
    zip(label_encoder.classes_, [int(x) for x in label_encoder.transform(label_encoder.classes_)])
)
print(f"Author encoding: {author_mapping}")

# Save label encoder
os.makedirs("./working", exist_ok=True)
with open("./working/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("./working/author_encoding.json", "w") as f:
    json.dump(author_mapping, f)

# =====================
# TRAINING CONFIGURATION
# =====================

print("Setting up 5-fold cross-validation training...")

# Use StratifiedKFold for cross-validation
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# We'll keep test data aside for final inference
test_texts = test_df["text_clean"].values

print(f"Total training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Number of folds: {N_FOLDS}")

# Clean up memory
gc.collect()

# =====================
# MODEL DESIGN: MultiPoolingDeBERTa
# =====================

class MultiPoolingDeBERTa(nn.Module):
    """
    DeBERTa-v3-large with multi-pooling strategy for author classification.
    Combines CLS pooling, mean pooling, and attention-weighted pooling
    to capture both sentence-level and token-level stylometric features.
    Includes a hierarchical Transformer encoder layer to refine representations.
    """

    def __init__(
        self, model_name="microsoft/deberta-v3-large", num_labels=3, dropout=0.15
    ):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        # Load pretrained weights with strict=False for custom architecture
        self.load_pretrained_weights(model_name, device)
        self.hidden_size = self.deberta.config.hidden_size

        # Hierarchical encoder module (single-layer Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hierarchical_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )

        self.pool_norm = nn.LayerNorm(self.hidden_size)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Dropout(dropout * 0.8),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.hidden_size // 4, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.attention_weights:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialize hierarchical encoder with small normal distribution
        for p in self.hierarchical_encoder.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        # Apply hierarchical encoder (re-encode token representations)
        # Use attention_mask to prevent attending to padding tokens
        src_key_padding_mask = (attention_mask == 0).bool()
        last_hidden = self.hierarchical_encoder(
            last_hidden,
            src_key_padding_mask=src_key_padding_mask,
        )

        # CLS pooling
        cls_pool = last_hidden[:, 0, :]

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        mask_sum = mask_expanded.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1e-9)
        mean_pool = (last_hidden * mask_expanded).sum(dim=1) / mask_sum

        # Attention-weighted pooling (use -1e4 for AMP compatibility)
        attn_scores = self.attention_weights(last_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_pool = (last_hidden * attn_weights.unsqueeze(-1)).sum(dim=1)
        # Guard against NaN from empty sequences
        attn_pool = torch.nan_to_num(attn_pool, nan=0.0)

        # Normalize pooling outputs (handle potential NaN)
        cls_pool = torch.nan_to_num(cls_pool, nan=0.0)
        mean_pool = torch.nan_to_num(mean_pool, nan=0.0)
        attn_pool = torch.nan_to_num(attn_pool, nan=0.0)
        cls_pool = self.pool_norm(cls_pool)
        mean_pool = self.pool_norm(mean_pool)
        attn_pool = self.pool_norm(attn_pool)

        # Concatenate all pooling strategies
        combined_pool = torch.cat([cls_pool, mean_pool, attn_pool], dim=1)

        # Multi-sample dropout for training (K=5 stochastic forward passes)
        if self.training:
            K = 5
            logits_list = []
            for _ in range(K):
                # Each pass through classifier has different dropout mask
                logits_k = self.classifier(combined_pool)
                logits_list.append(logits_k)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            logits = self.classifier(combined_pool)
        return logits

    def load_pretrained_weights(self, model_name, device):
        """Load pretrained weights with strict=False to handle custom architecture modifications."""
        try:
            pretrained = AutoModel.from_pretrained(model_name)
            pretrained_state = pretrained.state_dict()
            model_state = self.state_dict()
            filtered = {}
            for k, v in pretrained_state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered[k] = v
            self.load_state_dict(filtered, strict=False)
            del pretrained
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Warning during pretrained weight loading: {e}")
            print("  Continuing with randomly initialized weights for unmatched layers.")

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.05, weight=None, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# =====================
# DATASET CLASS
# =====================

class AuthorDataset(Dataset):
    """Dataset for author classification with DeBERTa"""

    def __init__(self, texts, labels=None, tokenizer=None, max_length=256, augment=False, word_dropout_prob=0.1):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.word_dropout_prob = word_dropout_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Apply word-dropout augmentation (only during training)
        if self.augment and np.random.random() < self.word_dropout_prob:
            # Split text into words
            words = text.split()
            if len(words) > 5:  # Only augment if enough words
                # Randomly drop tokens with word_dropout probability
                keep_mask = np.random.random(len(words)) > self.word_dropout_prob
                # Always keep at least some words
                if keep_mask.sum() >= 2:
                    words = [w for w, keep in zip(words, keep_mask) if keep]
                    text = " ".join(words)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Apply token-level cutoff augmentation (only during training)
        if self.augment and self.labels is not None and np.random.random() < 0.5:
            input_ids = encoding["input_ids"].flatten()
            # Get the actual sequence length (excluding padding)
            seq_len = (input_ids != self.tokenizer.pad_token_id).sum().item()
            if seq_len > 10:
                # Mask 15% of tokens (excluding special tokens)
                mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
                num_to_mask = max(1, int(0.15 * seq_len))
                # Choose random positions in the non-padding part
                mask_positions = np.random.choice(seq_len, size=num_to_mask, replace=False)
                for pos in mask_positions:
                    # Don't mask special tokens (CLS, SEP, etc.)
                    if pos > 0 and pos < seq_len - 1:
                        input_ids[pos] = mask_token_id
            encoding["input_ids"] = input_ids.unsqueeze(0)

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# =====================
# 5-FOLD CROSS-VALIDATION TRAINING
# =====================

print("\n" + "=" * 60)
print("5-Fold Cross-Validation Training with DeBERTa-v3-large Multi-Pooling")
print("=" * 60)

# Configuration
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.1

# Initialize tokenizer
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded: {model_name}")

# Prepare test dataset for later inference
test_dataset = AuthorDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Track all fold results
fold_results = []
best_fold_logloss = float("inf")
best_fold_idx = -1
best_model_state = None
all_fold_val_preds = []  # For OOF predictions
all_fold_val_labels = []  # For OOF labels
all_fold_test_preds = []  # For test ensemble

# SWA: store checkpoints for last 3 epochs per fold
SWA_EPOCHS = 3

# Iterate over folds
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["author_encoded"])):
    print("\n" + "=" * 50)
    print(f"  Fold {fold_idx + 1}/{N_FOLDS}")
    print("=" * 50)

    # Get split data
    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]

    y_train_fold = train_split["author_encoded"].values
    y_val_fold = val_split["author_encoded"].values

    train_texts_fold = train_split["text_clean"].values
    val_texts_fold = val_split["text_clean"].values

    # Compute class weights for balanced training
    class_counts_fold = np.bincount(y_train_fold)
    class_weights_fold = torch.tensor(
        [max(class_counts_fold) / count for count in class_counts_fold], dtype=torch.float32
    ).to(device)
    print(f"  Fold {fold_idx + 1} class counts: {class_counts_fold}")
    print(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    # Initialize model for this fold
    model = MultiPoolingDeBERTa(model_name=model_name, num_labels=3, dropout=0.15)
    model = model.to(device)
    # Load pretrained backbone weights with strict=False (custom architecture modifications)
    model.load_pretrained_weights(model_name, device)

    if fold_idx == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets with word-dropout augmentation for training
    train_dataset = AuthorDataset(train_texts_fold, y_train_fold, tokenizer, MAX_LENGTH, augment=True, word_dropout_prob=0.1)
    val_dataset = AuthorDataset(val_texts_fold, y_val_fold, tokenizer, MAX_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Loss function with label smoothing
    criterion = LabelSmoothingLoss(
        num_classes=3, smoothing=LABEL_SMOOTHING, weight=class_weights_fold
    )

    # Optimizer with layer-wise decay
    def get_optimizer_with_layerwise_decay(
        model, learning_rate=2e-5, weight_decay=0.01, layerwise_decay=0.95
    ):
        named_params = list(model.named_parameters())
        classifier_params = []
        attention_params = []
        pool_norm_params = []
        backbone_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            if "classifier" in name:
                classifier_params.append(param)
            elif "attention_weights" in name:
                attention_params.append(param)
            elif "pool_norm" in name:
                pool_norm_params.append(param)
            else:
                backbone_params.append((name, param))

        optimizer_grouped_parameters = [
            {
                "params": classifier_params,
                "lr": learning_rate,
                "weight_decay": weight_decay * 0.5,
            },
            {
                "params": attention_params,
                "lr": learning_rate,
                "weight_decay": weight_decay * 0.5,
            },
            {
                "params": pool_norm_params,
                "lr": learning_rate * 0.8,
                "weight_decay": weight_decay,
            },
        ]

        layer_params = {}
        for name, param in backbone_params:
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num not in layer_params:
                    layer_params[layer_num] = []
                layer_params[layer_num].append(param)
            elif "embeddings" in name:
                optimizer_grouped_parameters.append(
                    {
                        "params": param,
                        "lr": learning_rate * (layerwise_decay**12),
                        "weight_decay": weight_decay,
                    }
                )
            else:
                optimizer_grouped_parameters.append(
                    {
                        "params": param,
                        "lr": learning_rate * (layerwise_decay**12),
                        "weight_decay": weight_decay,
                    }
                )

        num_layers = len(layer_params)
        for layer_num in sorted(layer_params.keys()):
            decay_factor = layerwise_decay ** (num_layers - 1 - layer_num)
            optimizer_grouped_parameters.append(
                {
                    "params": layer_params[layer_num],
                    "lr": learning_rate * decay_factor,
                    "weight_decay": weight_decay,
                }
            )

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
        return optimizer

    # Differential layer warmup: keep first 18 DeBERTa layers at constant low LR,
    # warmup last 6 layers (19-24) and classifier/attention head over first 20% steps
    optimizer = get_optimizer_with_layerwise_decay(model, LEARNING_RATE, WEIGHT_DECAY)

    # Learning rate scheduler with differential warmup
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # Custom scheduler that handles differential warmup:
    # - Early layers (backbone layers 0-17): keep at constant initial_lr (no warmup)
    # - Late layers (backbone layers 18-23) and classifier/attention: apply linear warmup
    # - After warmup: cosine decay for all groups
    class DifferentialWarmupScheduler:
        def __init__(self, optimizer, num_warmup_steps, num_training_steps, model):
            self.optimizer = optimizer
            self.num_warmup_steps = max(1, num_warmup_steps)
            self.num_training_steps = max(1, num_training_steps)
            self._step = 0

            # Identify parameter groups: early layers (backbone layers 0-17), late layers (18-23), classifier/attention
            # We use the names of the first parameter in each group to determine the group type
            self.group_types = []
            for group in optimizer.param_groups:
                params = group['params']
                if len(params) > 0:
                    param = params[0]
                    # Check if this param belongs to classifier or attention
                    found = False
                    for name, p in model.named_parameters():
                        if p is param:
                            if 'classifier' in name or 'attention_weights' in name:
                                self.group_types.append('classifier_attention')
                                found = True
                            elif 'layer' in name:
                                # Extract layer number from name
                                match = re.search(r'layer\.(\d+)', name)
                                if match:
                                    layer_num = int(match.group(1))
                                    if layer_num <= 17:
                                        self.group_types.append('early')
                                    else:
                                        self.group_types.append('late')
                                    found = True
                                else:
                                    self.group_types.append('mid')
                                    found = True
                            else:
                                self.group_types.append('mid')
                                found = True
                            break
                    if not found:
                        self.group_types.append('mid')
                else:
                    self.group_types.append('mid')

            # Store initial LRs for each group
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            # Set min_lr for early layers (keep them nearly constant)
            self.min_lrs = []
            for gtype, base_lr in zip(self.group_types, self.base_lrs):
                if gtype == 'early':
                    self.min_lrs.append(base_lr * 0.01)  # Nearly freeze early layers
                elif gtype == 'classifier_attention':
                    self.min_lrs.append(base_lr * 0.05)  # Start from near-zero
                elif gtype == 'late':
                    self.min_lrs.append(base_lr * 0.5)   # Started with higher decay factor
                else:
                    self.min_lrs.append(base_lr * 0.1)   # Mid layers

        def step(self, step_count=None):
            if not hasattr(self, '_step'):
                self._step = 0
            if step_count is not None:
                self._step = step_count
            else:
                self._step += 1
            for i, group in enumerate(self.optimizer.param_groups):
                base_lr = self.base_lrs[i]
                min_lr = self.min_lrs[i]
                gtype = self.group_types[i]

                if self._step <= self.num_warmup_steps:
                    warmup_progress = float(self._step) / float(self.num_warmup_steps)
                    if gtype == 'early':
                        # Keep nearly frozen
                        lr = base_lr * (0.01 + 0.99 * warmup_progress * 0.1)
                    elif gtype == 'classifier_attention':
                        # Full warmup from near-zero
                        lr = base_lr * (0.05 + 0.95 * warmup_progress)
                    elif gtype == 'late':
                        # Warmup from half base
                        lr = base_lr * (0.5 + 0.5 * warmup_progress)
                    else:
                        # Warmup from 10% base
                        lr = base_lr * (0.1 + 0.9 * warmup_progress)
                else:
                    progress = float(self._step - self.num_warmup_steps) / float(
                        self.num_training_steps - self.num_warmup_steps
                    )
                    # Cosine decay to min_lr
                    lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))

                group['lr'] = lr

        def get_last_lr(self):
            return [group['lr'] for group in self.optimizer.param_groups]

        def state_dict(self):
            return {'step': getattr(self, '_step', 0)}

        def load_state_dict(self, state_dict):
            self._step = state_dict.get('step', 0)

    scheduler = DifferentialWarmupScheduler(optimizer, warmup_steps, total_steps, model)

    # Mixed precision
    scaler = GradScaler()

    # Early stopping
    best_val_logloss_fold = float("inf")
    patience = 4
    patience_counter = 0
    best_model_state_fold = None
    best_epoch_fold = 0

    # Training loop for this fold
    print(f"\n  Starting fold {fold_idx + 1} training")
    print(f"  Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")

    # Initialize SWA checkpoint list for this fold
    swa_checkpoints = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Step the differential warmup scheduler (handles per-group LR)
                scheduler.step()

            total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        avg_train_loss = total_train_loss / len(train_loader)

        # Collect SWA checkpoint for last SWA_EPOCHS epochs
        if epoch >= EPOCHS - SWA_EPOCHS:
            swa_checkpoints.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        # Validation
        model.eval()
        total_val_loss = 0
        all_val_preds_fold = []
        all_val_labels_fold = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()

                    probs = F.softmax(logits, dim=-1)
                    all_val_preds_fold.append(probs.cpu().numpy())
                    all_val_labels_fold.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_preds_fold = np.concatenate(all_val_preds_fold, axis=0)
        val_labels_fold = np.concatenate(all_val_labels_fold, axis=0)
        val_preds_fold = np.clip(val_preds_fold, 1e-15, 1 - 1e-15)
        val_preds_fold = val_preds_fold / val_preds_fold.sum(axis=1, keepdims=True)
        val_logloss_fold = log_loss(val_labels_fold, val_preds_fold)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Fold {fold_idx+1} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss_fold:.4f} | LR: {current_lr:.2e}")

        if val_logloss_fold < best_val_logloss_fold:
            best_val_logloss_fold = val_logloss_fold
            patience_counter = 0
            best_epoch_fold = epoch + 1
            best_model_state_fold = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✓ New best! Val LogLoss: {best_val_logloss_fold:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered at epoch {epoch+1}")
                break

    fold_results.append({
        "fold": fold_idx + 1,
        "best_epoch": best_epoch_fold,
        "best_val_logloss": best_val_logloss_fold
    })
    print(f"  Fold {fold_idx + 1} best Val LogLoss: {best_val_logloss_fold:.4f} (epoch {best_epoch_fold})")

    # Apply SWA: average the last SWA_EPOCHS checkpoints
    if len(swa_checkpoints) >= 2:
        print(f"  Applying SWA over {len(swa_checkpoints)} checkpoints...")
        # Average checkpoint state dicts (all tensors already on CPU)
        swa_state_dict = {}
        for key in swa_checkpoints[0].keys():
            # Stack tensors and compute mean along new dimension
            stacked = torch.stack([ckpt[key].float() for ckpt in swa_checkpoints], dim=0)
            swa_state_dict[key] = stacked.mean(dim=0)

        # Temporarily load SWA weights for evaluation
        current_device = next(model.parameters()).device
        model.load_state_dict(swa_state_dict)
        model = model.to(current_device)
        model.eval()

        # Compute validation logloss with SWA model
        total_val_loss_swa = 0
        all_val_preds_swa = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    total_val_loss_swa += loss.item()
                    probs = F.softmax(logits, dim=-1)
                    all_val_preds_swa.append(probs.cpu().numpy())
        avg_val_loss_swa = total_val_loss_swa / len(val_loader)
        val_preds_swa = np.concatenate(all_val_preds_swa, axis=0)
        val_preds_swa = np.clip(val_preds_swa, 1e-15, 1 - 1e-15)
        val_preds_swa = val_preds_swa / val_preds_swa.sum(axis=1, keepdims=True)
        val_logloss_swa = log_loss(val_labels_fold, val_preds_swa)
        print(f"  SWA Val LogLoss: {val_logloss_swa:.4f} (best epoch: {best_val_logloss_fold:.4f})")

        # Use SWA if it improves validation
        if val_logloss_swa < best_val_logloss_fold:
            print(f"    ✓ SWA improved validation! Using SWA model.")
            best_val_logloss_fold = val_logloss_swa
            best_model_state_fold = swa_state_dict
        else:
            # Fall back to best epoch model
            print(f"    SWA did not improve. Using best epoch model (epoch {best_epoch_fold}).")
            model.load_state_dict(best_model_state_fold)
            model = model.to(device)
            model.eval()
    else:
        # Fall back to best epoch model (fewer than SWA_EPOCHS checkpoints available)
        print(f"  Not enough SWA checkpoints: {len(swa_checkpoints)}. Using best epoch model (epoch {best_epoch_fold}).")
        model.load_state_dict(best_model_state_fold)
        model = model.to(device)
        model.eval()

    # Generate OOF predictions for this fold

    # Generate OOF predictions for this fold
    oof_preds_fold = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
            oof_preds_fold.append(probs.cpu().numpy())

    oof_preds_fold = np.concatenate(oof_preds_fold, axis=0)
    all_fold_val_preds.append(oof_preds_fold)
    all_fold_val_labels.append(val_labels_fold)

    # Generate test predictions for this fold
    test_preds_fold = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
            test_preds_fold.append(probs.cpu().numpy())

    test_preds_fold = np.concatenate(test_preds_fold, axis=0)
    all_fold_test_preds.append(test_preds_fold)

    # Track best model across folds
    if best_val_logloss_fold < best_fold_logloss:
        best_fold_logloss = best_val_logloss_fold
        best_fold_idx = fold_idx
        best_model_state = best_model_state_fold

    # Clean up memory for this fold
    del model, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =====================
# FOLD RESULTS SUMMARY
# =====================
print("\n" + "=" * 60)
print("Cross-Validation Results Summary")
print("=" * 60)
for result in fold_results:
    print(f"Fold {result['fold']}: Best Val LogLoss = {result['best_val_logloss']:.4f} (epoch {result['best_epoch']})")

avg_logloss = np.mean([r["best_val_logloss"] for r in fold_results])
std_logloss = np.std([r["best_val_logloss"] for r in fold_results])
print(f"\nAverage Val LogLoss across folds: {avg_logloss:.4f} ± {std_logloss:.4f}")
print(f"Best fold: {best_fold_idx + 1} with LogLoss: {best_fold_logloss:.4f}")

# =====================
# COMPUTE OOF SCORE
# =====================
# Construct full OOF predictions in original order
oof_preds_full = np.zeros((len(train_df), 3))
for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["author_encoded"])):
    oof_preds_full[val_idx] = all_fold_val_preds[fold_idx]

# Clip and normalize OOF predictions
oof_preds_full = np.clip(oof_preds_full, 1e-15, 1 - 1e-15)
oof_preds_full = oof_preds_full / oof_preds_full.sum(axis=1, keepdims=True)
oof_labels_full = train_df["author_encoded"].values
oof_logloss = log_loss(oof_labels_full, oof_preds_full)
print(f"\nOOF Validation LogLoss: {oof_logloss:.6f}")

# =====================
# TEST ENSEMBLE PREDICTION
# =====================
print("\n" + "=" * 60)
print("Creating Ensemble Test Predictions")
print("=" * 60)

# Average predictions from all folds (simple average)
ensemble_test_preds = np.mean(all_fold_test_preds, axis=0)

# Clip and normalize
ensemble_test_preds = np.clip(ensemble_test_preds, 1e-15, 1 - 1e-15)
row_sums = ensemble_test_preds.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)
ensemble_test_preds = ensemble_test_preds / row_sums

print(f"Ensemble test predictions shape: {ensemble_test_preds.shape}")

# Use best fold model for final predictions as well
print(f"\nUsing best fold (Fold {best_fold_idx + 1}) for final test predictions")

# Load best model
model = MultiPoolingDeBERTa(model_name=model_name, num_labels=3, dropout=0.15)
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

# Generate best fold test predictions
best_fold_test_preds = all_fold_test_preds[best_fold_idx]

# Clip and normalize best fold predictions
best_fold_test_preds = np.clip(best_fold_test_preds, 1e-15, 1 - 1e-15)
row_sums = best_fold_test_preds.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)
best_fold_test_preds = best_fold_test_preds / row_sums

# =====================
# CREATE SUBMISSION FILE
# =====================
print("\n" + "=" * 60)
print("Creating Submission File")
print("=" * 60)

# Create submission directory if it doesn't exist
os.makedirs("./submission", exist_ok=True)

# Load test IDs
test_ids = test_df["id"].values

# Get class names in the correct order from label encoder
class_names = label_encoder.classes_  # e.g., ['EAP', 'HPL', 'MWS']

# Use ensemble predictions for the final submission (more robust)
submission_dict = {"id": test_ids}
for i, class_name in enumerate(class_names):
    submission_dict[class_name] = ensemble_test_preds[:, i]

# Create submission dataframe
submission_df = pd.DataFrame(submission_dict)

# Ensure columns are in the expected order
submission_df = submission_df[["id", "EAP", "HPL", "MWS"]]

# Save submission file
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Submission columns: {submission_df.columns.tolist()}")

# Verify submission format
expected_columns = ["id", "EAP", "HPL", "MWS"]
assert (
    list(submission_df.columns) == expected_columns
), f"Column mismatch: {list(submission_df.columns)}"
assert len(submission_df) == len(
    test_df
), f"Row count mismatch: {len(submission_df)} vs {len(test_df)}"
print("✓ Submission format verified!")

# =====================
# CLEANUP
# =====================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")

print("\n" + "=" * 60)
print("Training and Evaluation Complete")
print("=" * 60)

# Final validation score (required for parser)
print(f"OOF Validation LogLoss: {oof_logloss:.6f}")
print(f"Best Fold Validation LogLoss: {best_fold_logloss:.6f}")