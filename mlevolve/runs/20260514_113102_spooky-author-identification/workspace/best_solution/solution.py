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

# Train/Validation split (StratifiedKFold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large with Custom Head (from Step 2)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention pooling over token hidden states"""

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len) - boolean mask (1 for valid, 0 for padding)
        # Prepare key_padding_mask for MultiheadAttention (True for padded tokens)
        key_padding_mask = ~attention_mask.bool()  # True for padding tokens

        # Apply multi-head attention
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_output = self.attn_dropout(attn_output)
        return attn_output


class SpookyAuthorClassifier(nn.Module):
    """DeBERTa-v3-large with MHA+Mean-Max pooling and multi-sample dropout"""

    def __init__(self, num_authors=3, dropout_rate=0.3, hidden_dim=256, n_dropouts=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            output_hidden_states=True,
            output_attentions=False,
        )

        # Freeze all backbone layers initially, will be unfrozen during two-stage training
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size

        # MHA pooling (learned attention)
        self.mha_pooling = MultiHeadAttentionPooling(self.hidden_size, num_heads=8)

        # Projection from 3 * hidden_size to hidden_dim (after concat mean, max, mha-pooled mean)
        self.projection = nn.Sequential(
            nn.Linear(3 * self.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Intermediate projection after MHA+Mean-Max concatenation
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        for module in [self.projection, self.intermediate, self.classifier]:
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

        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean pooling across sequence
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # (batch, hidden_size)

        # Max pooling across sequence
        # Set padding positions to very negative value for max
        inf_mask = (1.0 - mask_expanded) * -1e9
        max_pooled = torch.max(last_hidden + inf_mask, dim=1)[0]  # (batch, hidden_size)

        # MHA pooling - apply attention and then mean pool the attended output
        mha_output = self.mha_pooling(last_hidden, attention_mask)  # (batch, seq_len, hidden_size)
        mha_mean = torch.sum(mha_output * mask_expanded, dim=1) / sum_mask  # (batch, hidden_size)

        # Concatenate all three representations
        combined = torch.cat([mean_pooled, max_pooled, mha_mean], dim=1)  # (batch, 3*hidden_size)

        # Project down to hidden_dim
        features = self.projection(combined)

        # Apply intermediate projection
        features = self.intermediate(features)

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


class FocalLoss(nn.Module):
    """Focal Loss with adaptive gamma"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


model = SpookyAuthorClassifier(
    num_authors=3, dropout_rate=0.3, hidden_dim=256, n_dropouts=4
)
model.to(device)

# Focal Loss with adaptive gamma
criterion = FocalLoss(gamma=2.0)

# Count parameters for all layers (not just trainable initially)
all_backbone_params = []
all_head_params = []

for name, param in model.named_parameters():
    if "backbone" in name:
        all_backbone_params.append(param)
    else:
        all_head_params.append(param)

print(f"Total backbone params: {sum(p.numel() for p in all_backbone_params):,}")
print(f"Total head params: {sum(p.numel() for p in all_head_params):,}")
print(
    f"Total parameters: {sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# DATASET AND DATALOADER
# ============================================================
import random
import hashlib
import os
import pickle
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Pre-compute back-translation augmented texts
BACK_TRANSLATION_CACHE_PATH = "./working/back_translation_cache.pkl"

# Language pairs for back-translation
LANG_PAIRS = ["fra", "deu", "spa"]  # French, German, Spanish


def generate_back_translation(text, lang_code="fra"):
    """Generate back-translated version of text using Helsinki-NLP models"""
    try:
        # Forward translation
        forward_pipeline = pipeline(
            "translation_en_to_" + lang_code,
            model=f"Helsinki-NLP/opus-mt-en-{lang_code}",
            device=-1,  # Use CPU to avoid GPU memory issues
            max_length=512,
        )
        intermediate = forward_pipeline(text)[0]["translation_text"]

        # Backward translation
        backward_pipeline = pipeline(
            "translation_" + lang_code + "_to_en",
            model=f"Helsinki-NLP/opus-mt-{lang_code}-en",
            device=-1,
            max_length=512,
        )
        result = backward_pipeline(intermediate)[0]["translation_text"]
        return result
    except Exception as e:
        # Fallback to original text if translation fails
        return text


def precompute_back_translation(texts, cache_path=BACK_TRANSLATION_CACHE_PATH, augment_prob=0.15):
    """Pre-compute back-translation augmentations for training texts"""
    # Check if cache exists
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        # Verify cache matches input
        if len(cached_data) == len(texts):
            print(f"Loaded {len(cached_data)} cached back-translation samples")
            return cached_data

    print(f"Pre-computing back-translation augmentations for {len(texts)} texts...")
    print("This may take a while but only runs once...")

    augmented_texts = []
    for i, text in enumerate(texts):
        if random.random() < augment_prob:
            # Randomly select a language pair
            lang = random.choice(LANG_PAIRS)
            aug_text = generate_back_translation(text, lang)
            augmented_texts.append(aug_text)
        else:
            augmented_texts.append(text)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(texts)} texts...")

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(augmented_texts, f)
    print(f"Saved {len(augmented_texts)} back-translation samples to cache")

    return augmented_texts


class SpookyDataset(Dataset):
    def __init__(
        self,
        texts,
        labels=None,
        tokenizer=None,
        max_length=512,
        is_training=False,
        augment_texts=None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        # Store pre-computed augmented texts (same length as texts)
        self.augment_texts = augment_texts if augment_texts is not None else texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # During training, use pre-computed back-translation augmentation with 30% probability
        if self.is_training and random.random() < 0.30:
            text = str(self.augment_texts[idx])
        else:
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

# Pre-compute back-translation augmentations for training texts
augmented_train_texts = precompute_back_translation(
    train_texts_final, augment_prob=0.15
)

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length,
    is_training=True, augment_texts=augmented_train_texts
)
val_dataset = SpookyDataset(
    val_texts_final, val_labels_final, tokenizer, max_length,
    is_training=False
)
test_dataset = SpookyDataset(
    test_texts, None, tokenizer, max_length, is_training=False
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
# TWO-STAGE FINE-TUNING
# ============================================================
num_epochs = 30
patience = 8  # Increased patience due to two-stage training
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()
os.makedirs("./submission", exist_ok=True)

total_steps = len(train_loader) * num_epochs
two_stage_split = int(0.1 * total_steps)  # First 10% of steps for stage 1

# Adaptive gamma tracking for Focal Loss
current_gamma = 2.0
max_gamma = 4.0
gamma_increase_epochs = 0
last_val_loss = float("inf")


def setup_optimizer_and_scheduler(model, stage=1, total_steps=None):
    """Setup optimizer and scheduler for specific training stage"""
    global current_gamma, criterion

    if stage == 1:
        # Stage 1: Freeze backbone entirely, train only head
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Optimizer for head only
        head_params = [p for name, p in model.named_parameters() if "backbone" not in name and p.requires_grad]
        optimizer = AdamW(
            head_params,
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        # Stage 2: Unfreeze last 8 layers of backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        for layer in model.backbone.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Differential learning rates
        backbone_params = [p for name, p in model.named_parameters() if "backbone" in name and p.requires_grad]
        head_params = [p for name, p in model.named_parameters() if "backbone" not in name and p.requires_grad]

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
                {"params": head_params, "lr": 5e-5, "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # OneCycleLR scheduler
    if total_steps is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )
    else:
        scheduler = None

    return optimizer, scheduler


# Initial setup for Stage 1
current_stage = 1
optimizer, scheduler = setup_optimizer_and_scheduler(
    model, stage=1, total_steps=two_stage_split
)

print(f"\nStage 1: Training head only ({two_stage_split} steps)...")
print(f"Head params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

global_step = 0

for epoch in range(num_epochs):
    # Check if we should switch to Stage 2
    if current_stage == 1 and global_step >= two_stage_split:
        current_stage = 2
        print(f"\nSwitching to Stage 2: Unfreezing last 8 backbone layers...")
        remaining_steps = total_steps - global_step
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model, stage=2, total_steps=remaining_steps
        )
        backbone_unfrozen = sum(p.numel() for p in model.backbone.encoder.layer[-8:].parameters())
        print(f"Unfrozen backbone params: {backbone_unfrozen:,}")
        print(f"Total trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Check for stage transition mid-epoch
        if current_stage == 1 and global_step >= two_stage_split:
            break

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

        if scheduler is not None:
            scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1
        global_step += 1

        # Debug: Print LR every 100 steps
        if global_step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Step {global_step}/{total_steps} | LR: {current_lr:.2e} | Loss: {loss.item():.4f}")

    # If we broke out of the batch loop due to stage transition, finish the epoch differently
    if current_stage == 1 and global_step >= two_stage_split:
        # We already broke; continue to validation
        avg_train_loss = total_train_loss / max(1, num_train_batches)
    else:
        avg_train_loss = total_train_loss / max(1, num_train_batches)

    # Validation
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

            with autocast():
                logits = model(input_ids, attention_mask)
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

    current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Stage {current_stage} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    # Adaptive gamma for Focal Loss
    if avg_val_loss >= last_val_loss and current_stage == 2:
        gamma_increase_epochs += 1
        if gamma_increase_epochs >= 3 and current_gamma < max_gamma:
            current_gamma = min(current_gamma + 0.5, max_gamma)
            criterion.gamma = current_gamma
            gamma_increase_epochs = 0
            print(f"  ** Increased Focal Loss gamma to {current_gamma}")
    else:
        gamma_increase_epochs = 0
    last_val_loss = avg_val_loss

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # If we just transitioned to stage 2, the remaining steps may be very few; continue gracefully
    if current_stage == 2 and global_step >= total_steps:
        print("All scheduled training steps completed. Stopping training.")
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
