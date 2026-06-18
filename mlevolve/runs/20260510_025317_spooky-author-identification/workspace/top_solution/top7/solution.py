import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import (
    AutoTokenizer,
    ModernBertModel,
    BertTokenizer,
)
import os
import pickle
import re
import random

# ================================================================
# DATA LOADING AND SPLIT
# ================================================================

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

os.makedirs("./working", exist_ok=True)

# Encode target
le = LabelEncoder()
le.fit(train_df["author"])
train_df["author_encoded"] = le.transform(train_df["author"])

# Create stratified train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df["text"], train_df["author_encoded"]))

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print(
    f"Train size: {len(train_split)}, Val size: {len(val_split)}, Test size: {len(test_df)}"
)

# Save indices and splits
np.save("./working/train_indices.npy", train_idx)
np.save("./working/val_indices.npy", val_idx)
train_split.to_csv("./working/train_preprocessed.csv", index=False)
val_split.to_csv("./working/val_preprocessed.csv", index=False)

# ================================================================
# MODEL DESIGN - ModernBERT
# ================================================================

MODEL_ID = "answerdotai/ModernBERT-large"
CHAR_VOCAB_SIZE = 128
CHAR_EMBED_DIM = 32
CHAR_MAX_LEN = 128
NUM_LABELS = 3
MAX_LENGTH = 256  # segment length
STRIDE = 128      # overlap stride
NUM_SEGMENTS_MAX = 20  # max segments per text

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Simple char-level tokenizer using ASCII mapping
# Build char-to-id mapping: 0=PAD, 1=UNK, 2=CLS, 3=SEP, then ASCII 32-126
# Ensure vocab list covers indices 0..127 (128 total)
CHAR_VOCAB_LIST = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + [chr(i) for i in range(32, 127)]
# Pad the vocab list to exactly CHAR_VOCAB_SIZE entries
while len(CHAR_VOCAB_LIST) < CHAR_VOCAB_SIZE:
    CHAR_VOCAB_LIST.append('[UNK]')
char_to_id = {c: i for i, c in enumerate(CHAR_VOCAB_LIST)}
# Special token IDs for CharCNN
CHAR_PAD_ID = 0
CHAR_UNK_ID = 1
CHAR_CLS_ID = 2
CHAR_SEP_ID = 3

print(f"Loading ModernBERT-large backbone...")
bert_backbone = ModernBertModel.from_pretrained(MODEL_ID)
bert_backbone.config.hidden_dropout_prob = 0.3
bert_backbone.config.attention_probs_dropout_prob = 0.2

# Set all backbone parameters to require grad (full fine-tuning)
for param in bert_backbone.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# TEXT AUGMENTATION FUNCTION
# ================================================================

def augment_text_segment(text):
    """Apply a single random augmentation to a text segment with probability 0.5."""
    if random.random() > 0.5:
        return text
    aug_type = random.choice(['word_swap', 'punctuation_insert', 'vowel_noise', 'synonym_replace'])
    words = text.split()
    if not words:
        return text
    if aug_type == 'word_swap' and len(words) >= 2:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    elif aug_type == 'punctuation_insert':
        punct = random.choice(['.', ',', '!', '?', ';', ':'])
        pos = random.randint(0, len(text))
        return text[:pos] + punct + text[pos:]
    elif aug_type == 'vowel_noise':
        vowels = 'aeiouAEIOU'
        chars = list(text)
        for i in range(len(chars)):
            if chars[i] in vowels and random.random() < 0.1:
                chars[i] = random.choice(vowels)
        return ''.join(chars)
    elif aug_type == 'synonym_replace' and len(words) >= 1:
        # Simple synonym mapping (very basic, just swaps some common words)
        synonym_map = {
            'good': 'fine', 'bad': 'poor', 'big': 'large', 'small': 'tiny',
            'happy': 'glad', 'sad': 'unhappy', 'said': 'spoke', 'went': 'walked',
            'very': 'quite', 'much': 'many', 'could': 'might', 'would': 'shall',
        }
        word_idx = random.randint(0, len(words)-1)
        word_lower = words[word_idx].lower().strip('.,!?;:')
        if word_lower in synonym_map:
            old_word = words[word_idx]
            # Preserve case
            if old_word[0].isupper():
                words[word_idx] = synonym_map[word_lower].capitalize()
            else:
                words[word_idx] = synonym_map[word_lower]
        return ' '.join(words)
    return text

# ================================================================
# MODEL CLASS - TextFeatureEncoder
# ================================================================

class CharCNN(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=32, num_filters=64, filter_sizes=[2,3,4,5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self._output_dim = num_filters * len(filter_sizes)  # 256

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, num_filters, seq_len)
            # Global max pooling
            pooled = torch.max(conv_out, dim=2)[0]  # (batch, num_filters)
            conv_outs.append(pooled)
        x = torch.cat(conv_outs, dim=1)  # (batch, 256)
        x = self.dropout(x)
        return x

class MultiSegmentAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, segment_embeddings, mask=None):
        # segment_embeddings: (batch, num_segments, hidden_size)
        # mask: (batch, num_segments) - True for valid segments
        if mask is not None:
            # Invert mask for attention (True = need to attend)
            attn_mask = ~mask  # (batch, num_segments) - True for valid
            # MultiheadAttention expects key_padding_mask: True for positions to ignore
            key_padding_mask = ~attn_mask  # True for padding
        else:
            key_padding_mask = None

        attended, _ = self.attention(
            segment_embeddings, segment_embeddings, segment_embeddings,
            key_padding_mask=key_padding_mask,
        )
        attended = self.layer_norm(segment_embeddings + attended)
        # Mean pooling over valid segments
        if mask is not None:
            attended = attended * mask.unsqueeze(-1).float()
            pooled = attended.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-8)
        else:
            pooled = attended.mean(dim=1)
        return pooled, attended

class TextFeatureEncoder(nn.Module):
    def __init__(self, backbone, num_labels=3):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = backbone.config.hidden_size  # 768

        # Multi-segment attention pooling
        self.segment_attention = MultiSegmentAttention(
            hidden_size=self.hidden_size,
            num_heads=4,
            dropout=0.1,
        )

        # Character-level CNN
        self.char_cnn = CharCNN(
            vocab_size=CHAR_VOCAB_SIZE,
            embed_dim=CHAR_EMBED_DIM,
            num_filters=64,
            filter_sizes=[2, 3, 4, 5],
        )

        # MLP classifier
        total_features = self.hidden_size + 256  # 768 + 256 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.segment_attention, self.char_cnn, self.classifier]:
            for submodule in module.modules():
                if isinstance(submodule, (nn.Linear, nn.Conv1d)):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.Embedding):
                    nn.init.normal_(submodule.weight, mean=0, std=0.02)

    def forward(self, segment_input_ids, segment_attention_mask, char_input_ids):
        # segment_input_ids: (batch, num_segments, seq_len)
        # segment_attention_mask: (batch, num_segments, seq_len)
        # char_input_ids: (batch, char_max_len)
        batch_size, num_segments, seq_len = segment_input_ids.shape

        # Process each segment through backbone
        segment_embeddings_list = []
        # Compute valid mask: segment is valid if any token is non-padding
        segment_valid_mask = segment_attention_mask.sum(dim=2) > 0  # (batch, num_segments)

        for seg_idx in range(num_segments):
            seg_input = segment_input_ids[:, seg_idx, :]  # (batch, seq_len)
            seg_mask = segment_attention_mask[:, seg_idx, :]  # (batch, seq_len)

            # Skip segments that are all padding
            if seg_mask.sum() == 0:
                segment_embeddings_list.append(torch.zeros(batch_size, self.hidden_size, device=seg_input.device))
                continue

            with autocast('cuda'):
                outputs = self.backbone(input_ids=seg_input, attention_mask=seg_mask)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
            segment_embeddings_list.append(cls_emb)

        # Stack segment embeddings
        segment_embeddings = torch.stack(segment_embeddings_list, dim=1)  # (batch, num_segments, hidden_size)

        # Multi-segment attention pooling
        pooled_segments, _ = self.segment_attention(segment_embeddings, segment_valid_mask)  # (batch, hidden_size)

        # Character CNN features
        char_features = self.char_cnn(char_input_ids)  # (batch, 256)

        # Concatenate and classify
        combined = torch.cat([pooled_segments, char_features], dim=1)  # (batch, 1024)
        logits = self.classifier(combined)

        return logits

bert_backbone.to(device)
model = TextFeatureEncoder(bert_backbone, num_labels=NUM_LABELS)
model.to(device)
print(f"Model loaded on {device}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Separate learning rates: 1e-5 for backbone, 1e-4 for new layers
backbone_params = []
new_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

optimizer = AdamW([
    {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 0.1},
    {'params': new_params, 'lr': 1e-4, 'weight_decay': 0.1},
], lr=1e-5, eps=1e-8)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Backbone lr: 1e-5, New layers lr: 1e-4, weight_decay: 0.1")

# ================================================================
# DATASET CLASS
# ================================================================


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None,
                 max_length=256, stride=128, is_training=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.is_training = is_training

    def __len__(self):
        return len(self.texts)

    def _segment_text(self, text):
        """Split text into overlapping segments."""
        tokens = self.tokenizer.tokenize(text)
        segments = []
        # Handle very short texts
        if len(tokens) == 0:
            tokens = ['[PAD]']
        # Generate overlapping segments
        for start in range(0, len(tokens), self.stride):
            end = start + self.max_length
            segment_tokens = tokens[start:end]
            if len(segment_tokens) == 0:
                break
            segments.append(segment_tokens)
            if end >= len(tokens):
                break
        # Ensure at least one segment
        if len(segments) == 0:
            segments.append(tokens[:self.max_length])
        return segments

    def _text_to_char_ids(self, text):
        """Convert text to char-level token IDs using simple mapping."""
        chars = list(text[:CHAR_MAX_LEN - 2])  # reserve for [CLS] and [SEP]
        char_ids = [CHAR_CLS_ID] + [
            char_to_id.get(c, CHAR_UNK_ID) for c in chars
        ] + [CHAR_SEP_ID]
        # Pad to CHAR_MAX_LEN
        if len(char_ids) < CHAR_MAX_LEN:
            char_ids += [CHAR_PAD_ID] * (CHAR_MAX_LEN - len(char_ids))
        else:
            char_ids = char_ids[:CHAR_MAX_LEN]
        return torch.tensor(char_ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Apply augmentation for training
        if self.is_training:
            text = augment_text_segment(text)

        # Get char-level IDs
        char_ids = self._text_to_char_ids(text)

        # Segment text into overlapping windows
        segments = self._segment_text(text)

        # Cap number of segments to avoid OOM
        if len(segments) > NUM_SEGMENTS_MAX:
            # Randomly sample segments if training, else take first N
            if self.is_training:
                indices = sorted(random.sample(range(len(segments)), NUM_SEGMENTS_MAX))
                segments = [segments[i] for i in indices]
            else:
                segments = segments[:NUM_SEGMENTS_MAX]

        # Encode each segment
        segment_input_ids = []
        segment_attention_mask = []

        for seg_tokens in segments:
            # Convert tokens to ids and add special tokens
            input_ids = self.tokenizer.convert_tokens_to_ids(seg_tokens)
            # Add [CLS] and [SEP]
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
            # Truncate or pad to max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            else:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

            mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]
            segment_input_ids.append(input_ids)
            segment_attention_mask.append(mask)

        # Pad segments to fixed number
        num_segments = len(segments)
        if num_segments < NUM_SEGMENTS_MAX:
            pad_len = NUM_SEGMENTS_MAX - num_segments
            for _ in range(pad_len):
                segment_input_ids.append([self.tokenizer.pad_token_id] * self.max_length)
                segment_attention_mask.append([0] * self.max_length)

        item = {
            "segment_input_ids": torch.tensor(segment_input_ids, dtype=torch.long),
            "segment_attention_mask": torch.tensor(segment_attention_mask, dtype=torch.long),
            "char_input_ids": char_ids,
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ================================================================
# CREATE DATALOADERS
# ================================================================

train_labels = train_split["author_encoded"].values
val_labels = val_split["author_encoded"].values

train_dataset = TextDataset(
    train_split["text"].values, train_labels, tokenizer,
    max_length=MAX_LENGTH, stride=STRIDE, is_training=True
)
val_dataset = TextDataset(
    val_split["text"].values, val_labels, tokenizer,
    max_length=MAX_LENGTH, stride=STRIDE, is_training=False
)
test_dataset = TextDataset(
    test_df["text"].values, labels=None, tokenizer=tokenizer,
    max_length=MAX_LENGTH, stride=STRIDE, is_training=False
)

batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ================================================================
# TRAINING SETUP
# ================================================================

num_epochs = 10
gradient_accumulation_steps = 2

# CosineAnnealingWarmRestarts scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=2,  # initial restart period
    T_mult=2,  # multiply period after each restart
    eta_min=1e-6,
)

scaler = GradScaler('cuda')
best_val_loss = float("inf")
best_model_path = "./working/best_model.pt"
patience = 3
patience_counter = 0

print(f"Starting training for up to {num_epochs} epochs with early stopping (patience={patience})...")

# ================================================================
# TRAINING LOOP
# ================================================================

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        segment_input_ids = batch["segment_input_ids"].to(device, non_blocking=True)
        segment_attention_mask = batch["segment_attention_mask"].to(device, non_blocking=True)
        char_input_ids = batch["char_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast('cuda'):
            logits = model(
                segment_input_ids=segment_input_ids,
                segment_attention_mask=segment_attention_mask,
                char_input_ids=char_input_ids,
            )
            loss = criterion(logits, labels) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + step / len(train_loader))
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            segment_input_ids = batch["segment_input_ids"].to(device, non_blocking=True)
            segment_attention_mask = batch["segment_attention_mask"].to(device, non_blocking=True)
            char_input_ids = batch["char_input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(
                    segment_input_ids=segment_input_ids,
                    segment_attention_mask=segment_attention_mask,
                    char_input_ids=char_input_ids,
                )
                loss = criterion(logits, labels)

            total_val_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert labels to one-hot for log loss computation
    all_labels_onehot = np.zeros((len(all_labels), NUM_LABELS))
    all_labels_onehot[np.arange(len(all_labels)), all_labels] = 1

    eps_clip = 1e-15
    # Clip BEFORE computing log loss
    all_preds = np.clip(all_preds, eps_clip, 1 - eps_clip)
    log_loss_val = -np.mean(np.sum(all_labels_onehot * np.log(all_preds), axis=1))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {log_loss_val:.4f} [SAVED]"
        )
    else:
        patience_counter += 1
        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Log Loss: {log_loss_val:.4f} (patience {patience_counter}/{patience})"
        )

    if patience_counter >= patience:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# ================================================================
# FINAL EVALUATION ON BEST MODEL
# ================================================================

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds_val = []
all_labels_val = []
with torch.no_grad():
    for batch in val_loader:
        segment_input_ids = batch["segment_input_ids"].to(device, non_blocking=True)
        segment_attention_mask = batch["segment_attention_mask"].to(device, non_blocking=True)
        char_input_ids = batch["char_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(
                segment_input_ids=segment_input_ids,
                segment_attention_mask=segment_attention_mask,
                char_input_ids=char_input_ids,
            )
            probs = torch.softmax(logits, dim=-1)
        all_preds_val.append(probs.cpu().numpy())
        all_labels_val.append(labels.cpu().numpy())

all_preds_val = np.concatenate(all_preds_val, axis=0)
all_labels_val = np.concatenate(all_labels_val, axis=0)
all_labels_onehot = np.zeros((len(all_labels_val), NUM_LABELS))
all_labels_onehot[np.arange(len(all_labels_val)), all_labels_val] = 1
eps_clip = 1e-15
all_preds_val = np.clip(all_preds_val, eps_clip, 1 - eps_clip)
val_log_loss = -np.mean(np.sum(all_labels_onehot * np.log(all_preds_val), axis=1))

# ================================================================
# TEST INFERENCE
# ================================================================

all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        segment_input_ids = batch["segment_input_ids"].to(device, non_blocking=True)
        segment_attention_mask = batch["segment_attention_mask"].to(device, non_blocking=True)
        char_input_ids = batch["char_input_ids"].to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(
                segment_input_ids=segment_input_ids,
                segment_attention_mask=segment_attention_mask,
                char_input_ids=char_input_ids,
            )
            probs = torch.softmax(logits, dim=-1)
        all_test_preds.append(probs.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds, axis=0)
# Clip test predictions before normalization
eps_clip = 1e-15
all_test_preds = np.clip(all_test_preds, eps_clip, 1 - eps_clip)
all_test_preds = all_test_preds / all_test_preds.sum(axis=1, keepdims=True)

# ================================================================
# SUBMISSION
# ================================================================

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {val_log_loss}")