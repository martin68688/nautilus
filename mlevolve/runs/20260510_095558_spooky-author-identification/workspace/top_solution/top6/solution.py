import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import re
import os
import json
import math

# ============================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_id"] = train_df["author"].map(author_mapping)
y = train_df["author_id"].values


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def extract_stylometric_features(text_series):
    features = []
    # Extended archaic/supernatural word lists
    archaic_words = [
        "thee", "thou", "thy", "thine", "hath", "doth", "art", "wilt",
        "shalt", "dost", "ere", "hence", "thence", "whence", "wherefore",
        "betwixt", "amongst", "whilst", "methinks", "prithee", "forsooth",
    ]
    supernatural_words = [
        "apparition", "spectre", "spectral", "ghost", "phantom", "spirit",
        "demon", "devil", "satanic", "infernal", "hellish", "abyss",
        "void", "darkness", "shadow", "gloom", "horror", "terror",
        "dread", "fear", "scream", "shriek", "howl", "groan",
        "corpse", "skeleton", "grave", "tomb", "sepulchre", "coffin",
        "witch", "wizard", "magic", "curse", "omen", "prophecy",
        "eldritch", "cyclopean", "non-euclidean", "carcosa", "cthulhu",
        "r'lyeh", "yog-sothoth", "nyarlathotep", "azathoth", "necronomicon",
        "unmentionable", "unnamable", "indescribable", "inconceivable",
        "antediluvian", "primordial", "forbidden", "accursed", "blasphemous",
        "nevermore", "raven", "chamber", "ghastly", "grim", "phantasm",
        "sepulchre", "tomb", "corpse", "pallid", "countenance", "visage",
        "melancholy", "dreary", "weary", "chilling", "dismal",
        "nature", "spirit", "soul", "eternal", "mortal", "immortal",
        "creator", "creation", "monster", "fiend", "wretch", "daemon",
        "alpine", "glacier", "mountain", "cottage", "geneva",
    ]
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Top 5 most discriminative punctuation densities
        punct_top5 = {
            "num_commas": text.count(","),
            "num_periods": text.count("."),
            "num_exclamation": text.count("!"),
            "num_question": text.count("?"),
            "num_colons": text.count(":"),
        }

        # Readability: approximate syllables per word using vowel groups
        vowel_groups = len(re.findall(r'[aeiouy]+', text.lower()))
        syll_per_word = vowel_groups / max(len(words), 1)

        # Archaic word count
        archaic_count = sum(1 for w in words if w.lower().strip(".,!?;:\"'()[]-") in archaic_words)

        # Supernatural word count (expanded)
        supernatural_count = sum(1 for w in words if w.lower().strip(".,!?;:\"'()[]-") in supernatural_words)

        # Contractions count
        contractions_count = len(re.findall(r"\b\w+'\w+", text))

        # Word-level features
        word_lengths = [len(w) for w in words]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        std_word_length = np.std(word_lengths) if len(word_lengths) > 1 else 0

        # Sentence-level features
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
        std_sent_length = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

        feat = [
            avg_word_length,
            std_word_length,
            avg_sent_length,
            std_sent_length,
            punct_top5["num_commas"] / max(len(sentences), 1),
            punct_top5["num_periods"] / max(len(sentences), 1),
            punct_top5["num_exclamation"] / max(len(sentences), 1),
            punct_top5["num_question"] / max(len(sentences), 1),
            punct_top5["num_colons"] / max(len(sentences), 1),
            syll_per_word,
            archaic_count,
            supernatural_count,
            contractions_count,
        ]
        features.append(feat)
    return np.array(features)


train_texts = train_df["text"].values
test_texts = test_df["text"].values

# Note: model only uses raw text via DistilRoBERTa tokenizer, no TF-IDF or stylometric features needed
# Directly use all training data for the single fold training below
train_texts_fold = train_texts
val_texts = None  # Validation will use a separate split within training loop

# Simple train/val split for single model training (no K-fold leakage issues)
from sklearn.model_selection import train_test_split
train_texts_split, val_texts_split, y_train, y_val = train_test_split(
    train_texts, y, test_size=0.2, random_state=42, stratify=y
)
train_texts_fold = train_texts_split
val_texts = val_texts_split

# ============================================================
# MODEL DESIGN
# ============================================================


class AuthorshipModel(nn.Module):
    def __init__(self, num_authors=3, model_name="distilroberta-base", hidden_dim=128, dropout_rate=0.3):
        super(AuthorshipModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_dim = self.bert.config.hidden_size  # 768 for distilroberta-base

        # Multi-head attention pooling: learnable query vector attends to token embeddings
        self.dropout = nn.Dropout(dropout_rate)
        self.num_attention_heads = 8
        self.attention_head_dim = self.bert_hidden_dim // self.num_attention_heads
        self.query = nn.Parameter(torch.randn(1, self.num_attention_heads, self.attention_head_dim))
        self.key_proj = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.value_proj = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.attention_output = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.bert_hidden_dim)
        self.classifier = nn.Linear(self.bert_hidden_dim, num_authors)

        # Initialize query
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, bert_hidden_dim)

        # Multi-head attention pooling
        batch_size, seq_len, _ = hidden_states.shape

        # Project keys and values
        keys = self.key_proj(hidden_states)  # (batch, seq_len, bert_hidden_dim)
        values = self.value_proj(hidden_states)  # (batch, seq_len, bert_hidden_dim)

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)

        # Expand query: (1, num_heads, head_dim) -> (batch, num_heads, head_dim)
        query = self.query.expand(batch_size, -1, -1).unsqueeze(2)  # (batch, num_heads, 1, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.attention_head_dim ** 0.5)
        # Apply attention mask
        mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # (batch, 1, 1, seq_len)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, num_heads, 1, seq_len)

        # Apply attention to values
        context = torch.matmul(attention_weights, values)  # (batch, num_heads, 1, head_dim)
        context = context.squeeze(2).transpose(1, 2).contiguous()  # (batch, 1, num_heads * head_dim)
        context = context.view(batch_size, -1)  # (batch, bert_hidden_dim)

        # Output projection and residual
        context = self.attention_output(context)
        # Mean pool as residual connection
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        num_tokens = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / num_tokens
        combined = context + mean_pooled
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits


NUM_AUTHORS = 3

model = AuthorshipModel(
    num_authors=NUM_AUTHORS,
    model_name="distilroberta-base",
    hidden_dim=256,
    dropout_rate=0.5,
)
criterion = FocalLoss(gamma=2.0)
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.1,
)
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model device: {device}")

# ============================================================
# TRAINING AND EVALUATION
# ============================================================

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)


def tokenize_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


def word_dropout(input_ids, attention_mask, tokenizer, dropout_prob=0.05):
    """Randomly replace 5% of tokens with [MASK] token ID."""
    mask_token_id = tokenizer.mask_token_id
    input_ids = input_ids.clone()
    dropout_mask = torch.rand(input_ids.shape, device=input_ids.device) < dropout_prob
    # Don't mask special tokens (0=padding, 1=CLS, 2=SEP for DistilRoBERTa)
    special_tokens_mask = (input_ids == tokenizer.cls_token_id) | (input_ids == tokenizer.sep_token_id) | (input_ids == tokenizer.pad_token_id)
    dropout_mask = dropout_mask & ~special_tokens_mask
    input_ids[dropout_mask] = mask_token_id
    return input_ids, attention_mask


print("Tokenizing texts...")
train_input_ids, train_attention_masks = tokenize_texts(train_texts_fold, tokenizer)
val_input_ids, val_attention_masks = tokenize_texts(val_texts, tokenizer)
test_input_ids, test_attention_masks = tokenize_texts(test_texts, tokenizer)

batch_size = 16
train_dataset = torch.utils.data.TensorDataset(
    train_input_ids, train_attention_masks, y_train_tensor
)
val_dataset = torch.utils.data.TensorDataset(
    val_input_ids, val_attention_masks, y_val_tensor
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

num_epochs = 20
best_val_loss = float("inf")
best_model_state = None
patience_counter = 0
accumulation_steps = 4
scaler = torch.cuda.amp.GradScaler()
# SWA: store model states from each epoch (or best epochs)
epoch_model_states = []  # will store (val_log_loss, state_dict)
swa_num_epochs = 3  # number of best epochs to average for SWA inference

# Create scheduler with total steps
total_steps = len(train_loader) * num_epochs // accumulation_steps
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

patience = 3
best_val_loss = float("inf")

print("Starting training...")
optimizer.zero_grad()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        input_ids, attention_masks, labels = [b.to(device) for b in batch]

        # Word dropout augmentation
        if np.random.random() < 0.5:  # Apply 50% of the time
            input_ids, attention_masks = word_dropout(input_ids, attention_masks, tokenizer)

        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_masks)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_masks, labels = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_masks)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)
    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
    log_loss_value = 0.0
    for i in range(len(val_labels)):
        for j in range(3):
            y_ij = 1.0 if val_labels[i] == j else 0.0
            log_loss_value -= y_ij * math.log(val_preds_clipped[i, j])
    log_loss_value /= len(val_labels)
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {log_loss_value:.4f}"
    )
    # SWA: store each epoch's state (or only top-k best)
    epoch_model_states.append((log_loss_value, model.state_dict().copy()))

    if log_loss_value < best_val_loss:
        best_val_loss = log_loss_value
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  New best model! Val LogLoss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# SWA: average top 3 best checkpoints
epoch_model_states.sort(key=lambda x: x[0])  # sort by val log loss ascending
top_k_states = [state for _, state in epoch_model_states[:swa_num_epochs]]
if len(top_k_states) > 0:
    swa_state_dict = {}
    for key in top_k_states[0].keys():
        swa_state_dict[key] = torch.stack([state[key] for state in top_k_states]).float().mean(dim=0)
    model.load_state_dict(swa_state_dict)
    print(f"SWA applied: averaged top {len(top_k_states)} checkpoints")
else:
    model.load_state_dict(best_model_state)

# Final validation prediction
model.eval()
all_val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_masks, _ = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_masks)
        probs = torch.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())
val_preds = np.concatenate(all_val_preds, axis=0)
val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
final_val_loss = 0.0
for i in range(len(val_labels)):
    for j in range(3):
        y_ij = 1.0 if val_labels[i] == j else 0.0
        final_val_loss -= y_ij * math.log(val_preds_clipped[i, j])
final_val_loss /= len(val_labels)

print(f"Final Validation Score: {final_val_loss}")

# Test inference
print("Running test inference...")
test_dataset = torch.utils.data.TensorDataset(
    test_input_ids, test_attention_masks
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
all_test_preds = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_masks = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_masks)
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())
test_preds = np.concatenate(all_test_preds, axis=0)
test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")