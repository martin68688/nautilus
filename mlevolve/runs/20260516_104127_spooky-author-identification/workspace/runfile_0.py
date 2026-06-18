import os
os.sched_setaffinity(0, {17, 18, 19, 20, 21})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import string

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# ============================================================
# Path Configuration
# ============================================================
DATA_DIR = "./input"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
PATIENCE = 3
DROPOUT = 0.1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# Data Loading
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
print(
    f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# Stratified Split (CRITICAL: use indices directly to prevent INDEX_BUG)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["author_encoded"].values,
)
assert len(set(train_idx) & set(val_idx)) == 0, "Split overlap detected!"

train_texts = train_df["text"].values[train_idx]
train_labels = train_df["author_encoded"].values[train_idx]
val_texts = train_df["text"].values[val_idx]
val_labels = train_df["author_encoded"].values[val_idx]
test_texts = test_df["text"].values
test_ids = test_df["id"].values

print(
    f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)

# ============================================================
# DATA PREPARATION FOR BiLSTM AND Char-CNN
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA FOR BiLSTM AND Char-CNN")
print("=" * 60)

# ---------------------------
# BiLSTM: Word-level tokenization with GloVe
# ---------------------------
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.300d.txt")
MAX_WORDS = 30000
MAX_SEQ_LEN_WORD = 256

# Build word vocabulary from training texts
word_counter = Counter()
for text in train_texts:
    words = text.lower().split()
    word_counter.update(words)

# Keep top MAX_WORDS-2 words (reserve 0 for PAD, 1 for UNK)
word_vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in word_counter.most_common(MAX_WORDS - 2):
    word_vocab[word] = len(word_vocab)

def text_to_word_ids(text, vocab, max_len=MAX_SEQ_LEN_WORD):
    words = text.lower().split()
    ids = [vocab.get(w, 1) for w in words[:max_len]]
    # Pad to max_len
    ids = ids + [0] * (max_len - len(ids))
    return ids[:max_len]

# Tokenize all texts to word IDs
train_word_ids = np.array([text_to_word_ids(t, word_vocab) for t in train_texts], dtype=np.int64)
val_word_ids = np.array([text_to_word_ids(t, word_vocab) for t in val_texts], dtype=np.int64)
test_word_ids = np.array([text_to_word_ids(t, word_vocab) for t in test_texts], dtype=np.int64)

# Load GloVe embeddings
print("Loading GloVe embeddings...")
embedding_dim = 300
glove_vectors = {}
glove_path = os.path.join(DATA_DIR, "glove.6B.300d.txt")
if os.path.exists(glove_path):
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            glove_vectors[word] = vector
    print(f"Loaded {len(glove_vectors)} GloVe vectors")
else:
    print(f"GloVe file not found at {glove_path}. Random initialization will be used.")
    glove_vectors = {}

# Build embedding matrix
vocab_size = len(word_vocab)
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
for word, idx in word_vocab.items():
    if word in glove_vectors:
        embedding_matrix[idx] = glove_vectors[word]
    else:
        # Random init for words not in GloVe
        embedding_matrix[idx] = np.random.randn(embedding_dim) * 0.1

print(f"Embedding matrix shape: {embedding_matrix.shape}")

# ---------------------------
# Char-CNN: Character-level tokenization
# ---------------------------
# ASCII printable characters
ascii_chars = string.printable[:95]  # 95 printable chars
char_to_idx = {c: i+2 for i, c in enumerate(ascii_chars)}  # Reserve 0 for PAD, 1 for UNK
char_to_idx["<PAD>"] = 0
char_to_idx["<UNK>"] = 1
MAX_SEQ_LEN_CHAR = 512

def text_to_char_ids(text, vocab, max_len=MAX_SEQ_LEN_CHAR):
    ids = [vocab.get(c, 1) for c in text[:max_len]]
    ids = ids + [0] * (max_len - len(ids))
    return ids[:max_len]

train_char_ids = np.array([text_to_char_ids(t, char_to_idx) for t in train_texts], dtype=np.int64)
val_char_ids = np.array([text_to_char_ids(t, char_to_idx) for t in val_texts], dtype=np.int64)
test_char_ids = np.array([text_to_char_ids(t, char_to_idx) for t in test_texts], dtype=np.int64)

print(f"Char vocab size: {len(char_to_idx)}")
print(f"Train char IDs shape: {train_char_ids.shape}")

# Convert labels to tensors
train_labels_tensor = torch.LongTensor(train_labels)
val_labels_tensor = torch.LongTensor(val_labels)

# ============================================================
# FEATURE ENGINEERING - NONE (pure transformer fine-tuning)
# ============================================================
print("\n" + "=" * 60)
print("PURE TRANSFORMER FINE-TUNING (NO HAND-CRAFTED FEATURES)")
print("=" * 60)

# ============================================================
# MODEL ARCHITECTURES: DeBERTa, BiLSTM, Char-CNN
# ============================================================

# ----- BiLSTM Model -----
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch, seq_len, hidden*2)
        # mask: (batch, seq_len)
        attn_weights = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.attention = AttentionPooling(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        # Create mask for attention
        if attention_mask is None:
            mask = (input_ids != 0).float()
        else:
            mask = attention_mask.float()
        context = self.attention(lstm_out, mask)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits

# ----- Char-CNN Model -----
class CharCNNClassifier(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim=50, num_filters=100,
                 kernel_sizes=[2,3,4,5], num_classes=3, dropout=0.3,
                 word_embed_dim=768, use_word_embed=True):
        super().__init__()
        self.use_word_embed = use_word_embed

        # Char embedding
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        # Conv layers for character sequences
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])

        # DistilBERT for word-level features (frozen)
        if use_word_embed:
            self.distilbert = AutoModel.from_pretrained("distilbert-base-uncased")
            # Freeze DistilBERT
            for param in self.distilbert.parameters():
                param.requires_grad = False

            total_features = num_filters * len(kernel_sizes) + word_embed_dim
        else:
            total_features = num_filters * len(kernel_sizes)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_features, num_classes)

    def forward(self, char_ids, input_ids=None, attention_mask=None):
        # Character CNN path
        char_emb = self.char_embedding(char_ids)  # (batch, seq_len, char_embed_dim)
        char_emb = char_emb.permute(0, 2, 1)  # (batch, char_embed_dim, seq_len)

        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(char_emb)  # (batch, num_filters, seq_len)
            conv_out = torch.relu(conv_out)
            conv_out, _ = torch.max(conv_out, dim=-1)  # Global max pooling
            conv_outputs.append(conv_out)

        char_features = torch.cat(conv_outputs, dim=-1)  # (batch, num_filters * len(kernel_sizes))

        # Word-level DistilBERT path
        if self.use_word_embed and input_ids is not None:
            with torch.no_grad():
                bert_outputs = self.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # Mean pooling
                word_features = bert_outputs.last_hidden_state.mean(dim=1)  # (batch, 768)
            features = torch.cat([char_features, word_features], dim=-1)
        else:
            features = char_features

        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

# ============================================================
# Initialize tokenizer and models
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING MODELS")
print("=" * 60)

from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_AUTHORS, hidden_dropout_prob=DROPOUT, attention_probs_dropout_prob=DROPOUT)

# ----- DeBERTa Model -----
deberta_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=False,
)
deberta_model.to(device)

# ----- BiLSTM Model -----
bilstm_model = BiLSTMClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    embedding_matrix=embedding_matrix,
    hidden_dim=256,
    num_classes=NUM_AUTHORS,
    dropout=0.3
)
bilstm_model.to(device)

# ----- Char-CNN Model -----
char_cnn_model = CharCNNClassifier(
    char_vocab_size=len(char_to_idx),
    char_embed_dim=50,
    num_filters=100,
    kernel_sizes=[2,3,4,5],
    num_classes=NUM_AUTHORS,
    dropout=0.3,
    use_word_embed=True
)
char_cnn_model.to(device)

# Multi-sample dropout will be applied during training in the forward pass
MSD_K = 4
msd_dropout = nn.Dropout(DROPOUT)

# ============================================================
# Tokenize texts
# ============================================================
print("Tokenizing texts...")
train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

train_labels_tensor = torch.LongTensor(train_labels)
val_labels_tensor = torch.LongTensor(val_labels)

# ============================================================
# Prepare data loaders
# ============================================================
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels_tensor,
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    val_labels_tensor,
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

# BiLSTM datasets
train_bilstm_dataset = TensorDataset(
    torch.LongTensor(train_word_ids),
    train_labels_tensor,
)
val_bilstm_dataset = TensorDataset(
    torch.LongTensor(val_word_ids),
    val_labels_tensor,
)
test_bilstm_dataset = TensorDataset(
    torch.LongTensor(test_word_ids),
)

# Char-CNN datasets (need word-level tokens for DistilBERT)
train_charcnn_dataset = TensorDataset(
    torch.LongTensor(train_char_ids),
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels_tensor,
)
val_charcnn_dataset = TensorDataset(
    torch.LongTensor(val_char_ids),
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    val_labels_tensor,
)
test_charcnn_dataset = TensorDataset(
    torch.LongTensor(test_char_ids),
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
)

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
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# BiLSTM loaders
bilstm_train_loader = DataLoader(
    train_bilstm_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
bilstm_val_loader = DataLoader(
    val_bilstm_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
bilstm_test_loader = DataLoader(
    test_bilstm_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Char-CNN loaders
charcnn_train_loader = DataLoader(
    train_charcnn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
charcnn_val_loader = DataLoader(
    val_charcnn_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
charcnn_test_loader = DataLoader(
    test_charcnn_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# Optimizer and scheduler (Two-phase training)
# ============================================================

# Phase 1: Train only classifier head
# Simplified optimizer creation with weight decay groups (no fragile layer-wise discriminative LR)
def create_optimizer_with_discriminative_lr(model, encoder_lr=1e-5, classifier_lr=2e-5, weight_decay=0.01):
    """
    Create optimizer with separate learning rates for encoder and classifier.
    Uses standard parameter grouping without fragile layer-wise LR multipliers.
    """
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'layernorm']

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' in n and not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': classifier_lr,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' in n and any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': classifier_lr,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' not in n and not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': encoder_lr,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'classifier' not in n and any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': encoder_lr,
        },
    ]

    # Filter out empty groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g['params']) > 0]

    optimizer = AdamW(optimizer_grouped_parameters, lr=encoder_lr, weight_decay=weight_decay, eps=1e-8)
    return optimizer, optimizer_grouped_parameters

# Phase 1: Freeze encoder, train classifier
def freeze_encoder(model):
    """Freeze all base encoder parameters."""
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("Encoder frozen, classifier trainable")

def unfreeze_encoder(model):
    """Unfreeze all parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen for fine-tuning")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# SWA (Stochastic Weight Averaging)
swa_model = None
swa_n = 0
scaler = GradScaler() if torch.cuda.is_available() else None

# ============================================================
# Helper functions
# ============================================================
def compute_log_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(NUM_AUTHORS):
            if y_true[i] == j:
                loss -= np.log(y_pred_proba[i, j])
    return loss / n

def evaluate(model, loader, criterion, device, model_type="deberta"):
    model.eval()
    all_losses = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if model_type == "deberta":
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                with autocast(enabled=(scaler is not None)):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                loss = criterion(logits, labels)
            elif model_type == "bilstm":
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                with autocast(enabled=(scaler is not None)):
                    logits = model(input_ids=input_ids)
                probs = torch.softmax(logits, dim=1)
                loss = criterion(logits, labels)
            elif model_type == "charcnn":
                char_ids = batch[0].to(device)
                input_ids = batch[1].to(device)
                attention_mask = batch[2].to(device)
                labels = batch[3].to(device)
                with autocast(enabled=(scaler is not None)):
                    logits = model(char_ids=char_ids, input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)
                loss = criterion(logits, labels)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_losses.append(loss.item())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = np.mean(all_losses)
    logloss = compute_log_loss(all_labels, all_probs)
    acc = np.mean(np.argmax(all_probs, axis=1) == all_labels)
    return avg_loss, logloss, acc, all_probs

def train_model(model, train_loader, val_loader, model_type, device, num_epochs=10, lr=2e-5):
    """Generic training function for non-transformer models."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_logloss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if model_type == "bilstm":
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                logits = model(input_ids=input_ids)
                loss = criterion(logits, labels)
            elif model_type == "charcnn":
                char_ids = batch[0].to(device)
                input_ids = batch[1].to(device)
                attention_mask = batch[2].to(device)
                labels = batch[3].to(device)
                logits = model(char_ids=char_ids, input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        val_loss, val_logloss, val_acc, _ = evaluate(
            model, val_loader, criterion, device, model_type=model_type
        )

        print(f"{model_type} Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}")

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_val_logloss

# ============================================================
# TRAINING LOOP (Two-Phase Training with SWA)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING HuggingFace Classifier (Two-Phase Training with SWA)")
print("=" * 60)

def update_swa(swa_model, model, swa_n):
    """Update SWA model with running average of weights."""
    if swa_model is None:
        swa_model = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                swa_model[name] = param.data.clone()
        return swa_model, 1
    else:
        for name, param in model.named_parameters():
            if param.requires_grad and name in swa_model:
                swa_model[name] = (swa_model[name] * swa_n + param.data) / (swa_n + 1)
        return swa_model, swa_n + 1

def apply_swa(model, swa_model):
    """Temporarily apply SWA weights to the model (creates a copy)."""
    swa_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in swa_model:
            swa_state[name] = param.data.clone()
            param.data.copy_(swa_model[name])
    return swa_state

def restore_weights(model, swa_state):
    """Restore original weights after evaluation."""
    for name, param in model.named_parameters():
        if name in swa_state:
            param.data.copy_(swa_state[name])

best_val_logloss = float("inf")
best_epoch = 0
patience_counter = 0

# Phase 1: Train only classifier head (epochs 1-2)
print("\n" + "=" * 60)
print("PHASE 1: Training Classifier Head Only (Epochs 1-2)")
print("=" * 60)

# ----- Train DeBERTa -----
freeze_encoder(deberta_model)
optimizer, param_groups = create_optimizer_with_discriminative_lr(
    deberta_model, encoder_lr=1e-5, classifier_lr=2e-5
)

# Phase 1 scheduler: constant LR
phase1_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: 1.0
)

for epoch in range(2):  # Phase 1: 2 epochs
    deberta_model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = deberta_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
            optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    phase1_scheduler.step()

    avg_train_loss = total_train_loss / max(num_batches, 1)
    val_loss, val_logloss, val_acc, _ = evaluate(deberta_model, val_loader, criterion, device, model_type="deberta")

    print(
        f"DeBERTa Phase1 Epoch {epoch+1:2d}/2 | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(deberta_model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_phase1.pt"))
        print(f"  --> New best model saved (logloss: {val_logloss:.4f})")
    else:
        patience_counter += 1

# Phase 2: Unfreeze all layers, discriminative LR, SWA
print("\n" + "=" * 60)
print("PHASE 2: Full Fine-Tuning with Discriminative LR and SWA (Epochs 3-20)")
print("=" * 60)

unfreeze_encoder(deberta_model)
optimizer, param_groups = create_optimizer_with_discriminative_lr(
    deberta_model, encoder_lr=1e-5, classifier_lr=2e-5
)

# Phase 2 scheduler: cosine decay with warm restarts
phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=4, T_mult=2, eta_min=1e-6
)

# Reset SWA
swa_model = None
swa_n = 0

for phase2_epoch in range(NUM_EPOCHS - 2):  # Remaining epochs
    epoch = phase2_epoch + 2  # Continue epoch numbering
    deberta_model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = deberta_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
            optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    phase2_scheduler.step()

    avg_train_loss = total_train_loss / max(num_batches, 1)

    # Update SWA after each epoch in phase 2
    swa_model, swa_n = update_swa(swa_model, deberta_model, swa_n)

    # Evaluate using SWA averaged weights
    if swa_model is not None:
        swa_state = apply_swa(deberta_model, swa_model)
        val_loss, val_logloss, val_acc, _ = evaluate(deberta_model, val_loader, criterion, device, model_type="deberta")
        restore_weights(deberta_model, swa_state)
    else:
        val_loss, val_logloss, val_acc, _ = evaluate(deberta_model, val_loader, criterion, device, model_type="deberta")

    print(
        f"DeBERTa Phase2 Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        # Save SWA model weights
        if swa_model is not None:
            # Apply SWA before saving
            swa_state_save = apply_swa(deberta_model, swa_model)
            torch.save(deberta_model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_28803f7f9bd24c81aa65d812dbaa4e70.pt"))
            restore_weights(deberta_model, swa_state_save)
        else:
            torch.save(deberta_model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_28803f7f9bd24c81aa65d812dbaa4e70.pt"))
        print(f"  --> New best model saved (logloss: {val_logloss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best logloss: {best_val_logloss:.4f}"
            )
            break

# Load best DeBERTa model (apply SWA weights)
print(f"\nLoading best DeBERTa model from epoch {best_epoch} with validation logloss {best_val_logloss:.4f}")

if swa_model is not None:
    print("Applying SWA averaged weights to best model...")
    state_dict = torch.load(os.path.join(OUTPUT_DIR, "best_model_28803f7f9bd24c81aa65d812dbaa4e70.pt"), map_location=device)
    deberta_model.load_state_dict(state_dict, strict=False)
else:
    state_dict = torch.load(os.path.join(OUTPUT_DIR, "best_model_28803f7f9bd24c81aa65d812dbaa4e70.pt"), map_location=device)
    deberta_model.load_state_dict(state_dict, strict=False)

deberta_model.eval()

# ----- Evaluate DeBERTa on validation set -----
_, deberta_val_logloss, deberta_val_acc, deberta_val_probs = evaluate(
    deberta_model, val_loader, criterion, device, model_type="deberta"
)
print(f"DeBERTa validation logloss: {deberta_val_logloss:.6f}, accuracy: {deberta_val_acc:.4f}")

# ----- Train BiLSTM -----
print("\n" + "=" * 60)
print("TRAINING BiLSTM")
print("=" * 60)
bilstm_val_logloss = train_model(
    bilstm_model, bilstm_train_loader, bilstm_val_loader, "bilstm", device, num_epochs=10, lr=2e-4
)
print(f"BiLSTM best validation logloss: {bilstm_val_logloss:.6f}")

# Evaluate BiLSTM on validation set to get probs
_, _, _, bilstm_val_probs = evaluate(bilstm_model, bilstm_val_loader, criterion, device, model_type="bilstm")

# ----- Train Char-CNN -----
print("\n" + "=" * 60)
print("TRAINING Char-CNN")
print("=" * 60)
charcnn_val_logloss = train_model(
    char_cnn_model, charcnn_train_loader, charcnn_val_loader, "charcnn", device, num_epochs=10, lr=2e-4
)
print(f"Char-CNN best validation logloss: {charcnn_val_logloss:.6f}")

# Evaluate Char-CNN on validation set to get probs
_, _, _, charcnn_val_probs = evaluate(char_cnn_model, charcnn_val_loader, criterion, device, model_type="charcnn")

# ============================================================
# ENSEMBLE: Find optimal weights using validation logloss
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE: Finding Optimal Weights via Validation LogLoss")
print("=" * 60)

from scipy.optimize import minimize

val_labels_array = val_labels

def ensemble_logloss(weights, val_probs_list, val_labels):
    """Compute logloss for weighted ensemble of model predictions."""
    weights = np.abs(weights)  # Ensure non-negative
    weights = weights / weights.sum()  # Normalize
    ensemble_probs = sum(w * p for w, p in zip(weights, val_probs_list))
    eps = 1e-15
    ensemble_probs = np.clip(ensemble_probs, eps, 1 - eps)
    row_sums = ensemble_probs.sum(axis=1, keepdims=True)
    ensemble_probs = ensemble_probs / row_sums
    ensemble_probs = np.clip(ensemble_probs, eps, 1 - eps)

    loss = 0.0
    n = len(val_labels)
    for i in range(n):
        for j in range(NUM_AUTHORS):
            if val_labels[i] == j:
                loss -= np.log(ensemble_probs[i, j])
    return loss / n

# Initial weights (equal)
initial_weights = np.array([1/3, 1/3, 1/3])
val_probs_list = [deberta_val_probs, bilstm_val_probs, charcnn_val_probs]

# Optimize with constraints
bounds = [(0.0, 1.0)] * 3
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

result = minimize(
    ensemble_logloss,
    initial_weights,
    args=(val_probs_list, val_labels_array),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"Optimal ensemble weights: DeBERTa={optimal_weights[0]:.4f}, BiLSTM={optimal_weights[1]:.4f}, Char-CNN={optimal_weights[2]:.4f}")

# Compute ensemble validation logloss
ensemble_val_probs = sum(w * p for w, p in zip(optimal_weights, val_probs_list))
eps_val = 1e-15
ensemble_val_probs = np.clip(ensemble_val_probs, eps_val, 1 - eps_val)
row_sums = ensemble_val_probs.sum(axis=1, keepdims=True)
ensemble_val_probs = ensemble_val_probs / row_sums
ensemble_val_probs = np.clip(ensemble_val_probs, eps_val, 1 - eps_val)
ensemble_val_logloss = compute_log_loss(val_labels_array, ensemble_val_probs)
print(f"Ensemble validation logloss: {ensemble_val_logloss:.6f}")

# ============================================================
# TEST INFERENCE WITH ENSEMBLE
# ============================================================
print("\nPerforming test inference with ensemble...")

# DeBERTa test inference
deberta_model.eval()
deberta_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast(enabled=(scaler is not None)):
            outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        deberta_test_probs.append(probs.cpu().numpy())
deberta_test_probs = np.vstack(deberta_test_probs)

# BiLSTM test inference
bilstm_model.eval()
bilstm_test_probs = []
with torch.no_grad():
    for batch in bilstm_test_loader:
        input_ids = batch[0].to(device)
        logits = bilstm_model(input_ids=input_ids)
        probs = torch.softmax(logits, dim=1)
        bilstm_test_probs.append(probs.cpu().numpy())
bilstm_test_probs = np.vstack(bilstm_test_probs)

# Char-CNN test inference
char_cnn_model.eval()
charcnn_test_probs = []
with torch.no_grad():
    for batch in charcnn_test_loader:
        char_ids = batch[0].to(device)
        input_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        logits = char_cnn_model(char_ids=char_ids, input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        charcnn_test_probs.append(probs.cpu().numpy())
charcnn_test_probs = np.vstack(charcnn_test_probs)

# Weighted ensemble
test_probs = sum(w * p for w, p in zip(optimal_weights, [deberta_test_probs, bilstm_test_probs, charcnn_test_probs]))

eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission_df.to_csv("./submission/submission_28803f7f9bd24c81aa65d812dbaa4e70.csv", index=False)
print(f"\nSubmission saved to ./submission/submission_28803f7f9bd24c81aa65d812dbaa4e70.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"\nFinal Validation Score: {ensemble_val_logloss:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
