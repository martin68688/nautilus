import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import warnings
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if needed
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])

# ===================== DATA PROCESSING & FEATURE ENGINEERING =====================

# Tokenization and vocabulary building
max_seq_len = 150
embedding_dim = 300  # Using GloVe 300d embeddings

def tokenize(text):
    """Simple whitespace tokenizer with basic cleaning"""
    if not isinstance(text, str):
        return []
    # Convert to lowercase and split on whitespace
    tokens = text.lower().split()
    # Remove punctuation from tokens (keep alphanumeric)
    cleaned = []
    for token in tokens:
        token = ''.join(c for c in token if c.isalnum())
        if token:
            cleaned.append(token)
    return cleaned

def compute_stylometric_features(text):
    """Compute stylometric features for a given text."""
    if not isinstance(text, str) or len(text) == 0:
        return np.zeros(6)

    words = text.split()
    num_words = len(words)
    num_chars = len(text)

    # Average word length
    if num_words > 0:
        avg_word_len = sum(len(w) for w in words) / num_words
    else:
        avg_word_len = 0

    # Log-transformed sentence length (approximate by words)
    sentence_len = np.log1p(num_words)

    # Punctuation density
    punct_count = sum(1 for c in text if c in '.,!?;:()[]{}""''')
    punct_density = punct_count / max(num_chars, 1)

    # Stopword ratio
    tokens_lower = [w.lower() for w in words]
    stopword_count = sum(1 for t in tokens_lower if t in stop_words)
    stopword_ratio = stopword_count / max(num_words, 1)

    # Type-token ratio (unique word ratio)
    unique_words = len(set(tokens_lower))
    unique_ratio = unique_words / max(num_words, 1)

    # Coleman-Lieu readability index (simplified)
    # = 0.0588 * L - 0.296 * S - 15.8, where L = average chars per 100 words, S = average sentences per 100 words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    L = (num_chars / num_words) * 100 if num_words > 0 else 0
    S = (num_sentences / max(num_words, 1)) * 100
    coleman_lieu = 0.0588 * L - 0.296 * S - 15.8 if num_words > 0 else 0

    return np.array([avg_word_len, sentence_len, punct_density, stopword_ratio, unique_ratio, coleman_lieu])

# Build vocabulary from training data
print("Building vocabulary...")
word_counts = Counter()
for text in train_df["text"]:
    tokens = tokenize(text)
    word_counts.update(tokens)

# Create vocabulary mapping
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in word_counts.most_common():
    vocab[word] = len(vocab)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Build random embedding matrix (Xavier uniform initialization)
print("Building embedding matrix with Xavier initialization...")
np.random.seed(42)
embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(np.float32)
embedding_matrix[0] = 0.0  # <PAD> token stays zero

print(f"Embedding matrix created: {embedding_matrix.shape}")

def text_to_sequence(text, vocab, max_len):
    """Convert text to sequence of indices with padding/truncation"""
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # Truncate
    if len(indices) > max_len:
        indices = indices[:max_len]
    # Pad
    indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)

# Split BEFORE tokenization
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

train_df_train = train_df.iloc[train_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train split size: {len(train_df_train)}, Val split size: {len(train_df_val)}")

# Create sequences
print("Creating sequences...")
X_train_seq = np.stack([text_to_sequence(text, vocab, max_seq_len) for text in train_df_train["text"]])
X_val_seq = np.stack([text_to_sequence(text, vocab, max_seq_len) for text in train_df_val["text"]])
X_test_seq = np.stack([text_to_sequence(text, vocab, max_seq_len) for text in test_df["text"]])

y_train = train_df_train["author_encoded"].values
y_val = train_df_val["author_encoded"].values

print(f"Train sequences: {X_train_seq.shape}, Val sequences: {X_val_seq.shape}, Test sequences: {X_test_seq.shape}")

# Compute stylometric features
print("Computing stylometric features...")
stylo_features_train = np.array([compute_stylometric_features(text) for text in train_df_train["text"]])
stylo_features_val = np.array([compute_stylometric_features(text) for text in train_df_val["text"]])
stylo_features_test = np.array([compute_stylometric_features(text) for text in test_df["text"]])

# Normalize stylometric features
scaler = StandardScaler()
stylo_features_train = scaler.fit_transform(stylo_features_train)
stylo_features_val = scaler.transform(stylo_features_val)
stylo_features_test = scaler.transform(stylo_features_test)

print(f"Stylometric features shape - Train: {stylo_features_train.shape}, Val: {stylo_features_val.shape}, Test: {stylo_features_test.shape}")

# ===================== MODEL DESIGN =====================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, num_classes=3,
                 kernel_sizes=[2, 3, 4, 5], num_filters=128, dropout=0.5, stylo_dim=6):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,  # Finetune GloVe embeddings
            padding_idx=0
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                      kernel_size=k, padding=k-1)
            for k in kernel_sizes
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        # Stylometric MLP
        self.stylo_mlp = nn.Sequential(
            nn.Linear(stylo_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters + 32, num_classes)

    def forward(self, x, stylo_features=None):
        # x shape: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        emb = emb.permute(0, 2, 1)  # (batch, embedding_dim, seq_len) for Conv1d
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            conv_out = conv(emb)  # (batch, num_filters, seq_len + padding)
            conv_out = self.batch_norms[i](conv_out)
            conv_out = F.relu(conv_out)
            # Global max pooling over sequence dimension
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)
        combined = torch.cat(conv_outputs, dim=1)  # (batch, len(kernel_sizes) * num_filters)

        # Process stylometric features if provided
        if stylo_features is not None:
            stylo_out = self.stylo_mlp(stylo_features)  # (batch, 32)
            combined = torch.cat([combined, stylo_out], dim=1)  # (batch, len(kernel_sizes)*num_filters + 32)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TextCNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    embedding_matrix=embedding_matrix,
    num_classes=3,
    kernel_sizes=[2, 3, 4, 5],
    num_filters=128,
    dropout=0.3,
    stylo_dim=6
).to(device)

# Loss function with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_pred = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_pred, dim=1))

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Optimizer - all parameters with same LR since we train from scratch
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# ===================== TRAINING & EVALUATION =====================

class TextDataset(Dataset):
    def __init__(self, sequences, stylo_features, labels=None):
        self.sequences = torch.LongTensor(sequences)
        self.stylo_features = torch.FloatTensor(stylo_features)
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.stylo_features[idx], self.labels[idx]
        return self.sequences[idx], self.stylo_features[idx]

# Create datasets and dataloaders
train_dataset = TextDataset(X_train_seq, stylo_features_train, y_train)
val_dataset = TextDataset(X_val_seq, stylo_features_val, y_val)
test_dataset = TextDataset(X_test_seq, stylo_features_test)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)

# Exponential Moving Average
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Warmup + Cosine Annealing scheduler
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr / 1e-4, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

num_epochs = 30
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)

scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6)

ema = EMA(model, decay=0.999)
ema.register()

best_val_loss = float("inf")
best_model_state = None
patience = 5
patience_counter = 0

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_X, batch_stylo, batch_y in train_loader:
        batch_X, batch_stylo, batch_y = batch_X.to(device), batch_stylo.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X, stylo_features=batch_stylo)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Step scheduler after each batch
        ema.update()
        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(logits, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # Use EMA for validation
    ema.apply_shadow()
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_labels_list = []

    with torch.no_grad():
        for batch_X, batch_stylo, batch_y in val_loader:
            batch_X, batch_stylo, batch_y = batch_X.to(device), batch_stylo.to(device), batch_y.to(device)
            logits = model(batch_X, stylo_features=batch_stylo)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_labels_list.append(batch_y.cpu().numpy())
            _, predicted = torch.max(logits, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    ema.restore()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels_list)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_log_loss = log_loss(val_labels, val_probs)

    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} LogLoss: {val_log_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "./working/best_model.pt")
        # Also save EMA weights
        ema_shadow_copy = ema.shadow.copy()
        torch.save(ema_shadow_copy, "./working/ema_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model and compute final validation score
model.load_state_dict(best_model_state)
# Apply EMA for final evaluation
ema = EMA(model, decay=0.999)
ema.register()
ema_shadow = torch.load("./working/ema_model.pt", map_location=device)
ema.shadow = ema_shadow
ema.apply_shadow()

model.eval()
val_probs = []
with torch.no_grad():
    for batch_X, batch_stylo, _ in val_loader:
        batch_X, batch_stylo = batch_X.to(device), batch_stylo.to(device)
        logits = model(batch_X, stylo_features=batch_stylo)
        probs = torch.softmax(logits, dim=1)
        val_probs.append(probs.cpu().numpy())

val_probs = np.concatenate(val_probs)
val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
final_val_score = log_loss(y_val, val_probs)

print(f"Final Validation Score: {final_val_score}")

# Generate test predictions
print("Generating test predictions...")
# Apply EMA for inference
ema.apply_shadow()
model.eval()
test_probs = []
with torch.no_grad():
    for batch_X, batch_stylo in test_loader:
        batch_X, batch_stylo = batch_X.to(device), batch_stylo.to(device)
        logits = model(batch_X, stylo_features=batch_stylo)
        probs = torch.softmax(logits, dim=1)
        test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# Restore original weights
ema.restore()

# Create submission file
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")