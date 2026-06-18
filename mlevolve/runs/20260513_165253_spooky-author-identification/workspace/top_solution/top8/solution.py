import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import warnings
from collections import Counter

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
embedding_dim = 128  # Reduced from 300 because we'll train from scratch

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

# Build randomly initialized embedding matrix (trainable)
np.random.seed(42)
embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(np.float32)
embedding_matrix[0] = 0.0  # <PAD> token stays zero
print(f"Created randomly initialized embedding matrix of shape {embedding_matrix.shape}")

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

# ===================== MODEL DESIGN =====================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, num_classes=3,
                 kernel_sizes=[2, 3, 4, 5], num_filters=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,  # Train embeddings from scratch
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
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
    dropout=0.3
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
    def __init__(self, sequences, labels=None):
        self.sequences = torch.LongTensor(sequences)
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

# Create datasets and dataloaders
train_dataset = TextDataset(X_train_seq, y_train)
val_dataset = TextDataset(X_val_seq, y_val)
test_dataset = TextDataset(X_test_seq)

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
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
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
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
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
    for batch_X, _ in val_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
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
    for batch_X in test_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
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