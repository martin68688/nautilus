import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, load_npz, save_npz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import re
import string
import os
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA LOADING AND STRATIFIED SPLIT
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_label"] = train_df["author"].map(author_map)
train_texts = train_df["text"].values
train_labels = train_df["author_label"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Stratified split (70-15-15 split; we use val for early stopping, rest for train)
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_texts, train_labels, test_size=0.15, stratify=train_labels, random_state=42
)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def extract_stylometric_features(texts):
    features = []
    exclamation_pattern = re.compile(r"!")
    question_pattern = re.compile(r"\?")
    semicolon_pattern = re.compile(r";")
    colon_pattern = re.compile(r":")
    dash_pattern = re.compile(r"—|–|-")
    quote_pattern = re.compile(r'["\'\u2018\u2019\u201c\u201d]')
    ellipsis_pattern = re.compile(r"\.\.\.|…")
    comma_pattern = re.compile(r",")
    period_pattern = re.compile(r"\.")

    lovecraft_indicators = re.compile(
        r"\b(cyclopean|eldritch|non-euclidean|antediluvian|\w+ious\b|ae|aeon|cryptic|monolith|gibbous|noisome|miasmal|squamous|ichor|tellurian|primordial|caduceus|chthonian|nebulous|prodigious|unutterable|nameless|inconceivable|blasphemous|daemoniac)\b",
        re.IGNORECASE,
    )
    poe_indicators = re.compile(
        r"\b(nevermore|chamber|rave|grim|ghastly|terrified|hideous|grotesque|sepulchre|pallid|countenance|appalling|shriek|howl|demons|spectre)\b",
        re.IGNORECASE,
    )
    shelley_indicators = re.compile(
        r"\b(whose|whom|thou|thee|thy|hath|doth|artificial|philosophy|principle|nature|existence|truth|perception|sublime|infinite|eternal|mortal|immortal|virtue|passion)\b",
        re.IGNORECASE,
    )

    for text in texts:
        text_lower = text.lower()
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = max(1, len(re.findall(r"[.!?]+", text)))

        features.append(char_count)
        features.append(word_count)
        features.append(sentence_count)
        features.append(word_count / max(1, sentence_count))
        features.append(char_count / max(1, word_count))
        features.append(len(exclamation_pattern.findall(text)) / max(1, char_count))
        features.append(len(question_pattern.findall(text)) / max(1, char_count))
        features.append(len(semicolon_pattern.findall(text)) / max(1, char_count))
        features.append(len(colon_pattern.findall(text)) / max(1, char_count))
        features.append(len(dash_pattern.findall(text)) / max(1, char_count))
        features.append(len(quote_pattern.findall(text)) / max(1, char_count))
        features.append(len(ellipsis_pattern.findall(text)) / max(1, char_count))
        features.append(len(comma_pattern.findall(text)) / max(1, char_count))
        features.append(len(period_pattern.findall(text)) / max(1, char_count))

        upper_count = sum(1 for c in text if c.isupper())
        features.append(upper_count / max(1, char_count))
        digit_count = sum(1 for c in text if c.isdigit())
        features.append(digit_count / max(1, char_count))

        words = text_lower.split()
        word_lengths = [len(w) for w in words]
        features.append(np.mean(word_lengths) if word_lengths else 0)
        features.append(np.std(word_lengths) if word_lengths else 0)
        features.append(np.median(word_lengths) if word_lengths else 0)
        features.append(max(word_lengths) if word_lengths else 0)
        features.append(min(word_lengths) if word_lengths else 0)

        short_words = sum(1 for w in words if len(w) < 4)
        features.append(short_words / max(1, len(words)))
        long_words = sum(1 for w in words if len(w) > 8)
        features.append(long_words / max(1, len(words)))

        lovecraft_matches = len(lovecraft_indicators.findall(text_lower))
        poe_matches = len(poe_indicators.findall(text_lower))
        shelley_matches = len(shelley_indicators.findall(text_lower))
        features.append(lovecraft_matches / max(1, word_count))
        features.append(poe_matches / max(1, word_count))
        features.append(shelley_matches / max(1, word_count))

        unique_words = len(set(words))
        features.append(unique_words / max(1, len(words)))
        punct_types = sum(
            1 for punct in ["!", "?", ";", ":", "—", ","] if punct in text
        )
        features.append(punct_types / max(1, char_count))
        cap_sequences = len(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text))
        features.append(cap_sequences / max(1, sentence_count))

        stopwords = set(
            [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "as",
                "is",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "can",
                "could",
                "may",
                "might",
                "shall",
                "should",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "its",
                "our",
                "their",
                "not",
                "no",
                "nor",
                "so",
                "if",
                "then",
                "than",
                "too",
                "very",
                "just",
                "also",
                "more",
                "most",
                "some",
                "any",
                "each",
                "every",
                "all",
                "both",
                "few",
                "several",
                "such",
                "only",
                "own",
                "same",
                "other",
                "another",
                "much",
                "many",
                "many",
            ]
        )
        stopword_count = sum(
            1 for w in words if w.rstrip(string.punctuation) in stopwords
        )
        features.append(stopword_count / max(1, len(words)))
        contraction_matches = len(re.findall(r"\b\w+'\w+", text_lower))
        features.append(contraction_matches / max(1, word_count))

    return np.array(features).reshape(len(texts), -1)


train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(test_texts)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

# Character n-grams
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=15000, lowercase=True
)
train_char = char_vectorizer.fit_transform(X_train_texts)
val_char = char_vectorizer.transform(X_val_texts)
test_char = char_vectorizer.transform(test_texts)

# Word n-grams (TF-IDF)
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=15000,
    sublinear_tf=True,
    lowercase=True,
    strip_accents="unicode",
    stop_words=None,
    min_df=3,
    max_df=0.85,
)
train_word = word_vectorizer.fit_transform(X_train_texts)
val_word = word_vectorizer.transform(X_val_texts)
test_word = word_vectorizer.transform(test_texts)

# Author vocabulary overlap features - build ONLY from training split
author_texts_full = {0: [], 1: [], 2: []}
for text, label in zip(X_train_texts, y_train_labels):
    author_texts_full[label].append(text.lower())

author_word_sets = {}
for author_id in [0, 1, 2]:
    words_raw = " ".join(author_texts_full[author_id]).split()
    words_clean = [
        w.strip(string.punctuation)
        for w in words_raw
        if len(w.strip(string.punctuation)) > 2
    ]
    word_counter = Counter(words_clean)
    author_word_sets[author_id] = set([w for w, _ in word_counter.most_common(300)])


def extract_vocab_signature(texts):
    features = []
    for text in texts:
        words_set = set(
            w.strip(string.punctuation).lower()
            for w in text.split()
            if len(w.strip(string.punctuation)) > 2
        )
        overlap_eap = len(words_set & author_word_sets[0]) / max(1, len(words_set))
        overlap_hpl = len(words_set & author_word_sets[1]) / max(1, len(words_set))
        overlap_mws = len(words_set & author_word_sets[2]) / max(1, len(words_set))
        features.append([overlap_eap, overlap_hpl, overlap_mws])
    return np.array(features)


train_vocab = extract_vocab_signature(X_train_texts)
val_vocab = extract_vocab_signature(X_val_texts)
test_vocab = extract_vocab_signature(test_texts)

# Combine all features
train_stylo_sparse = csr_matrix(train_stylo_scaled)
val_stylo_sparse = csr_matrix(val_stylo_scaled)
test_stylo_sparse = csr_matrix(test_stylo_scaled)
train_vocab_sparse = csr_matrix(train_vocab)
val_vocab_sparse = csr_matrix(val_vocab)
test_vocab_sparse = csr_matrix(test_vocab)

X_train_combined = hstack(
    [train_char, train_word, train_stylo_sparse, train_vocab_sparse]
)
X_val_combined = hstack([val_char, val_word, val_stylo_sparse, val_vocab_sparse])
X_test_combined = hstack([test_char, test_word, test_stylo_sparse, test_vocab_sparse])

# Cache feature info
n_char = train_char.shape[1]
n_word = train_word.shape[1]
n_stylo = train_stylo.shape[1]
n_vocab = train_vocab.shape[1]
feature_info = {
    "n_char_features": n_char,
    "n_word_features": n_word,
    "n_stylo_features": n_stylo,
    "n_vocab_features": n_vocab,
    "total_features": X_train_combined.shape[1],
}

# ============================================================
# 3. BUILD VOCABULARY FOR MLP (simple word-level bag-of-words)
# ============================================================
# We will use the sparse TF-IDF features directly as input to an MLP.
# Convert to dense for model input (using small batches).
# We'll implement a simple MLP classifier on the combined sparse features.


class SparseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Dataset class to yield dense mini-batches from sparse matrix
class SpMatrixDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_dense = torch.tensor(self.X[idx].toarray().flatten().astype(np.float32))
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return x_dense, label


train_dataset = SpMatrixDataset(X_train_combined, y_train_labels)
val_dataset = SpMatrixDataset(X_val_combined, y_val_labels)

batch_size = 256
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)

# ============================================================
# 4. MODEL, LOSS, OPTIMIZER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
input_dim = X_train_combined.shape[1]
model = SparseMLP(input_dim=input_dim, hidden_dim=512, num_classes=3, dropout=0.3).to(
    device
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ============================================================
# 5. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
epochs = 120
best_val_logloss = float("inf")
patience_counter = 0
patience = 7

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with autocast():
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            with autocast():
                logits = model(x_batch)
                probs = F.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(y_batch.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
    val_logloss = -np.mean(
        np.sum(np.eye(3)[val_targets] * np.log(val_preds_clipped), axis=1)
    )

    scheduler.step(val_logloss)

    print(
        f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}'
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print(f"Training complete. Best validation log loss: {best_val_logloss:.6f}")

# ============================================================
# 6. FINAL VALIDATION SCORE AND TEST PREDICTIONS
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

# Quick validation re-evaluation for final score
val_preds = []
val_targets = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        with autocast():
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)
        val_preds.append(probs.cpu().numpy())
        val_targets.append(y_batch.numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
score = -np.mean(np.sum(np.eye(3)[val_targets] * np.log(val_preds_clipped), axis=1))

print(f"Final Validation Score: {score}")

# ============================================================
# 7. GENERATE SUBMISSION
# ============================================================
# Create test dataset and loader
test_dataset = SpMatrixDataset(X_test_combined, np.zeros(X_test_combined.shape[0]))
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

test_preds = []
with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        with autocast():
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)
test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)
submission = submission[["id", "EAP", "HPL", "MWS"]]
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")