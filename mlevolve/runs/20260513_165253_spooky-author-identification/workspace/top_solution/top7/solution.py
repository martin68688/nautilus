import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import re
import string
import os
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
submission_template = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])

# ===================== FEATURE ENGINEERING =====================


def extract_stylometric_features(text_series):
    """Extract comprehensive stylometric features from text"""
    features = []
    for text in text_series:
        if not isinstance(text, str):
            features.append([0] * 25)
            continue
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        word_lengths = [len(w) for w in words]
        avg_word_len = np.mean(word_lengths) if word_lengths else 0
        max_word_len = max(word_lengths) if word_lengths else 0
        min_word_len = min(word_lengths) if word_lengths else 0
        std_word_len = np.std(word_lengths) if word_lengths else 0
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
        max_sent_len = max(sent_lengths) if sent_lengths else 0
        min_sent_len = min(sent_lengths) if sent_lengths else 0
        std_sent_len = np.std(sent_lengths) if sent_lengths else 0
        punct_counts = {}
        for p in string.punctuation:
            punct_counts[p] = text.count(p)
        total_punct = sum(punct_counts.values())
        punct_ratio = total_punct / max(len(words), 1)
        specific_punct = [
            punct_counts.get(p, 0)
            for p in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")"]
        ]
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        space_ratio = sum(1 for c in text if c.isspace()) / max(len(text), 1)
        special_char_ratio = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        ) / max(len(text), 1)
        char_count = len(text)
        word_count = len(words)
        type_token_ratio = len(set(words)) / max(word_count, 1)
        unique_word_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
        long_word_ratio = sum(1 for w in words if len(w) > 6) / max(len(words), 1)
        very_long_word_ratio = sum(1 for w in words if len(w) > 10) / max(len(words), 1)
        stopwords = {
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
            "by",
            "with",
            "from",
        }
        stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / max(
            word_count, 1
        )
        feature_vec = [
            avg_word_len,
            max_word_len,
            min_word_len,
            std_word_len,
            avg_sent_len,
            max_sent_len,
            min_sent_len,
            std_sent_len,
            total_punct,
            punct_ratio,
            *specific_punct,
            uppercase_ratio,
            digit_ratio,
            space_ratio,
            special_char_ratio,
            type_token_ratio,
            unique_word_ratio,
            long_word_ratio,
            very_long_word_ratio,
            stopword_ratio,
        ]
        features.append(feature_vec)
    return np.array(features)


# Split BEFORE scaling and feature engineering
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

train_df_train = train_df.iloc[train_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train split size: {len(train_df_train)}, Val split size: {len(train_df_val)}")

# Feature engineering on train split only
print("Extracting stylometric features...")
train_stylo_train = extract_stylometric_features(train_df_train["text"])
train_stylo_val = extract_stylometric_features(train_df_val["text"])
test_stylo = extract_stylometric_features(test_df["text"])
print(f"Stylometric features shape: {train_stylo_train.shape}")

print("Extracting character n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True,
    min_df=5,
    max_df=0.85,
)
char_tfidf_train = char_vectorizer.fit_transform(train_df_train["text"])
char_tfidf_val = char_vectorizer.transform(train_df_val["text"])
char_tfidf_test = char_vectorizer.transform(test_df["text"])
print(f"Char n-gram TF-IDF shape: {char_tfidf_train.shape}")

print("Extracting word n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=8000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.90,
    stop_words="english",
)
word_tfidf_train = word_vectorizer.fit_transform(train_df_train["text"])
word_tfidf_val = word_vectorizer.transform(train_df_val["text"])
word_tfidf_test = word_vectorizer.transform(test_df["text"])
print(f"Word n-gram TF-IDF shape: {word_tfidf_train.shape}")

print("Extracting additional text statistics...")


def text_statistics(text_series):
    """Extract additional text statistics features"""
    features = []
    for text in text_series:
        if not isinstance(text, str):
            features.append([0] * 5)
            continue
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Punctuation diversity
        punct_types = set(
            c for c in text if c in string.punctuation
        )
        punct_diversity = len(punct_types) / max(len(string.punctuation), 1)

        # Sentence complexity: avg clauses per sentence
        clause_markers = len(re.findall(r"\b(and|or|but|because|although|while|when|after|before|if|that|which)\b", text.lower()))
        clause_per_sent = clause_markers / max(len(sentences), 1)

        # Vocabulary richness: hapax legomena ratio (words appearing only once)
        word_freq = {}
        for w in words:
            w_lower = w.lower().strip(string.punctuation)
            if w_lower:
                word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
        hapax_count = sum(1 for v in word_freq.values() if v == 1)
        hapax_ratio = hapax_count / max(len(word_freq), 1)

        # Average syllables per word (approximate by counting vowel groups)
        def count_syllables(word):
            word = word.lower().strip(string.punctuation)
            if not word:
                return 0
            vowels = "aeiouy"
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(count, 1)

        syllable_counts = [count_syllables(w) for w in words if w.strip(string.punctuation)]
        avg_syllables = np.mean(syllable_counts) if syllable_counts else 0

        # Flesch reading ease (approximate)
        total_syllables = sum(syllable_counts)
        total_words = len([w for w in words if w.strip(string.punctuation)])
        total_sentences = len(sentences)
        if total_words > 0 and total_sentences > 0:
            reading_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
            reading_ease = max(0, min(100, reading_ease))
        else:
            reading_ease = 0

        feature_vec = [
            punct_diversity,
            clause_per_sent,
            hapax_ratio,
            avg_syllables,
            reading_ease,
        ]
        features.append(feature_vec)
    return np.array(features)


train_text_stats_train = text_statistics(train_df_train["text"])
train_text_stats_val = text_statistics(train_df_val["text"])
test_text_stats = text_statistics(test_df["text"])
print(f"Text statistics features shape: {train_text_stats_train.shape}")

# Scale on train split only
scaler_stylo = StandardScaler()
train_stylo_scaled = scaler_stylo.fit_transform(train_stylo_train)
val_stylo_scaled = scaler_stylo.transform(train_stylo_val)
test_stylo_scaled = scaler_stylo.transform(test_stylo)

scaler_stats = StandardScaler()
train_text_stats_scaled = scaler_stats.fit_transform(train_text_stats_train)
val_text_stats_scaled = scaler_stats.transform(train_text_stats_val)
test_text_stats_scaled = scaler_stats.transform(test_text_stats)

# Combine features
from scipy.sparse import hstack, csr_matrix

X_train_combined = hstack(
    [
        csr_matrix(train_stylo_scaled),
        csr_matrix(train_text_stats_scaled),
        char_tfidf_train,
        word_tfidf_train,
    ]
)
X_val_combined = hstack(
    [
        csr_matrix(val_stylo_scaled),
        csr_matrix(val_text_stats_scaled),
        char_tfidf_val,
        word_tfidf_val,
    ]
)
X_test_combined = hstack(
    [
        csr_matrix(test_stylo_scaled),
        csr_matrix(test_text_stats_scaled),
        char_tfidf_test,
        word_tfidf_test,
    ]
)

# Feature selection on train split only
selector = SelectKBest(mutual_info_classif, k=min(15000, X_train_combined.shape[1]))
X_train_selected = selector.fit_transform(X_train_combined, train_df_train["author_encoded"])
X_val_selected = selector.transform(X_val_combined)
X_test_selected = selector.transform(X_test_combined)

X_train_split = X_train_selected
X_val_split = X_val_selected
y_train_split = train_df_train["author_encoded"].values
y_val_split = train_df_val["author_encoded"].values

print(f"Train split: {X_train_split.shape}, Val split: {X_val_split.shape}")

# ===================== MODEL DESIGN =====================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=3,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, input_dim) — treat as sequence of length 1
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.input_proj(x)  # (batch, 1, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        # Mean pooling over sequence dimension
        x = x.mean(dim=1)
        return self.classifier(x)


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """Apply mixup augmentation on batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = X_train_split.shape[1]
model = FeatureTransformer(
    input_dim=input_dim,
    num_classes=3,
    d_model=256,
    nhead=4,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.1,
).to(device)

# Loss function with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimization with AdamW
optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-4, weight_decay=1e-4, betas=(0.9, 0.999)
)

# Cosine annealing with warm restarts scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Gradual unfreezing setup
# Layer structure: input_proj, pos_encoder, transformer_encoder.layers[0], transformer_encoder.layers[1], norm, classifier
# We'll freeze input_proj and first transformer layer initially
def set_grad_for_module(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_layers(model, freeze_until_epoch):
    """Freeze layers progressively based on epoch."""
    # Initially freeze embedding layer and first transformer layer
    set_grad_for_module(model.input_proj, False)
    set_grad_for_module(model.transformer_encoder.layers[0], False)
    # Unfreeze embedding after 5 epochs
    if freeze_until_epoch >= 5:
        set_grad_for_module(model.input_proj, True)
    # Unfreeze first transformer layer after 8 epochs
    if freeze_until_epoch >= 8:
        set_grad_for_module(model.transformer_encoder.layers[0], True)


# Initial freeze
freeze_layers(model, 0)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ===================== TRAINING & EVALUATION =====================

# Create datasets and dataloaders
train_dataset = TensorDataset(
    torch.FloatTensor(
        X_train_split.toarray() if hasattr(X_train_split, "toarray") else X_train_split
    ),
    torch.LongTensor(y_train_split),
)
val_dataset = TensorDataset(
    torch.FloatTensor(
        X_val_split.toarray() if hasattr(X_val_split, "toarray") else X_val_split
    ),
    torch.LongTensor(y_val_split),
)
test_dataset = TensorDataset(
    torch.FloatTensor(
        X_test_selected.toarray()
        if hasattr(X_test_selected, "toarray")
        else X_test_selected
    )
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
)

num_epochs = 50
best_val_loss = float("inf")
best_model_state = None
patience = 10
patience_counter = 0

# OneCycleLR scheduler
scheduler_onecycle = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
)

print("Starting training with gradual unfreezing and mixup...")
for epoch in range(num_epochs):
    # Apply gradual unfreezing schedule
    freeze_layers(model, epoch)

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Apply mixup augmentation
        mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=0.4, device=device)

        optimizer.zero_grad()
        logits = model(mixed_X)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler_onecycle.step()

        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(logits, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

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

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels_list)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_log_loss = log_loss(val_labels, val_probs)

    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} LogLoss: {val_log_loss:.4f}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "./working/best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model and compute final validation score
model.load_state_dict(best_model_state)
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
final_val_score = log_loss(y_val_split, val_probs)

print(f"Final Validation Score: {final_val_score}")

# Generate test predictions
print("Generating test predictions...")
model.eval()
test_probs = []
with torch.no_grad():
    for batch_X in test_loader:
        batch_X = batch_X[0].to(device)
        logits = model(batch_X)
        probs = torch.softmax(logits, dim=1)
        test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

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