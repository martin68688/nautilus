import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import re
import string
import os
import warnings
from scipy.sparse import hstack, save_npz, load_npz

warnings.filterwarnings("ignore")

# ============================================================
# Data Processing and Feature Engineering
# ============================================================


def load_data():
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")
    return train, test


def extract_sentence_stats(text_series):
    features = pd.DataFrame()
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["unique_char_ratio"] = text_series.apply(
        lambda x: len(set(str(x).lower())) / (len(str(x)) + 1)
    )
    features["capital_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    features["first_word_caps"] = text_series.apply(
        lambda x: 1 if str(x)[0].isupper() else 0
    )
    for punct in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "..."]:
        features[f"punct_{punct}"] = text_series.str.count(re.escape(punct))
    features["total_punct"] = features[
        [c for c in features.columns if c.startswith("punct_")]
    ].sum(axis=1)
    features["digit_count"] = text_series.str.count(r"\d")
    features["special_chars"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace())
    )
    return features


def extract_style_features(text_series):
    features = pd.DataFrame()
    features["syllable_count"] = text_series.apply(
        lambda x: len(re.findall(r"[aeiouy]+", str(x).lower()))
    )
    features["syllables_per_word"] = features["syllable_count"] / (
        text_series.str.split().str.len() + 1
    )
    words_per_sentence = text_series.str.split().str.len()
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * words_per_sentence
        - 84.6 * (features["syllable_count"] / (words_per_sentence + 1))
    )
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
            "shall",
            "should",
            "may",
            "might",
            "must",
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
            "mine",
            "yours",
            "hers",
            "ours",
            "theirs",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        ]
    )
    features["stopword_ratio"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).lower().split()
            if w.strip(string.punctuation) in stopwords
        )
        / (len(str(x).split()) + 1)
    )
    conjunctions = set(
        [
            "and",
            "but",
            "or",
            "yet",
            "so",
            "for",
            "nor",
            "because",
            "although",
            "while",
            "since",
            "unless",
            "if",
            "when",
            "where",
            "whether",
            "after",
            "before",
            "until",
            "once",
            "as",
        ]
    )
    features["conjunction_density"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).lower().split()
            if w.strip(string.punctuation) in conjunctions
        )
        / (len(str(x).split()) + 1)
    )
    features["ttr"] = text_series.apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )
    features["long_words_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 8)
        / (len(str(x).split()) + 1)
    )
    return features


def preprocess_data():
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Authors distribution:\n{train['author'].value_counts()}")

    X = train["text"].values
    y = train["author"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
    )

    train_stats = extract_sentence_stats(pd.Series(X_train))
    train_style = extract_style_features(pd.Series(X_train))
    val_stats = extract_sentence_stats(pd.Series(X_val))
    val_style = extract_style_features(pd.Series(X_val))
    test_stats = extract_sentence_stats(test["text"])
    test_style = extract_style_features(test["text"])

    train_stylo = pd.concat([train_stats, train_style], axis=1)
    val_stylo = pd.concat([val_stats, val_style], axis=1)
    test_stylo = pd.concat([test_stats, test_style], axis=1)

    train_stylo = train_stylo.replace([np.inf, -np.inf], np.nan).fillna(0)
    val_stylo = val_stylo.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_stylo = test_stylo.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    train_stylo_scaled = scaler.fit_transform(train_stylo)
    val_stylo_scaled = scaler.transform(val_stylo)
    test_stylo_scaled = scaler.transform(test_stylo)

    train_stylo_df = pd.DataFrame(train_stylo_scaled, columns=train_stylo.columns)
    val_stylo_df = pd.DataFrame(val_stylo_scaled, columns=train_stylo.columns)
    test_stylo_df = pd.DataFrame(test_stylo_scaled, columns=train_stylo.columns)

    train_text_series = pd.Series(X_train)
    val_text_series = pd.Series(X_val)
    test_text_series = test["text"]

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        max_features=5000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    char_train_features = char_vectorizer.fit_transform(train_text_series)
    char_val_features = char_vectorizer.transform(val_text_series)
    char_test_features = char_vectorizer.transform(test_text_series)

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        min_df=3,
        max_df=0.95,
    )
    word_train_features = word_vectorizer.fit_transform(train_text_series)
    word_val_features = word_vectorizer.transform(val_text_series)
    word_test_features = word_vectorizer.transform(test_text_series)

    train_tfidf = hstack([char_train_features, word_train_features])
    val_tfidf = hstack([char_val_features, word_val_features])
    test_tfidf = hstack([char_test_features, word_test_features])

    os.makedirs("./working", exist_ok=True)
    np.save("./working/train_stylo.npy", train_stylo_df.values.astype(np.float32))
    np.save("./working/val_stylo.npy", val_stylo_df.values.astype(np.float32))
    np.save("./working/test_stylo.npy", test_stylo_df.values.astype(np.float32))
    save_npz("./working/train_tfidf.npz", train_tfidf.astype(np.float32))
    save_npz("./working/val_tfidf.npz", val_tfidf.astype(np.float32))
    save_npz("./working/test_tfidf.npz", test_tfidf.astype(np.float32))

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    np.save("./working/y_train.npy", y_train_encoded)
    np.save("./working/y_val.npy", y_val_encoded)
    np.save("./working/y_train_orig.npy", y_train)
    np.save("./working/y_val_orig.npy", y_val)
    np.save("./working/author_labels.npy", le.classes_)
    test_ids = test["id"].values
    # Save as fixed-length string array to avoid object dtype loading issues
    test_ids_bytes = np.array([s.encode('utf-8') for s in test_ids])
    np.save("./working/test_ids.npy", test_ids_bytes)

    print(f"\nProcessed data shapes:")
    print(f"Train stylo: {train_stylo_df.shape}")
    print(f"Train TF-IDF: {train_tfidf.shape}")
    print(f"Val stylo: {val_stylo_df.shape}")
    print(f"Val TF-IDF: {val_tfidf.shape}")
    print(f"Test stylo: {test_stylo_df.shape}")
    print(f"Test TF-IDF: {test_tfidf.shape}")


# ============================================================
# Dataset and Model Definition
# ============================================================


class SpookyDataset(Dataset):
    def __init__(self, tfidf_features, stylo_features, labels=None):
        self.tfidf = torch.FloatTensor(tfidf_features)
        self.stylo = torch.FloatTensor(stylo_features)
        self.labels = labels
        if labels is not None:
            self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.tfidf)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tfidf[idx], self.stylo[idx], self.labels[idx]
        return self.tfidf[idx], self.stylo[idx]


class CrossAttentionFusion(nn.Module):
    def __init__(self, tfidf_dim, stylo_dim, d_model=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.tfidf_proj = nn.Linear(tfidf_dim, d_model)
        self.stylo_proj = nn.Linear(stylo_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, tfidf, stylo):
        # Project both modalities to same dimension
        tfidf_proj = self.tfidf_proj(tfidf).unsqueeze(1)  # (B, 1, d_model)
        stylo_proj = self.stylo_proj(stylo).unsqueeze(1)    # (B, 1, d_model)

        # Cross-attention: stylo as query, tfidf as key/value
        attn_out, _ = self.cross_attention(
            stylo_proj, tfidf_proj, tfidf_proj
        )  # (B, 1, d_model)

        # Gated residual connection
        gate_input = torch.cat([stylo_proj, attn_out], dim=-1)
        gate_val = self.gate(gate_input)  # (B, 1, d_model)
        gated_out = gate_val * stylo_proj + (1 - gate_val) * attn_out

        return self.output_proj(gated_out.squeeze(1))  # (B, d_model)


class MultiInputClassifier(nn.Module):
    def __init__(
        self, tfidf_dim, stylo_dim, hidden_size=512, num_labels=3, dropout=0.3
    ):
        super().__init__()
        self.tfidf_branch = nn.Sequential(
            nn.Linear(tfidf_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stylo_branch = nn.Sequential(
            nn.Linear(stylo_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attention = CrossAttentionFusion(
            tfidf_dim=hidden_size // 2,
            stylo_dim=128,
            d_model=256,
            num_heads=4,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, tfidf, stylo):
        tfidf_out = self.tfidf_branch(tfidf)
        stylo_out = self.stylo_branch(stylo)
        fused = self.cross_attention(tfidf_out, stylo_out)
        logits = self.classifier(fused)
        return logits


# ============================================================
# Training and Evaluation
# ============================================================


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        tfidf, stylo, labels = batch
        tfidf = tfidf.to(device)
        stylo = stylo.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(tfidf, stylo)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            tfidf, stylo, labels = batch
            tfidf = tfidf.to(device)
            stylo = stylo.to(device)
            labels = labels.to(device)
            with autocast():
                logits = model(tfidf, stylo)
                loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    eps = 1e-15
    all_preds = np.clip(all_preds, eps, 1 - eps)
    all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)
    n = len(all_labels)
    log_loss = 0.0
    for i in range(n):
        log_loss += np.log(all_preds[i, all_labels[i]])
    log_loss = -log_loss / n
    return total_loss / num_batches, log_loss, all_preds


def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            tfidf, stylo = batch
            tfidf = tfidf.to(device)
            stylo = stylo.to(device)
            with autocast():
                logits = model(tfidf, stylo)
                probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfidf = load_npz("./working/train_tfidf.npz").toarray()
    val_tfidf = load_npz("./working/val_tfidf.npz").toarray()
    test_tfidf = load_npz("./working/test_tfidf.npz").toarray()
    train_stylo = np.load("./working/train_stylo.npy")
    val_stylo = np.load("./working/val_stylo.npy")
    test_stylo = np.load("./working/test_stylo.npy")
    y_train = np.load("./working/y_train.npy")
    y_val = np.load("./working/y_val.npy")
    test_ids_bytes = np.load("./working/test_ids.npy")
    test_ids = np.array([s.decode('utf-8') for s in test_ids_bytes])

    train_dataset = SpookyDataset(train_tfidf, train_stylo, y_train)
    val_dataset = SpookyDataset(val_tfidf, val_stylo, y_val)
    test_dataset = SpookyDataset(test_tfidf, test_stylo)

    batch_size = 64
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

    model = MultiInputClassifier(
        tfidf_dim=train_tfidf.shape[1],
        stylo_dim=train_stylo.shape[1],
        hidden_size=512,
        num_labels=3,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)
    )

    # Warmup for 5 epochs then cosine annealing
    warmup_epochs = 5
    total_epochs = 35
    warmup_steps = len(train_loader) * warmup_epochs
    total_steps = len(train_loader) * total_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    best_val_logloss = float("inf")
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(total_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_logloss, val_preds = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        print(
            f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    _, final_val_logloss, _ = validate(model, val_loader, criterion, device)

    test_preds = predict(model, test_loader, device)
    eps = 1e-15
    test_preds = np.clip(test_preds, eps, 1 - eps)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "EAP": test_preds[:, 0],
            "HPL": test_preds[:, 1],
            "MWS": test_preds[:, 2],
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"Final Validation Score: {final_val_logloss}")


if __name__ == "__main__":
    preprocess_data()
    train_and_evaluate()
