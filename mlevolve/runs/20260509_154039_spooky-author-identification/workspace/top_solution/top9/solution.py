import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
import re
import string
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import log_loss
import os
import gc
import joblib
import scipy

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# ============================================================
# 2. AUTHOR LABEL ENCODING
# ============================================================
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_map)
y = train_df["label"].values

# ============================================================
# 3. STRATIFIED SPLIT (train 80%, val 10%, test 10%)
# ============================================================
X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
    train_df["text"].values, y, test_size=0.2, stratify=y, random_state=42
)

X_val_texts, X_test_texts, y_val, y_test = train_test_split(
    X_temp_texts, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


# ============================================================
# 4. FEATURE ENGINEERING FUNCTIONS
# ============================================================
def extract_sentence_stats(texts):
    """Extract numerical sentence-level statistics that capture authorial style."""
    features = []
    for text in texts:
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        char_count = len(text)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Punctuation frequencies (per character)
        comma_freq = text.count(",") / max(char_count, 1)
        period_freq = text.count(".") / max(char_count, 1)
        exclaim_freq = text.count("!") / max(char_count, 1)
        question_freq = text.count("?") / max(char_count, 1)
        semicolon_freq = text.count(";") / max(char_count, 1)
        colon_freq = text.count(":") / max(char_count, 1)
        dash_freq = text.count("\u2014") + text.count("\u2013") + text.count("-")
        dash_freq /= max(char_count, 1)
        quote_freq = text.count('"') + text.count("'") / max(char_count, 1)

        # Capitalization patterns
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(
            len([c for c in text if c.isalpha()]), 1
        )
        first_word_cap_ratio = sum(1 for s in sentences if s and s[0].isupper()) / max(
            sentence_count, 1
        )

        # Function word density (common author-specific patterns)
        function_words = set(
            [
                "the",
                "and",
                "of",
                "to",
                "a",
                "in",
                "that",
                "was",
                "is",
                "it",
                "with",
                "as",
                "for",
                "on",
                "but",
                "by",
                "his",
                "had",
                "not",
                "be",
                "at",
                "from",
                "were",
                "are",
                "so",
                "have",
                "has",
                "been",
                "or",
                "he",
                "you",
                "I",
                "my",
                "me",
                "she",
                "her",
                "they",
                "them",
                "their",
                "we",
                "our",
                "us",
                "all",
                "an",
                "no",
                "do",
                "did",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "will",
                "can",
                "must",
                "now",
                "then",
                "there",
                "here",
                "very",
                "such",
                "some",
                "any",
                "every",
                "each",
                "both",
                "own",
                "this",
                "that",
                "these",
                "those",
                "who",
                "whom",
                "whose",
                "which",
                "what",
                "when",
                "where",
                "why",
                "how",
                "if",
                "though",
                "while",
                "because",
                "since",
                "until",
                "before",
                "after",
                "about",
                "between",
                "through",
                "during",
                "without",
                "within",
                "upon",
                "into",
                "over",
                "under",
                "above",
                "below",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
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
                "also",
                "moreover",
                "furthermore",
                "however",
                "therefore",
                "thus",
                "nevertheless",
                "meanwhile",
                "afterward",
            ]
        )
        token_lower = [w.lower().strip(string.punctuation) for w in words]
        fn_words = [w for w in token_lower if w in function_words]
        func_word_ratio = len(fn_words) / max(len(token_lower), 1)

        # Unique word ratio (vocabulary richness)
        unique_words = len(set(token_lower))
        vocab_richness = unique_words / max(word_count, 1)

        # Stopword ratio
        stop_words = set(
            [
                "the",
                "and",
                "of",
                "to",
                "a",
                "in",
                "that",
                "is",
                "was",
                "it",
                "with",
                "as",
                "for",
                "on",
                "but",
                "by",
                "his",
                "had",
                "not",
                "be",
                "at",
                "from",
                "were",
                "are",
                "so",
                "have",
                "has",
                "been",
                "or",
                "he",
                "you",
                "i",
                "my",
                "me",
                "she",
                "her",
                "they",
                "them",
                "their",
                "we",
                "our",
                "us",
                "all",
                "an",
                "no",
                "do",
                "did",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "will",
                "can",
                "must",
                "now",
                "then",
                "there",
                "here",
                "very",
                "such",
                "some",
                "any",
                "every",
                "each",
                "both",
                "own",
                "this",
                "that",
                "these",
                "those",
                "who",
                "whom",
                "whose",
                "which",
                "what",
                "when",
                "where",
                "why",
                "how",
                "if",
                "though",
                "while",
                "because",
                "since",
                "until",
                "before",
                "after",
                "about",
                "between",
                "through",
                "during",
                "without",
                "within",
                "upon",
                "into",
                "over",
                "under",
                "above",
                "below",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
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
                "also",
                "moreover",
                "furthermore",
                "however",
                "therefore",
                "thus",
                "nevertheless",
                "meanwhile",
                "afterward",
            ]
        )
        token_lower_words = [
            w.lower().strip(string.punctuation) for w in words if w.strip()
        ]
        stopword_count = sum(1 for w in token_lower_words if w in stop_words)
        stopword_ratio = stopword_count / max(len(token_lower_words), 1)

        features.append(
            [
                char_count,
                word_count,
                sentence_count,
                avg_word_len,
                avg_sent_len,
                comma_freq,
                period_freq,
                exclaim_freq,
                question_freq,
                semicolon_freq,
                colon_freq,
                dash_freq,
                quote_freq,
                uppercase_ratio,
                first_word_cap_ratio,
                func_word_ratio,
                vocab_richness,
                stopword_ratio,
            ]
        )
    return np.array(features)


def extract_char_ngrams(texts, vectorizer=None, fit=False):
    """Extract character n-grams (2-5 grams) that capture morphology and punctuation patterns."""
    if fit:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=15000,
            sublinear_tf=True,
            min_df=5,
            max_df=0.8,
        )
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    else:
        return vectorizer.transform(texts)


def extract_word_ngrams(texts, vectorizer=None, fit=False):
    """Extract word n-grams (1-3 grams) that capture vocabulary and phrase patterns."""
    if fit:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=15000,
            sublinear_tf=True,
            min_df=5,
            max_df=0.8,
            stop_words=None,
        )
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    else:
        return vectorizer.transform(texts)


def extract_pos_patterns(texts, vectorizer=None, fit=False):
    """
    Extract POS tag n-grams to capture syntactic patterns.
    Using a simple heuristic: tag words based on suffixes and patterns.
    """

    def tag_sentence(text):
        words = text.split()
        tags = []
        for w in words:
            w_clean = w.lower().strip(string.punctuation)
            if not w_clean:
                tags.append("PUNCT")
            elif w_clean in ["the", "a", "an"]:
                tags.append("DET")
            elif w_clean in ["is", "am", "are", "was", "were", "be", "been", "being"]:
                tags.append("VERB_BE")
            elif w_clean in ["have", "has", "had", "having"]:
                tags.append("VERB_HAVE")
            elif w_clean in ["do", "does", "did", "doing", "done"]:
                tags.append("VERB_DO")
            elif w_clean in [
                "will",
                "would",
                "can",
                "could",
                "shall",
                "should",
                "may",
                "might",
                "must",
            ]:
                tags.append("MODAL")
            elif w_clean.endswith("ing"):
                tags.append("VBG")
            elif w_clean.endswith("ed"):
                tags.append("VBD")
            elif w_clean.endswith("ly"):
                tags.append("ADV")
            elif (
                w_clean.endswith("tion")
                or w_clean.endswith("sion")
                or w_clean.endswith("ment")
            ):
                tags.append("NN")
            elif (
                w_clean.endswith("ness")
                or w_clean.endswith("ity")
                or w_clean.endswith("ship")
            ):
                tags.append("NN")
            elif (
                w_clean.endswith("able")
                or w_clean.endswith("ible")
                or w_clean.endswith("ful")
            ):
                tags.append("ADJ")
            elif (
                w_clean.endswith("ous")
                or w_clean.endswith("ive")
                or w_clean.endswith("al")
            ):
                tags.append("ADJ")
            elif w_clean.endswith("s") and len(w_clean) > 2:
                tags.append("NNS")
            elif len(w_clean) <= 3 and w_clean.isalpha():
                tags.append("FUNC")
            elif w_clean[0].isupper() and len(w_clean) > 1:
                tags.append("NNP")
            else:
                tags.append("OTHER")
        return " ".join(tags)

    pos_texts = [tag_sentence(t) for t in texts]

    if fit:
        vectorizer = CountVectorizer(
            ngram_range=(2, 4), max_features=5000, min_df=3, max_df=0.9
        )
        features = vectorizer.fit_transform(pos_texts)
        return features, vectorizer
    else:
        return vectorizer.transform(pos_texts)


# ============================================================
# 5. EXTRACT FEATURES (FIT ONLY ON TRAINING DATA)
# ============================================================
# Sentence-level statistics
train_stats = extract_sentence_stats(X_train_texts)
val_stats = extract_sentence_stats(X_val_texts)
test_stats = extract_sentence_stats(X_test_texts)

# Character n-grams (fit on training only)
char_features_train, char_vectorizer = extract_char_ngrams(X_train_texts, fit=True)
char_features_val = extract_char_ngrams(X_val_texts, vectorizer=char_vectorizer)
char_features_test = extract_char_ngrams(X_test_texts, vectorizer=char_vectorizer)

# Word n-grams (fit on training only)
word_features_train, word_vectorizer = extract_word_ngrams(X_train_texts, fit=True)
word_features_val = extract_word_ngrams(X_val_texts, vectorizer=word_vectorizer)
word_features_test = extract_word_ngrams(X_test_texts, vectorizer=word_vectorizer)

# POS pattern features (fit on training only)
pos_features_train, pos_vectorizer = extract_pos_patterns(X_train_texts, fit=True)
pos_features_val = extract_pos_patterns(X_val_texts, vectorizer=pos_vectorizer)
pos_features_test = extract_pos_patterns(X_test_texts, vectorizer=pos_vectorizer)

# ============================================================
# 6. COMBINE ALL FEATURES INTO SPARSE MATRIX
# ============================================================
train_stats_sparse = csr_matrix(train_stats)
val_stats_sparse = csr_matrix(val_stats)
test_stats_sparse = csr_matrix(test_stats)

X_train = hstack(
    [train_stats_sparse, char_features_train, word_features_train, pos_features_train]
)
X_val = hstack(
    [val_stats_sparse, char_features_val, word_features_val, pos_features_val]
)
X_test = hstack(
    [test_stats_sparse, char_features_test, word_features_test, pos_features_test]
)

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")
print(f"Test features shape: {X_test.shape}")

# ============================================================
# 7. PREPARE TEST DATA FOR SUBMISSION (ALL TEST SAMPLES)
# ============================================================
test_texts_full = test_df["text"].values
test_stats_full = csr_matrix(extract_sentence_stats(test_texts_full))
test_char_full = extract_char_ngrams(test_texts_full, vectorizer=char_vectorizer)
test_word_full = extract_word_ngrams(test_texts_full, vectorizer=word_vectorizer)
test_pos_full = extract_pos_patterns(test_texts_full, vectorizer=pos_vectorizer)

X_test_full = hstack([test_stats_full, test_char_full, test_word_full, test_pos_full])

# ============================================================
# 8. SAVE PROCESSED DATA FOR NEXT STEPS
# ============================================================
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)
np.save("./working/y_test.npy", y_test)

save_npz("./working/X_train.npz", X_train)
save_npz("./working/X_val.npz", X_val)
save_npz("./working/X_test.npz", X_test)
save_npz("./working/X_test_full.npz", X_test_full)

joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
joblib.dump(pos_vectorizer, "./working/pos_vectorizer.pkl")

test_ids = test_df["id"].values
np.save("./working/test_ids.npy", test_ids)

np.save("./working/X_train_texts.npy", X_train_texts)
np.save("./working/X_val_texts.npy", X_val_texts)
np.save("./working/X_test_texts.npy", X_test_texts)
np.save("./working/test_texts_full.npy", test_texts_full)

print("Data processing and feature engineering completed successfully.")
print(
    f"Train samples: {len(y_train)}, Val samples: {len(y_val)}, Test samples: {len(y_test)}"
)
print(f"Feature dimensions: {X_train.shape[1]}")

# ============================================================
# 9. SENTENCE-BERT FEATURE EXTRACTION (FROZEN BACKBONE) - SMALLER MODEL
# ============================================================
print("Extracting S-BERT embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use all-MiniLM-L6-v2 for lower memory usage (384-dim) to allow TF-IDF fusion
model_name = "all-MiniLM-L6-v2"
sbert_model = SentenceTransformer(model_name)
sbert_model = sbert_model.to(device)

# Extract embeddings (frozen - no gradient computation needed)
with torch.no_grad():
    train_embeddings = sbert_model.encode(
        X_train_texts.tolist(), convert_to_tensor=True, batch_size=128
    )
    val_embeddings = sbert_model.encode(
        X_val_texts.tolist(), convert_to_tensor=True, batch_size=128
    )
    test_embeddings = sbert_model.encode(
        test_texts_full.tolist(), convert_to_tensor=True, batch_size=128
    )

print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Val embeddings shape: {val_embeddings.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")


# ============================================================
# 10. CUSTOM DATASET FOR ON-THE-FLY TF-IDF CONVERSION
# ============================================================
class TfidfSbertDataset(torch.utils.data.Dataset):
    """Loads precomputed TF-IDF sparse matrix and SBERT embeddings on-the-fly."""

    def __init__(self, sbert_embeddings, tfidf_path, labels=None):
        self.sbert_embeddings = sbert_embeddings
        self.tfidf = scipy.sparse.load_npz(tfidf_path)
        self.labels = labels

    def __len__(self):
        return len(self.sbert_embeddings)

    def __getitem__(self, idx):
        sbert_vec = self.sbert_embeddings[idx]
        tfidf_dense = torch.from_numpy(self.tfidf[idx].toarray()).float().squeeze(0)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return sbert_vec, tfidf_dense, label
        else:
            return sbert_vec, tfidf_dense


# ============================================================
# 11. MULTI-INPUT FUSION NETWORK
# ============================================================
sbert_dim = train_embeddings.shape[1]  # 384
tfidf_dim = X_train.shape[1]  # 35018
num_classes = 3
hidden_dim = 256
dropout_rate = 0.2


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value):
        # query, key, value: (batch, seq_len, d_model) - we have seq_len=1
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output.squeeze(1)  # (batch, d_model)


class MultiInputFusionNet(nn.Module):
    def __init__(self, sbert_dim, tfidf_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()
        # TF-IDF projection MLP (35018 -> 512 -> 128)
        self.tfidf_fc1 = nn.Linear(tfidf_dim, 512)
        self.tfidf_fc2 = nn.Linear(512, 128)
        self.tfidf_norm = nn.LayerNorm(128)
        # SBERT projection to match TF-IDF projected dimension
        self.sbert_proj = nn.Linear(sbert_dim, 128)
        self.sbert_norm = nn.LayerNorm(128)

        # Multi-head cross-attention: SBERT queries TF-IDF, and TF-IDF queries SBERT
        self.cross_attn_sbert2tfidf = CrossAttention(d_model=128, num_heads=4)
        self.cross_attn_tfidf2sbert = CrossAttention(d_model=128, num_heads=4)

        # Classifier head on fused vector (128*3 = 384)
        self.fc1 = nn.Linear(384, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sbert_vec, tfidf_vec):
        # Project both modalities to common 128-dim space
        tfidf_out = self.dropout(self.relu(self.tfidf_fc1(tfidf_vec)))
        tfidf_out = self.tfidf_fc2(tfidf_out)
        tfidf_out = self.tfidf_norm(tfidf_out)

        sbert_out = self.sbert_proj(sbert_vec)
        sbert_out = self.sbert_norm(sbert_out)

        # Add sequence dimension for cross-attention (seq_len=1)
        sbert_seq = sbert_out.unsqueeze(1)  # (batch, 1, 128)
        tfidf_seq = tfidf_out.unsqueeze(1)  # (batch, 1, 128)

        # Cross-attention: SBERT as query, TF-IDF as key/value
        attended_sbert = self.cross_attn_sbert2tfidf(sbert_seq, tfidf_seq, tfidf_seq)
        # Cross-attention: TF-IDF as query, SBERT as key/value
        attended_tfidf = self.cross_attn_tfidf2sbert(tfidf_seq, sbert_seq, sbert_seq)

        # Concatenate original projections with attended outputs
        fused = torch.cat([sbert_out, attended_sbert, attended_tfidf], dim=1)  # 384-dim

        # Classifier head
        x = self.dropout(self.relu(self.fc1(fused)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


classifier = MultiInputFusionNet(
    sbert_dim, tfidf_dim, hidden_dim, num_classes, dropout_rate
)
classifier = classifier.to(device)

# ============================================================
# 12. TRAINING SETUP WITH MULTI-INPUT DATALOADERS
# ============================================================
# Keep embeddings on CPU; DataLoader will move to device in training loop

train_labels = torch.tensor(y_train, dtype=torch.long)
val_labels = torch.tensor(y_val, dtype=torch.long)

batch_size = 1024

# Use custom datasets with on-the-fly TF-IDF conversion
train_dataset = TfidfSbertDataset(
    train_embeddings, "./working/X_train.npz", labels=y_train
)
val_dataset = TfidfSbertDataset(val_embeddings, "./working/X_val.npz", labels=y_val)
test_dataset = TfidfSbertDataset(test_embeddings, "./working/X_test_full.npz")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=0,
)

# ============================================================
# 13. TRAINING SETUP
# ============================================================
criterion = nn.CrossEntropyLoss()

# AdamW with cosine annealing and linear warmup
optimizer = torch.optim.AdamW(
    classifier.parameters(), lr=2e-5, weight_decay=1e-3
)

# Calculate total steps and warmup steps
epochs = 60
patience = 7
batch_size = 1024
train_steps_per_epoch = len(train_loader)  # ~14 steps with batch_size=1024 and 14096 samples
total_steps = epochs * train_steps_per_epoch
warmup_steps = 70  # ~5 epochs * (14096/1024) ≈ 70

class WarmupCosineSchedule:
    """Linear warmup then cosine annealing."""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Linear warmup
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                # Cosine annealing
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
            param_group['lr'] = lr
        return self.optimizer.param_groups[0]['lr']

scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

# ============================================================
# 14. TRAINING LOOP WITH MULTI-INPUT DATA
# ============================================================
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None
first_epoch_state = None

print("\nStarting training...")
for epoch in range(epochs):
    classifier.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_sbert, batch_tfidf, batch_labels in train_loader:
        batch_sbert = batch_sbert.to(device)
        batch_tfidf = batch_tfidf.to(device)
        batch_labels = batch_labels.to(device)
        # Embeddings already moved to device in DataLoader/on-the-fly
        logits = classifier(batch_sbert, batch_tfidf)
        loss = criterion(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        current_lr = scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    classifier.eval()
    total_val_loss = 0
    num_val_batches = 0
    all_val_probs = []
    all_val_targets = []

    with torch.no_grad():
        for batch_sbert, batch_tfidf, batch_labels in val_loader:
            batch_sbert = batch_sbert.to(device)
            batch_tfidf = batch_tfidf.to(device)
            batch_labels = batch_labels.to(device)
            # Embeddings already moved to device in DataLoader/on-the-fly
            logits = classifier(batch_sbert, batch_tfidf)
            loss = criterion(logits, batch_labels)

            total_val_loss += loss.item()
            num_val_batches += 1

            probs = F.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_targets.append(batch_labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_probs = np.concatenate(all_val_probs, axis=0)
    val_targets = np.concatenate(all_val_targets, axis=0)

    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    row_sums = val_probs_clipped.sum(axis=1, keepdims=True)
    val_probs_normalized = val_probs_clipped / row_sums
    val_probs_normalized = np.clip(val_probs_normalized, epsilon, 1 - epsilon)
    row_sums = val_probs_normalized.sum(axis=1, keepdims=True)
    val_probs_normalized = val_probs_normalized / row_sums

    val_log_loss = log_loss(val_targets, val_probs_normalized)

    print(
        f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f} | LR: {current_lr:.6f}"
    )

    if first_epoch_state is None:
        first_epoch_state = classifier.state_dict().copy()
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_log_loss = val_log_loss
        patience_counter = 0
        best_model_state = classifier.state_dict().copy()
        torch.save(best_model_state, "./working/best_fusion_classifier.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# ============================================================
# 15. LOAD BEST MODEL AND COMPUTE FINAL VALIDATION SCORE
# ============================================================
if best_model_state is None:
    best_model_state = first_epoch_state
classifier.load_state_dict(best_model_state)
classifier.eval()

with torch.no_grad():
    all_val_probs = []
    for batch_sbert, batch_tfidf, _ in val_loader:
        batch_sbert = batch_sbert.to(device)
        batch_tfidf = batch_tfidf.to(device)
        logits = classifier(batch_sbert, batch_tfidf)
        probs = F.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)

epsilon = 1e-15
val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
row_sums = val_probs_clipped.sum(axis=1, keepdims=True)
val_probs_normalized = val_probs_clipped / row_sums
val_probs_normalized = np.clip(val_probs_normalized, epsilon, 1 - epsilon)
row_sums = val_probs_normalized.sum(axis=1, keepdims=True)
val_probs_normalized = val_probs_normalized / row_sums

final_val_score = log_loss(val_labels.cpu().numpy(), val_probs_normalized)

# ============================================================
# 16. TEST INFERENCE AND SUBMISSION FILE
# ============================================================
print("\nGenerating test predictions...")

classifier.eval()
all_test_probs = []
with torch.no_grad():
    for batch_sbert, batch_tfidf in test_loader:
        batch_sbert = batch_sbert.to(device)
        batch_tfidf = batch_tfidf.to(device)
        logits = classifier(batch_sbert, batch_tfidf)
        probs = F.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

test_probs_clipped = np.clip(test_probs, epsilon, 1 - epsilon)
row_sums = test_probs_clipped.sum(axis=1, keepdims=True)
test_probs_normalized = test_probs_clipped / row_sums

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_normalized[:, 0],
        "HPL": test_probs_normalized[:, 1],
        "MWS": test_probs_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# ============================================================
# 17. PRINT FINAL VALIDATION SCORE
# ============================================================
print(f"Final Validation Score: {final_val_score}")

# Clean up
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()