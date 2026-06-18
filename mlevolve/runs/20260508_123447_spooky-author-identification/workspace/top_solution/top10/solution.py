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
    # NEW: Character n-gram entropy (vocabulary richness)
    features["char_entropy_2"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 2)
    )
    features["char_entropy_3"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 3)
    )
    features["char_entropy_4"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 4)
    )
    # NEW: Sentence length distribution features
    features["sent_len_mean"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[0]
    )
    features["sent_len_std"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[1]
    )
    features["sent_len_skew"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[2]
    )
    features["sent_len_kurtosis"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[3]
    )
    # NEW: Punctuation bigram frequencies
    features["punct_bigram_dash_space"] = text_series.apply(
        lambda x: len(re.findall(r'—\s', str(x)))
    )
    features["punct_bigram_space_dash"] = text_series.apply(
        lambda x: len(re.findall(r'\s—', str(x)))
    )
    # NEW: Paragraph-level features (approximated by double newline counts)
    features["paragraph_count"] = text_series.str.count(r'\n\n')
    features["paragraph_ratio"] = features["paragraph_count"] / (features["word_count"] + 1)
    # NEW: Sentence count
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.findall(r'[.!?]+', str(x)))
    )
    features["words_per_sentence"] = features["word_count"] / (features["sentence_count"] + 1)
    return features


def _ngram_entropy(text, n):
    """Compute entropy of character n-grams in text."""
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams:
        return 0.0
    total = len(ngrams)
    freq = {}
    for ng in ngrams:
        freq[ng] = freq.get(ng, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy


def _sent_len_stats(text):
    """Compute mean, std, skew, kurtosis of word counts per sentence."""
    sentences = re.split(r'[.!?]+', str(text))
    sent_lens = [len(s.split()) for s in sentences if len(s.strip()) > 0]
    if len(sent_lens) < 2:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.array(sent_lens, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr)) + 1e-8
    skew = float(np.mean(((arr - mean) / std) ** 3))
    kurtosis = float(np.mean(((arr - mean) / std) ** 4) - 3.0)
    return (mean, std, skew, kurtosis)


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
    # NEW: Specific stopword patterns ("the" for Lovecraft, "but yet" for Poe)
    features["stopword_the_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) == "the")
        / (len(str(x).split()) + 1)
    )
    features["stopword_but_yet"] = text_series.str.contains(r'\bbut yet\b', case=False, na=False).astype(int)
    features["stopword_and_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) == "and")
        / (len(str(x).split()) + 1)
    )
    features["stopword_of_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) == "of")
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
    # NEW: Archaic/period-specific word flags
    archaic_words = ['thou', 'thee', 'thy', 'thine', 'hath', 'doth', 'art', 'wilt',
                     'canst', 'dost', 'didst', 'hast', 'shalt', 'whence', 'thence',
                     'hither', 'thither', 'wherefore', 'methinks', 'forsooth', 'prithee',
                     'ere', 'whilst', 'betwixt', 'unto', 'thrice', 'nay', 'yea']
    features["archaic_word_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split()
                      if w.strip(string.punctuation) in archaic_words)
    )
    word_count_col = text_series.str.split().str.len()
    features["archaic_word_ratio"] = features["archaic_word_count"] / (word_count_col + 1)
    # NEW: Pronoun usage patterns
    first_person = set(['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'])
    second_person = set(['you', 'your', 'yours', 'yourself', 'yourselves'])
    third_person = set(['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'])
    features["first_person_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) in first_person)
        / (len(str(x).split()) + 1)
    )
    features["second_person_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) in second_person)
        / (len(str(x).split()) + 1)
    )
    features["third_person_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) in third_person)
        / (len(str(x).split()) + 1)
    )
    # NEW: Repeated word patterns (stuttering/emphasis)
    features["repeated_words"] = text_series.apply(
        lambda x: len(re.findall(r'\b(\w+)\s+\1\b', str(x).lower()))
    )
    # NEW: Question and exclamation density
    features["question_count"] = text_series.str.count(r'\?')
    features["exclamation_count"] = text_series.str.count(r'!')
    features["quote_count"] = text_series.str.count(r'"')
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

    # Generate and save character IDs for CNN
    max_char_len = 500
    train_char_ids = np.array([text_to_char_ids(t, max_char_len) for t in X_train], dtype=np.int32)
    val_char_ids = np.array([text_to_char_ids(t, max_char_len) for t in X_val], dtype=np.int32)
    test_char_ids = np.array([text_to_char_ids(t, max_char_len) for t in test["text"]], dtype=np.int32)
    np.save("./working/train_char_ids.npy", train_char_ids)
    np.save("./working/val_char_ids.npy", val_char_ids)
    np.save("./working/test_char_ids.npy", test_char_ids)

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
    print(f"Train char_ids: {train_char_ids.shape}")
    print(f"Val stylo: {val_stylo_df.shape}")
    print(f"Val TF-IDF: {val_tfidf.shape}")
    print(f"Val char_ids: {val_char_ids.shape}")
    print(f"Test stylo: {test_stylo_df.shape}")
    print(f"Test TF-IDF: {test_tfidf.shape}")
    print(f"Test char_ids: {test_char_ids.shape}")


# ============================================================
# Dataset and Model Definition
# ============================================================


# Character vocabulary for CNN
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"-()[]{}@#$%^&*_+=<>/\\|~` "
CHAR2IDX = {ch: idx + 1 for idx, ch in enumerate(CHAR_VOCAB)}  # 0 for padding
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1

def text_to_char_ids(text, max_len=500):
    text = str(text).lower()
    ids = [CHAR2IDX.get(ch, 0) for ch in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


class SpookyDataset(Dataset):
    def __init__(self, tfidf_features, stylo_features, char_ids, labels=None):
        self.tfidf = torch.FloatTensor(tfidf_features)
        self.stylo = torch.FloatTensor(stylo_features)
        self.char_ids = torch.LongTensor(char_ids)
        self.labels = labels
        if labels is not None:
            self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.tfidf)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tfidf[idx], self.stylo[idx], self.char_ids[idx], self.labels[idx]
        return self.tfidf[idx], self.stylo[idx], self.char_ids[idx]


class CharCNN(nn.Module):
    """Character-level 1D-CNN for morphological and orthographic patterns."""
    def __init__(self, char_vocab_size=70, char_embed_dim=48, dropout=0.6):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_embed_dim, out_channels=48, kernel_size=k)
            for k in [3, 4, 5, 7]
        ])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(48 * 4, 128)  # 4 kernels * 48 filters

    def forward(self, char_ids):
        # char_ids: (B, max_seq_len)
        x = self.char_embedding(char_ids)  # (B, L, embed_dim)
        x = x.permute(0, 2, 1)  # (B, embed_dim, L)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(x)  # (B, 48, L - k + 1)
            conv_out = torch.relu(conv_out)
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (B, 48)
            conv_outs.append(conv_out)
        out = torch.cat(conv_outs, dim=1)  # (B, 192)
        out = self.dropout(out)
        out = self.projection(out)  # (B, 128)
        return out


class MultiHeadGating(nn.Module):
    """Efficient multi-head gating fusion with per-author importance weights.
    Uses direct gating instead of full cross-attention to avoid resource issues."""
    def __init__(self, d_model=256, num_heads=4, num_labels=3, dropout=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Multi-head projections for each modality input
        self.tfidf_proj = nn.Linear(d_model, d_model)
        self.stylo_proj = nn.Linear(d_model, d_model)
        self.cnn_proj = nn.Linear(d_model, d_model)

        # Per-head gating weights (learned importance per author)
        self.gate_weights = nn.Parameter(torch.randn(num_heads, num_labels))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, tfidf_feat, stylo_feat, cnn_feat):
        # Each feature: (B, d_model)
        # Project each modality
        tfidf_out = self.tfidf_proj(tfidf_feat)  # (B, d_model)
        stylo_out = self.stylo_proj(stylo_feat)   # (B, d_model)
        cnn_out = self.cnn_proj(cnn_feat)         # (B, d_model)

        # Stack and reshape for multi-head processing
        # Each head processes a different combination of modalities
        B = tfidf_out.size(0)

        # Split into heads
        tfidf_h = tfidf_out.view(B, self.num_heads, self.head_dim)  # (B, H, d_h)
        stylo_h = stylo_out.view(B, self.num_heads, self.head_dim)
        cnn_h = cnn_out.view(B, self.num_heads, self.head_dim)

        # Compute head outputs: each head learns a weighted combination of modalities
        # Using learned gate weights averaged across labels for class-agnostic gating
        gate = torch.softmax(self.gate_weights, dim=0)  # (H, num_labels)
        # Average over label dimension to get per-head importance
        gate = gate.mean(dim=1)  # (H,)
        gate = gate.unsqueeze(0).unsqueeze(-1)  # (1, H, 1)

        # Apply gating: average of modalities with learned head importance
        # Each head learns to emphasize different modality combinations
        head_outputs = (tfidf_h + stylo_h + cnn_h) / 3.0  # (B, H, d_h)
        head_outputs = head_outputs * gate  # (B, H, d_h)

        # Flatten heads back to full dimension
        fused = head_outputs.view(B, self.d_model)  # (B, d_model)

        # Apply residual connection with the average of original projections
        avg_input = (tfidf_out + stylo_out + cnn_out) / 3.0
        fused = fused + avg_input

        # Layer normalization and output projection
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        out = self.output_proj(fused)
        return out


class CrossAttentionFusion(nn.Module):
    def __init__(self, tfidf_dim, stylo_dim, cnn_dim=128, d_model=256, num_heads=8, dropout=0.5):
        super().__init__()
        self.tfidf_proj = nn.Linear(tfidf_dim, d_model)
        self.stylo_proj = nn.Linear(stylo_dim, d_model)
        self.cnn_proj = nn.Linear(cnn_dim, d_model)
        self.multi_head_gating = MultiHeadGating(
            d_model=d_model, num_heads=num_heads, num_labels=3, dropout=dropout
        )

    def forward(self, tfidf, stylo, cnn):
        tfidf_proj = self.tfidf_proj(tfidf)  # (B, d_model)
        stylo_proj = self.stylo_proj(stylo)  # (B, d_model)
        cnn_proj = self.cnn_proj(cnn)  # (B, d_model)
        fused = self.multi_head_gating(tfidf_proj, stylo_proj, cnn_proj)
        return fused


class MultiInputClassifier(nn.Module):
    def __init__(
        self, tfidf_dim, stylo_dim, char_vocab_size=70, max_char_len=500,
        hidden_size=384, num_labels=3, dropout=0.6
    ):
        super().__init__()
        self.tfidf_branch = nn.Sequential(
            nn.Linear(tfidf_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stylo_branch = nn.Sequential(
            nn.Linear(stylo_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            char_embed_dim=48,
            dropout=dropout,
        )
        self.cross_attention = CrossAttentionFusion(
            tfidf_dim=hidden_size // 2,
            stylo_dim=128,
            cnn_dim=128,
            d_model=256,
            num_heads=4,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels),
        )
        self.max_char_len = max_char_len

    def forward(self, tfidf, stylo, char_ids):
        tfidf_out = self.tfidf_branch(tfidf)
        stylo_out = self.stylo_branch(stylo)
        char_out = self.char_cnn(char_ids)
        fused = self.cross_attention(tfidf_out, stylo_out, char_out)
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
        tfidf, stylo, char_ids, labels = batch
        tfidf = tfidf.to(device)
        stylo = stylo.to(device)
        char_ids = char_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(tfidf, stylo, char_ids)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        # Stronger gradient clipping to prevent overfitting
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
            tfidf, stylo, char_ids, labels = batch
            tfidf = tfidf.to(device)
            stylo = stylo.to(device)
            char_ids = char_ids.to(device)
            labels = labels.to(device)
            with autocast():
                logits = model(tfidf, stylo, char_ids)
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
            tfidf, stylo, char_ids = batch
            tfidf = tfidf.to(device)
            stylo = stylo.to(device)
            char_ids = char_ids.to(device)
            with autocast():
                logits = model(tfidf, stylo, char_ids)
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
    train_char_ids = np.load("./working/train_char_ids.npy")
    val_char_ids = np.load("./working/val_char_ids.npy")
    test_char_ids = np.load("./working/test_char_ids.npy")
    y_train = np.load("./working/y_train.npy")
    y_val = np.load("./working/y_val.npy")
    test_ids_bytes = np.load("./working/test_ids.npy")
    test_ids = np.array([s.decode('utf-8') for s in test_ids_bytes])

    train_dataset = SpookyDataset(train_tfidf, train_stylo, train_char_ids, y_train)
    val_dataset = SpookyDataset(val_tfidf, val_stylo, val_char_ids, y_val)
    test_dataset = SpookyDataset(test_tfidf, test_stylo, test_char_ids)

    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = MultiInputClassifier(
        tfidf_dim=train_tfidf.shape[1],
        stylo_dim=train_stylo.shape[1],
        char_vocab_size=CHAR_VOCAB_SIZE,
        max_char_len=500,
        hidden_size=512,
        num_labels=3,
        dropout=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.999)
    )

    total_epochs = 25
    warmup_epochs = 3

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=5e-6
            )
        ],
        milestones=[warmup_epochs]
    )
    scaler = GradScaler()

    best_val_logloss = float("inf")
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(total_epochs):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_logloss, val_preds = validate(
            model, val_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {current_lr:.2e}"
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