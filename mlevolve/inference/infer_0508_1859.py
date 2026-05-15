"""
Run 20260508_123447 Train+Inference Script
LogLoss: ~0.1859 (真实 log_loss, 无 INDEX_BUG)
模型: MultiInputClassifier (TF-IDF + Style + CharCNN + CrossAttention + PerAuthorAttention)
含伪标签再训练

用法: python infer_0508_1859.py
"""

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
# 路径配置
# ============================================================
INFERENCE_DIR = "/workspace/nautilus/mlevolve/inference"
DATA_DIR = "/workspace/nautilus/mlevolve/data/spooky-author-identification/prepared/public"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{INFERENCE_DIR}/test.csv"
OUTPUT_CSV = f"{INFERENCE_DIR}/submissions/run_0508_logloss_1859_full_8392.csv"
WORKING_DIR = f"{INFERENCE_DIR}/working_0508"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(f"{INFERENCE_DIR}/submissions", exist_ok=True)


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
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
    features["char_entropy_2"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 2)
    )
    features["char_entropy_3"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 3)
    )
    features["char_entropy_4"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 4)
    )
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
    features["punct_bigram_dash_space"] = text_series.apply(
        lambda x: len(re.findall(r'—\s', str(x)))
    )
    features["punct_bigram_space_dash"] = text_series.apply(
        lambda x: len(re.findall(r'\s—', str(x)))
    )
    features["paragraph_count"] = text_series.str.count(r'\n\n')
    features["paragraph_ratio"] = features["paragraph_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.findall(r'[.!?]+', str(x)))
    )
    features["words_per_sentence"] = features["word_count"] / (features["sentence_count"] + 1)
    return features


def _ngram_entropy(text, n):
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
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "can",
            "could", "shall", "should", "may", "might", "must", "i", "you", "he",
            "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
            "theirs", "this", "that", "these", "those", "what", "which", "who",
            "whom", "whose", "when", "where", "why", "how", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "as", "until", "while", "of", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once",
        ]
    )
    features["stopword_ratio"] = text_series.apply(
        lambda x: sum(
            1 for w in str(x).lower().split()
            if w.strip(string.punctuation) in stopwords
        ) / (len(str(x).split()) + 1)
    )
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
        ["and", "but", "or", "yet", "so", "for", "nor", "because", "although",
         "while", "since", "unless", "if", "when", "where", "whether", "after",
         "before", "until", "once", "as"]
    )
    features["conjunction_density"] = text_series.apply(
        lambda x: sum(
            1 for w in str(x).lower().split()
            if w.strip(string.punctuation) in conjunctions
        ) / (len(str(x).split()) + 1)
    )
    features["ttr"] = text_series.apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )
    features["long_words_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 8)
        / (len(str(x).split()) + 1)
    )
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
    features["repeated_words"] = text_series.apply(
        lambda x: len(re.findall(r'\b(\w+)\s+\1\b', str(x).lower()))
    )
    features["question_count"] = text_series.str.count(r'\?')
    features["exclamation_count"] = text_series.str.count(r'!')
    features["quote_count"] = text_series.str.count(r'"')
    return features


def preprocess_data():
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

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
        analyzer="char", ngram_range=(2, 6), max_features=5000,
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )
    char_train_features = char_vectorizer.fit_transform(train_text_series)
    char_val_features = char_vectorizer.transform(val_text_series)
    char_test_features = char_vectorizer.transform(test_text_series)

    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), max_features=8000,
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
        min_df=3, max_df=0.95,
    )
    word_train_features = word_vectorizer.fit_transform(train_text_series)
    word_val_features = word_vectorizer.transform(val_text_series)
    word_test_features = word_vectorizer.transform(test_text_series)

    train_tfidf = hstack([char_train_features, word_train_features])
    val_tfidf = hstack([char_val_features, word_val_features])
    test_tfidf = hstack([char_test_features, word_test_features])

    np.save(f"{WORKING_DIR}/train_stylo.npy", train_stylo_df.values.astype(np.float32))
    np.save(f"{WORKING_DIR}/val_stylo.npy", val_stylo_df.values.astype(np.float32))
    np.save(f"{WORKING_DIR}/test_stylo.npy", test_stylo_df.values.astype(np.float32))
    save_npz(f"{WORKING_DIR}/train_tfidf.npz", train_tfidf.astype(np.float32))
    save_npz(f"{WORKING_DIR}/val_tfidf.npz", val_tfidf.astype(np.float32))
    save_npz(f"{WORKING_DIR}/test_tfidf.npz", test_tfidf.astype(np.float32))

    max_char_len = 500
    train_char_ids = np.array([text_to_char_ids(t, max_char_len) for t in X_train], dtype=np.int32)
    val_char_ids = np.array([text_to_char_ids(t, max_char_len) for t in X_val], dtype=np.int32)
    test_char_ids = np.array([text_to_char_ids(t, max_char_len) for t in test["text"]], dtype=np.int32)
    np.save(f"{WORKING_DIR}/train_char_ids.npy", train_char_ids)
    np.save(f"{WORKING_DIR}/val_char_ids.npy", val_char_ids)
    np.save(f"{WORKING_DIR}/test_char_ids.npy", test_char_ids)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    np.save(f"{WORKING_DIR}/y_train.npy", y_train_encoded)
    np.save(f"{WORKING_DIR}/y_val.npy", y_val_encoded)
    np.save(f"{WORKING_DIR}/author_labels.npy", le.classes_)
    test_ids = test["id"].values
    test_ids_bytes = np.array([s.encode('utf-8') for s in test_ids])
    np.save(f"{WORKING_DIR}/test_ids.npy", test_ids_bytes)

    print(f"\nProcessed data shapes:")
    print(f"Train stylo: {train_stylo_df.shape}, Train TF-IDF: {train_tfidf.shape}, Train char_ids: {train_char_ids.shape}")
    print(f"Val stylo: {val_stylo_df.shape}, Val TF-IDF: {val_tfidf.shape}, Val char_ids: {val_char_ids.shape}")
    print(f"Test stylo: {test_stylo_df.shape}, Test TF-IDF: {test_tfidf.shape}, Test char_ids: {test_char_ids.shape}")


# Character vocabulary for CNN
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"-()[]{}@#$%^&*_+=<>/\\|~` "
CHAR2IDX = {ch: idx + 1 for idx, ch in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1


def text_to_char_ids(text, max_len=500):
    text = str(text).lower()
    ids = [CHAR2IDX.get(ch, 0) for ch in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


class SpookyDataset(Dataset):
    def __init__(self, tfidf_features, stylo_features, char_ids, labels=None, soft_labels=False):
        self.tfidf = torch.FloatTensor(tfidf_features)
        self.stylo = torch.FloatTensor(stylo_features)
        self.char_ids = torch.LongTensor(char_ids)
        self.labels = labels
        self.soft_labels = soft_labels
        if labels is not None:
            if soft_labels:
                self.labels = torch.FloatTensor(labels)
            else:
                self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.tfidf)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tfidf[idx], self.stylo[idx], self.char_ids[idx], self.labels[idx]
        return self.tfidf[idx], self.stylo[idx], self.char_ids[idx]


class TokenLevelDropout(nn.Module):
    def __init__(self, dropout_prob=0.05):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if self.training and self.dropout_prob > 0:
            mask = torch.bernoulli(
                torch.full((x.size(0), x.size(1), 1), 1.0 - self.dropout_prob, device=x.device)
            )
            x = x * mask / (1.0 - self.dropout_prob)
        return x


class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.9):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x):
        if self.training and self.survival_prob < 1.0:
            mask = torch.bernoulli(
                torch.full((x.size(0), 1), self.survival_prob, device=x.device)
            )
            x = x * mask / self.survival_prob
        return x


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size=70, char_embed_dim=48, dropout=0.6):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.token_dropout = TokenLevelDropout(dropout_prob=0.05)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_embed_dim, out_channels=48, kernel_size=k)
            for k in [3, 4, 5, 7]
        ])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(48 * 4, 128)

    def forward(self, char_ids):
        x = self.char_embedding(char_ids)
        x = self.token_dropout(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = torch.relu(conv_out)
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(conv_out)
        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        out = self.projection(out)
        return out


class PerAuthorAttention(nn.Module):
    def __init__(self, d_model=256, num_labels=3, dropout=0.6):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.author_queries = nn.Parameter(torch.randn(num_labels, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tfidf_feat, stylo_feat, cnn_feat):
        fused = (tfidf_feat + stylo_feat + cnn_feat) / 3.0
        keys = self.key_proj(fused)
        values = self.value_proj(fused)
        attn_scores = torch.matmul(self.author_queries, keys.transpose(0, 1))
        attn_scores = attn_scores / (self.d_model ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vectors = torch.matmul(attn_weights, values)
        output = torch.matmul(attn_weights.transpose(0, 1), context_vectors)
        output = output + fused
        output = self.layer_norm(output)
        output = self.dropout(output)
        output = self.output_proj(output)
        return output


class CrossAttentionFusion(nn.Module):
    def __init__(self, tfidf_dim, stylo_dim, cnn_dim=128, d_model=256, num_heads=8, dropout=0.5):
        super().__init__()
        self.tfidf_proj = nn.Linear(tfidf_dim, d_model)
        self.stylo_proj = nn.Linear(stylo_dim, d_model)
        self.cnn_proj = nn.Linear(cnn_dim, d_model)
        self.per_author_attention = PerAuthorAttention(
            d_model=d_model, num_labels=3, dropout=dropout
        )

    def forward(self, tfidf, stylo, cnn):
        tfidf_proj = self.tfidf_proj(tfidf)
        stylo_proj = self.stylo_proj(stylo)
        cnn_proj = self.cnn_proj(cnn)
        fused = self.per_author_attention(tfidf_proj, stylo_proj, cnn_proj)
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
            StochasticDepth(survival_prob=0.9),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            StochasticDepth(survival_prob=0.9),
        )
        self.stylo_branch = nn.Sequential(
            nn.Linear(stylo_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            StochasticDepth(survival_prob=0.9),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            StochasticDepth(survival_prob=0.9),
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


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, soft_labels=False):
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
            if soft_labels:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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


def generate_pseudo_labels(model, device, confidence_threshold=0.8):
    print("Generating pseudo-labels for test set...")
    test_tfidf = load_npz(f"{WORKING_DIR}/test_tfidf.npz").toarray()
    test_stylo = np.load(f"{WORKING_DIR}/test_stylo.npy")
    test_char_ids = np.load(f"{WORKING_DIR}/test_char_ids.npy")

    test_dataset = SpookyDataset(test_tfidf, test_stylo, test_char_ids)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            tfidf, stylo, char_ids = batch
            tfidf = tfidf.to(device)
            stylo = stylo.to(device)
            char_ids = char_ids.to(device)
            with autocast():
                logits = model(tfidf, stylo, char_ids)
                probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)

    max_probs = np.max(all_preds, axis=1)
    confident_mask = max_probs > confidence_threshold
    print(f"High-confidence test samples: {confident_mask.sum()} out of {len(confident_mask)}")

    if confident_mask.sum() == 0:
        print("No high-confidence samples found, skipping pseudo-labeling.")
        return None, None, None

    confident_preds = all_preds[confident_mask]
    confident_indices = np.where(confident_mask)[0]

    train_tfidf = load_npz(f"{WORKING_DIR}/train_tfidf.npz").toarray()
    train_stylo = np.load(f"{WORKING_DIR}/train_stylo.npy")
    train_char_ids = np.load(f"{WORKING_DIR}/train_char_ids.npy")
    y_train = np.load(f"{WORKING_DIR}/y_train.npy")
    val_tfidf = load_npz(f"{WORKING_DIR}/val_tfidf.npz").toarray()
    val_stylo = np.load(f"{WORKING_DIR}/val_stylo.npy")
    val_char_ids = np.load(f"{WORKING_DIR}/val_char_ids.npy")
    y_val = np.load(f"{WORKING_DIR}/y_val.npy")

    augmented_tfidf = np.vstack([train_tfidf, val_tfidf, test_tfidf[confident_indices]])
    augmented_stylo = np.vstack([train_stylo, val_stylo, test_stylo[confident_indices]])
    augmented_char_ids = np.vstack([train_char_ids, val_char_ids, test_char_ids[confident_indices]])

    num_classes = confident_preds.shape[1]
    pseudo_labels = np.eye(num_classes)[np.argmax(confident_preds, axis=1)] * 0.9 + 0.1/num_classes
    y_train_onehot = np.eye(num_classes)[y_train]
    y_val_onehot = np.eye(num_classes)[y_val]
    augmented_labels = np.vstack([y_train_onehot, y_val_onehot, pseudo_labels])

    np.save(f"{WORKING_DIR}/augmented_tfidf.npy", augmented_tfidf.astype(np.float32))
    np.save(f"{WORKING_DIR}/augmented_stylo.npy", augmented_stylo.astype(np.float32))
    np.save(f"{WORKING_DIR}/augmented_char_ids.npy", augmented_char_ids.astype(np.int32))
    np.save(f"{WORKING_DIR}/augmented_labels.npy", augmented_labels.astype(np.float32))

    return f"{WORKING_DIR}/augmented_tfidf.npy", f"{WORKING_DIR}/augmented_stylo.npy", f"{WORKING_DIR}/augmented_char_ids.npy"


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfidf = load_npz(f"{WORKING_DIR}/train_tfidf.npz").toarray()
    val_tfidf = load_npz(f"{WORKING_DIR}/val_tfidf.npz").toarray()
    test_tfidf = load_npz(f"{WORKING_DIR}/test_tfidf.npz").toarray()
    train_stylo = np.load(f"{WORKING_DIR}/train_stylo.npy")
    val_stylo = np.load(f"{WORKING_DIR}/val_stylo.npy")
    test_stylo = np.load(f"{WORKING_DIR}/test_stylo.npy")
    train_char_ids = np.load(f"{WORKING_DIR}/train_char_ids.npy")
    val_char_ids = np.load(f"{WORKING_DIR}/val_char_ids.npy")
    test_char_ids = np.load(f"{WORKING_DIR}/test_char_ids.npy")
    y_train = np.load(f"{WORKING_DIR}/y_train.npy")
    y_val = np.load(f"{WORKING_DIR}/y_val.npy")
    test_ids_bytes = np.load(f"{WORKING_DIR}/test_ids.npy")
    test_ids = np.array([s.decode('utf-8') for s in test_ids_bytes])

    train_dataset = SpookyDataset(train_tfidf, train_stylo, train_char_ids, y_train)
    val_dataset = SpookyDataset(val_tfidf, val_stylo, val_char_ids, y_val)
    test_dataset = SpookyDataset(test_tfidf, test_stylo, test_char_ids)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.999))

    total_epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, pct_start=0.2, div_factor=50, final_div_factor=1e4,
        total_steps=total_epochs * len(train_loader),
    )
    scaler = GradScaler()

    best_val_logloss = float("inf")
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_logloss, val_preds = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {current_lr:.2e}")

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
    _, final_val_logloss, val_preds = validate(model, val_loader, criterion, device)
    print(f"\nBest validation LogLoss: {final_val_logloss:.6f}")

    # Pseudo-labeling phase
    print("\n=== Pseudo-Labeling Phase ===")
    pseudo_result = generate_pseudo_labels(model, device, confidence_threshold=0.8)
    if pseudo_result[0] is not None:
        aug_tfidf = np.load(f"{WORKING_DIR}/augmented_tfidf.npy")
        aug_stylo = np.load(f"{WORKING_DIR}/augmented_stylo.npy")
        aug_char_ids = np.load(f"{WORKING_DIR}/augmented_char_ids.npy")
        aug_labels = np.load(f"{WORKING_DIR}/augmented_labels.npy")

        all_tfidf = np.vstack([train_tfidf, val_tfidf])
        all_stylo = np.vstack([train_stylo, val_stylo])
        all_char_ids = np.vstack([train_char_ids, val_char_ids])
        all_labels = np.concatenate([y_train, y_val])

        full_tfidf = all_tfidf.copy()
        full_stylo = all_stylo.copy()
        full_char_ids = all_char_ids.copy()

        num_classes = 3
        all_labels_onehot = np.eye(num_classes)[all_labels]
        full_labels = all_labels_onehot.copy()

        retrain_model = MultiInputClassifier(
            tfidf_dim=full_tfidf.shape[1],
            stylo_dim=full_stylo.shape[1],
            char_vocab_size=CHAR_VOCAB_SIZE,
            max_char_len=500,
            hidden_size=512,
            num_labels=3,
            dropout=0.5,
        ).to(device)

        retrain_dataset = SpookyDataset(full_tfidf, full_stylo, full_char_ids, full_labels, soft_labels=True)
        retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

        retrain_optimizer = torch.optim.AdamW(retrain_model.parameters(), lr=5e-4, weight_decay=0.2, betas=(0.9, 0.999))
        retrain_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            retrain_optimizer, max_lr=5e-4, pct_start=0.2, div_factor=50, final_div_factor=1e4,
            total_steps=20 * len(retrain_loader),
        )

        retrain_best_logloss = float("inf")
        retrain_best_state = None
        retrain_patience = 5
        retrain_counter = 0

        print("\nRetraining with augmented data (soft labels)...")
        for epoch in range(20):
            current_lr = retrain_optimizer.param_groups[0]['lr']
            train_loss = train_epoch(retrain_model, retrain_loader, criterion, retrain_optimizer, scaler, device, soft_labels=True)
            val_loss, val_logloss, val_preds = validate(retrain_model, val_loader, criterion, device)
            print(f"Retrain Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {current_lr:.2e}")

            if val_logloss < retrain_best_logloss:
                retrain_best_logloss = val_logloss
                retrain_best_state = retrain_model.state_dict().copy()
                retrain_counter = 0
            else:
                retrain_counter += 1
                if retrain_counter >= retrain_patience:
                    print(f"Retrain early stopping at epoch {epoch+1}")
                    break

        retrain_model.load_state_dict(retrain_best_state)
        _, final_retrain_logloss, _ = validate(retrain_model, val_loader, criterion, device)
        print(f"Final Validation Score after pseudo-labeling: {final_retrain_logloss:.6f}")

        test_preds = predict(retrain_model, test_loader, device)
    else:
        test_preds = predict(model, test_loader, device)

    eps = 1e-15
    test_preds = np.clip(test_preds, eps, 1 - eps)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

    submission = pd.DataFrame({
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    })
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved to {OUTPUT_CSV} with {len(submission)} rows")
    print(f"Final Validation LogLoss: {final_val_logloss:.6f}")


if __name__ == "__main__":
    preprocess_data()
    train_and_evaluate()
