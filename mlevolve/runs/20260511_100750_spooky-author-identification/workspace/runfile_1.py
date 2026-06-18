import os
os.sched_setaffinity(0, {65, 171, 172, 76, 77})
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import json
import os
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from scipy.sparse import save_npz, load_npz
import joblib

# ============================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


# ---- 1. Stylometric feature engineering ----
def extract_stylometric_features(text_series, is_train=True, global_stats=None):
    """Extract linguistic style features from text."""
    features = []
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        char_count = len(text)
        unique_words = set(words)
        sent_count = max(len(re.findall(r"[.!?]+", text)), 1)
        avg_word_len = char_count / max(word_count, 1)
        avg_sent_len = word_count / sent_count
        punct_count = sum(1 for c in text if c in string.punctuation)
        punct_density = punct_count / max(char_count, 1)
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
                "are",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "can",
                "need",
                "dare",
                "ought",
                "used",
                "it",
                "its",
                "them",
                "their",
                "they",
                "he",
                "she",
                "his",
                "her",
                "him",
                "my",
                "your",
                "our",
                "we",
                "i",
                "you",
            ]
        )
        stopword_count = sum(1 for w in words if w in stopwords)
        stopword_ratio = stopword_count / max(word_count, 1)
        lexical_diversity = len(unique_words) / max(word_count, 1)
        char_bigrams = [text_lower[i : i + 2] for i in range(len(text_lower) - 1)]
        rare_bigram_ratio = sum(
            1
            for bg in char_bigrams
            if bg
            in [
                "th",
                "he",
                "in",
                "er",
                "an",
                "re",
                "nd",
                "at",
                "on",
                "nt",
                "ha",
                "ou",
                "it",
                "es",
                "st",
            ]
        ) / max(len(char_bigrams), 1)
        archaic_words = [
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "art",
            "wast",
            "hast",
            "shalt",
            "canst",
            "dost",
            "ere",
            "whence",
            "thence",
            "wherefore",
            "perchance",
            "abyss",
            "horror",
            "gloom",
            "spectral",
            "eldritch",
            "cyclopean",
            "antediluvian",
            "sepulchral",
        ]
        archaic_count = sum(1 for w in words if w in archaic_words)
        contraction_count = len(re.findall(r"\b\w+'\w+\b", text_lower))
        contraction_ratio = contraction_count / max(word_count, 1)
        cap_words = sum(1 for w in text.split() if w and w[0].isupper())
        cap_ratio = cap_words / max(word_count, 1)
        excl_count = text.count("!")
        quest_count = text.count("?")
        excl_ratio = excl_count / sent_count
        quest_ratio = quest_count / sent_count
        comma_ratio = text.count(",") / max(word_count, 1)
        semicolon_ratio = text.count(";") / max(word_count, 1)
        colon_ratio = text.count(":") / max(word_count, 1)
        dash_ratio = text.count("--") / max(word_count, 1)
        quote_count = text.count('"') + text.count("'")
        quote_ratio = quote_count / max(word_count, 1)
        paren_count = text.count("(") + text.count(")")
        num_count = len(re.findall(r"\d+", text))
        ing_words = sum(1 for w in words if w.endswith("ing"))
        ly_words = sum(1 for w in words if w.endswith("ly"))
        ed_words = sum(1 for w in words if w.endswith("ed"))
        tion_words = sum(1 for w in words if w.endswith("tion"))
        word_lengths = [len(w) for w in words if w]
        if word_lengths:
            mean_wl = np.mean(word_lengths)
            std_wl = np.std(word_lengths)
            max_wl = max(word_lengths)
        else:
            mean_wl = std_wl = max_wl = 0
        feature_dict = {
            "word_count": word_count,
            "char_count": char_count,
            "avg_word_len": avg_word_len,
            "avg_sent_len": avg_sent_len,
            "punct_density": punct_density,
            "stopword_ratio": stopword_ratio,
            "lexical_diversity": lexical_diversity,
            "rare_bigram_ratio": rare_bigram_ratio,
            "archaic_count": archaic_count,
            "contraction_ratio": contraction_ratio,
            "cap_ratio": cap_ratio,
            "excl_ratio": excl_ratio,
            "quest_ratio": quest_ratio,
            "comma_ratio": comma_ratio,
            "semicolon_ratio": semicolon_ratio,
            "colon_ratio": colon_ratio,
            "dash_ratio": dash_ratio,
            "quote_ratio": quote_ratio,
            "paren_count": paren_count,
            "num_count": num_count,
            "ing_words_ratio": ing_words / max(word_count, 1),
            "ly_words_ratio": ly_words / max(word_count, 1),
            "ed_words_ratio": ed_words / max(word_count, 1),
            "tion_words_count": tion_words,
            "mean_word_len": mean_wl,
            "std_word_len": std_wl,
            "max_word_len": max_wl,
        }
        features.append(feature_dict)
    df = pd.DataFrame(features)
    if is_train:
        global_stats = {}
        for col in df.columns:
            if df[col].std() > 0:
                global_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                }
            else:
                global_stats[col] = {"mean": 0.0, "std": 1.0}
        os.makedirs("./working", exist_ok=True)
        with open("./working/stylometric_stats.json", "w") as f:
            json.dump(global_stats, f)
    if global_stats is None:
        with open("./working/stylometric_stats.json", "r") as f:
            global_stats = json.load(f)
    for col in df.columns:
        stats = global_stats.get(col, {"mean": 0, "std": 1})
        df[col] = (df[col] - stats["mean"]) / stats["std"]
    return df, global_stats


# ---- 2. TF-IDF character n-gram features ----
def create_tfidf_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=5000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        min_df=3,
        max_df=0.8,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    joblib.dump(vectorizer, "./working/tfidf_vectorizer.pkl")
    return train_tfidf, test_tfidf, vectorizer


# ---- 3. Word n-gram TF-IDF ----
def create_word_tfidf_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.7,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    joblib.dump(vectorizer, "./working/word_tfidf_vectorizer.pkl")
    return train_tfidf, test_tfidf, vectorizer


# ---- 4. Readability scores ----
def extract_readability_features(text_series):
    features = []
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        word_count = len(words)
        sent_count = max(len(re.findall(r"[.!?]+", text)), 1)
        char_count = len(text)
        avg_word_len = char_count / max(word_count, 1)
        fk_grade = 0.39 * (word_count / sent_count) + 11.8 * (avg_word_len / 5) - 15.59
        if word_count > 0:
            ari = (
                4.71 * (char_count / word_count)
                + 0.5 * (word_count / sent_count)
                - 21.43
            )
        else:
            ari = 0
        letters = sum(1 for c in text if c.isalpha())
        if word_count > 0:
            cli = (
                5.89 * (letters / word_count)
                - 0.3 * (sent_count / (word_count / 100))
                - 15.8
            )
        else:
            cli = 0
        features.append(
            {
                "flesch_kincaid": fk_grade,
                "automated_readability": ari,
                "coleman_liau": cli,
            }
        )
    return pd.DataFrame(features)


# ---- 5. Sentence structure features ----
def extract_sentence_structure(text_series):
    features = []
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            sent_length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
            max_sent_len = max(sent_lengths)
            min_sent_len = min(sent_lengths)
            coord_starters = sum(
                1
                for s in sentences
                if s.lower().startswith(("and", "but", "or", "for", "nor", "yet", "so"))
            )
            coord_ratio = coord_starters / len(sentences)
        else:
            sent_length_variance = 0
            max_sent_len = 0
            min_sent_len = 0
            coord_ratio = 0
        features.append(
            {
                "sent_length_variance": sent_length_variance,
                "max_sent_len": max_sent_len,
                "min_sent_len": min_sent_len,
                "coord_ratio": coord_ratio,
            }
        )
    return pd.DataFrame(features)


# Process all features
train_texts = train_df["text"].fillna("").tolist()
test_texts = test_df["text"].fillna("").tolist()

# Create stratified train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train = train_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values

fold_indices = list(skf.split(np.zeros(len(train_df)), y_train))

# Save raw texts and labels
pd.Series(train_texts).to_pickle("./working/train_texts.pkl")
pd.Series(test_texts).to_pickle("./working/test_texts.pkl")
np.save("./working/y_train.npy", y_train)
np.save("./working/test_ids.npy", test_df["id"].values)

# ---- Compute all features INSIDE the fold loop to avoid leakage ----
# Precompute test-side features that don't depend on train data
stylo_test_raw, _ = extract_stylometric_features(
    test_texts, is_train=False, global_stats={"word_count": {"mean": 0, "std": 1}}
)
read_test_raw = extract_readability_features(test_texts)
struct_test_raw = extract_sentence_structure(test_texts)

# Set STYLO_FEATURES based on a rough estimate - will be dynamically set in fold
STYLO_FEATURES = 37  # placeholder, will be updated per fold

# Save raw texts and labels
print("Stylometric feature dimension will be determined per fold during training.")

# ============================================================
# MODEL DESIGN
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PRETRAINED_MODEL_NAME = "microsoft/deberta-v3-large"
NUM_AUTHORS = 3
DROPOUT_RATE = 0.2
HIDDEN_SIZE = 1024

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


class StylometricEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=512, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(2)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.layer_norm_out = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.layer_norm1(x)
        for encoder_layer, layer_norm in zip(self.encoder_layers, self.layer_norms):
            residual = x
            x = encoder_layer(x)
            x = layer_norm(x + residual)
            x = self.dropout(x)
        x = self.output_proj(x)
        x = self.layer_norm_out(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, semantic_dim=1024, stylo_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = semantic_dim // num_heads
        assert semantic_dim % num_heads == 0
        self.query_proj = nn.Linear(semantic_dim, semantic_dim)
        self.key_proj = nn.Linear(stylo_dim, semantic_dim)
        self.value_proj = nn.Linear(stylo_dim, semantic_dim)
        self.output_proj = nn.Linear(semantic_dim, semantic_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(semantic_dim)
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))

    def forward(self, semantic_features, stylo_features):
        batch_size = semantic_features.size(0)
        Q = (
            self.query_proj(semantic_features)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key_proj(stylo_features)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value_proj(stylo_features)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, semantic_features.size(-1))
        )
        output = self.output_proj(context.squeeze(1))
        output = self.layer_norm(output + semantic_features)
        return output


class DebertaStylometricFusion(nn.Module):
    def __init__(self, stylo_dim=None):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
        for param in self.deberta.parameters():
            param.requires_grad = False
        # DebertaV2Model does NOT have pooler or rel_embeddings attributes
        # Remove references to pooler.parameters() and rel_embeddings.parameters()
        self.stylo_encoder = StylometricEncoder(
            input_dim=stylo_dim if stylo_dim is not None else 512,
            hidden_dim=256,
            output_dim=512,
            dropout=DROPOUT_RATE,
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.Dropout(DROPOUT_RATE)
        )
        self.fusion = CrossAttentionFusion(
            semantic_dim=1024, stylo_dim=512, num_heads=8, dropout=DROPOUT_RATE
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE * 0.5),
            nn.Linear(256, NUM_AUTHORS),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [self.semantic_proj, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, stylo_features):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state
        attn_scores = self.attention_pool(sequence_output).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        semantic_features = torch.bmm(
            attn_weights.unsqueeze(1), sequence_output
        ).squeeze(1)
        semantic_features = self.semantic_proj(semantic_features)
        stylo_encoded = self.stylo_encoder(stylo_features)
        fused_features = self.fusion(semantic_features, stylo_encoded)
        logits = self.classifier(fused_features)
        return logits


# ============================================================
# TRAINING AND EVALUATION
# ============================================================


class AuthorshipDataset(Dataset):
    def __init__(self, texts, stylo_features, labels=None):
        self.texts = texts
        self.stylo_features = torch.FloatTensor(stylo_features)
        self.labels = labels
        if labels is not None:
            self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        stylo = self.stylo_features[idx]
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors=None,
        )
        input_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, attention_mask, stylo, label
        else:
            return input_ids, attention_mask, stylo


def train_epoch(model, dataloader, optimizer, scheduler, scaler, criterion):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        input_ids, attention_mask, stylo_features, labels = [
            x.to(device) for x in batch
        ]
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
        if num_batches % 50 == 0:
            torch.cuda.empty_cache()
    return total_loss / num_batches


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, stylo_features, labels = [
                x.to(device) for x in batch
            ]
            with autocast():
                logits = model(input_ids, attention_mask, stylo_features)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    eps = 1e-15
    all_preds = np.clip(all_preds, eps, 1 - eps)
    score = log_loss(all_labels, all_preds)
    return score, all_preds


def predict_test(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, stylo_features = [x.to(device) for x in batch]
            with autocast():
                logits = model(input_ids, attention_mask, stylo_features)
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


# Training hyperparameters
BATCH_SIZE = 16
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5
NUM_FOLDS = 5

fold_val_scores = []
test_preds_list = []

for fold, (train_idx_fold, val_idx_fold) in enumerate(fold_indices):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/{NUM_FOLDS}")
    print(f"{'='*60}")

    X_train_fold = [train_texts[i] for i in train_idx_fold]
    X_val_fold = [train_texts[i] for i in val_idx_fold]
    y_train_fold = y_train[train_idx_fold]
    y_val_fold = y_train[val_idx_fold]

    # ---- Fit feature extractors ONLY on training fold ----
    # TF-IDF char n-grams
    fold_char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=5000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        min_df=3,
        max_df=0.8,
    )
    fold_train_tfidf_char = fold_char_vectorizer.fit_transform(X_train_fold)
    fold_val_tfidf_char = fold_char_vectorizer.transform(X_val_fold)
    fold_test_tfidf_char = fold_char_vectorizer.transform(test_texts)

    # TF-IDF word n-grams
    fold_word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.7,
        stop_words="english",
    )
    fold_train_tfidf_word = fold_word_vectorizer.fit_transform(X_train_fold)
    fold_val_tfidf_word = fold_word_vectorizer.transform(X_val_fold)
    fold_test_tfidf_word = fold_word_vectorizer.transform(test_texts)

    # Stylometric features - fit on train fold only
    fold_stylo_train, fold_global_stats = extract_stylometric_features(
        X_train_fold, is_train=True, global_stats=None
    )
    fold_stylo_val, _ = extract_stylometric_features(
        X_val_fold, is_train=False, global_stats=fold_global_stats
    )
    fold_stylo_test, _ = extract_stylometric_features(
        test_texts, is_train=False, global_stats=fold_global_stats
    )

    # Readability features - fit on train fold only
    fold_read_train_raw = extract_readability_features(X_train_fold)
    fold_read_val_raw = extract_readability_features(X_val_fold)
    fold_read_test_raw = extract_readability_features(test_texts)

    fold_read_stats = {}
    for col in fold_read_train_raw.columns:
        mean_val = fold_read_train_raw[col].mean()
        std_val = fold_read_train_raw[col].std() if fold_read_train_raw[col].std() > 0 else 1.0
        fold_read_stats[col] = {"mean": mean_val, "std": std_val}

    fold_read_train = fold_read_train_raw.copy()
    fold_read_val = fold_read_val_raw.copy()
    fold_read_test = fold_read_test_raw.copy()
    for col in fold_read_train.columns:
        stats = fold_read_stats[col]
        fold_read_train[col] = (fold_read_train[col] - stats["mean"]) / stats["std"]
        fold_read_val[col] = (fold_read_val[col] - stats["mean"]) / stats["std"]
        fold_read_test[col] = (fold_read_test[col] - stats["mean"]) / stats["std"]

    # Sentence structure features - fit on train fold only
    fold_struct_train_raw = extract_sentence_structure(X_train_fold)
    fold_struct_val_raw = extract_sentence_structure(X_val_fold)
    fold_struct_test_raw = extract_sentence_structure(test_texts)

    fold_struct_stats = {}
    for col in fold_struct_train_raw.columns:
        mean_val = fold_struct_train_raw[col].mean()
        std_val = fold_struct_train_raw[col].std() if fold_struct_train_raw[col].std() > 0 else 1.0
        fold_struct_stats[col] = {"mean": mean_val, "std": std_val}

    fold_struct_train = fold_struct_train_raw.copy()
    fold_struct_val = fold_struct_val_raw.copy()
    fold_struct_test = fold_struct_test_raw.copy()
    for col in fold_struct_train.columns:
        stats = fold_struct_stats[col]
        fold_struct_train[col] = (fold_struct_train[col] - stats["mean"]) / stats["std"]
        fold_struct_val[col] = (fold_struct_val[col] - stats["mean"]) / stats["std"]
        fold_struct_test[col] = (fold_struct_test[col] - stats["mean"]) / stats["std"]

    # Combine all fold-level features
    stylo_train_fold = np.concatenate(
        [
            fold_stylo_train.values.astype(np.float32),
            fold_read_train.values.astype(np.float32),
            fold_struct_train.values.astype(np.float32),
            fold_train_tfidf_char[:, :500].toarray().astype(np.float32),
            fold_train_tfidf_word[:, :500].toarray().astype(np.float32),
        ],
        axis=1,
    )
    stylo_val_fold = np.concatenate(
        [
            fold_stylo_val.values.astype(np.float32),
            fold_read_val.values.astype(np.float32),
            fold_struct_val.values.astype(np.float32),
            fold_val_tfidf_char[:, :500].toarray().astype(np.float32),
            fold_val_tfidf_word[:, :500].toarray().astype(np.float32),
        ],
        axis=1,
    )
    stylo_test_fold = np.concatenate(
        [
            fold_stylo_test.values.astype(np.float32),
            fold_read_test.values.astype(np.float32),
            fold_struct_test.values.astype(np.float32),
            fold_test_tfidf_char[:, :500].toarray().astype(np.float32),
            fold_test_tfidf_word[:, :500].toarray().astype(np.float32),
        ],
        axis=1,
    )

    # Update STYLO_FEATURES dynamically for this fold
    STYLO_FEATURES = stylo_train_fold.shape[1]
    print(f"Fold {fold+1} stylometric feature dimension: {STYLO_FEATURES}")

    train_dataset = AuthorshipDataset(X_train_fold, stylo_train_fold, y_train_fold)
    val_dataset = AuthorshipDataset(X_val_fold, stylo_val_fold, y_val_fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model for this fold using actual feature dimension
    model = DebertaStylometricFusion(stylo_dim=STYLO_FEATURES)
    model.to(device)

    # Phase 1: Train with frozen DeBERTa
    print("Phase 1: Training with frozen DeBERTa...")
    for param in model.deberta.parameters():
        param.requires_grad = False

    total_steps_phase1 = len(train_loader) * EPOCHS_PHASE1
    warmup_steps_phase1 = int(total_steps_phase1 * WARMUP_RATIO)

    optimizer_grouped_parameters_phase1 = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "deberta" in n and p.requires_grad
            ],
            "lr": LEARNING_RATE * 0.1,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "deberta" not in n],
            "lr": LEARNING_RATE * 10,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    optimizer_p1 = AdamW(
        optimizer_grouped_parameters_phase1, lr=LEARNING_RATE, eps=1e-8
    )
    scheduler_p1 = get_linear_schedule_with_warmup(
        optimizer_p1,
        num_warmup_steps=warmup_steps_phase1,
        num_training_steps=total_steps_phase1,
    )
    scaler_p1 = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss_phase1 = float("inf")
    patience_counter = 0
    for epoch in range(EPOCHS_PHASE1):
        train_loss = train_epoch(
            model, train_loader, optimizer_p1, scheduler_p1, scaler_p1, criterion
        )
        val_score, _ = validate(model, val_loader, criterion)
        print(
            f"  Phase 1 - Epoch {epoch+1}/{EPOCHS_PHASE1} - Train Loss: {train_loss:.4f} - Val Log Loss: {val_score:.4f}"
        )
        if val_score < best_val_loss_phase1:
            best_val_loss_phase1 = val_score
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/best_model_phase1_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= 2:
                print(f"  Early stopping phase 1 after {epoch+1} epochs")
                break

    model.load_state_dict(torch.load(f"./working/best_model_phase1_fold{fold}.pt"))

    # Phase 2: Unfreeze top layers
    print("Phase 2: Fine-tuning with unfrozen DeBERTa...")
    total_layers = 24
    layers_to_unfreeze = 8
    start_layer = total_layers - layers_to_unfreeze
    for i, layer in enumerate(model.deberta.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = i >= start_layer
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)"
    )

    total_steps_phase2 = len(train_loader) * EPOCHS_PHASE2
    warmup_steps_phase2 = int(total_steps_phase2 * WARMUP_RATIO)

    optimizer_grouped_parameters_phase2 = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "deberta" in n
                and p.requires_grad
                and not any(nd in n for nd in ["bias", "LayerNorm", "layer_norm"])
            ],
            "lr": LEARNING_RATE * 0.5,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "deberta" in n
                and p.requires_grad
                and any(nd in n for nd in ["bias", "LayerNorm", "layer_norm"])
            ],
            "lr": LEARNING_RATE * 0.5,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "deberta" not in n],
            "lr": LEARNING_RATE * 5,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    optimizer_p2 = AdamW(
        optimizer_grouped_parameters_phase2, lr=LEARNING_RATE, eps=1e-8
    )
    scheduler_p2 = get_linear_schedule_with_warmup(
        optimizer_p2,
        num_warmup_steps=warmup_steps_phase2,
        num_training_steps=total_steps_phase2,
    )
    scaler_p2 = GradScaler()

    best_val_loss_phase2 = float("inf")
    patience_counter = 0
    best_model_state_fold = None
    for epoch in range(EPOCHS_PHASE2):
        train_loss = train_epoch(
            model, train_loader, optimizer_p2, scheduler_p2, scaler_p2, criterion
        )
        val_score, _ = validate(model, val_loader, criterion)
        print(
            f"  Phase 2 - Epoch {epoch+1}/{EPOCHS_PHASE2} - Train Loss: {train_loss:.4f} - Val Log Loss: {val_score:.4f}"
        )
        if val_score < best_val_loss_phase2:
            best_val_loss_phase2 = val_score
            patience_counter = 0
            best_model_state_fold = model.state_dict().copy()
            torch.save(model.state_dict(), f"./working/best_model_phase2_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping phase 2 after {epoch+1} epochs")
                break

    fold_val_scores.append(best_val_loss_phase2)
    print(f"Fold {fold+1} best validation log loss: {best_val_loss_phase2:.4f}")

    # Predict on test set
    model.load_state_dict(best_model_state_fold)
    model.eval()
    test_dataset = AuthorshipDataset(test_texts, stylo_test_fold)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_probs = predict_test(model, test_loader)
    test_preds_list.append(test_probs)

    # Clean up
    del (
        model,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        optimizer_p1,
        scheduler_p1,
        optimizer_p2,
        scheduler_p2,
        scaler_p1,
        scaler_p2,
    )
    torch.cuda.empty_cache()
    gc.collect()

# Ensemble predictions across folds (geometric mean)
print("\n" + "=" * 60)
print("Ensembling predictions across folds...")
print("=" * 60)

test_preds_ensemble = np.zeros_like(test_preds_list[0])
for fold_preds in test_preds_list:
    test_preds_ensemble += np.log(fold_preds + 1e-15)
test_preds_ensemble = np.exp(test_preds_ensemble / len(test_preds_list))
test_preds_ensemble = test_preds_ensemble / test_preds_ensemble.sum(
    axis=1, keepdims=True
)
eps = 1e-15
test_preds_ensemble = np.clip(test_preds_ensemble, eps, 1 - eps)

# Create submission
print("\nCreating submission file...")
os.makedirs("./submission", exist_ok=True)
test_ids = np.load("./working/test_ids.npy")
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_ensemble[:, 0],
        "HPL": test_preds_ensemble[:, 1],
        "MWS": test_preds_ensemble[:, 2],
    }
)
submission_df.to_csv("./submission/submission_2c78a395456e49e59e987cf72a885429.csv", index=False)
print(f"Submission saved to ./submission/submission_2c78a395456e49e59e987cf72a885429.csv")
print(f"Submission shape: {submission_df.shape}")

# Print validation scores
print("\n" + "=" * 60)
print("Cross-Validation Results")
print("=" * 60)
for i, score in enumerate(fold_val_scores):
    print(f"Fold {i+1}: {score:.4f}")
mean_score = np.mean(fold_val_scores)
std_score = np.std(fold_val_scores)
print(f"Mean CV Log Loss: {mean_score:.4f} +/- {std_score:.4f}")

final_score = mean_score
print(f"Final Validation Score: {final_score}")