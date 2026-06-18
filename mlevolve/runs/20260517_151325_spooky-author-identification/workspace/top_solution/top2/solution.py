"""
Merged: Spooky Author Identification
DeBERTa-v3-large fine-tuning + Handcrafted Features + XGBoost + Logistic Regression + Ensemble
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
author_map = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print(f"Label encoding: {author_map}")

# ============================================================
# STRATIFIED SPLIT (NO INDEX_BUG) - Keep only raw text, no handcrafted features
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]

assert len(set(train_idx) & set(val_idx)) == 0, "INDEX BUG detected!"
print(
    f"Training: {len(X_train_texts)}, Validation: {len(X_val_texts)}, Test: {len(test_df)}"
)

# ============================================================
# HELPER FUNCTION: Syllable Counter
# ============================================================
def _count_syllables(word):
    word = word.lower().strip()
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if count == 0:
        count = 1
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

# ============================================================
# HANDCRAFTED FEATURE EXTRACTION FUNCTIONS
# ============================================================

def extract_stylometric_features(texts):
    """Extract 30 stylometric features per text."""
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0] * 30)
            continue
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            features.append([0] * 30)
            continue
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        sent_count = max(len(sentences), 1)
        char_count = len(text)
        avg_word_len = char_count / max(word_count, 1)
        avg_sent_len = word_count / sent_count

        upper_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        lower_ratio = sum(1 for c in text if c.islower()) / max(char_count, 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
        whitespace_ratio = sum(1 for c in text if c.isspace()) / max(char_count, 1)

        punct_chars = string.punctuation
        total_punct = sum(1 for c in text if c in punct_chars)
        pct_period = text.count(".") / max(total_punct, 1)
        pct_comma = text.count(",") / max(total_punct, 1)
        pct_exclaim = text.count("!") / max(total_punct, 1)
        pct_question = text.count("?") / max(total_punct, 1)
        pct_semi = text.count(";") / max(total_punct, 1)
        pct_colon = text.count(":") / max(total_punct, 1)
        pct_dash = text.count("-") / max(total_punct, 1)
        pct_quote = text.count('"') / max(total_punct, 1)
        pct_apost = text.count("'") / max(total_punct, 1)
        pct_paren = (text.count("(") + text.count(")")) / max(total_punct, 1)
        pct_backtick = text.count("`") / max(total_punct, 1)
        pct_other = max(
            0,
            1
            - (
                pct_period
                + pct_comma
                + pct_exclaim
                + pct_question
                + pct_semi
                + pct_colon
                + pct_dash
                + pct_quote
                + pct_apost
                + pct_paren
                + pct_backtick
            ),
        )

        char_diversity = len(set(text.lower())) / max(char_count, 1)
        long_words = sum(1 for w in words if len(w) >= 7) / max(word_count, 1)
        capitalized = sum(1 for w in words if len(w) > 0 and w[0].isupper()) / max(
            word_count, 1
        )
        all_caps = sum(1 for w in words if len(w) > 1 and w.isupper()) / max(
            word_count, 1
        )

        sent_lens = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lens) if len(sent_lens) > 0 else 0
        sent_len_var = np.var(sent_lens) if len(sent_lens) > 0 else 0

        function_words = {
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
            "that",
            "which",
            "who",
            "whom",
            "this",
            "these",
            "those",
            "it",
            "its",
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
            "can",
            "could",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "not",
            "no",
            "nor",
            "all",
            "each",
            "every",
            "both",
            "some",
            "any",
            "many",
            "much",
            "few",
            "little",
            "more",
            "most",
            "other",
            "such",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
        }
        function_word_ratio = sum(
            1 for w in words if w.lower() in function_words
        ) / max(word_count, 1)

        archaic_words = {
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "dost",
            "art",
            "wilt",
            "shalt",
            "canst",
            "didst",
            "hast",
            "whence",
            "thence",
            "whither",
            "thither",
            "wherefore",
            "therefor",
            "perchance",
            "ere",
            "unto",
            "doth",
            "hath",
        }
        emotional_words = {
            "fear",
            "terrible",
            "horrible",
            "dreadful",
            "awful",
            "hideous",
            "monstrous",
            "gruesome",
            "ghastly",
            "frightful",
            "shocking",
            "horrifying",
            "terrifying",
            "frightening",
            "alarming",
            "disturbing",
            "creepy",
            "eerie",
            "sinister",
            "ominous",
            "menacing",
            "threatening",
            "forbidding",
            "beautiful",
            "wonderful",
            "splendid",
            "magnificent",
            "glorious",
            "delightful",
            "pleasant",
            "charming",
            "lovely",
            "excellent",
            "superb",
            "joyful",
            "happy",
            "peaceful",
            "serene",
            "calm",
            "tranquil",
            "despair",
            "wretched",
            "miserable",
            "gloomy",
            "dreary",
            "sombre",
            "melancholy",
            "woeful",
            "sorrowful",
            "anguish",
            "torment",
            "agony",
        }
        lovecraft_words = {
            "eldritch",
            "cyclopean",
            "blasphemous",
            "unspeakable",
            "antediluvian",
            "nameless",
            "unutterable",
            "nonhuman",
            "cosmic",
            "gibbering",
            "daemoniac",
            "loathsome",
            "noisome",
            "fevered",
            "unwholesome",
            "preternatural",
            "squamous",
            "ichor",
            "gargoyle",
            "miasmal",
            "arkham",
            "insmouth",
            "rlyeh",
            "cthulhu",
            "dunwich",
        }
        sub_conj_words = {
            "although",
            "because",
            "since",
            "unless",
            "while",
            "whereas",
            "if",
            "when",
            "whenever",
            "wherever",
            "after",
            "before",
            "though",
            "even",
        }

        archaic_ratio = sum(1 for w in words if w.lower() in archaic_words) / max(
            word_count, 1
        )
        emotional_ratio = sum(1 for w in words if w.lower() in emotional_words) / max(
            word_count, 1
        )
        lovecraft_ratio = sum(1 for w in words if w.lower() in lovecraft_words) / max(
            word_count, 1
        )
        sub_conj_ratio = sum(1 for w in words if w.lower() in sub_conj_words) / max(
            word_count, 1
        )

        feats = [
            char_count,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            pct_period,
            pct_comma,
            pct_exclaim,
            pct_question,
            pct_semi,
            pct_colon,
            pct_dash,
            pct_quote,
            pct_apost,
            pct_paren,
            pct_backtick,
            pct_other,
            char_diversity,
            long_words,
            capitalized,
            all_caps,
            sent_len_std,
            sent_len_var,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            sub_conj_ratio,
        ]
        features.append(feats[:30])
    return np.array(features, dtype=np.float64)

def create_readability_features(texts):
    """Extract 4 readability features per text."""
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0, 0, 0, 0])
            continue
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            features.append([0, 0, 0, 0])
            continue
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        sent_count = max(len(sentences), 1)
        syllables = sum(_count_syllables(w) for w in words)
        char_count = len(text.replace(" ", ""))
        flesch = (
            206.835
            - 1.015 * (word_count / sent_count)
            - 84.6 * (syllables / max(word_count, 1))
        )
        flesch = max(0, min(100, flesch))
        ari = (
            4.71 * (char_count / max(word_count, 1))
            + 0.5 * (word_count / sent_count)
            - 21.43
        )
        ari = max(0, ari)
        avg_syllables = syllables / max(word_count, 1)
        complex_words = sum(1 for w in words if _count_syllables(w) >= 3)
        complex_ratio = complex_words / max(word_count, 1)
        features.append([flesch, ari, avg_syllables, complex_ratio])
    return np.array(features, dtype=np.float64)

def create_pos_tag_approximation(texts):
    """Extract 5 POS approximation features per text."""
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ship",
        "dom",
        "hood",
    ]
    verb_suffixes = ["ing", "ed", "en", "ize", "ify", "ate"]
    adj_suffixes = ["ous", "ive", "ful", "less", "able", "ible", "al", "ic", "ical"]
    adv_suffixes = ["ly", "ward", "wise"]
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        words = text.split()
        word_count = max(len(words), 1)
        noun_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in noun_suffixes)
        )
        verb_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in verb_suffixes)
        )
        adj_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in adj_suffixes)
        )
        adv_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in adv_suffixes)
        )
        content_words = noun_count + verb_count + adj_count + adv_count
        content_word_ratio = content_words / word_count
        features.append(
            [
                noun_count / word_count,
                verb_count / word_count,
                adj_count / word_count,
                adv_count / word_count,
                content_word_ratio,
            ]
        )
    return np.array(features, dtype=np.float64)

# ============================================================
# EXTRACT HANDCRAFTED FEATURES
# ============================================================
print("Extracting handcrafted features...")

train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(test_df["text"].values)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)

train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

print(
    f"Stylometric: {train_stylo_filtered.shape[1]}, Readability: {train_read_scaled.shape[1]}, POS: {train_pos_scaled.shape[1]}"
)

# ============================================================
# N-GRAM FEATURES
# ============================================================
print("Extracting n-gram features...")

char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vectorizer_short.fit_transform(X_train_texts)
val_char_short = char_vectorizer_short.transform(X_val_texts)
test_char_short = char_vectorizer_short.transform(test_df["text"].values)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_train_texts)
val_char_med = char_vectorizer_med.transform(X_val_texts)
test_char_med = char_vectorizer_med.transform(test_df["text"].values)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_train_texts)
val_char_long = char_vectorizer_long.transform(X_val_texts)
test_char_long = char_vectorizer_long.transform(test_df["text"].values)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vectorizer.fit_transform(X_train_texts)
val_word = word_vectorizer.transform(X_val_texts)
test_word = word_vectorizer.transform(test_df["text"].values)

def extract_punctuation_sequence(text):
    return "".join([c for c in str(text) if c in string.punctuation]) if text else ""

all_texts_for_punct = np.concatenate(
    [X_train_texts, X_val_texts, test_df["text"].values]
)
punct_sequences = [extract_punctuation_sequence(t) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(X_train_texts)
n_val = len(X_val_texts)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + n_val]
test_punct = punct_features_all[n_train + n_val :]

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Sparse train shape: {train_sparse.shape}")

# ============================================================
# COMPUTE REGRESSION TARGETS (handcrafted features for multi-task learning)
# ============================================================
print("Computing regression targets for multi-task learning...")

# Extract stylometric features for regression targets
regression_stylo = extract_stylometric_features(X_train_texts)
regression_read = create_readability_features(X_train_texts)
regression_pos = create_pos_tag_approximation(X_train_texts)

# Normalize targets to zero mean, unit variance
regression_targets_full = np.hstack([regression_stylo[:, :25], regression_read, regression_pos])
stylo_mean = regression_targets_full.mean(axis=0)
stylo_std = regression_targets_full.std(axis=0) + 1e-8
regression_targets_normalized = (regression_targets_full - stylo_mean) / stylo_std

regression_dim = regression_targets_normalized.shape[1]
print(f"Regression target dimension: {regression_dim}")

# ============================================================
# DEBERTA FINE-TUNING WITH MULTI-TASK LEARNING
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE WITH MULTI-TASK LEARNING")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom model with multi-task learning heads
class DebertaMultiTask(nn.Module):
    def __init__(self, model_name, num_labels, reg_dim, dropout=0.1):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        hidden_size = self.backbone.config.hidden_size
        # Classification head (reuse the existing one from backbone)
        self.classifier = self.backbone.classifier
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, reg_dim),
        )

    def forward(self, input_ids, attention_mask, labels=None, regression_targets=None, alpha=0.1):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        cls_embedding = hidden_states[:, 0, :]

        logits = self.classifier(cls_embedding)
        reg_output = self.regressor(cls_embedding)

        loss = None
        if labels is not None:
            loss_cls = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
            if regression_targets is not None:
                loss_reg = nn.MSELoss()(reg_output, regression_targets)
                loss = loss_cls + alpha * loss_reg
            else:
                loss = loss_cls

        return type('Output', (), {
            'logits': logits,
            'loss': loss,
            'regression_output': reg_output
        })()

model = DebertaMultiTask(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    reg_dim=regression_dim,
    dropout=DROPOUT
)
model.to(device)

train_encodings = tokenizer(
    list(X_train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    list(X_val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_encodings = tokenizer(
    list(test_df["text"].values),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

# Include regression targets in training dataset
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(y_train_labels, dtype=torch.long),
    torch.tensor(regression_targets_normalized, dtype=torch.float32),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(y_val_labels, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
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

total_steps = len(train_loader) * NUM_EPOCHS
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler() if torch.cuda.is_available() else None

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

def evaluate_deberta(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

# Track top-3 checkpoints for temporal ensemble
top_checkpoints = []  # list of (val_logloss, epoch, filepath)
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

# Alpha scheduler: linear decay from 0.1 to 0 over NUM_EPOCHS
def get_alpha(epoch, num_epochs, initial_alpha=0.1):
    return max(0.0, initial_alpha * (1 - epoch / num_epochs))

for epoch in range(NUM_EPOCHS):
    alpha = get_alpha(epoch, NUM_EPOCHS, 0.1)
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        regression_targets = batch[3].to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                regression_targets=regression_targets,
                alpha=alpha,
            )
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches
    # Compute validation log loss for model selection
    _, val_acc, val_probs = evaluate_deberta(model, val_loader)
    val_logloss = compute_log_loss(y_val_labels, val_probs)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Alpha: {alpha:.4f} | Train Loss: {avg_train_loss:.4f} | Val Log Loss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )
    # Save checkpoint for this epoch
    checkpoint_path = f"{WORKING_DIR}/deberta_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    # Track top-3 checkpoints
    top_checkpoints.append((val_logloss, epoch + 1, checkpoint_path))
    top_checkpoints.sort(key=lambda x: x[0])  # sort by log loss ascending
    if len(top_checkpoints) > 3:
        # Remove the worst checkpoint file to save space
        worst_path = top_checkpoints.pop(-1)[2]
        if os.path.exists(worst_path):
            os.remove(worst_path)

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        # Also save the single best checkpoint for compatibility
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_deberta_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val log loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest DeBERTa model: epoch {best_epoch}, val log loss: {best_val_loss:.4f}")
print(f"Top-3 checkpoints for temporal ensemble:")
for ll, ep, path in top_checkpoints:
    print(f"  Epoch {ep}: val log loss = {ll:.4f}")

# Use temporal ensemble of top-3 checkpoints
def get_temporal_ensemble_probs_and_embeddings(model, top_checkpoints, embeddings_loader, probs_loader):
    """Average probabilities and embeddings from top-K checkpoints with inverse log loss weighting."""
    all_probs = []
    all_embeddings = []
    weights = []

    for ll, ep, ckpt_path in top_checkpoints:
        # Reload checkpoint
        state_dict = torch.load(ckpt_path, map_location=device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        model.to(device)
        model.eval()

        # Extract probabilities
        batch_probs = []
        with torch.no_grad():
            for batch in probs_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                batch_probs.append(probs.cpu().numpy())
        all_probs.append(np.vstack(batch_probs))

        # Extract embeddings from backbone's DeBERTa
        batch_embs = []
        with torch.no_grad():
            for batch in embeddings_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                with autocast():
                    # Access backbone directly for hidden states
                    outputs = model.backbone.deberta(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.last_hidden_state
                    cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                batch_embs.append(cls_embeddings)
        all_embeddings.append(np.vstack(batch_embs))

        # Weight by inverse log loss (lower log loss = higher weight)
        weight = 1.0 / (ll + 1e-10)
        weights.append(weight)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Weighted average
    avg_probs = np.zeros_like(all_probs[0])
    avg_embeddings = np.zeros_like(all_embeddings[0])
    for w, p, e in zip(weights, all_probs, all_embeddings):
        avg_probs += w * p
        avg_embeddings += w * e

    return avg_probs, avg_embeddings

# Create data loaders for ensemble extraction
val_probs_loader = DataLoader(
    TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(y_val_labels, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_probs_loader = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Load best model for compatibility (will be overwritten by temporal ensemble if top-3 available)
if len(top_checkpoints) > 1:
    print("\nUsing temporal ensemble of top-3 checkpoints for inference...")
    # Create no-label loaders for ensemble extraction
    train_loader_no_labels = DataLoader(
        TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    val_loader_no_labels = DataLoader(
        TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader_no_labels = DataLoader(
        TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    # Get ensemble probabilities and embeddings for validation and test
    deberta_val_probs, val_emb_ensemble = get_temporal_ensemble_probs_and_embeddings(
        model, top_checkpoints, val_loader_no_labels, val_probs_loader
    )
    deberta_test_probs, test_emb_ensemble = get_temporal_ensemble_probs_and_embeddings(
        model, top_checkpoints, test_loader_no_labels, test_probs_loader
    )
    # Also get train embeddings from best model (not averaged, to save compute)
    state_dict = torch.load(top_checkpoints[0][2], map_location=device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
else:
    print("\nOnly one checkpoint available, using single best model...")
    state_dict = torch.load(f"{WORKING_DIR}/best_deberta_model.pt", map_location=device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)

print(f"DeBERTa temporal ensemble validation log loss: {compute_log_loss(y_val_labels, deberta_val_probs):.4f}")

# ============================================================
# EXTRACT DEBERTA EMBEDDINGS (from temporal ensemble)
# ============================================================
print("\nUsing DeBERTa embeddings from temporal ensemble...")

# The embeddings were already extracted in get_temporal_ensemble_probs_and_embeddings
# Use the ensemble embeddings directly
train_embeddings = np.zeros((len(X_train_texts), 1024))  # placeholder, will be filled below
val_embeddings = val_emb_ensemble
test_embeddings = test_emb_ensemble

# Extract train embeddings from best checkpoint (not from ensemble to save compute)
print("Extracting train DeBERTa embeddings from best checkpoint...")
train_loader_no_labels_train = DataLoader(
    TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
model.load_state_dict(torch.load(top_checkpoints[0][2], map_location=device))
model.to(device)
model.eval()
train_batch_embs = []
with torch.no_grad():
    for batch in train_loader_no_labels_train:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            # Access backbone directly for hidden states
            outputs = model.backbone.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.last_hidden_state
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
        train_batch_embs.append(cls_embeddings)
train_embeddings = np.vstack(train_batch_embs)
print(f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")

# ============================================================
# XGBOOST
# ============================================================
print("\nTraining XGBoost classifier...")
xgb_train_features = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled, train_embeddings]
)
xgb_val_features = np.hstack(
    [val_stylo_filtered, val_read_scaled, val_pos_scaled, val_embeddings]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled, test_embeddings]
)
print(f"XGBoost train features: {xgb_train_features.shape}")

# Step 1: Train initial XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    xgb_train_features,
    y_train_labels,
    eval_set=[(xgb_val_features, y_val_labels)],
    verbose=False,
)

xgb_val_probs_initial = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs_initial = xgb_model.predict_proba(xgb_test_features)
print(
    f"Initial XGBoost validation log loss: {compute_log_loss(y_val_labels, xgb_val_probs_initial):.4f}"
)

# Use initial XGBoost model (no pseudolabeling to avoid data leakage)
print("Using initial XGBoost model (no pseudolabeling).")
xgb_val_probs = xgb_val_probs_initial
xgb_test_probs = xgb_test_probs_initial

print(
    f"XGBoost validation log loss: {compute_log_loss(y_val_labels, xgb_val_probs):.4f}"
)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
lr_model.fit(train_sparse, y_train_labels)

lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)
print(
    f"Logistic Regression validation log loss: {compute_log_loss(y_val_labels, lr_val_probs):.4f}"
)

# ============================================================
# DEBERTA VALIDATION & TEST PROBS (using temporal ensemble already computed above)
# ============================================================
# Note: deberta_val_probs and deberta_test_probs were already computed from temporal ensemble
print(f"DeBERTa validation log loss (temporal ensemble): {compute_log_loss(y_val_labels, deberta_val_probs):.4f}")

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "deberta": deberta_val_probs,
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
}

best_ll = float("inf")
best_weights = None
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = (
            w1 * val_probas["deberta"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(y_val_labels, ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"deberta": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

score = best_ll

# ============================================================
# GENERATE SUBMISSION
# ============================================================
test_probas = {
    "deberta": deberta_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
ensemble_test_probs = (
    best_weights["deberta"] * test_probas["deberta"]
    + best_weights["xgboost"] * test_probas["xgboost"]
    + best_weights["lr"] * test_probas["lr"]
)

eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

author_cols = label_encoder.classes_.tolist()
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        author_cols[0]: ensemble_test_probs[:, 0],
        author_cols[1]: ensemble_test_probs[:, 1],
        author_cols[2]: ensemble_test_probs[:, 2],
    }
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"\nFinal Validation Score: {score:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()