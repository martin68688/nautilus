"""
Run 20260509_185008 Train+Inference Script
LogLoss: ~0.2013 (真实 log_loss, 无 INDEX_BUG)
模型: DeBERTa-v3-large fine-tuning + XGBoost + Logistic Regression + 集成

用法: python infer_0509_185008_0201.py
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
# 路径配置
# ============================================================
INFERENCE_DIR = "./input"
DATA_DIR = "./input"
TRAIN_CSV = "./input/train.csv"
TEST_CSV = "./input/test.csv"
OUTPUT_CSV = "./submission/submission.csv"
WORKING_DIR = "./working"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

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

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT
# ============================================================
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)

# ============================================================
# HANDCRAFTED FEATURES
# ============================================================


def extract_stylometric_features(texts):
    function_words = {
        "the",
        "and",
        "a",
        "an",
        "in",
        "of",
        "to",
        "is",
        "was",
        "it",
        "that",
        "for",
        "on",
        "with",
        "as",
        "by",
        "at",
        "from",
        "be",
        "have",
        "are",
        "were",
        "had",
        "not",
        "but",
        "what",
        "all",
        "this",
        "or",
        "which",
        "they",
        "he",
        "she",
        "its",
        "my",
        "we",
        "our",
        "no",
        "so",
        "if",
        "about",
        "up",
        "out",
        "do",
        "does",
        "did",
        "has",
        "been",
        "being",
        "him",
        "his",
        "her",
        "their",
        "them",
        "these",
        "those",
        "some",
        "any",
        "each",
        "every",
    }
    archaic_words = {
        "thee",
        "thou",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "dost",
        "didst",
        "wert",
        "art",
        "hast",
        "shalt",
        "wilt",
        "canst",
        "couldst",
        "wouldst",
        "shouldst",
        "whence",
        "thence",
        "hither",
        "thither",
        "wherefore",
        "perchance",
        "ere",
        "whilst",
        "alas",
        "betwixt",
        "amongst",
        "dwell",
        "behold",
        "forsooth",
        "methinks",
        "prithee",
        "anon",
        "hark",
        "lo",
        "twas",
        "tis",
        "oft",
        "ne'er",
        "o'er",
        "e'er",
        "seldom",
        "yonder",
    }
    emotional_words = {
        "fear",
        "terror",
        "horror",
        "dread",
        "anguish",
        "agony",
        "despair",
        "gloom",
        "shadow",
        "darkness",
        "ghost",
        "spectre",
        "phantom",
        "mystery",
        "secret",
        "strange",
        "weird",
        "awful",
        "dreadful",
        "frightful",
        "terrible",
        "horrible",
        "sorrow",
        "grief",
        "woe",
        "mournful",
        "lament",
        "desolate",
        "solemn",
        "melancholy",
        "somber",
        "dreary",
        "bleak",
        "sad",
        "grim",
        "eerie",
        "uncanny",
        "supernatural",
        "ominous",
        "sinister",
        "macabre",
        "grotesque",
        "hideous",
        "loathsome",
        "repulsive",
        "disgust",
        "repugnant",
    }
    lovecraft_words = {
        "cyclopean",
        "eldritch",
        "cryptic",
        "antiquarian",
        "blasphemous",
        "non-euclidean",
        "cosmic",
        "primordial",
        "unknown",
        "nameless",
        "unnameable",
        "miasmal",
        "fungoid",
        "squamous",
        "rugose",
        "ichorous",
        "gibbous",
        "faint",
        "indescribable",
        "inconceivable",
        "unfathomable",
        "immeasurable",
        "abyss",
        "void",
        "chaos",
        "ancient",
        "forbidden",
        "accursed",
        "cursed",
        "mortal",
        "immortal",
        "entity",
        "being",
        "creature",
        "monstrosity",
        "abomination",
        "cadaverous",
        "ghastly",
        "hideous",
        "noisome",
        "noxious",
        "pestilential",
        "putrid",
        "rank",
        "stale",
        "tainted",
        "vile",
        "coven",
        "cult",
        "charnel",
        "sepulchral",
        "tomb",
        "vault",
        "crypt",
        "arkham",
        "insmouth",
        "dunwich",
        "yuggoth",
        "cthulhu",
        "yog-sothoth",
        "necronomicon",
        "r'lyeh",
        "kadath",
        "leng",
        "hyperborean",
    }
    sub_conjunctions = {
        "although",
        "because",
        "since",
        "unless",
        "while",
        "whereas",
        "if",
        "though",
        "even",
        "once",
        "until",
        "after",
        "before",
        "when",
        "whenever",
        "wherever",
        "whereas",
        "whereupon",
        "while",
    }
    features = []
    for text in texts:
        text_str = str(text).lower()
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_chars = len(text_str)
        n_words = len(words)
        n_sents = len(sentences)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        n_upper = sum(1 for c in text if c.isupper())
        n_lower = sum(1 for c in text if c.islower())
        n_digits = sum(1 for c in text if c.isdigit())
        n_whitespace = sum(1 for c in text if c.isspace())
        upper_ratio = n_upper / n_chars if n_chars > 0 else 0
        lower_ratio = n_lower / n_chars if n_chars > 0 else 0
        digit_ratio = n_digits / n_chars if n_chars > 0 else 0
        whitespace_ratio = n_whitespace / n_chars if n_chars > 0 else 0
        punct_ratios = []
        for punct in string.punctuation:
            count = text_str.count(punct)
            punct_ratios.append(count / n_chars if n_chars > 0 else 0)
        char_diversity = len(set(text_str)) / n_chars if n_chars > 0 else 0
        long_words_ratio = (
            sum(1 for w in words if len(w) >= 7) / n_words if n_words > 0 else 0
        )
        capitalized_words = sum(1 for w in words if w and w[0].isupper())
        capitalized_ratio = capitalized_words / n_words if n_words > 0 else 0
        all_caps_words = sum(1 for w in words if w.isupper())
        all_caps_ratio = all_caps_words / n_words if n_words > 0 else 0
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if sent_lengths else 0
        sent_len_var = np.var(sent_lengths) if sent_lengths else 0
        function_word_count = sum(1 for w in words if w in function_words)
        function_word_ratio = function_word_count / n_words if n_words > 0 else 0
        archaic_count = sum(1 for w in words if w in archaic_words)
        archaic_ratio = archaic_count / n_words if n_words > 0 else 0
        emotional_count = sum(1 for w in words if w in emotional_words)
        emotional_ratio = emotional_count / n_words if n_words > 0 else 0
        lovecraft_count = sum(1 for w in words if w in lovecraft_words)
        lovecraft_ratio = lovecraft_count / n_words if n_words > 0 else 0
        sub_conj_count = sum(1 for w in words if w in sub_conjunctions)
        sub_conj_ratio = sub_conj_count / n_words if n_words > 0 else 0
        feature_vector = [
            n_chars,
            n_words,
            n_sents,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            *punct_ratios,
            char_diversity,
            long_words_ratio,
            capitalized_ratio,
            all_caps_ratio,
            sent_len_std,
            sent_len_var,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            sub_conj_ratio,
        ]
        features.append(feature_vector)
    return np.array(features)


def create_readability_features(texts):
    vowels = "aeiouy"

    def count_syllables(word):
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    features = []
    for text in texts:
        text_str = str(text)
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text_str.split()
        n_words = len(words)
        n_sents = len(sentences)
        n_syllables = sum(count_syllables(w) for w in words) if words else 0
        if n_words > 0 and n_sents > 0:
            avg_syllables_per_word = n_syllables / n_words
            avg_words_per_sent = n_words / n_sents
            flesch = (
                206.835 - 1.015 * avg_words_per_sent - 84.6 * avg_syllables_per_word
            )
        else:
            flesch = 0
            avg_syllables_per_word = 0
        n_chars = sum(len(w) for w in words) if words else 0
        if n_words > 0 and n_sents > 0:
            ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43
        else:
            ari = 0
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / n_words if n_words > 0 else 0
        features.append([flesch, ari, avg_syllables_per_word, complex_ratio])
    return np.array(features)


def create_pos_tag_approximation(texts):
    function_words_set = {
        "the",
        "and",
        "a",
        "an",
        "in",
        "of",
        "to",
        "is",
        "was",
        "it",
        "that",
        "for",
        "on",
        "with",
        "as",
        "by",
        "at",
        "from",
        "be",
        "have",
        "are",
        "were",
        "had",
        "not",
        "but",
        "what",
        "all",
        "this",
        "or",
        "which",
        "they",
        "he",
        "she",
        "its",
        "my",
        "we",
        "our",
        "no",
        "so",
        "if",
        "about",
        "up",
        "out",
        "do",
        "does",
        "did",
        "has",
        "been",
        "being",
        "him",
        "his",
        "her",
        "their",
        "them",
        "these",
        "those",
        "some",
        "any",
        "each",
        "every",
    }
    noun_suffixes = (
        "tion",
        "ment",
        "ness",
        "ity",
        "ence",
        "ance",
        "ist",
        "ism",
        "ship",
        "dom",
    )
    verb_suffixes = ("ed", "ing", "ate", "ize", "ise", "ify", "en", "ish")
    adj_suffixes = (
        "ous",
        "ful",
        "less",
        "able",
        "ible",
        "al",
        "ive",
        "ic",
        "ical",
        "ish",
    )
    adv_suffixes = ("ly", "ward", "wise")
    features = []
    for text in texts:
        words = str(text).lower().split()
        n_words = len(words)
        if n_words == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        noun_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in noun_suffixes)
        )
        verb_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in verb_suffixes)
        )
        adj_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in adj_suffixes)
        )
        adv_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in adv_suffixes)
        )
        content_count = sum(1 for w in words if w not in function_words_set)
        noun_ratio = noun_count / n_words
        verb_ratio = verb_count / n_words
        adj_ratio = adj_count / n_words
        adv_ratio = adv_count / n_words
        content_ratio = content_count / n_words
        features.append([noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_ratio])
    return np.array(features)


# ============================================================
# HANDCRAFTED FEATURES (continued)
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

# ============================================================
# CHARACTER & WORD N-GRAM + PUNCTUATION FEATURES
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
    return "".join([c for c in text if c in string.punctuation]) if text else ""


all_texts_for_punct = np.concatenate(
    [X_train_texts, X_val_texts, test_df["text"].values]
)
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
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
# DEBERTA FINE-TUNING
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
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

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(y_train_labels, dtype=torch.long),
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

# ============================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            focal_loss = alpha_t.gather(0, targets) * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Class weights for focal loss alpha (balanced)
class_counts = np.bincount(y_train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
alpha_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

criterion = FocalLoss(gamma=2.0, alpha=alpha_weights)
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


best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, _ = evaluate_deberta(model, val_loader)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_deberta_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest DeBERTa model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
model.load_state_dict(
    torch.load(f"{WORKING_DIR}/best_deberta_model.pt", map_location=device)
)

# ============================================================
# EXTRACT DEBERTA EMBEDDINGS
# ============================================================
print("\nExtracting DeBERTa embeddings...")


def extract_embeddings(model, loader):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)


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

train_embeddings = extract_embeddings(model, train_loader_no_labels)
val_embeddings = extract_embeddings(model, val_loader_no_labels)
test_embeddings = extract_embeddings(model, test_loader_no_labels)
print(
    f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}"
)

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

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
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
# DEBERTA VALIDATION & TEST PROBS
# ============================================================
print("\nGetting DeBERTa probabilities...")
val_loader_eval = DataLoader(
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
_, deberta_val_loss, deberta_val_probs = evaluate_deberta(model, val_loader_eval)
print(f"DeBERTa validation log loss: {deberta_val_loss:.4f}")

test_loader_eval = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
model.eval()
all_test_probs = []
with torch.no_grad():
    for batch in test_loader_eval:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())
deberta_test_probs = np.vstack(all_test_probs)

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

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSubmission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())
print(f"\nFinal Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
