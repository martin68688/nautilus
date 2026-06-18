"""
Spooky Author Identification - Merged Solution
DeBERTa-v3-large + XGBoost + Logistic Regression + Weighted Ensemble
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
# PATH CONFIGURATION
# ============================================================
DATA_DIR = "./input"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
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
author_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print(f"Label encoding: {author_mapping}")

# ============================================================
# STRATIFIED SPLIT (NO INDEX_BUG)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
assert len(set(train_idx) & set(val_idx)) == 0, "Train/val indices overlap!"

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]
test_texts = test_df["text"].values

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_texts)}"
)
print(f"Train class dist: {np.bincount(y_train_labels)}")
print(f"Val class dist: {np.bincount(y_val_labels)}")

# ============================================================
# HANDCRAFTED FEATURE FUNCTIONS
# ============================================================

# Word lists
ARCHAIC_WORDS = set(
    [
        "thee",
        "thou",
        "thy",
        "thine",
        "thyself",
        "ye",
        "hath",
        "doth",
        "dost",
        "doth",
        "art",
        "wert",
        "shalt",
        "wilt",
        "canst",
        "didst",
        "hadst",
        "hast",
        "mayst",
        "mightst",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "betwixt",
        "unto",
        "nay",
        "yea",
        "oft",
        "ofttimes",
        "ne'er",
        "o'er",
        "e'er",
        "tis",
        "twas",
        "twill",
        "twere",
        "thou'rt",
        "thou'lt",
        "thou'dst",
        "prithee",
        "forsooth",
        "perchance",
        "methinks",
        "alas",
        "anon",
        "wherefore",
        "henceforth",
        "thereupon",
        "whereupon",
        "herewith",
        "therewith",
        "wherein",
        "whereto",
        "wherefrom",
        "hitherto",
        "thenceforth",
        "thenceforward",
    ]
)

EMOTIONAL_WORDS = set(
    [
        "horror",
        "terror",
        "dread",
        "fear",
        "afraid",
        "scared",
        "frightened",
        "terrified",
        "dismay",
        "anguish",
        "agony",
        "torment",
        "suffering",
        "pain",
        "despair",
        "hopeless",
        "gloom",
        "shadow",
        "darkness",
        "night",
        "death",
        "dead",
        "corpse",
        "ghost",
        "spirit",
        "phantom",
        "spectre",
        "apparition",
        "demon",
        "devil",
        "monster",
        "creature",
        "beast",
        "madness",
        "insanity",
        "crazed",
        "lunatic",
        "frantic",
        "wild",
        "savage",
        "furious",
        "rage",
        "wrath",
        "fury",
        "hatred",
        "loathing",
        "disgust",
        "revulsion",
        "abhorrence",
        "weep",
        "weeping",
        "tears",
        "sob",
        "sobbing",
        "lament",
        "mourn",
        "grieve",
        "grief",
        "sorrow",
        "woe",
        "misery",
        "sadness",
        "melancholy",
        "desolate",
        "forsaken",
        "lonely",
        "solemn",
        "grim",
        "macabre",
        "grotesque",
        "hideous",
        "dreadful",
        "awful",
        "terrible",
        "fearful",
        "frightful",
        "horrible",
        "horrid",
        "shocking",
        "appalling",
        "ghastly",
        "cursed",
        "haunted",
        "unholy",
        "wicked",
        "evil",
        "sinister",
        "ominous",
        "portentous",
    ]
)

LOVECRAFT_WORDS = set(
    [
        "cyclopean",
        "eldritch",
        "cryptic",
        "ancient",
        "alien",
        "void",
        "abyss",
        "cosmic",
        "nameless",
        "unspeakable",
        "unutterable",
        "indescribable",
        "inconceivable",
        "unimaginable",
        "ineffable",
        "blasphemous",
        "coven",
        "miasmatic",
        "squamous",
        "rugose",
        "ichor",
        "gibbous",
        "noisome",
        "cacodaemonic",
        "carcass",
        "charnel",
        "catacomb",
        "sepulchre",
        "necropolis",
        "primordial",
        "antediluvian",
        "prehuman",
        "nonhuman",
        "accursed",
        "maddening",
        "tenebrous",
        "ululate",
        "amorphous",
        "ichor",
        "fungoid",
        "euphoric",
        "batrachian",
        "nightmare",
        "hideous",
        "cataleptic",
        "cerement",
        "labyrinthine",
        "immemorial",
        "vendigo",
        "shoggoth",
        "cthulhu",
        "r'lyeh",
        "yogg",
        "nyarlathotep",
        "azathoth",
        "yuggoth",
        "kadath",
    ]
)

FUNCTION_WORDS = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "because",
        "as",
        "while",
        "when",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "by",
        "with",
        "without",
        "from",
        "upon",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
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
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "themselves",
    ]
)

SUB_CONJ_WORDS = set(
    [
        "after",
        "although",
        "as",
        "because",
        "before",
        "if",
        "once",
        "since",
        "than",
        "that",
        "though",
        "till",
        "unless",
        "until",
        "when",
        "whenever",
        "where",
        "wherever",
        "whether",
        "while",
        "why",
        "how",
        "even",
        "provided",
        "supposing",
        "whereas",
        "whereupon",
        "lest",
        "ere",
        "albeit",
    ]
)


def count_syllables(word):
    word = word.lower()
    vowel_runs = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and vowel_runs > 1:
        vowel_runs -= 1
    return max(1, vowel_runs)


def flesch_reading_ease(text, total_words, total_sentences, total_syllables):
    if total_words == 0 or total_sentences == 0:
        return 0.0
    return (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )


def automated_readability_index(text, total_chars, total_words, total_sentences):
    if total_words == 0 or total_sentences == 0:
        return 0.0
    return (
        4.71 * (total_chars / total_words)
        + 0.5 * (total_words / total_sentences)
        - 21.43
    )


def extract_stylometric_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 30)
            continue
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_chars = len(text_str)
        n_words = len(words)
        n_sentences = len(sentences)
        avg_word_len = n_chars / max(1, n_words)
        avg_sent_len = n_words / max(1, n_sentences)
        n_upper = sum(1 for c in text_str if c.isupper())
        n_lower = sum(1 for c in text_str if c.islower())
        n_digits = sum(1 for c in text_str if c.isdigit())
        n_whitespace = sum(1 for c in text_str if c.isspace())
        upper_ratio = n_upper / max(1, n_chars)
        lower_ratio = n_lower / max(1, n_chars)
        digit_ratio = n_digits / max(1, n_chars)
        whitespace_ratio = n_whitespace / max(1, n_chars)
        punct_counts = []
        for p in ".,;:!?\"'-()[]{}":
            punct_counts.append(text_str.count(p) / max(1, n_chars))
        char_diversity = len(set(text_str.lower())) / max(1, n_chars)
        long_words = sum(1 for w in words if len(w) >= 7) / max(1, n_words)
        capitalized = sum(1 for w in words if w[0].isupper()) / max(1, n_words)
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1) / max(1, n_words)
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0.0
        words_lower = [w.lower().strip(string.punctuation) for w in words]
        words_lower = [w for w in words_lower if len(w) > 0]
        function_word_count = sum(1 for w in words_lower if w in FUNCTION_WORDS)
        function_word_ratio = function_word_count / max(1, len(words_lower))
        archaic_count = sum(1 for w in words_lower if w in ARCHAIC_WORDS)
        archaic_ratio = archaic_count / max(1, len(words_lower))
        emotional_count = sum(1 for w in words_lower if w in EMOTIONAL_WORDS)
        emotional_ratio = emotional_count / max(1, len(words_lower))
        lovecraft_count = sum(1 for w in words_lower if w in LOVECRAFT_WORDS)
        lovecraft_ratio = lovecraft_count / max(1, len(words_lower))
        sub_conj_count = sum(1 for w in words_lower if w in SUB_CONJ_WORDS)
        sub_conj_ratio = sub_conj_count / max(1, len(words_lower))
        feat = (
            [
                n_chars,
                n_words,
                n_sentences,
                avg_word_len,
                avg_sent_len,
                upper_ratio,
                lower_ratio,
                digit_ratio,
                whitespace_ratio,
            ]
            + punct_counts[:12]
            + [
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
        )
        if len(feat) < 30:
            feat.extend([0.0] * (30 - len(feat)))
        feat = feat[:30]
        features.append(feat)
    return np.array(features, dtype=np.float64)


def create_readability_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0, 0.0, 0.0, 0.0])
            continue
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_chars = len(text_str)
        n_words = len(words)
        n_sentences = len(sentences)
        total_syllables = sum(count_syllables(w) for w in words)
        fre = flesch_reading_ease(text_str, n_words, n_sentences, total_syllables)
        ari = automated_readability_index(text_str, n_chars, n_words, n_sentences)
        avg_syllables = total_syllables / max(1, n_words)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / max(1, n_words)
        features.append([fre, ari, avg_syllables, complex_ratio])
    return np.array(features, dtype=np.float64)


def create_pos_tag_approximation(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        text_str = str(text)
        words = text_str.split()
        words_lower = [w.lower().strip(string.punctuation) for w in words]
        words_lower = [w for w in words_lower if len(w) > 0]
        n_words = len(words_lower)
        if n_words == 0:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        noun_like = sum(
            1
            for w in words_lower
            if re.search(
                r"(tion|sion|ment|ness|ity|ance|ence|ism|ship|ist|age|dom)$", w
            )
        )
        verb_like = sum(
            1
            for w in words_lower
            if re.search(r"(ed|ing|ate|ize|ify|en|ise|fy)$", w) and len(w) > 3
        )
        adj_like = sum(
            1
            for w in words_lower
            if re.search(
                r"(ous|al|ful|less|able|ible|ic|ive|ent|ant|ish|like|some)$", w
            )
        )
        adv_like = sum(
            1 for w in words_lower if re.search(r"(ly|ward|wise)$", w) and len(w) > 2
        )
        noun_ratio = noun_like / n_words
        verb_ratio = verb_like / n_words
        adj_ratio = adj_like / n_words
        adv_ratio = adv_like / n_words
        content_words = sum(
            1 for w in words_lower if len(w) >= 4 and w not in FUNCTION_WORDS
        )
        content_ratio = content_words / n_words
        features.append([noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_ratio])
    return np.array(features, dtype=np.float64)


def extract_punctuation_sequence(text):
    if not isinstance(text, str):
        return ""
    return "".join([c for c in text if c in string.punctuation])


# ============================================================
# APPLY FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Stylometric features
print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(test_texts)
print(f"  Stylometric features shape: train {train_stylo.shape}")

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)
print(f"  Stylometric features after variance filter: {train_stylo_filtered.shape[1]}")

# Readability features
print("Extracting readability features...")
train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_texts)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

# POS approximation features
print("Extracting POS approximation features...")
train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_texts)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# N-gram features
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
test_char_short = char_vectorizer_short.transform(test_texts)

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
test_char_med = char_vectorizer_med.transform(test_texts)

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
test_char_long = char_vectorizer_long.transform(test_texts)

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
test_word = word_vectorizer.transform(test_texts)

# Punctuation sequence features
print("Extracting punctuation sequence features...")
train_punct_seqs = [extract_punctuation_sequence(t) for t in X_train_texts]
val_punct_seqs = [extract_punctuation_sequence(t) for t in X_val_texts]
test_punct_seqs = [extract_punctuation_sequence(t) for t in test_texts]

all_punct_seqs = train_punct_seqs + val_punct_seqs + test_punct_seqs
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_vectorizer.fit(all_punct_seqs)
train_punct = punct_vectorizer.transform(train_punct_seqs)
val_punct = punct_vectorizer.transform(val_punct_seqs)
test_punct = punct_vectorizer.transform(test_punct_seqs)

# Combine sparse features
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
    list(test_texts),
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
