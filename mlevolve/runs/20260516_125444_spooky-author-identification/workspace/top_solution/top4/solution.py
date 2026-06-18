"""
Merged Script: Spooky Author Identification
Ensemble: DeBERTa-v3-large + XGBoost + Logistic Regression
"""

import pandas as pd
import numpy as np
import os
import re
import string
import warnings
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & CONFIGURATION
# ============================================================
TRAIN_CSV = "./input/train.csv"
TEST_CSV = "./input/test.csv"
OUTPUT_CSV = "./submission/submission.csv"
WORKING_DIR = "./working"

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

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# WORD LISTS FOR FEATURE ENGINEERING
# ============================================================
ARCHAIC_WORDS = set(
    [
        "thou",
        "thee",
        "thy",
        "thine",
        "doth",
        "hath",
        "art",
        "wert",
        "shalt",
        "wilt",
        "canst",
        "didst",
        "hast",
        "dost",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "o'er",
        "ne'er",
        "e'er",
        "oft",
        "ofttimes",
        "betwixt",
        "behold",
        "beseech",
        "childe",
        "doth",
        "erewhile",
        "fain",
        "forsooth",
        "hark",
        "hie",
        "hitherto",
        "laden",
        "morrow",
        "nigh",
        "nought",
        "perchance",
        "prithee",
        "sayeth",
        "spake",
        "tarry",
        "thence",
        "theretofore",
        "unto",
        "wherefore",
        "whereof",
        "wherein",
        "wherewith",
        "wot",
        "ye",
        "yon",
        "yonder",
        "methinks",
        "methought",
        "alas",
        "anon",
        "ay",
        "betimes",
        "certes",
        "durst",
        "erewhile",
        "even",
        "evermore",
        "fain",
        "farewell",
        "gainsay",
        "hapless",
        "lorn",
        "mayhap",
        "mischance",
        "perforce",
        "quoth",
        "sooth",
        "thrice",
        "twain",
        "twas",
        "twill",
        "twixt",
        "wight",
    ]
)

POSITIVE_EMOTION_WORDS = set(
    [
        "joy",
        "happy",
        "delight",
        "pleasure",
        "beautiful",
        "wonderful",
        "love",
        "hope",
        "peace",
        "gentle",
        "sweet",
        "calm",
        "bright",
        "glad",
        "cheerful",
        "blessed",
        "bliss",
        "ecstasy",
        "rapture",
        "serene",
        "tranquil",
        "radiant",
        "splendid",
        "glorious",
        "magnificent",
        "exquisite",
        "enchant",
        "charm",
        "merry",
        "jolly",
        "festive",
        "exult",
        "rejoice",
        "triumph",
        "euphoria",
    ]
)

NEGATIVE_EMOTION_WORDS = set(
    [
        "fear",
        "terror",
        "horror",
        "dread",
        "anguish",
        "agony",
        "sorrow",
        "grief",
        "despair",
        "sad",
        "dark",
        "shadow",
        "gloom",
        "mourn",
        "weep",
        "suffer",
        "pain",
        "anguish",
        "misery",
        "wretched",
        "dismal",
        "mournful",
        "hopeless",
        "desolate",
        "forlorn",
        "bleak",
        "somber",
        "melancholy",
        "doleful",
        "lament",
        "dirge",
        "macabre",
        "gruesome",
        "ghastly",
        "hideous",
        "appalling",
        "shocking",
        "frightful",
        "dreadful",
        "awful",
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
        "when",
        "where",
        "which",
        "who",
        "whom",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
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
        "none",
        "nothing",
        "never",
        "very",
        "too",
        "quite",
        "rather",
        "some",
        "any",
        "each",
        "every",
        "all",
        "both",
        "few",
        "several",
        "many",
        "much",
        "more",
        "most",
        "other",
        "another",
        "such",
        "own",
        "same",
        "so",
        "as",
        "than",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "over",
        "under",
        "above",
        "below",
        "into",
        "through",
        "between",
        "before",
        "after",
        "about",
        "from",
        "to",
        "up",
        "down",
        "out",
        "off",
        "upon",
    ]
)

LOVECRAFTIAN_WORDS = set(
    [
        "cyclopean",
        "non-euclidean",
        "eldritch",
        "antediluvian",
        "chthonian",
        "cryptic",
        "primordial",
        "immemorial",
        "squamous",
        "rugose",
        "ichor",
        "gibbous",
        "cthulhu",
        "unspeakable",
        "unnameable",
        "indescribable",
        "inconceivable",
        "fecund",
        "loathsome",
        "repulsive",
        "blasphemous",
        "abysmal",
        "abyssal",
        "daemonic",
        "daemon",
        "sorcerous",
        "accursed",
        "maddening",
        "nightmare",
        "nightmarish",
        "foetid",
        "fetid",
        "putrid",
        "noxious",
        "perverse",
        "ancient",
        "elder",
        "forbidding",
        "gigantic",
        "colossal",
        "monstrous",
        "hellish",
        "infernal",
        "otherworldly",
        "unearthly",
        "supernatural",
        "preternatural",
    ]
)

SUB_CONJUNCTIONS = set(
    [
        "after",
        "although",
        "as",
        "because",
        "before",
        "even",
        "if",
        "once",
        "since",
        "that",
        "though",
        "till",
        "unless",
        "until",
        "when",
        "whenever",
        "where",
        "wherever",
        "while",
        "whereas",
        "whilst",
        "provided",
    ]
)

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def extract_stylometric_features(texts):
    """Extract 33 stylometric features from texts."""
    n = len(texts)
    features = np.zeros((n, 33), dtype=np.float32)

    for i, text in enumerate(texts):
        text = str(text)
        if not text.strip():
            continue

        text_len = len(text)
        words = text.split()
        word_count = len(words)
        sent_endings = re.findall(r"[.!?]+", text)
        sent_count = max(len(sent_endings), 1)
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0

        char_counts = {
            "upper": sum(1 for c in text if c.isupper()),
            "lower": sum(1 for c in text if c.islower()),
            "digit": sum(1 for c in text if c.isdigit()),
            "whitespace": sum(1 for c in text if c.isspace()),
        }

        punct_marks = [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "..."]
        punct_counts = {}
        for p in punct_marks:
            punct_counts[p] = text.count(p)

        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / max(text_len, 1)

        long_words_count = sum(1 for w in words if len(w) >= 7)
        capitalized_count = sum(1 for w in words if w and w[0].isupper())
        all_caps_count = sum(1 for w in words if len(w) > 1 and w.isupper())

        sent_lengths = []
        current_sent = ""
        for char in text:
            current_sent += char
            if char in ".!?":
                sent_lengths.append(len(current_sent.split()))
                current_sent = ""
        if current_sent.strip():
            sent_lengths.append(len(current_sent.split()))

        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 0 else 0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 0 else 0

        words_lower = [w.lower().strip(string.punctuation) for w in words]
        words_clean = [w for w in words_lower if w]
        total_clean = max(len(words_clean), 1)

        function_word_count = sum(1 for w in words_clean if w in FUNCTION_WORDS)
        function_word_ratio = function_word_count / total_clean

        archaic_count = sum(1 for w in words_clean if w in ARCHAIC_WORDS)
        archaic_ratio = archaic_count / total_clean

        positive_count = sum(1 for w in words_clean if w in POSITIVE_EMOTION_WORDS)
        negative_count = sum(1 for w in words_clean if w in NEGATIVE_EMOTION_WORDS)
        emotional_ratio = (positive_count + negative_count) / total_clean

        lovecraft_count = sum(1 for w in words_clean if w in LOVECRAFTIAN_WORDS)
        lovecraft_ratio = lovecraft_count / total_clean

        sub_conj_count = sum(1 for w in words_clean if w in SUB_CONJUNCTIONS)
        sub_conj_ratio = sub_conj_count / total_clean

        features[i, 0] = text_len
        features[i, 1] = word_count
        features[i, 2] = sent_count
        features[i, 3] = avg_word_len
        features[i, 4] = text_len / max(sent_count, 1)
        features[i, 5] = char_counts["upper"] / max(text_len, 1)
        features[i, 6] = char_counts["lower"] / max(text_len, 1)
        features[i, 7] = char_counts["digit"] / max(text_len, 1)
        features[i, 8] = char_counts["whitespace"] / max(text_len, 1)
        features[i, 9] = punct_counts["."] / max(text_len, 1) * 100
        features[i, 10] = punct_counts[","] / max(text_len, 1) * 100
        features[i, 11] = punct_counts["!"] / max(text_len, 1) * 100
        features[i, 12] = punct_counts["?"] / max(text_len, 1) * 100
        features[i, 13] = punct_counts[";"] / max(text_len, 1) * 100
        features[i, 14] = punct_counts[":"] / max(text_len, 1) * 100
        features[i, 15] = punct_counts["-"] / max(text_len, 1) * 100
        features[i, 16] = punct_counts['"'] / max(text_len, 1) * 100
        features[i, 17] = punct_counts["'"] / max(text_len, 1) * 100
        features[i, 18] = punct_counts["("] / max(text_len, 1) * 100
        features[i, 19] = punct_counts[")"] / max(text_len, 1) * 100
        features[i, 20] = text.count("...") / max(text_len, 1) * 100
        features[i, 21] = char_diversity
        features[i, 22] = long_words_count / max(word_count, 1)
        features[i, 23] = capitalized_count / max(word_count, 1)
        features[i, 24] = all_caps_count / max(word_count, 1)
        features[i, 25] = sent_len_std
        features[i, 26] = sent_len_var
        features[i, 27] = function_word_ratio
        features[i, 28] = archaic_ratio
        features[i, 29] = emotional_ratio
        features[i, 30] = lovecraft_ratio
        features[i, 31] = sub_conj_ratio
        features[i, 32] = (positive_count + 1) / (negative_count + 1)

    return features

def create_readability_features(texts):
    """Extract 4 readability features from texts."""

    def count_syllables(word):
        word = word.lower().strip(string.punctuation)
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
        return count

    n = len(texts)
    features = np.zeros((n, 4), dtype=np.float32)

    for i, text in enumerate(texts):
        text = str(text)
        if not text.strip():
            continue

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)

        words = text.split()
        num_words = max(len(words), 1)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / num_words

        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / num_words

        total_chars = sum(len(w) for w in words)

        fre = (
            206.835
            - 1.015 * (num_words / num_sentences)
            - 84.6 * (total_syllables / num_words)
        )
        fre = max(0, min(100, fre))

        ari = (
            4.71 * (total_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        )

        features[i, 0] = fre
        features[i, 1] = ari
        features[i, 2] = avg_syllables
        features[i, 3] = complex_ratio

    return features

def create_pos_tag_approximation(texts):
    """Approximate POS tagging via suffix patterns."""
    NOUN_SUFFIXES = (
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
        "ist",
        "ism",
        "age",
        "ure",
        "tude",
    )
    VERB_SUFFIXES = ("ed", "ing", "ize", "ise", "ify", "ate", "en", "er", "est")
    ADJ_SUFFIXES = (
        "ous",
        "ive",
        "ful",
        "al",
        "able",
        "ible",
        "ic",
        "ical",
        "ish",
        "less",
        "like",
        "y",
    )
    ADV_SUFFIXES = ("ly", "wards", "wise")

    n = len(texts)
    features = np.zeros((n, 5), dtype=np.float32)

    for i, text in enumerate(texts):
        text = str(text)
        if not text.strip():
            continue

        words = text.split()
        total_words = max(len(words), 1)

        noun_count = 0
        verb_count = 0
        adj_count = 0
        adv_count = 0
        content_count = 0

        for w in words:
            w_lower = w.lower().strip(string.punctuation)
            if not w_lower:
                continue

            if w_lower.endswith(NOUN_SUFFIXES):
                noun_count += 1
            if w_lower.endswith(VERB_SUFFIXES):
                verb_count += 1
            if w_lower.endswith(ADJ_SUFFIXES):
                adj_count += 1
            if w_lower.endswith(ADV_SUFFIXES):
                adv_count += 1
            if w_lower not in FUNCTION_WORDS:
                content_count += 1

        features[i, 0] = noun_count / total_words
        features[i, 1] = verb_count / total_words
        features[i, 2] = adj_count / total_words
        features[i, 3] = adv_count / total_words
        features[i, 4] = content_count / total_words

    return features

def compute_log_loss(y_true, y_pred_proba):
    """Multi-class logarithmic loss."""
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
# STRATIFIED SPLIT (NO INDEX_BUG)
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
# HANDCRAFTED FEATURES EXTRACTION
# ============================================================
print("\nExtracting handcrafted features...")

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
    f"Stylo features: {train_stylo_filtered.shape[1]}, Read: {train_read_scaled.shape[1]}, POS: {train_pos_scaled.shape[1]}"
)

# ============================================================
# N-GRAM FEATURES
# ============================================================
print("\nExtracting n-gram features...")

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

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler() if torch.cuda.is_available() else None

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
        # Save only the classifier and pooler to avoid architecture mismatch
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_deberta_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest DeBERTa model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
# Load with strict=False to handle potential pooler key mismatches
state_dict = torch.load(f"{WORKING_DIR}/best_deberta_model.pt", map_location=device)
model_state = model.state_dict()
# Filter out keys that don't exist in current model or have size mismatch
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if k in model_state and v.shape == model_state[k].shape
}
model.load_state_dict(filtered_state_dict, strict=False)
print(f"Loaded {len(filtered_state_dict)}/{len(model_state)} parameters (skipped pooler if mismatched)")

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
# XGBOOST CLASSIFIER
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
# LOGISTIC REGRESSION CLASSIFIER
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
# DEBERTA FINAL PREDICTIONS
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
deberta_val_loss, _, deberta_val_probs = evaluate_deberta(model, val_loader_eval)
print(f"DeBERTa validation log loss: {deberta_val_loss:.4f}")

model.eval()
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
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
