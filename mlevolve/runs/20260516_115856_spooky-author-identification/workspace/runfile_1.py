import os
os.sched_setaffinity(0, {30, 31})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
OUTPUT_CSV = "./submission/submission_a9a2d9e744d34655bfb958c65497f679.csv"
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
# FEATURE ENGINEERING FUNCTIONS
# ============================================================
def extract_stylometric_features(texts):
    """
    Extract 30 stylometric features from texts:
    - Basic stats: text length, word count, sentence count
    - Average word/sentence lengths
    - Character type ratios (upper, lower, digit, whitespace)
    - Punctuation ratios (12 types)
    - Vocabulary diversity, long word ratio
    - Capitalization patterns
    - Function words, archaic words, emotional words, Lovecraft-specific words
    - Sentence length variability
    """
    n_samples = len(texts)
    features = np.zeros((n_samples, 30), dtype=np.float32)

    # Word lists for author style detection
    function_words = set(
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
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "can",
            "could",
            "must",
            "ought",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "when",
            "where",
            "why",
            "how",
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
            "her",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "his",
            "hers",
            "its",
            "ours",
            "theirs",
            "myself",
            "yourself",
            "himself",
            "herself",
            "itself",
            "ourselves",
            "yourselves",
            "themselves",
        ]
    )

    archaic_words = set(
        [
            "thou",
            "thee",
            "thy",
            "thine",
            "ye",
            "hath",
            "doth",
            "dost",
            "art",
            "wert",
            "shalt",
            "wilt",
            "canst",
            "couldst",
            "wouldst",
            "shouldst",
            "mayst",
            "mightst",
            "thrice",
            "whence",
            "thence",
            "whither",
            "thither",
            "hither",
            "wherein",
            "whereof",
            "wherewith",
            "therein",
            "thereof",
            "therewith",
            "herein",
            "hereof",
            "herewith",
            "ere",
            "betwixt",
            "twixt",
            "unto",
            "nay",
            "yea",
            "perchance",
            "methinks",
            "methought",
            "forsooth",
            "albeit",
            "anon",
            "ne'er",
            "o'er",
            "e'en",
            "oft",
            "oftentimes",
            "ofttimes",
            "erewhile",
            "erstwhile",
            "wherefore",
            "whereupon",
        ]
    )

    emotional_words = set(
        [
            "dread",
            "horror",
            "terror",
            "fear",
            "fright",
            "panic",
            "alarm",
            "anguish",
            "agony",
            "torment",
            "suffering",
            "pain",
            "woe",
            "sorrow",
            "grief",
            "despair",
            "hopeless",
            "gloom",
            "melancholy",
            "somber",
            "dark",
            "shadow",
            "death",
            "dead",
            "corpse",
            "ghost",
            "specter",
            "phantom",
            "apparition",
            "spirit",
            "demon",
            "devil",
            "fiend",
            "monster",
            "beast",
            "creature",
            "evil",
            "wicked",
            "vile",
            "hideous",
            "grotesque",
            "ghastly",
            "grim",
            "macabre",
            "morbid",
            "sinister",
            "ominous",
            "portentous",
            "foreboding",
            "eerie",
            "uncanny",
            "weird",
            "strange",
            "bizarre",
            "odd",
            "peculiar",
            "curious",
            "mysterious",
            "secret",
            "unknown",
            "unseen",
            "invisible",
            "supernatural",
            "unnatural",
            "otherworldly",
            "ethereal",
            "unearthly",
        ]
    )

    lovecraft_words = set(
        [
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "azathoth",
            "nyarlathotep",
            "shoggoth",
            "eldritch",
            "cyclopean",
            "non-euclidean",
            "non euclidean",
            "arkham",
            "miskatonic",
            "insmouth",
            "dunwich",
            "kadath",
            "leng",
            "plateau",
            "antediluvian",
            "prehuman",
            "pre-human",
            "primordial",
            "aeon",
            "immemorial",
            "unspeakable",
            "unnameable",
            "indescribable",
            "ineffable",
            "nameless",
            "blasphemous",
            "abominable",
            "loathsome",
            "noisome",
            "foetid",
            "stench",
            "fragrant",
            "squamous",
            "rugose",
            "ichor",
            "gibbous",
            "necronomicon",
            "book",
            "manuscript",
            "cryptic",
            "esoteric",
            "occult",
            "arcane",
            "forbidden",
            "cosmic",
            "galactic",
            "spheres",
            "dimension",
            "abyss",
            "void",
            "chaos",
            "madness",
            "insanity",
            "crazed",
            "lurking",
            "dwelling",
            "sleeping",
            "dreaming",
            "dead",
            "undead",
            "slimy",
            "gelatinous",
        ]
    )

    for i, text in enumerate(texts):
        text = str(text)
        n_chars = len(text)
        words = text.split()
        n_words = len(words)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        n_sents = max(len(sentences), 1)

        # Basic stats
        features[i, 0] = n_chars  # text length
        features[i, 1] = n_words  # word count
        features[i, 2] = n_sents  # sentence count
        features[i, 3] = n_chars / max(n_words, 1)  # avg word length
        features[i, 4] = n_words / n_sents  # avg sentence length (words)

        # Character type ratios
        upper_count = sum(1 for c in text if c.isupper())
        lower_count = sum(1 for c in text if c.islower())
        digit_count = sum(1 for c in text if c.isdigit())
        space_count = sum(1 for c in text if c.isspace())
        features[i, 5] = upper_count / max(n_chars, 1)  # uppercase ratio
        features[i, 6] = lower_count / max(n_chars, 1)  # lowercase ratio
        features[i, 7] = digit_count / max(n_chars, 1)  # digit ratio
        features[i, 8] = space_count / max(n_chars, 1)  # whitespace ratio

        # Punctuation ratios
        punct_types = [".", ",", ";", ":", "!", "?", "-", '"', "'", "(", ")", "—"]
        for j, punct in enumerate(punct_types):
            punct_count = text.count(punct)
            features[i, 9 + j] = punct_count / max(n_chars, 1)  # punct type j ratio

        # Vocabulary diversity
        unique_words = len(set(w.lower() for w in words))
        features[i, 21] = unique_words / max(n_words, 1)  # type-token ratio

        # Long words (≥7 chars) ratio
        long_words = sum(1 for w in words if len(w) >= 7)
        features[i, 22] = long_words / max(n_words, 1)

        # Capitalization patterns
        capitalized_words = sum(1 for w in words if w[0].isupper() if w)
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features[i, 23] = capitalized_words / max(n_words, 1)  # capitalized ratio
        features[i, 24] = all_caps_words / max(n_words, 1)  # all caps ratio

        # Sentence length variability
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            features[i, 25] = np.std(sent_lengths)  # sent len std
            features[i, 26] = np.var(sent_lengths)  # sent len variance

        # Word category ratios
        lower_words = [
            w.lower().strip(string.punctuation)
            for w in words
            if w.strip(string.punctuation)
        ]
        n_valid_words = max(len(lower_words), 1)

        func_word_count = sum(1 for w in lower_words if w in function_words)
        features[i, 27] = func_word_count / n_valid_words  # function word ratio

        archaic_count = sum(1 for w in lower_words if w in archaic_words)
        features[i, 28] = archaic_count / n_valid_words  # archaic word ratio

        emotional_count = sum(1 for w in lower_words if w in emotional_words)
        lovecraft_count = sum(1 for w in lower_words if w in lovecraft_words)
        features[i, 29] = (
            emotional_count + lovecraft_count
        ) / n_valid_words  # combined stylistic ratio

    return features


def create_readability_features(texts):
    """
    Extract 4 readability features:
    - Flesch Reading Ease score
    - Automated Readability Index (ARI)
    - Average syllables per word
    - Complex word ratio (words with 3+ syllables)
    """
    n_samples = len(texts)
    features = np.zeros((n_samples, 4), dtype=np.float32)

    # English syllable counts (simplified heuristic)
    def count_syllables(word):
        word = word.lower().strip(string.punctuation)
        if not word:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e"):
            count = max(count - 1, 1)
        if word.endswith("le") and len(word) > 2:
            count = max(count + 1, 1)
        return max(count, 1)

    for i, text in enumerate(texts):
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]

        n_words = len(words)
        n_sents = max(len(sentences), 1)

        if n_words == 0:
            continue

        # Count syllables and complex words
        total_syllables = 0
        complex_words = 0
        for word in words:
            syl_count = count_syllables(word)
            total_syllables += syl_count
            if syl_count >= 3:
                complex_words += 1

        avg_syllables = total_syllables / n_words
        complex_ratio = complex_words / n_words

        # Flesch Reading Ease
        fre = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * avg_syllables
        fre = np.clip(fre, 0, 100)
        features[i, 0] = fre

        # Automated Readability Index
        n_chars = sum(len(w) for w in words)
        ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43
        ari = max(ari, 0)
        features[i, 1] = ari

        # Average syllables
        features[i, 2] = avg_syllables

        # Complex word ratio
        features[i, 3] = complex_ratio

    return features


def create_pos_tag_approximation(texts):
    """
    Approximate POS tagging using suffix patterns (5 features):
    - Noun-like suffix ratio
    - Verb-like suffix ratio
    - Adjective-like suffix ratio
    - Adverb-like suffix ratio
    - Content word ratio (non-function words)
    """
    n_samples = len(texts)
    features = np.zeros((n_samples, 5), dtype=np.float32)

    # Suffix patterns for POS approximation
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "dom",
        "ship",
        "ism",
        "ist",
        "ity",
        "age",
        "ure",
        "ery",
        "ory",
        "tude",
        "cule",
        "cle",
        "ice",
        "ine",
        "ing",
    ]
    verb_suffixes = [
        "ate",
        "ify",
        "ize",
        "ise",
        "en",
        "ish",
        "ess",
        "end",
        "eed",
        "ain",
        "eep",
        "ell",
        "ind",
        "and",
        "old",
    ]
    adj_suffixes = [
        "ful",
        "less",
        "ous",
        "ive",
        "able",
        "ible",
        "al",
        "ial",
        "ical",
        "ish",
        "like",
        "y",
        "ent",
        "ant",
        "ary",
        "ory",
        "some",
        "ward",
        "en",
    ]
    adv_suffixes = ["ly", "ward", "wards", "wise", "ways", "style"]

    function_words = set(
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
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "can",
            "could",
            "must",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "when",
            "where",
            "why",
            "how",
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
        ]
    )

    for i, text in enumerate(texts):
        text = str(text)
        words = text.split()
        n_words = max(len(words), 1)

        noun_count = 0
        verb_count = 0
        adj_count = 0
        adv_count = 0
        content_count = 0

        for word in words:
            word_lower = word.lower().strip(string.punctuation)
            if not word_lower:
                continue

            # Check suffixes
            for suffix in noun_suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                    noun_count += 1
                    break
            for suffix in verb_suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                    verb_count += 1
                    break
            for suffix in adj_suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                    adj_count += 1
                    break
            for suffix in adv_suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                    adv_count += 1
                    break

            # Check if content word
            if word_lower not in function_words:
                content_count += 1

        features[i, 0] = noun_count / n_words  # noun-like suffix ratio
        features[i, 1] = verb_count / n_words  # verb-like suffix ratio
        features[i, 2] = adj_count / n_words  # adjective-like suffix ratio
        features[i, 3] = adv_count / n_words  # adverb-like suffix ratio
        features[i, 4] = content_count / n_words  # content word ratio

    return features


# ============================================================
# HANDCRAFTED FEATURES
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
print(f"Stylometric features after variance filtering: {train_stylo_filtered.shape[1]}")

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
    return "".join([c for c in str(text) if c in string.punctuation]) if text else ""


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

print(f"Train sparse features: {train_sparse.shape}")
print(f"Val sparse features: {val_sparse.shape}")
print(f"Test sparse features: {test_sparse.shape}")

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
deberta_val_loss, _, deberta_val_probs = evaluate_deberta(model, val_loader_eval)
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
