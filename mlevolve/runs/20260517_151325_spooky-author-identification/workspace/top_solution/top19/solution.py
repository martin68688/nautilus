import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
)
import numpy as np
import pandas as pd
import os
import re
import warnings
import gc
import string
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import hstack, save_npz, csr_matrix
from collections import Counter
from sentence_transformers import SentenceTransformer
import pickle

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
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50
WARMUP_RATIO = 0.1
PATIENCE = 7
DROPOUT = 0.3

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
# STRATIFIED SPLIT - NO INDEX_BUG
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)

assert len(set(train_idx) & set(val_idx)) == 0, "CRITICAL: Train/val overlap detected!"

# ============================================================
# FEATURE 1: STYLOMETRIC FINGERPRINT FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING STYLOMETRIC FINGERPRINT FEATURES")
print("=" * 60)

# Define word lists
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
        "wast",
        "wert",
        "shalt",
        "wilt",
        "canst",
        "didst",
        "hadst",
        "cometh",
        "maketh",
        "taketh",
        "upon",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "anon",
        "perchance",
        "forsooth",
        "bethink",
        "bethought",
        "betwixt",
        "twixt",
        "nigh",
        "methinks",
        "methought",
        "prithee",
        "pray",
        "twas",
        "twere",
        "twill",
        "dread",
        "horror",
        "ghastly",
        "spectral",
        "phantom",
        "abyss",
        "eldritch",
        "cyclopean",
        "squamous",
        "ichor",
        "fungus",
        "fester",
        "putrid",
        "gibber",
    ]
)

emotional_words = set(
    [
        "fear",
        "terror",
        "dread",
        "horror",
        "panic",
        "alarm",
        "fright",
        "scare",
        "anguish",
        "agony",
        "despair",
        "grief",
        "sorrow",
        "woe",
        "misery",
        "gloom",
        "darkness",
        "shadow",
        "night",
        "gloomy",
        "somber",
        "dreary",
        "weird",
        "strange",
        "odd",
        "peculiar",
        "curious",
        "mysterious",
        "uncanny",
        "fabulous",
        "prodigious",
        "monstrous",
        "hideous",
        "terrible",
        "awful",
    ]
)

lovecraft_words = set(
    [
        "eldritch",
        "cyclopean",
        "squamous",
        "ichor",
        "fungus",
        "fester",
        "putrid",
        "gibber",
        "non-euclidean",
        "antediluvian",
        "carcosa",
        "cthulhu",
        "yog-sothoth",
        "nyarlathotep",
        "shoggoth",
        "miskatonic",
        "arkham",
        "innsmouth",
        "kadath",
        "ulthar",
        "hyperborean",
        "aklo",
        "necronomicon",
        "alhazred",
        "sarnath",
    ]
)

function_words = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
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
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "not",
        "no",
        "nor",
    ]
)

def extract_stylometric_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""

        text_len = len(text)
        words = text.split()
        word_count = len(words) if words else 1
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = len(sentences) if sentences else 1

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        char_counts = {
            "upper": sum(1 for c in text if c.isupper()),
            "lower": sum(1 for c in text if c.islower()),
            "digit": sum(1 for c in text if c.isdigit()),
            "whitespace": sum(1 for c in text if c.isspace()),
            "punct": sum(1 for c in text if c in string.punctuation),
        }

        total_chars = max(text_len, 1)
        char_ratios = {
            "upper_ratio": char_counts["upper"] / total_chars,
            "lower_ratio": char_counts["lower"] / total_chars,
            "digit_ratio": char_counts["digit"] / total_chars,
            "whitespace_ratio": char_counts["whitespace"] / total_chars,
            "punct_ratio": char_counts["punct"] / total_chars,
        }

        punct_ratios = {}
        for p in string.punctuation:
            count = text.count(p)
            punct_ratios[f"punct_{p}_ratio"] = count / total_chars

        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / min(max(unique_chars, 1), 100)

        long_words_ratio = (
            sum(1 for w in words if len(w) > 6) / word_count if word_count > 0 else 0
        )

        capitalized_ratio = (
            sum(1 for w in words if w[0].isupper() and w.isalpha()) / word_count
            if word_count > 0
            else 0
        )

        all_caps_ratio = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
            if word_count > 0
            else 0
        )

        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 0 else 0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 0 else 0

        words_lower = [w.lower().strip(string.punctuation) for w in words]
        words_clean = [w for w in words_lower if w]
        total_clean_words = max(len(words_clean), 1)

        function_word_count = sum(1 for w in words_clean if w in function_words)
        function_word_ratio = function_word_count / total_clean_words

        archaic_word_count = sum(1 for w in words_clean if w in archaic_words)
        archaic_ratio = archaic_word_count / total_clean_words

        emotional_word_count = sum(1 for w in words_clean if w in emotional_words)
        emotional_ratio = emotional_word_count / total_clean_words

        lovecraft_word_count = sum(1 for w in words_clean if w in lovecraft_words)
        lovecraft_ratio = lovecraft_word_count / total_clean_words

        first_person = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"]
        first_person_ratio = (
            sum(1 for w in words_clean if w in first_person) / total_clean_words
        )

        interjections = ["oh", "ah", "alas", "hark", "hush", "hist", "lo", "behold"]
        interjection_ratio = (
            sum(1 for w in words_clean if w in interjections) / total_clean_words
        )

        q_excl_ratio = (
            (text.count("?") + text.count("!")) / sent_count if sent_count > 0 else 0
        )

        feature_vector = [
            text_len,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            char_diversity,
            long_words_ratio,
            capitalized_ratio,
            char_ratios["upper_ratio"],
            char_ratios["lower_ratio"],
            char_ratios["digit_ratio"],
            char_ratios["whitespace_ratio"],
            char_ratios["punct_ratio"],
            all_caps_ratio,
            sent_len_std,
            sent_len_var,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            first_person_ratio,
            interjection_ratio,
            q_excl_ratio,
        ]

        for p in string.punctuation:
            feature_vector.append(punct_ratios[f"punct_{p}_ratio"])

        features.append(feature_vector)

    return np.array(features)

print("Extracting stylometric features...")
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

print(
    f"Stylometric features: train={train_stylo_filtered.shape}, val={val_stylo_filtered.shape}, test={test_stylo_filtered.shape}"
)

# ============================================================
# FEATURE 2: READABILITY AND COMPLEXITY FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING READABILITY & COMPLEXITY FEATURES")
print("=" * 60)

def count_syllables(word):
    word = word.lower()
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
    return max(count, 1)

def create_readability_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""

        words = text.split()
        word_count = len(words) if words else 1
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = len(sentences) if sentences else 1
        char_count = sum(len(w) for w in words)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count

        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_word_ratio = complex_words / word_count

        flesch = (
            206.835
            - 1.015 * (word_count / sent_count)
            - 84.6 * (total_syllables / word_count)
        )
        flesch = max(0, min(100, flesch))

        ari = 4.71 * (char_count / word_count) + 0.5 * (word_count / sent_count) - 21.43
        ari = max(0, ari)

        L = (char_count / word_count) * 100
        S = (sent_count / word_count) * 100
        coleman = 0.0588 * L - 0.296 * S - 15.8
        coleman = max(0, coleman)

        fog = 0.4 * ((word_count / sent_count) + 100 * (complex_words / word_count))
        fog = max(0, fog)

        smog = 1.0430 * np.sqrt(complex_words * (30 / sent_count)) + 3.1291
        smog = max(0, smog)

        feature_vector = [
            flesch,
            ari,
            avg_syllables,
            complex_word_ratio,
            coleman,
            fog,
            smog,
        ]
        features.append(feature_vector)

    return np.array(features)

print("Extracting readability features...")
train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

print(
    f"Readability features: train={train_read_scaled.shape}, val={val_read_scaled.shape}, test={test_read_scaled.shape}"
)

# ============================================================
# FEATURE 3: POS TAG APPROXIMATION FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING POS TAG APPROXIMATION FEATURES")
print("=" * 60)

def create_pos_tag_approximation(texts):
    noun_suffixes = (
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ship",
        "ity",
    )
    verb_suffixes = ("ed", "ing", "ate", "ify", "ize", "en", "ish")
    adj_suffixes = (
        "able",
        "ible",
        "ful",
        "less",
        "ous",
        "al",
        "ic",
        "ive",
        "ish",
        "like",
    )
    adv_suffixes = ("ly", "ward", "wise")

    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""

        words = text.split()
        word_count = len(words) if words else 1

        noun_count = sum(1 for w in words if w.lower().endswith(noun_suffixes))
        verb_count = sum(1 for w in words if w.lower().endswith(verb_suffixes))
        adj_count = sum(1 for w in words if w.lower().endswith(adj_suffixes))
        adv_count = sum(1 for w in words if w.lower().endswith(adv_suffixes))

        noun_ratio = noun_count / word_count
        verb_ratio = verb_count / word_count
        adj_ratio = adj_count / word_count
        adv_ratio = adv_count / word_count

        content_word_count = noun_count + verb_count + adj_count + adv_count
        content_word_ratio = content_word_count / word_count

        feature_vector = [
            noun_ratio,
            verb_ratio,
            adj_ratio,
            adv_ratio,
            content_word_ratio,
        ]
        features.append(feature_vector)

    return np.array(features)

print("Extracting POS approximation features...")
train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

print(
    f"POS features: train={train_pos_scaled.shape}, val={val_pos_scaled.shape}, test={test_pos_scaled.shape}"
)

# ============================================================
# FEATURE 4: CHARACTER N-GRAM FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING CHARACTER N-GRAM FEATURES")
print("=" * 60)

char_vec_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vec_short.fit_transform(X_train_texts)
val_char_short = char_vec_short.transform(X_val_texts)
test_char_short = char_vec_short.transform(test_df["text"].values)
print(f"Short char n-grams: {train_char_short.shape}")

char_vec_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vec_med.fit_transform(X_train_texts)
val_char_med = char_vec_med.transform(X_val_texts)
test_char_med = char_vec_med.transform(test_df["text"].values)
print(f"Medium char n-grams: {train_char_med.shape}")

char_vec_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 8),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vec_long.fit_transform(X_train_texts)
val_char_long = char_vec_long.transform(X_val_texts)
test_char_long = char_vec_long.transform(test_df["text"].values)
print(f"Long char n-grams: {train_char_long.shape}")

# ============================================================
# FEATURE 5: WORD N-GRAM FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING WORD N-GRAM FEATURES")
print("=" * 60)

word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vec.fit_transform(X_train_texts)
val_word = word_vec.transform(X_val_texts)
test_word = word_vec.transform(test_df["text"].values)
print(f"Word n-grams: {train_word.shape}")

# ============================================================
# FEATURE 6: PUNCTUATION PATTERN FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING PUNCTUATION PATTERN FEATURES")
print("=" * 60)

def extract_punctuation_sequence(text):
    if not isinstance(text, str):
        return ""
    return "".join([c for c in text if c in string.punctuation])

all_texts_punct = np.concatenate([X_train_texts, X_val_texts, test_df["text"].values])
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_punct]
punct_vec = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vec.fit_transform(punct_sequences)

n_train = len(X_train_texts)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + len(X_val_texts)]
test_punct = punct_features_all[n_train + len(X_val_texts) :]
print(f"Punctuation features: {train_punct.shape}")

# ============================================================
# FEATURE 7: SEMANTIC EMBEDDINGS FROM SENTENCE TRANSFORMER
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING SEMANTIC EMBEDDINGS (all-MiniLM-L6-v2)")
print("=" * 60)

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

batch_size = 256
train_embeddings_list = []
for i in range(0, len(X_train_texts), batch_size):
    batch = X_train_texts[i : i + batch_size]
    embeddings = semantic_model.encode(
        batch, show_progress_bar=False, convert_to_numpy=True
    )
    train_embeddings_list.append(embeddings)
train_semantic_emb = np.vstack(train_embeddings_list)

val_embeddings_list = []
for i in range(0, len(X_val_texts), batch_size):
    batch = X_val_texts[i : i + batch_size]
    embeddings = semantic_model.encode(
        batch, show_progress_bar=False, convert_to_numpy=True
    )
    val_embeddings_list.append(embeddings)
val_semantic_emb = np.vstack(val_embeddings_list)

test_embeddings_list = []
for i in range(0, len(test_df), batch_size):
    batch = test_df["text"].values[i : i + batch_size]
    embeddings = semantic_model.encode(
        batch, show_progress_bar=False, convert_to_numpy=True
    )
    test_embeddings_list.append(embeddings)
test_semantic_emb = np.vstack(test_embeddings_list)

print(
    f"Semantic embeddings: train={train_semantic_emb.shape}, val={val_semantic_emb.shape}, test={test_semantic_emb.shape}"
)

# ============================================================
# FEATURE 8: TEXT STRUCTURE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING TEXT STRUCTURE FEATURES")
print("=" * 60)

stop_words = set(
    [
        "a",
        "an",
        "the",
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
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
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
    ]
)

def extract_text_structure_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""

        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)

        words = text.split()
        word_count = len(words) if words else 1
        words_lower = [
            w.lower().strip(string.punctuation)
            for w in words
            if w.strip(string.punctuation)
        ]

        unique_words = len(set(words_lower))
        unique_ratio = unique_words / max(len(words_lower), 1)

        stop_word_count = sum(1 for w in words_lower if w in stop_words)
        stop_word_ratio = stop_word_count / max(len(words_lower), 1)

        sentences = re.split(r"[.!?]+", text)
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sent_lengths:
            sent_len_q1 = np.percentile(sent_lengths, 25)
            sent_len_median = np.percentile(sent_lengths, 50)
            sent_len_q3 = np.percentile(sent_lengths, 75)
            sent_len_iqr = sent_len_q3 - sent_len_q1
        else:
            sent_len_q1 = sent_len_median = sent_len_q3 = sent_len_iqr = 0

        feature_vector = [
            para_count,
            unique_ratio,
            stop_word_ratio,
            sent_len_median,
            sent_len_iqr,
        ]
        features.append(feature_vector)

    return np.array(features)

print("Extracting text structure features...")
train_structure = extract_text_structure_features(X_train_texts)
val_structure = extract_text_structure_features(X_val_texts)
test_structure = extract_text_structure_features(test_df["text"].values)

structure_scaler = StandardScaler()
train_structure_scaled = structure_scaler.fit_transform(train_structure)
val_structure_scaled = structure_scaler.transform(val_structure)
test_structure_scaled = structure_scaler.transform(test_structure)

print(
    f"Structure features: train={train_structure_scaled.shape}, val={val_structure_scaled.shape}, test={test_structure_scaled.shape}"
)

# ============================================================
# COMBINE ALL HANDCRAFTED FEATURES
# ============================================================
print("\n" + "=" * 60)
print("COMBINING ALL HANDCRAFTED FEATURES")
print("=" * 60)

train_dense = np.hstack(
    [
        train_stylo_filtered,
        train_read_scaled,
        train_pos_scaled,
        train_structure_scaled,
    ]
)
val_dense = np.hstack(
    [
        val_stylo_filtered,
        val_read_scaled,
        val_pos_scaled,
        val_structure_scaled,
    ]
)
test_dense = np.hstack(
    [
        test_stylo_filtered,
        test_read_scaled,
        test_pos_scaled,
        test_structure_scaled,
    ]
)

print(
    f"Dense features: train={train_dense.shape}, val={val_dense.shape}, test={test_dense.shape}"
)

train_sparse = hstack(
    [
        train_char_short,
        train_char_med,
        train_char_long,
        train_word,
        train_punct,
    ]
).tocsr()
val_sparse = hstack(
    [
        val_char_short,
        val_char_med,
        val_char_long,
        val_word,
        val_punct,
    ]
).tocsr()
test_sparse = hstack(
    [
        test_char_short,
        test_char_med,
        test_char_long,
        test_word,
        test_punct,
    ]
).tocsr()

print(
    f"Sparse features: train={train_sparse.shape}, val={val_sparse.shape}, test={test_sparse.shape}"
)

# ============================================================
# MODEL ARCHITECTURE: SimpleClassifier (using pre-extracted features)
# ============================================================
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE DESIGN")
print("=" * 60)

class SimpleAuthorClassifier(nn.Module):
    def __init__(
        self,
        dense_input_dim,
        semantic_input_dim,
        sparse_input_dim,
        num_labels=3,
        dropout=0.3,
    ):
        super(SimpleAuthorClassifier, self).__init__()

        self.bn_dense = nn.BatchNorm1d(dense_input_dim)
        self.bn_semantic = nn.BatchNorm1d(semantic_input_dim)

        self.dense_branch = nn.Sequential(
            nn.Linear(dense_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.sparse_branch = nn.Sequential(
            nn.Linear(sparse_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        combined_dim = 64 + 64 + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, dense_feats, semantic_emb, sparse_feats):
        dense = self.bn_dense(dense_feats)
        dense_out = self.dense_branch(dense)

        sem = self.bn_semantic(semantic_emb)
        sem_out = self.semantic_branch(sem)

        sparse_out = self.sparse_branch(sparse_feats)

        combined = torch.cat([dense_out, sem_out, sparse_out], dim=1)
        logits = self.classifier(combined)
        return logits

print(f"Model: SimpleAuthorClassifier (3-branch MLP)")
print(f"Dense input dim: {train_dense.shape[1]}")
print(f"Semantic input dim: {train_semantic_emb.shape[1]}")
print(f"Sparse input dim: {train_sparse.shape[1]}")
print(f"Dropout: {DROPOUT}")
print(f"Loss: CrossEntropyLoss with label_smoothing=0.1")
print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print("=" * 60)

# ============================================================
# DATASET SETUP
# ============================================================
print("\n" + "=" * 60)
print("SETTING UP DATALOADERS")
print("=" * 60)

# Convert sparse to dense for PyTorch
train_sparse_dense = train_sparse.toarray().astype(np.float32)
val_sparse_dense = val_sparse.toarray().astype(np.float32)
test_sparse_dense = test_sparse.toarray().astype(np.float32)

train_dense_t = torch.tensor(train_dense, dtype=torch.float32)
train_semantic_t = torch.tensor(train_semantic_emb, dtype=torch.float32)
train_sparse_t = torch.tensor(train_sparse_dense, dtype=torch.float32)
train_labels_t = torch.tensor(y_train_labels, dtype=torch.long)

val_dense_t = torch.tensor(val_dense, dtype=torch.float32)
val_semantic_t = torch.tensor(val_semantic_emb, dtype=torch.float32)
val_sparse_t = torch.tensor(val_sparse_dense, dtype=torch.float32)
val_labels_t = torch.tensor(y_val_labels, dtype=torch.long)

test_dense_t = torch.tensor(test_dense, dtype=torch.float32)
test_semantic_t = torch.tensor(test_semantic_emb, dtype=torch.float32)
test_sparse_t = torch.tensor(test_sparse_dense, dtype=torch.float32)

train_dataset = TensorDataset(train_dense_t, train_semantic_t, train_sparse_t, train_labels_t)
val_dataset = TensorDataset(val_dense_t, val_semantic_t, val_sparse_t, val_labels_t)
test_dataset = TensorDataset(test_dense_t, test_semantic_t, test_sparse_t)

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

print(
    f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches"
)

# ============================================================
# INITIALIZE MODEL
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING SIMPLE AUTHOR CLASSIFIER")
print("=" * 60)

model = SimpleAuthorClassifier(
    dense_input_dim=train_dense.shape[1],
    semantic_input_dim=train_semantic_emb.shape[1],
    sparse_input_dim=train_sparse_dense.shape[1],
    num_labels=NUM_AUTHORS,
    dropout=DROPOUT,
)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)

print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# METRIC FUNCTIONS
# ============================================================

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

def evaluate_model(model, loader, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            dense_feats = batch[0].to(device)
            sem_emb = batch[1].to(device)
            sparse_feats = batch[2].to(device)
            labels = batch[3].to(device)

            logits = model(dense_feats, sem_emb, sparse_feats)
            probs = torch.softmax(logits, dim=1)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    avg_loss = total_loss / max(num_batches, 1)

    return logloss, acc, all_preds, avg_loss

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        dense_feats = batch[0].to(device)
        sem_emb = batch[1].to(device)
        sparse_feats = batch[2].to(device)
        labels = batch[3].to(device)

        optimizer.zero_grad()

        logits = model(dense_feats, sem_emb, sparse_feats)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / max(num_batches, 1)

    val_logloss, val_acc, val_probs, val_loss = evaluate_model(
        model, val_loader, criterion
    )

    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(
            model.state_dict(), os.path.join(WORKING_DIR, "best_model.pt")
        )
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")

# ============================================================
# LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FINAL PREDICTIONS")
print("=" * 60)

model.load_state_dict(
    torch.load(os.path.join(WORKING_DIR, "best_model.pt"), map_location=device)
)

# Validation predictions
val_logloss, val_acc, val_probs, _ = evaluate_model(model, val_loader, criterion)
print(f"Best model validation log loss: {val_logloss:.4f}")

# Test predictions
model.eval()
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        dense_feats = batch[0].to(device)
        sem_emb = batch[1].to(device)
        sparse_feats = batch[2].to(device)
        logits = model(dense_feats, sem_emb, sparse_feats)
        probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)

# Apply probability clipping and normalization
eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

# ============================================================
# CREATE SUBMISSION FILE
# ============================================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

class_names = label_encoder.classes_

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        class_names[0]: test_probs[:, 0],
        class_names[1]: test_probs[:, 1],
        class_names[2]: test_probs[:, 2],
    }
)

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
final_score = val_logloss
print(f"Final Validation Score: {final_score:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
