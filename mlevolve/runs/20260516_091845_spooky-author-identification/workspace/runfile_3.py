import os
os.sched_setaffinity(0, {221})
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
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler
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
OUTPUT_CSV = "./submission/submission_7b6d04e4fd11481db4a43a1904361bd4.csv"
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

# --- Stylometric Features (30 dimensions) ---
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
        "when",
        "while",
        "where",
        "who",
        "which",
        "that",
        "this",
        "these",
        "those",
        "is",
        "was",
        "were",
        "are",
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
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
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
        "since",
    ]
)

ARCHAIC_WORDS = set(
    [
        "thou",
        "thee",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "dost",
        "canst",
        "shalt",
        "wilt",
        "art",
        "wert",
        "didst",
        "hadst",
        "hast",
        "wherein",
        "whereof",
        "whereon",
        "whereunto",
        "thereof",
        "therein",
        "thereon",
        "thereunto",
        "bethought",
        "methinks",
        "perchance",
        "forsooth",
        "alas",
        "anon",
        "ere",
        "hither",
        "hence",
        "thence",
        "thither",
        "whence",
        "whither",
        "wherefore",
        "hark",
        "list",
        "prithee",
        "wert",
        "wilt",
        "durst",
        "quoth",
    ]
)

LOVECRAFT_WORDS = set(
    [
        "eldritch",
        "cyclopean",
        "antediluvian",
        "squamous",
        "ichor",
        "foetid",
        "amorphous",
        "non",
        "euclidean",
        "gibbous",
        "blasphemous",
        "unspeakable",
        "nameless",
        "indescribable",
        "noisome",
        "festering",
        "miasmal",
        "fœtid",
        "cryptic",
        "spectral",
        "hideous",
        "monstrous",
        "cosmic",
        "chaotic",
        "daemoniac",
        "carcass",
        "pentagram",
        "necronomicon",
        "cthulhu",
        "yog",
        "sothoth",
        "shoggoth",
        "azathoth",
        "nyarlathotep",
        "yuggoth",
        "rlyeh",
    ]
)

EMOTIONAL_WORDS = set(
    [
        "fear",
        "dread",
        "terror",
        "horror",
        "afraid",
        "scared",
        "frightened",
        "terrified",
        "anxious",
        "despair",
        "hopeless",
        "gloomy",
        "somber",
        "melancholy",
        "mournful",
        "sorrow",
        "anguish",
        "agony",
        "torment",
        "suffering",
        "pain",
        "dark",
        "shadow",
        "gloom",
        "dismal",
        "dreadful",
        "awful",
        "hideous",
        "ghastly",
        "macabre",
        "grotesque",
        "unnatural",
        "weird",
        "eerie",
        "uncanny",
        "strange",
        "mysterious",
        "ominous",
        "portentous",
        "sinister",
        "menacing",
        "threatening",
        "forbidding",
    ]
)

def extract_stylometric_features(texts):
    """Extract 30 stylometric features per text"""
    n = len(texts)
    features = np.zeros((n, 30))

    for i, text in enumerate(texts):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        text_lower = text.lower()
        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]

        char_count = len(text)
        word_count = len(words)
        sent_count = max(len(sentences), 1)

        # Basic stats (0-4)
        features[i, 0] = char_count
        features[i, 1] = word_count
        features[i, 2] = sent_count
        features[i, 3] = char_count / word_count if word_count > 0 else 0
        features[i, 4] = char_count / sent_count if sent_count > 0 else 0

        # Character type ratios (5-8)
        if char_count > 0:
            upper_count = sum(1 for c in text if c.isupper())
            lower_count = sum(1 for c in text if c.islower())
            digit_count = sum(1 for c in text if c.isdigit())
            space_count = sum(1 for c in text if c.isspace())
            features[i, 5] = upper_count / char_count
            features[i, 6] = lower_count / char_count
            features[i, 7] = digit_count / char_count
            features[i, 8] = space_count / char_count

        # Punctuation ratios (9-20)
        all_punct = [".", ",", "!", "?", ":", ";", '"', "'", "-", "(", ")", "—"]
        for j, p in enumerate(all_punct):
            count = text.count(p)
            features[i, 9 + j] = count / char_count if char_count > 0 else 0

        # Vocabulary diversity (21)
        unique_words = set(words)
        features[i, 21] = len(unique_words) / max(len(words), 1)

        # Long words and capitalization (22-24)
        features[i, 22] = sum(1 for w in words if len(w) > 6) / max(word_count, 1)
        capitalized_ratio = sum(1 for w in words if w[0].isupper()) / max(word_count, 1)
        features[i, 23] = capitalized_ratio

        # Sentence-level variability (25-26)
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            features[i, 25] = np.std(sent_lengths)
            features[i, 26] = np.var(sent_lengths)
        else:
            features[i, 25] = 0
            features[i, 26] = 0

        # Stylistic content ratios (27-30)
        words_lower = [w.lower().strip(string.punctuation) for w in words]
        words_lower = [w for w in words_lower if len(w) > 0]

        if len(words_lower) > 0:
            function_word_count = sum(1 for w in words_lower if w in FUNCTION_WORDS)
            archaic_count = sum(1 for w in words_lower if w in ARCHAIC_WORDS)
            emotional_count = sum(1 for w in words_lower if w in EMOTIONAL_WORDS)
            lovecraft_count = sum(1 for w in words_lower if w in LOVECRAFT_WORDS)

            features[i, 27] = function_word_count / len(words_lower)
            features[i, 28] = archaic_count / len(words_lower)
            features[i, 29] = emotional_count / len(words_lower)
            features[i, 24] = lovecraft_count / len(words_lower)

    return features

def create_readability_features(texts):
    """Extract 4 readability features"""
    n = len(texts)
    features = np.zeros((n, 4))

    for i, text in enumerate(texts):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        word_count = len(words)
        sent_count = max(len(sentences), 1)

        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = "aeiouy"
            count = 0
            prev_vowel = False
            for char in word:
                if char in vowels:
                    if not prev_vowel:
                        count += 1
                        prev_vowel = True
                else:
                    prev_vowel = False
            return max(count, 1)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count if word_count > 0 else 0

        # Flesch Reading Ease
        features[i, 0] = (
            206.835 - 1.015 * (word_count / sent_count) - 84.6 * avg_syllables
        )
        # Automated Readability Index (ARI)
        char_count = sum(len(w) for w in words)
        features[i, 1] = (
            4.71 * (char_count / word_count) + 0.5 * (word_count / sent_count) - 21.43
        )
        # Average syllables
        features[i, 2] = avg_syllables
        # Complex word ratio (words with >= 3 syllables)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        features[i, 3] = complex_words / word_count if word_count > 0 else 0

    return features

def create_pos_tag_approximation(texts):
    """Extract 5 POS approximation features using suffix patterns"""
    n = len(texts)
    features = np.zeros((n, 5))

    for i, text in enumerate(texts):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        words = text.split()
        word_count = len(words)
        if word_count == 0:
            continue

        noun_suffixes = [
            "tion",
            "sion",
            "ment",
            "ness",
            "ity",
            "ance",
            "ence",
            "dom",
            "ist",
            "ism",
        ]
        verb_suffixes = [
            "ate",
            "ize",
            "ify",
            "en",
            "ish",
            "ure",
            "ed",
            "ing",
            "ize",
            "ise",
        ]
        adj_suffixes = [
            "ous",
            "eous",
            "ious",
            "al",
            "ial",
            "ical",
            "able",
            "ible",
            "ful",
            "less",
            "ive",
            "ative",
        ]
        adv_suffixes = ["ly", "ward", "wards", "wise"]

        lower_words = [w.lower() for w in words]

        noun_count = sum(
            1 for w in lower_words if any(w.endswith(suf) for suf in noun_suffixes)
        )
        verb_count = sum(
            1 for w in lower_words if any(w.endswith(suf) for suf in verb_suffixes)
        )
        adj_count = sum(
            1 for w in lower_words if any(w.endswith(suf) for suf in adj_suffixes)
        )
        adv_count = sum(
            1 for w in lower_words if any(w.endswith(suf) for suf in adv_suffixes)
        )

        features[i, 0] = noun_count / word_count
        features[i, 1] = verb_count / word_count
        features[i, 2] = adj_count / word_count
        features[i, 3] = adv_count / word_count

        content_count = sum(
            1 for w in lower_words if w.strip(string.punctuation) not in FUNCTION_WORDS
        )
        features[i, 4] = content_count / word_count

    return features

def extract_rhythm_features(texts):
    """Extract rhythm and structural patterns - 10 features"""
    n = len(texts)
    features = np.zeros((n, 10))

    for i, text in enumerate(texts):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        if len(sentences) == 0:
            continue

        # 0-2: Sentence start patterns
        starts = []
        for s in sentences:
            first_word = s.split()[0].lower() if s.split() else ""
            starts.append(first_word)

        conjunction_starts = [
            "and",
            "but",
            "or",
            "yet",
            "so",
            "for",
            "nor",
            "then",
            "however",
            "therefore",
        ]
        features[i, 0] = sum(1 for st in starts if st in conjunction_starts) / len(
            starts
        )
        features[i, 1] = sum(1 for st in starts if st == "the") / len(starts)
        pronoun_starts = ["he", "she", "it", "they", "we", "i", "you", "this", "that"]
        features[i, 2] = sum(1 for st in starts if st in pronoun_starts) / len(starts)

        # 3-4: Sentence length rhythm
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 3:
            diffs = [
                abs(sent_lengths[j] - sent_lengths[j - 1])
                for j in range(1, len(sent_lengths))
            ]
            features[i, 3] = np.mean(diffs) / max(sent_lengths) if sent_lengths else 0
            features[i, 4] = (
                np.std(diffs) / max(np.mean(diffs), 1) if len(diffs) > 1 else 0
            )

        # 5-6: Comma usage patterns
        words = text.split()
        total_commas = text.count(",")
        features[i, 5] = total_commas / len(sentences)
        features[i, 6] = total_commas / len(words) if len(words) > 0 else 0

        # 7: Question density
        features[i, 7] = text.count("?") / len(sentences)
        # 8: Exclamation density
        features[i, 8] = text.count("!") / len(sentences)
        # 9: Ellipsis usage
        features[i, 9] = text.count("...") / len(sentences)

    return features

def extract_sophistication_gradient(texts):
    """Extract features showing how vocabulary complexity changes through text - 5 features"""
    n = len(texts)
    features = np.zeros((n, 5))

    for i, text in enumerate(texts):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        words = text.split()
        if len(words) < 6:
            continue

        # Split into 5 segments
        segment_size = len(words) // 5
        segments = []
        for j in range(5):
            start = j * segment_size
            end = start + segment_size if j < 4 else len(words)
            segments.append(words[start:end])

        # Track avg word length across segments (complexity gradient)
        avg_lengths = []
        for seg in segments:
            if seg:
                avg_lengths.append(np.mean([len(w) for w in seg]))
            else:
                avg_lengths.append(0)

        # 0-1: Linear trend of complexity
        if len(avg_lengths) >= 3:
            features[i, 0] = np.polyfit(range(len(avg_lengths)), avg_lengths, 1)[0]
            features[i, 1] = np.std(avg_lengths) / max(np.mean(avg_lengths), 0.1)

        # 2-3: Rare word gradient (words > 8 chars)
        rare_ratios = []
        for seg in segments:
            if seg:
                rare_count = sum(1 for w in seg if len(w) > 8)
                rare_ratios.append(rare_count / len(seg))
            else:
                rare_ratios.append(0)

        if len(rare_ratios) >= 3:
            features[i, 2] = np.polyfit(range(len(rare_ratios)), rare_ratios, 1)[0]
            features[i, 3] = np.std(rare_ratios) / max(np.mean(rare_ratios), 0.01)

        # 4: Overall lexical richness variability
        window = min(100, len(words))
        if window > 10:
            type_token_ratios = []
            for j in range(0, len(words) - window + 1, window // 2):
                window_words = words[j : j + window]
                unique = len(set(w.lower() for w in window_words))
                type_token_ratios.append(unique / len(window_words))
            features[i, 4] = np.std(type_token_ratios) if type_token_ratios else 0

    return features

# Extract all handcrafted features
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

# New: rhythm and sophistication gradient features
print("  Extracting rhythm features...")
train_rhythm = extract_rhythm_features(X_train_texts)
val_rhythm = extract_rhythm_features(X_val_texts)
test_rhythm = extract_rhythm_features(test_df["text"].values)

rhythm_scaler = StandardScaler()
train_rhythm_scaled = rhythm_scaler.fit_transform(train_rhythm)
val_rhythm_scaled = rhythm_scaler.transform(val_rhythm)
test_rhythm_scaled = rhythm_scaler.transform(test_rhythm)

print("  Extracting sophistication gradient features...")
train_soph = extract_sophistication_gradient(X_train_texts)
val_soph = extract_sophistication_gradient(X_val_texts)
test_soph = extract_sophistication_gradient(test_df["text"].values)

soph_scaler = StandardScaler()
train_soph_scaled = soph_scaler.fit_transform(train_soph)
val_soph_scaled = soph_scaler.transform(val_soph)
test_soph_scaled = soph_scaler.transform(test_soph)

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
print(f"Sparse train shape (before chi2): {train_sparse.shape}")

# Chi-squared feature selection for n-grams
print("Applying chi-squared feature selection (k=10000)...")
chi2_scaler = MaxAbsScaler()
train_sparse_scaled = chi2_scaler.fit_transform(train_sparse)
val_sparse_scaled = chi2_scaler.transform(val_sparse)
test_sparse_scaled = chi2_scaler.transform(test_sparse)

chi2_selector = SelectKBest(chi2, k=10000)
train_sparse = chi2_selector.fit_transform(train_sparse_scaled, y_train_labels)
val_sparse = chi2_selector.transform(val_sparse_scaled)
test_sparse = chi2_selector.transform(test_sparse_scaled)
print(f"Sparse train shape (after chi2): {train_sparse.shape}")

# ============================================================
# DEBERTA FINE-TUNING WITH 3-FOLD CV AND STOCHASTIC DEPTH
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE WITH 3-FOLD CV")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
NUM_FOLDS = 3
NUM_EPOCHS_CV = 6
GRADIENT_ACCUMULATION_STEPS = 2
LR_CV = 1e-5

class StochasticDepthDeberta(nn.Module):
    """Wrapper to add stochastic depth to DeBERTa forward pass"""
    def __init__(self, base_model, drop_rate=0.2):
        super().__init__()
        self.base_model = base_model
        self.drop_rate = drop_rate
        # Count the number of transformer layers for stochastic depth
        if hasattr(base_model, 'deberta') and hasattr(base_model.deberta, 'encoder'):
            self.num_layers = len(base_model.deberta.encoder.layer)
        else:
            self.num_layers = 0
        self.dropout_layer = nn.Dropout(drop_rate)

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False):
        # Use the full model forward pass (handles embeddings, encoder, pooler, classifier)
        # We pass return_dict=True to get consistent outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract logits and hidden states
        logits = outputs.logits
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = None

        # Apply dropout to logits for regularization
        if self.training:
            logits = self.dropout_layer(logits)

        if output_hidden_states and hidden_states is not None:
            return logits, hidden_states

        if labels is not None:
            # Loss is already computed by base model if we passed labels
            loss = outputs.loss
            if loss is None:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
            return loss, logits
        return logits

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Prepare all training data
all_texts = train_df["text"].values
all_labels = y_train_full.copy()
n_total = len(all_texts)

# Encode test set once
TOKENIZERS_PARALLELISM = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

test_encodings = tokenizer(
    list(test_df["text"].values),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

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

# 3-fold CV training and embedding + probability extraction
oof_embeddings = np.zeros((n_total, 1024), dtype=np.float32)
test_embeddings_cv = np.zeros((len(test_df), 1024), dtype=np.float32)
oof_probas = np.zeros((n_total, NUM_AUTHORS), dtype=np.float32)
test_probas_cv = np.zeros((len(test_df), NUM_AUTHORS), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
    print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")

    # Split data for this fold
    fold_train_texts = all_texts[train_idx]
    fold_val_texts = all_texts[val_idx]
    fold_train_labels = all_labels[train_idx]
    fold_val_labels = all_labels[val_idx]

    # Tokenize for this fold
    train_encodings = tokenizer(
        list(fold_train_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        list(fold_val_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(fold_train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(fold_val_labels, dtype=torch.long),
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

    # Initialize model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )
    model.to(device)
    model = StochasticDepthDeberta(model, drop_rate=0.2)

    total_steps = len(train_loader) * NUM_EPOCHS_CV // GRADIENT_ACCUMULATION_STEPS
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR_CV, eps=1e-8)

    # Cosine scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS_CV):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast():
                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        all_val_preds = []
        all_val_labels_batch = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                with autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels_batch.append(labels.cpu().numpy())

        all_val_preds = np.vstack(all_val_preds)
        all_val_labels_batch = np.concatenate(all_val_labels_batch)
        val_loss = compute_log_loss(all_val_labels_batch, all_val_preds)
        val_acc = np.mean(np.argmax(all_val_preds, axis=1) == all_val_labels_batch)

        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS_CV} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Fold {fold+1} best val loss: {best_val_loss:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_state, strict=False)
    model.to(device)
    model.eval()

    # Extract OOF probabilities for val samples (for ensemble)
    val_loader_no_labels = DataLoader(
        TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    with torch.no_grad():
        all_oof_probas = []
        for batch in val_loader_no_labels:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)
            all_oof_probas.append(probs.cpu().numpy())
        oof_probas[val_idx] = np.vstack(all_oof_probas)

    # Extract OOF embeddings for val samples (for XGBoost)
    with torch.no_grad():
        all_oof_embeddings = []
        for batch in val_loader_no_labels:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                _, hidden_states = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            all_oof_embeddings.append(cls_embeddings)
        oof_embeddings[val_idx] = np.vstack(all_oof_embeddings)

    # Extract test probabilities for this fold (for ensemble)
    with torch.no_grad():
        all_test_probs = []
        all_test_embeddings = []
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)
                _, hidden_states = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            all_test_probs.append(probs.cpu().numpy())
            all_test_embeddings.append(cls_embeddings)
        test_probas_cv += np.vstack(all_test_probs) / NUM_FOLDS
        test_embeddings_cv += np.vstack(all_test_embeddings) / NUM_FOLDS

    # Clean up
    del model, best_model_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# The existing train/val split (X_train_texts, X_val_texts) already defines our split.
# We used the FIRST split (X_train_texts) as training set for handcrafted features.
# OOF embeddings from 3-fold CV align with the full dataset order (all_texts).
# We need to map back to the original indices of the FIRST split.
# Find which indices from all_texts correspond to X_train_texts and X_val_texts
# by matching text content.
train_orig_idx = np.where(np.isin(all_texts, X_train_texts))[0]
val_orig_idx = np.where(np.isin(all_texts, X_val_texts))[0]
# Ensure lengths match
assert len(train_orig_idx) == len(X_train_texts), f"Mismatch: {len(train_orig_idx)} vs {len(X_train_texts)}"
assert len(val_orig_idx) == len(X_val_texts), f"Mismatch: {len(val_orig_idx)} vs {len(X_val_texts)}"
# Extract embeddings for the original train/val split
train_embeddings = oof_embeddings[train_orig_idx]
val_embeddings = oof_embeddings[val_orig_idx]
test_embeddings = test_embeddings_cv

# Extract DeBERTa OOF probabilities for the original train/val split for ensemble
deberta_train_probs = oof_probas[train_orig_idx]
deberta_val_probs = oof_probas[val_orig_idx]
deberta_test_probs = test_probas_cv

print(f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
print(f"DeBERTa val probs shape: {deberta_val_probs.shape}")

# ============================================================
# XGBOOST (with all handcrafted features + embeddings)
# ============================================================
print("\nTraining XGBoost classifier...")
xgb_train_features = np.hstack(
    [
        train_stylo_filtered,
        train_read_scaled,
        train_pos_scaled,
        train_rhythm_scaled,
        train_soph_scaled,
        train_embeddings,
    ]
)
xgb_val_features = np.hstack(
    [
        val_stylo_filtered,
        val_read_scaled,
        val_pos_scaled,
        val_rhythm_scaled,
        val_soph_scaled,
        val_embeddings,
    ]
)
xgb_test_features = np.hstack(
    [
        test_stylo_filtered,
        test_read_scaled,
        test_pos_scaled,
        test_rhythm_scaled,
        test_soph_scaled,
        test_embeddings,
    ]
)
print(f"XGBoost train features: {xgb_train_features.shape}")

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
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
# DEBERTA VALIDATION & TEST PROBS (from OOF probabilities of 3-fold CV)
# ============================================================
print("\nGetting DeBERTa probabilities from OOF predictions of 3-fold CV...")

# We already have deberta_val_probs, deberta_test_probs from above
# Compute validation log loss on the OOF predictions that align with our val split
deberta_val_ll = compute_log_loss(y_val_labels, deberta_val_probs)
print(f"DeBERTa OOF validation log loss (val split): {deberta_val_ll:.4f}")

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION (without DeBERTa standalone probs)
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
    "deberta": deberta_val_probs,
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
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
if "deberta" in best_weights and best_weights["deberta"] > 0:
    test_probas["deberta"] = deberta_test_probs
    ensemble_test_probs = (
        best_weights["deberta"] * test_probas["deberta"]
        + best_weights["xgboost"] * test_probas["xgboost"]
        + best_weights["lr"] * test_probas["lr"]
    )
else:
    ensemble_test_probs = (
        best_weights["xgboost"] * test_probas["xgboost"]
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
score = best_ll
print(f"\nFinal Validation Score: {score:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
