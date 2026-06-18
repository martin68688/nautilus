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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD

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

np.random.seed(RANDOM_STATE)

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
# STRATIFIED SPLIT (using indices directly to avoid INDEX_BUG)
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
# HANDCRAFTED FEATURES FUNCTIONS
# ============================================================
def extract_stylometric_features(texts):
    """Extract 45+ stylometric features from text."""
    features_list = []
    for text in texts:
        text = str(text)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        chars = list(text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(words) == 0:
            features_list.append([0] * 51)
            continue

        word_count = len(words)
        char_count = len(chars)
        sent_count = max(len(sentences), 1)
        avg_word_len = char_count / word_count if word_count > 0 else 0
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0

        # Character ratios
        upper_ratio = (
            sum(1 for c in chars if c.isupper()) / char_count if char_count > 0 else 0
        )
        lower_ratio = (
            sum(1 for c in chars if c.islower()) / char_count if char_count > 0 else 0
        )
        digit_ratio = (
            sum(1 for c in chars if c.isdigit()) / char_count if char_count > 0 else 0
        )
        whitespace_ratio = (
            sum(1 for c in chars if c.isspace()) / char_count if char_count > 0 else 0
        )

        # Punctuation ratios
        punct_marks = [".", ",", ";", ":", "!", "?", "-", '"', "'", "(", ")", "—"]
        punct_ratios = [
            text.count(p) / word_count if word_count > 0 else 0 for p in punct_marks
        ]

        # Word-level features
        word_set = set(w.lower() for w in words)
        char_diversity = len(word_set) / word_count if word_count > 0 else 0
        long_words = (
            sum(1 for w in words if len(w) > 6) / word_count if word_count > 0 else 0
        )
        capitalized = (
            sum(1 for i, w in enumerate(words) if w[0].isupper() and i > 0) / word_count
            if word_count > 0
            else 0
        )
        all_caps = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
            if word_count > 0
            else 0
        )

        # Sentence length variation
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 0 else 0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 0 else 0

        # Function words
        function_words = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "and",
            "but",
            "or",
            "if",
            "because",
            "although",
            "since",
            "while",
        }
        function_word_ratio = (
            sum(1 for w in words if w.lower() in function_words) / word_count
            if word_count > 0
            else 0
        )

        # Archaic words
        archaic_words = {
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "whence",
            "thence",
        }
        archaic_ratio = (
            sum(1 for w in words if w.lower() in archaic_words) / word_count
            if word_count > 0
            else 0
        )

        # Emotional words
        emotional_words = {
            "love",
            "hate",
            "fear",
            "joy",
            "despair",
            "hope",
            "dread",
            "horror",
            "terror",
            "sorrow",
        }
        emotional_ratio = (
            sum(1 for w in words if w.lower() in emotional_words) / word_count
            if word_count > 0
            else 0
        )

        # Lovecraft-specific words
        lovecraft_words = {
            "eldritch",
            "cyclopean",
            "cosmic",
            "unspeakable",
            "antediluvian",
            "cthulhu",
        }
        lovecraft_ratio = (
            sum(1 for w in words if w.lower() in lovecraft_words) / word_count
            if word_count > 0
            else 0
        )

        # Subordinate conjunctions
        sub_conjunctions = {
            "because",
            "since",
            "after",
            "although",
            "though",
            "while",
            "until",
            "unless",
            "as",
            "if",
            "before",
            "when",
            "where",
        }
        sub_conj_ratio = (
            sum(1 for w in words if w.lower() in sub_conjunctions) / word_count
            if word_count > 0
            else 0
        )

        # --- NEW FEATURES (15 more) ---
        # Sentence complexity: average number of clauses per sentence (approximated by commas + conjunctions)
        comma_count = text.count(",")
        and_count = len(re.findall(r'\b(?:and|but|or|so|for|nor|yet)\b', text.lower()))
        sentence_complexity = (comma_count + and_count) / sent_count if sent_count > 0 else 0

        # Hyphenation density
        hyphen_count = text.count("-")
        hyphenation_density = hyphen_count / word_count if word_count > 0 else 0

        # Quotation patterns
        single_quote_count = text.count("'")
        double_quote_count = text.count('"')
        quotation_density = (single_quote_count + double_quote_count) / char_count if char_count > 0 else 0

        # Capitalization entropy
        if char_count > 0:
            upper_count = sum(1 for c in chars if c.isupper())
            lower_count = sum(1 for c in chars if c.islower())
            other_count = char_count - upper_count - lower_count
            props = [upper_count/char_count, lower_count/char_count, other_count/char_count]
            props = [p for p in props if p > 0]
            cap_entropy = -sum(p * np.log2(p) for p in props)
        else:
            cap_entropy = 0.0

        # Word length percentiles (10th, 25th, 75th, 90th)
        word_lengths = [len(w) for w in words]
        if len(word_lengths) >= 4:
            p10 = np.percentile(word_lengths, 10)
            p25 = np.percentile(word_lengths, 25)
            p75 = np.percentile(word_lengths, 75)
            p90 = np.percentile(word_lengths, 90)
        else:
            p10 = p25 = p75 = p90 = 0.0

        # Hapax legomena (words occurring exactly once)
        word_freq = {}
        for w in words:
            wl = w.lower()
            word_freq[wl] = word_freq.get(wl, 0) + 1
        hapax_count = sum(1 for v in word_freq.values() if v == 1)
        hapax_ratio = hapax_count / word_count if word_count > 0 else 0

        # Type-token ratio
        type_token_ratio = len(word_set) / word_count if word_count > 0 else 0

        # Punctuation n-gram entropy (length 5-7)
        punct_seq = "".join([c for c in text if c in string.punctuation])
        punct_ngram_counts = {}
        for n in range(5, 8):
            for i in range(len(punct_seq) - n + 1):
                ng = punct_seq[i:i+n]
                punct_ngram_counts[ng] = punct_ngram_counts.get(ng, 0) + 1
        total_punct_ngrams = sum(punct_ngram_counts.values())
        if total_punct_ngrams > 0:
            punct_ngram_entropy = -sum((c/total_punct_ngrams) * np.log2(c/total_punct_ngrams) for c in punct_ngram_counts.values())
        else:
            punct_ngram_entropy = 0.0

        features = [
            char_count,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            *punct_ratios,
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
            # New features
            sentence_complexity,
            hyphenation_density,
            quotation_density,
            cap_entropy,
            p10,
            p25,
            p75,
            p90,
            hapax_ratio,
            type_token_ratio,
            punct_ngram_entropy,
        ]
        features_list.append(features)

    return np.array(features_list, dtype=np.float32)


def create_readability_features(texts):
    """Extract 9 readability features (4 existing + 5 new)."""
    features_list = []
    for text in texts:
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(words) == 0 or len(sentences) == 0:
            features_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
            continue

        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            if word.endswith("e") and len(word) > 2:
                word = word[:-1]
            vowels = "aeiou"
            count = 0
            prev_vowel = False
            for char in word:
                if char in vowels and not prev_vowel:
                    count += 1
                    prev_vowel = True
                else:
                    prev_vowel = False
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        total_chars = sum(len(w) for w in words)

        flesch = (
            206.835
            - 1.015 * (len(words) / len(sentences))
            - 84.6 * (total_syllables / len(words))
        )
        ari = (
            4.71 * (total_chars / len(words))
            + 0.5 * (len(words) / len(sentences))
            - 21.43
        )
        avg_syllables = total_syllables / len(words) if len(words) > 0 else 0
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / len(words) if len(words) > 0 else 0

        # New readability scores
        # Coleman-Liau Index
        L = (total_chars / len(words)) * 100 if len(words) > 0 else 0  # letters per 100 words
        S = (len(sentences) / len(words)) * 100 if len(words) > 0 else 0  # sentences per 100 words
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8

        # SMOG Index
        smog = 1.0430 * (complex_words * 30 / len(sentences))**0.5 + 3.1291 if len(sentences) > 0 else 0

        # Gunning-Fog Index
        fog = 0.4 * ((len(words) / len(sentences)) + 100 * (complex_words / len(words))) if len(sentences) > 0 else 0

        # Automated Readability Index (already have ARI from above, but compute separately)
        # Already have ARI = ari

        # LIX (Swedish readability index)
        long_words_lix = sum(1 for w in words if len(w) > 6)
        lix = (len(words) / len(sentences)) + (100 * long_words_lix / len(words)) if len(sentences) > 0 else 0

        features_list.append([flesch, ari, avg_syllables, complex_ratio, coleman_liau, smog, fog, ari, lix])

    return np.array(features_list, dtype=np.float32)


def create_pos_tag_approximation(texts):
    """Extract 15 POS approximation features (5 existing + 10 new)."""
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
        "ism",
    ]
    verb_suffixes = ["ed", "ing", "en", "ize", "ate", "ify", "es", "s", "th"]
    adj_suffixes = ["able", "ible", "al", "ful", "ic", "ive", "ous", "less", "y", "ish"]
    adv_suffixes = ["ly", "ward", "wise"]
    function_words = {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "and",
        "but",
        "or",
        "if",
        "because",
        "although",
        "since",
        "while",
    }

    # Verb tense markers
    past_tense_markers = ["ed", "t", "en"]
    present_tense_markers = ["s", "es", "ing"]
    future_tense_markers = ["will", "shall", "would"]

    # Adjective intensity markers
    intensity_markers = ["very", "quite", "extremely", "rather", "pretty", "somewhat", "fairly"]

    features_list = []
    for text in texts:
        text = str(text)
        words = text.split()

        if len(words) == 0:
            features_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            continue

        total = len(words)

        def count_suffix_matches(suffixes):
            count = 0
            for word in words:
                w = word.lower()
                for suffix in suffixes:
                    if w.endswith(suffix):
                        count += 1
                        break
            return count

        noun_ratio = count_suffix_matches(noun_suffixes) / total if total > 0 else 0
        verb_ratio = count_suffix_matches(verb_suffixes) / total if total > 0 else 0
        adj_ratio = count_suffix_matches(adj_suffixes) / total if total > 0 else 0
        adv_ratio = count_suffix_matches(adv_suffixes) / total if total > 0 else 0
        function_count = sum(1 for w in words if w.lower() in function_words)
        content_ratio = 1.0 - (function_count / total) if total > 0 else 0

        # New: Verb tense markers
        past_tense_count = count_suffix_matches(past_tense_markers)
        present_tense_count = count_suffix_matches(present_tense_markers)
        past_tense_ratio = past_tense_count / total if total > 0 else 0
        present_tense_ratio = present_tense_count / total if total > 0 else 0
        future_tense_count = sum(1 for w in words if w.lower() in future_tense_markers)
        future_tense_ratio = future_tense_count / total if total > 0 else 0

        # New: Adjective intensity markers
        intensity_count = sum(1 for w in words if w.lower() in intensity_markers)
        intensity_ratio = intensity_count / total if total > 0 else 0

        # New: Adverb density (already have adv_ratio from suffix matching, but add more specific adverb patterns)
        adverb_patterns = ["ly", "ward", "wise", "where", "here", "there"]
        adverb_count = count_suffix_matches(adverb_patterns)
        adverb_density = adverb_count / total if total > 0 else 0

        features_list.append(
            [
                noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_ratio,
                past_tense_ratio, present_tense_ratio, future_tense_ratio,
                intensity_ratio, adverb_density
            ]
        )

    return np.array(features_list, dtype=np.float32)


# ============================================================
# HANDCRAFTED FEATURES EXTRACTION
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING HANDCRAFTED FEATURES")
print("=" * 60)

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

print(f"Stylometric features: {train_stylo_filtered.shape[1]}")
print(f"Readability features: {train_read_scaled.shape[1]}")
print(f"POS features: {train_pos_scaled.shape[1]}")


# ============================================================
# CHARACTER & WORD N-GRAM + PUNCTUATION FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING N-GRAM FEATURES")
print("=" * 60)

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


train_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_train_texts]
val_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_val_texts]
test_punct_sequences = [extract_punctuation_sequence(str(t)) for t in test_df["text"].values]

punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
train_punct = punct_vectorizer.fit_transform(train_punct_sequences)
val_punct = punct_vectorizer.transform(val_punct_sequences)
test_punct = punct_vectorizer.transform(test_punct_sequences)

# New: Additional n-gram configurations at different max_features
char_vectorizer_1 = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=2000, sublinear_tf=True)
train_char_1 = char_vectorizer_1.fit_transform(X_train_texts)
val_char_1 = char_vectorizer_1.transform(X_val_texts)
test_char_1 = char_vectorizer_1.transform(test_df["text"].values)

char_vectorizer_2 = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), max_features=4000, sublinear_tf=True)
train_char_2 = char_vectorizer_2.fit_transform(X_train_texts)
val_char_2 = char_vectorizer_2.transform(X_val_texts)
test_char_2 = char_vectorizer_2.transform(test_df["text"].values)

char_vectorizer_3 = TfidfVectorizer(analyzer="char", ngram_range=(4, 8), max_features=6000, sublinear_tf=True)
train_char_3 = char_vectorizer_3.fit_transform(X_train_texts)
val_char_3 = char_vectorizer_3.transform(X_val_texts)
test_char_3 = char_vectorizer_3.transform(test_df["text"].values)

word_vectorizer_2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=2000, sublinear_tf=True, min_df=2)
train_word_2 = word_vectorizer_2.fit_transform(X_train_texts)
val_word_2 = word_vectorizer_2.transform(X_val_texts)
test_word_2 = word_vectorizer_2.transform(test_df["text"].values)

word_vectorizer_3 = TfidfVectorizer(analyzer="word", ngram_range=(2, 4), max_features=4000, sublinear_tf=True, min_df=2)
train_word_3 = word_vectorizer_3.fit_transform(X_train_texts)
val_word_3 = word_vectorizer_3.transform(X_val_texts)
test_word_3 = word_vectorizer_3.transform(test_df["text"].values)

word_vectorizer_4 = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), max_features=6000, sublinear_tf=True, min_df=2)
train_word_4 = word_vectorizer_4.fit_transform(X_train_texts)
val_word_4 = word_vectorizer_4.transform(X_val_texts)
test_word_4 = word_vectorizer_4.transform(test_df["text"].values)

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct,
     train_char_1, train_char_2, train_char_3, train_word_2, train_word_3, train_word_4]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct,
     val_char_1, val_char_2, val_char_3, val_word_2, val_word_3, val_word_4]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct,
     test_char_1, test_char_2, test_char_3, test_word_2, test_word_3, test_word_4]
).tocsr()
print(f"Sparse train shape: {train_sparse.shape}")

# ============================================================
# LOG LOSS FUNCTION
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

# ============================================================
# DENSE FEATURES COMBINATION (handcrafted + sparse dim reduction)
# ============================================================
print("\n" + "=" * 60)
print("COMBINING FEATURES FOR TREE MODELS")
print("=" * 60)

# Use TF-IDF sparse features directly without DeBERTa embeddings
# Reduce sparse dimensionality with SVD for tree models
from sklearn.decomposition import TruncatedSVD

print("Performing SVD on sparse features...")
svd = TruncatedSVD(n_components=300, random_state=RANDOM_STATE)
train_sparse_svd = svd.fit_transform(train_sparse)
val_sparse_svd = svd.transform(val_sparse)
test_sparse_svd = svd.transform(test_sparse)
print(f"SVD train shape: {train_sparse_svd.shape}")

# Combine dense features
train_dense = np.hstack([train_stylo_filtered, train_read_scaled, train_pos_scaled, train_sparse_svd])
val_dense = np.hstack([val_stylo_filtered, val_read_scaled, val_pos_scaled, val_sparse_svd])
test_dense = np.hstack([test_stylo_filtered, test_read_scaled, test_pos_scaled, test_sparse_svd])
print(f"Dense train features: {train_dense.shape}")

# ============================================================
# XGBOOST - DEEP MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST DEEP")
print("=" * 60)

xgb_deep = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=12,
    learning_rate=0.03,
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
xgb_deep.fit(
    train_dense,
    y_train_labels,
    eval_set=[(val_dense, y_val_labels)],
    verbose=False,
)

xgb_deep_val_probs = xgb_deep.predict_proba(val_dense)
xgb_deep_test_probs = xgb_deep.predict_proba(test_dense)
print(
    f"XGBoost Deep validation log loss: {compute_log_loss(y_val_labels, xgb_deep_val_probs):.4f}"
)

# ============================================================
# XGBOOST - SHALLOW MODEL (regularized)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST SHALLOW")
print("=" * 60)

xgb_shallow = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=1.0,
    reg_lambda=2.0,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_shallow.fit(
    train_sparse_svd,
    y_train_labels,
    eval_set=[(val_sparse_svd, y_val_labels)],
    verbose=False,
)

xgb_shallow_val_probs = xgb_shallow.predict_proba(val_sparse_svd)
xgb_shallow_test_probs = xgb_shallow.predict_proba(test_sparse_svd)
print(
    f"XGBoost Shallow validation log loss: {compute_log_loss(y_val_labels, xgb_shallow_val_probs):.4f}"
)

# ============================================================
# XGBOOST - CHARACTER N-GRAM SPECIALIZED
# ============================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST CHARACTER N-GRAM SPECIALIZED")
print("=" * 60)

# For character n-gram specialized, use only character n-gram features
char_ngram_features = hstack([
    train_char_short, train_char_med, train_char_long,
    train_char_1, train_char_2, train_char_3,
    train_punct
]).tocsr()
char_ngram_features_val = hstack([
    val_char_short, val_char_med, val_char_long,
    val_char_1, val_char_2, val_char_3,
    val_punct
]).tocsr()
char_ngram_features_test = hstack([
    test_char_short, test_char_med, test_char_long,
    test_char_1, test_char_2, test_char_3,
    test_punct
]).tocsr()

svd_char = TruncatedSVD(n_components=200, random_state=RANDOM_STATE)
train_char_svd = svd_char.fit_transform(char_ngram_features)
val_char_svd = svd_char.transform(char_ngram_features_val)
test_char_svd = svd_char.transform(char_ngram_features_test)

xgb_char = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_char.fit(
    train_char_svd,
    y_train_labels,
    eval_set=[(val_char_svd, y_val_labels)],
    verbose=False,
)

xgb_char_val_probs = xgb_char.predict_proba(val_char_svd)
xgb_char_test_probs = xgb_char.predict_proba(test_char_svd)
print(
    f"XGBoost Char-Specialized validation log loss: {compute_log_loss(y_val_labels, xgb_char_val_probs):.4f}"
)

# ============================================================
# LIGHTGBM
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    num_leaves=40,
    learning_rate=0.03,
    n_estimators=1000,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=5,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="multiclass",
    num_class=NUM_AUTHORS,
    metric="multi_logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    train_dense,
    y_train_labels,
    eval_set=[(val_dense, y_val_labels)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)

lgb_val_probs = lgb_model.predict_proba(val_dense)
lgb_test_probs = lgb_model.predict_proba(test_dense)
print(
    f"LightGBM validation log loss: {compute_log_loss(y_val_labels, lgb_val_probs):.4f}"
)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

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
# ENSEMBLE WEIGHT OPTIMIZATION (grid search over weights)
# ============================================================
print("\n" + "=" * 60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("=" * 60)

val_probas = {
    "xgb_deep": xgb_deep_val_probs,
    "xgb_shallow": xgb_shallow_val_probs,
    "xgb_char": xgb_char_val_probs,
    "lgb": lgb_val_probs,
    "lr": lr_val_probs,
}

best_ll = float("inf")
best_weights = None

# Grid search over weights with 0.05 step
for w1 in np.arange(0.05, 0.5, 0.05):
    for w2 in np.arange(0.05, 0.5, 0.05):
        for w3 in np.arange(0.05, 0.5, 0.05):
            for w4 in np.arange(0.05, 0.5, 0.05):
                w5 = 1.0 - w1 - w2 - w3 - w4
                if w5 < 0.05 or w5 > 0.8:
                    continue
                ensemble_proba = (
                    w1 * val_probas["xgb_deep"]
                    + w2 * val_probas["xgb_shallow"]
                    + w3 * val_probas["xgb_char"]
                    + w4 * val_probas["lgb"]
                    + w5 * val_probas["lr"]
                )
                ll = compute_log_loss(y_val_labels, ensemble_proba)
                if ll < best_ll:
                    best_ll = ll
                    best_weights = {
                        "xgb_deep": w1,
                        "xgb_shallow": w2,
                        "xgb_char": w3,
                        "lgb": w4,
                        "lr": w5,
                    }

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

test_probas = {
    "xgb_deep": xgb_deep_test_probs,
    "xgb_shallow": xgb_shallow_test_probs,
    "xgb_char": xgb_char_test_probs,
    "lgb": lgb_test_probs,
    "lr": lr_test_probs,
}

ensemble_test_probs = (
    best_weights["xgb_deep"] * test_probas["xgb_deep"]
    + best_weights["xgb_shallow"] * test_probas["xgb_shallow"]
    + best_weights["xgb_char"] * test_probas["xgb_char"]
    + best_weights["lgb"] * test_probas["lgb"]
    + best_weights["lr"] * test_probas["lr"]
)

eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

# Use the correct submission format from sample_submission.csv
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