"""
Merged Spooky Author Identification Pipeline
Multi-view Transformer with Contrastive Learning + Multi-Scale Stylistic Fusion
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DebertaV2Model,
    RobertaModel,
    AlbertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import string
from collections import Counter
import pickle

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_CLASSES = 3
DEBERTA_MODEL = "microsoft/deberta-v3-large"
ROBERTA_MODEL = "roberta-base"
ALBERT_MODEL = "albert-base-v2"
HIDDEN_DIM = 768
ROBERTA_HIDDEN = 768
ALBERT_HIDDEN = 768
FUSION_DIM = 512
DROPOUT_RATE = 0.3
TEMPERATURE = 0.1
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
MAX_EPOCHS = 40
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
PATIENCE = 5
MAX_LENGTH = 256
GRAD_CLIP_MAX_NORM = 1.0
CONTRASTIVE_ALPHA = 0.7

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors distribution:\n{train_df['author'].value_counts()}")

# Encode labels
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
print(
    f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
)

# ============================================================
# 2. STRATIFIED TRAIN/VALIDATION SPLIT
# ============================================================
print("\n" + "=" * 60)
print("CREATING STRATIFIED SPLIT")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, val_idx = next(
    skf.split(train_df["text"].values, train_df["author_encoded"].values)
)

train_texts = train_df["text"].values[train_idx]
val_texts = train_df["text"].values[val_idx]
train_labels = train_df["author_encoded"].values[train_idx]
val_labels = train_df["author_encoded"].values[val_idx]
test_texts = test_df["text"].values

assert len(set(train_idx) & set(val_idx)) == 0, "ERROR: Train/Val index overlap!"
print(
    f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)
print(f"Train label distribution: {np.bincount(train_labels)}")
print(f"Val label distribution: {np.bincount(val_labels)}")

# ============================================================
# 3. TEXT CLEANING FUNCTIONS
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_word_counts(text):
    w = str(text).lower().split()
    return len(w), len(set(w))

def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2:
        count += 1
    if count == 0:
        count += 1
    return count

# ============================================================
# 4. STYLOMETRIC FEATURE EXTRACTION
# ============================================================
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
        "so",
        "which",
        "that",
        "this",
        "those",
        "these",
        "what",
        "who",
        "whom",
        "whose",
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
        "almost",
        "enough",
        "even",
        "little",
        "much",
        "still",
        "thus",
        "there",
        "here",
        "hence",
        "thence",
        "then",
        "now",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "thereby",
        "therein",
        "thereupon",
        "am",
        "is",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "upon",
        "into",
        "within",
        "without",
        "through",
        "between",
        "among",
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "around",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "beyond",
        "by",
        "down",
        "during",
        "except",
        "for",
        "from",
        "in",
        "inside",
        "near",
        "of",
        "off",
        "on",
        "out",
        "outside",
        "over",
        "past",
        "through",
        "to",
        "toward",
        "under",
        "until",
        "up",
        "with",
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
        "whence",
        "thence",
        "hence",
        "wherefore",
        "thereof",
        "therein",
        "thereupon",
        "herein",
        "hereof",
        "hereupon",
        "aforesaid",
        "hither",
        "thither",
        "whither",
        "erst",
        "ere",
        "ne",
        "perchance",
        "methinks",
        "forsooth",
        "anon",
        "betwixt",
        "mayhap",
        "morrow",
        "nigh",
        "oft",
        "prithee",
        "twas",
        "wert",
        "wilt",
        "shalt",
        "canst",
        "dost",
        "doth",
        "art",
        "didst",
        "hast",
        "hadst",
        "wert",
        "couldst",
        "wouldst",
        "shouldst",
        "thyself",
        "yea",
        "nay",
        "wherewith",
        "albeit",
        "howbeit",
        "whereof",
        "whereon",
        "whereto",
        "wherein",
        "whereby",
        "whereupon",
        "whereat",
        "wherefrom",
        "whereinto",
        "whereunder",
        "wherewithal",
    ]
)

EMOTIONAL_WORDS = set(
    [
        "love",
        "hate",
        "fear",
        "dread",
        "horror",
        "terror",
        "anguish",
        "agony",
        "passion",
        "desire",
        "grief",
        "sorrow",
        "joy",
        "bliss",
        "rapture",
        "ecstasy",
        "woe",
        "despair",
        "hope",
        "yearning",
        "longing",
        "wonder",
        "awe",
        "reverence",
        "admiration",
        "pity",
        "compassion",
        "sympathy",
        "affection",
        "tenderness",
        "warmth",
        "coldness",
        "hatred",
        "loathing",
        "contempt",
        "scorn",
        "disdain",
        "envy",
        "jealousy",
        "wrath",
        "fury",
        "rage",
        "ire",
        "spleen",
        "melancholy",
        "gloom",
        "gloominess",
        "sadness",
        "dejection",
        "misery",
        "suffering",
        "pain",
        "torment",
        "torture",
        "distress",
        "anxiety",
        "apprehension",
        "alarm",
        "panic",
        "fright",
        "shock",
        "amazement",
        "astonishment",
        "surprise",
        "bewilderment",
        "confusion",
        "delight",
        "pleasure",
        "satisfaction",
        "contentment",
        "fulfillment",
    ]
)

LOVECRAFT_WORDS = set(
    [
        "eldritch",
        "cyclopean",
        "gaunt",
        "gibbous",
        "ichor",
        "antediluvian",
        "miasmal",
        "squamous",
        "rugose",
        "fungoid",
        "noisome",
        "madness",
        "abyss",
        "blasphemous",
        "cryptic",
        "daemoniac",
        "frenzy",
        "gargantuan",
        "hideous",
        "immemorial",
        "lurking",
        "monstrous",
        "nameless",
        "ominous",
        "primordial",
        "spectral",
        "uncanny",
        "unmentionable",
        "void",
        "cosmic",
        "charnel",
        "catacomb",
        "necronomicon",
        "cthulhu",
        "yog-sothoth",
        "azathoth",
        "nyarlathotep",
        "shoggoth",
        "miskatonic",
        "arkham",
        "kadath",
        "leng",
        "rlyeh",
        "ys",
        "hyperborean",
        "mu",
        "lemuria",
    ]
)

def extract_stylometric_features(texts):
    features = []
    for text in texts:
        text = clean_text(text)
        chars = len(text)
        w = text.split()
        word_count = len(w)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        sent_count = len(sentences) if sentences else 1

        avg_word_len = chars / word_count if word_count > 0 else 0
        avg_sent_len = len(w) / sent_count if sent_count > 0 else 0

        upper_ratio = sum(1 for c in text if c.isupper()) / chars if chars > 0 else 0
        lower_ratio = sum(1 for c in text if c.islower()) / chars if chars > 0 else 0
        digit_ratio = sum(1 for c in text if c.isdigit()) / chars if chars > 0 else 0
        whitespace_ratio = (
            sum(1 for c in text if c.isspace()) / chars if chars > 0 else 0
        )

        punct_counts = []
        for p in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "..."]:
            count = text.count(p)
            punct_counts.append(count / chars if chars > 0 else 0)

        unique_words = len(set(w_.lower() for w_ in w))
        type_token_ratio = unique_words / word_count if word_count > 0 else 0
        long_words_ratio = (
            sum(1 for w_ in w if len(w_) > 6) / word_count if word_count > 0 else 0
        )

        function_word_ratio = (
            sum(1 for w_ in w if w_.lower() in FUNCTION_WORDS) / word_count
            if word_count > 0
            else 0
        )
        archaic_ratio = (
            sum(1 for w_ in w if w_.lower() in ARCHAIC_WORDS) / word_count
            if word_count > 0
            else 0
        )
        emotional_ratio = (
            sum(1 for w_ in w if w_.lower() in EMOTIONAL_WORDS) / word_count
            if word_count > 0
            else 0
        )
        lovecraft_ratio = (
            sum(1 for w_ in w if w_.lower() in LOVECRAFT_WORDS) / word_count
            if word_count > 0
            else 0
        )

        feature_vector = [
            chars,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            *punct_counts,
            type_token_ratio,
            long_words_ratio,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
        ]
        features.append(feature_vector)
    return np.array(features)

print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(train_texts)
val_stylo = extract_stylometric_features(val_texts)
test_stylo = extract_stylometric_features(test_texts)
print(f"Stylometric features shape: {train_stylo.shape}")

feature_names = [
    "char_count",
    "word_count",
    "sent_count",
    "avg_word_len",
    "avg_sent_len",
    "upper_ratio",
    "lower_ratio",
    "digit_ratio",
    "whitespace_ratio",
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "-",
    '"',
    "'",
    "(",
    ")",
    "...",
    "type_token_ratio",
    "long_words_ratio",
    "func_word_ratio",
    "archaic_ratio",
    "emotional_ratio",
    "lovecraft_ratio",
]
print(f"Total: {len(feature_names)} stylometric features")

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

# ============================================================
# 5. READABILITY FEATURES
# ============================================================
def create_readability_features(texts):
    features = []
    for text in texts:
        text = clean_text(text)
        words = text.split()
        word_count = len(words)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        sent_count = len(sentences) if sentences else 1

        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count if word_count > 0 else 0

        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_word_ratio = complex_words / word_count if word_count > 0 else 0

        flesch = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * avg_syllables
        char_count = len(text)
        ari = 4.71 * (char_count / word_count) + 0.5 * (word_count / sent_count) - 21.43

        features.append([flesch, ari, avg_syllables, complex_word_ratio])
    return np.array(features)

print("\nExtracting readability features...")
train_read = create_readability_features(train_texts)
val_read = create_readability_features(val_texts)
test_read = create_readability_features(test_texts)
print(f"Readability features shape: {train_read.shape}")

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

# ============================================================
# 6. PART-OF-SPEECH APPROXIMATION FEATURES
# ============================================================
def create_pos_approximation_features(texts):
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
        "ure",
    ]
    verb_suffixes = ["ed", "ing", "ate", "ify", "ize", "ise", "en", "er", "est"]
    adj_suffixes = [
        "ous",
        "al",
        "ful",
        "less",
        "ive",
        "able",
        "ible",
        "ic",
        "ical",
        "ant",
        "ent",
        "ory",
        "ary",
    ]
    adv_suffixes = ["ly", "ward", "wards", "wise", "ways"]

    features = []
    for text in texts:
        text = clean_text(text)
        w = text.lower().split()
        word_count = len(w)
        if word_count == 0:
            features.append([0, 0, 0, 0, 0])
            continue

        noun_count = sum(1 for w_ in w if any(w_.endswith(s) for s in noun_suffixes))
        verb_count = sum(1 for w_ in w if any(w_.endswith(s) for s in verb_suffixes))
        adj_count = sum(1 for w_ in w if any(w_.endswith(s) for s in adj_suffixes))
        adv_count = sum(1 for w_ in w if any(w_.endswith(s) for s in adv_suffixes))
        content_words = sum(1 for w_ in w if w_ not in FUNCTION_WORDS)

        features.append(
            [
                noun_count / word_count,
                verb_count / word_count,
                adj_count / word_count,
                adv_count / word_count,
                content_words / word_count,
            ]
        )
    return np.array(features)

print("\nExtracting POS approximation features...")
train_pos = create_pos_approximation_features(train_texts)
val_pos = create_pos_approximation_features(val_texts)
test_pos = create_pos_approximation_features(test_texts)
print(f"POS features shape: {train_pos.shape}")

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# 7. N-GRAM FEATURES
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
train_char_short = char_vectorizer_short.fit_transform(train_texts)
val_char_short = char_vectorizer_short.transform(val_texts)
test_char_short = char_vectorizer_short.transform(test_texts)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(train_texts)
val_char_med = char_vectorizer_med.transform(val_texts)
test_char_med = char_vectorizer_med.transform(test_texts)

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
train_word = word_vectorizer.fit_transform(train_texts)
val_word = word_vectorizer.transform(val_texts)
test_word = word_vectorizer.transform(test_texts)

def extract_punctuation_sequence(text):
    return "".join(c for c in text if c in string.punctuation)

all_texts_for_punct = np.concatenate([train_texts, val_texts, test_texts])
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(train_texts)
n_val = len(val_texts)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + n_val]
test_punct = punct_features_all[n_train + n_val :]

print(f"Char n-gram (2-4): {train_char_short.shape[1]} features")
print(f"Char n-gram (4-6): {train_char_med.shape[1]} features")
print(f"Word n-gram (1-3): {train_word.shape[1]} features")
print(f"Punctuation patterns: {train_punct.shape[1]} features")

train_sparse = hstack(
    [train_char_short, train_char_med, train_word, train_punct]
).tocsr()
val_sparse = hstack([val_char_short, val_char_med, val_word, val_punct]).tocsr()
test_sparse = hstack([test_char_short, test_char_med, test_word, test_punct]).tocsr()
print(f"\nCombined sparse features: {train_sparse.shape}")

# ============================================================
# 8. CONCATENATE DENSE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("CONCATENATING DENSE FEATURES")
print("=" * 60)

train_dense = np.hstack([train_stylo_scaled, train_read_scaled, train_pos_scaled])
val_dense = np.hstack([val_stylo_scaled, val_read_scaled, val_pos_scaled])
test_dense = np.hstack([test_stylo_scaled, test_read_scaled, test_pos_scaled])

print(f"Handcrafted dense features: {train_dense.shape}")
print(f"Sparse features: {train_sparse.shape}")

# Save processed features
print("\n" + "=" * 60)
print("SAVING PROCESSED FEATURES")
print("=" * 60)

np.save("./working/train_dense.npy", train_dense)
np.save("./working/val_dense.npy", val_dense)
np.save("./working/test_dense.npy", test_dense)
np.save("./working/train_labels.npy", train_labels)
np.save("./working/val_labels.npy", val_labels)

with open("./working/stylo_scaler.pkl", "wb") as f:
    pickle.dump(stylo_scaler, f)
with open("./working/read_scaler.pkl", "wb") as f:
    pickle.dump(read_scaler, f)
with open("./working/pos_scaler.pkl", "wb") as f:
    pickle.dump(pos_scaler, f)
with open("./working/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✓ All features saved successfully!")

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MultiViewContrastiveAuthorClassifier(nn.Module):
    def __init__(self, stylometric_dim=39, num_classes=NUM_CLASSES):
        super().__init__()
        self.deberta_config = AutoConfig.from_pretrained(DEBERTA_MODEL)
        self.deberta_config.hidden_dropout_prob = DROPOUT_RATE
        self.deberta_config.attention_probs_dropout_prob = DROPOUT_RATE
        self.deberta = DebertaV2Model.from_pretrained(
            DEBERTA_MODEL, config=self.deberta_config
        )

        self.roberta_config = AutoConfig.from_pretrained(ROBERTA_MODEL)
        self.roberta_config.hidden_dropout_prob = DROPOUT_RATE
        self.roberta_config.attention_probs_dropout_prob = DROPOUT_RATE
        self.roberta = RobertaModel.from_pretrained(
            ROBERTA_MODEL, config=self.roberta_config
        )

        self.albert_config = AutoConfig.from_pretrained(ALBERT_MODEL)
        self.albert_config.hidden_dropout_prob = DROPOUT_RATE
        self.albert_config.attention_probs_dropout_prob = DROPOUT_RATE
        self.albert = AlbertModel.from_pretrained(
            ALBERT_MODEL, config=self.albert_config
        )

        self.deberta_proj = nn.Linear(self.deberta_config.hidden_size, FUSION_DIM)
        self.roberta_proj = nn.Linear(self.roberta_config.hidden_size, FUSION_DIM)
        self.albert_proj = nn.Linear(self.albert_config.hidden_size, FUSION_DIM)

        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=FUSION_DIM, num_heads=8, dropout=DROPOUT_RATE, batch_first=True
        )

        self.deberta_norm = nn.LayerNorm(FUSION_DIM)
        self.roberta_norm = nn.LayerNorm(FUSION_DIM)
        self.albert_norm = nn.LayerNorm(FUSION_DIM)

        self.stylometric_encoder = nn.Sequential(
            nn.Linear(stylometric_dim, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(FUSION_DIM * 2, 1024),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes),
        )

        self.contrastive_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        self.view_attention = nn.Parameter(torch.ones(3) / 3.0)
        self._init_weights()

    def _init_weights(self):
        for module in [self.deberta_proj, self.roberta_proj, self.albert_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in self.stylometric_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.feature_extractor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.contrastive_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids_deberta,
        attention_mask_deberta,
        input_ids_roberta,
        attention_mask_roberta,
        input_ids_albert,
        attention_mask_albert,
        stylometric_features=None,
        return_embeddings=False,
    ):
        debert_outputs = self.deberta(
            input_ids=input_ids_deberta, attention_mask=attention_mask_deberta
        )
        debert_emb = debert_outputs.last_hidden_state[:, 0, :]
        debert_proj = self.deberta_norm(self.deberta_proj(debert_emb))

        roberta_outputs = self.roberta(
            input_ids=input_ids_roberta, attention_mask=attention_mask_roberta
        )
        roberta_emb = roberta_outputs.last_hidden_state[:, 0, :]
        roberta_proj = self.roberta_norm(self.roberta_proj(roberta_emb))

        albert_outputs = self.albert(
            input_ids=input_ids_albert, attention_mask=attention_mask_albert
        )
        albert_emb = albert_outputs.last_hidden_state[:, 0, :]
        albert_proj = self.albert_norm(self.albert_proj(albert_emb))

        views_stacked = torch.stack([debert_proj, roberta_proj, albert_proj], dim=1)
        view_weights = F.softmax(self.view_attention, dim=0)
        views_weighted = views_stacked * view_weights.view(1, 3, 1)

        fused_output, attention_weights = self.attention_fusion(
            views_weighted, views_weighted, views_weighted
        )
        fused_representation = fused_output.mean(dim=1)

        if stylometric_features is not None:
            stylo_encoded = self.stylometric_encoder(stylometric_features)
            combined = torch.cat([fused_representation, stylo_encoded], dim=1)
        else:
            batch_size = fused_representation.size(0)
            stylo_zeros = torch.zeros(
                batch_size, FUSION_DIM, device=fused_representation.device
            )
            combined = torch.cat([fused_representation, stylo_zeros], dim=1)

        features = self.feature_extractor(combined)
        logits = self.classifier(features)

        if return_embeddings:
            contrastive_emb = self.contrastive_proj(features)
            contrastive_emb = F.normalize(contrastive_emb, dim=1, p=2)
            return logits, contrastive_emb
        return logits

class MultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, embeddings, labels):
        ce_loss = self.ce_loss(logits, labels)
        batch_size = embeddings.size(0)
        device = embeddings.device

        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels_expanded = labels.unsqueeze(1).expand(-1, batch_size)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask

        exp_similarity = torch.exp(similarity_matrix)
        sum_exp = exp_similarity.sum(
            dim=1, keepdim=True
        ) - exp_similarity.diag().unsqueeze(1)
        pos_exp = (exp_similarity * positive_mask).sum(dim=1)
        positive_count = positive_mask.sum(dim=1)
        valid_mask = positive_count > 0

        if valid_mask.sum() > 0:
            contrastive_loss = -torch.log(
                pos_exp[valid_mask] / (sum_exp[valid_mask].squeeze() + 1e-8)
            ).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        total_loss = self.alpha * ce_loss + (1 - self.alpha) * contrastive_loss
        return total_loss, ce_loss, contrastive_loss

# ============================================================
# TOKENIZATION
# ============================================================
print("\n" + "=" * 60)
print("TOKENIZING TEXT FOR MULTI-VIEW ARCHITECTURE")
print("=" * 60)

print("Loading tokenizers...")
deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
albert_tokenizer = AutoTokenizer.from_pretrained(ALBERT_MODEL)

def tokenize_texts(texts, tokenizer, max_length):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

print("Tokenizing training data...")
train_deberta = tokenize_texts(train_texts, deberta_tokenizer, MAX_LENGTH)
train_roberta = tokenize_texts(train_texts, roberta_tokenizer, MAX_LENGTH)
train_albert = tokenize_texts(train_texts, albert_tokenizer, MAX_LENGTH)

print("Tokenizing validation data...")
val_deberta = tokenize_texts(val_texts, deberta_tokenizer, MAX_LENGTH)
val_roberta = tokenize_texts(val_texts, roberta_tokenizer, MAX_LENGTH)
val_albert = tokenize_texts(val_texts, albert_tokenizer, MAX_LENGTH)

print("Tokenizing test data...")
test_deberta = tokenize_texts(test_texts, deberta_tokenizer, MAX_LENGTH)
test_roberta = tokenize_texts(test_texts, roberta_tokenizer, MAX_LENGTH)
test_albert = tokenize_texts(test_texts, albert_tokenizer, MAX_LENGTH)

print("Tokenization complete!")

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING MULTI-VIEW CONTRASTIVE MODEL")
print("=" * 60)

stylometric_dim = train_dense.shape[1]
model = MultiViewContrastiveAuthorClassifier(
    stylometric_dim=stylometric_dim, num_classes=NUM_CLASSES
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# SETUP DATALOADERS, OPTIMIZER, SCHEDULER, LOSS
# ============================================================
print("\n" + "=" * 60)
print("SETTING UP TRAINING COMPONENTS")
print("=" * 60)

train_dataset = TensorDataset(
    train_deberta["input_ids"],
    train_deberta["attention_mask"],
    train_roberta["input_ids"],
    train_roberta["attention_mask"],
    train_albert["input_ids"],
    train_albert["attention_mask"],
    torch.tensor(train_dense, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)

val_dataset = TensorDataset(
    val_deberta["input_ids"],
    val_deberta["attention_mask"],
    val_roberta["input_ids"],
    val_roberta["attention_mask"],
    val_albert["input_ids"],
    val_albert["attention_mask"],
    torch.tensor(val_dense, dtype=torch.float32),
    torch.tensor(val_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Differential learning rates - Use mutually exclusive parameter groups
# Pre-identify backbone parameters (from transformer models)
transformer_prefixes = ["deberta.", "roberta.", "albert."]
custom_layer_keywords = [
    "proj.",
    "attention_fusion.",
    "stylometric_encoder.",
    "feature_extractor.",
    "classifier.",
    "contrastive_proj.",
    "view_attention",
]

no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]

def is_transformer_param(name):
    return any(name.startswith(prefix) for prefix in transformer_prefixes)

def is_custom_param(name):
    return any(kw in name for kw in custom_layer_keywords)

# Collect all transformer backbone parameters (will use lower LR + weight_decay)
transformer_params_decay = []
transformer_params_no_decay = []
# Collect all custom/head parameters (will use higher LR)
custom_params_decay = []
custom_params_no_decay = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if is_custom_param(name):
        # Custom/head layers use higher LR
        if any(nd in name for nd in no_decay):
            custom_params_no_decay.append(param)
        else:
            custom_params_decay.append(param)
    elif is_transformer_param(name):
        # Transformer backbone layers use lower LR
        if any(nd in name for nd in no_decay):
            transformer_params_no_decay.append(param)
        else:
            transformer_params_decay.append(param)
    else:
        # Any remaining (shouldn't happen, but just in case) treat as custom
        if any(nd in name for nd in no_decay):
            custom_params_no_decay.append(param)
        else:
            custom_params_decay.append(param)

optimizer_grouped_parameters = [
    {
        "params": transformer_params_decay,
        "weight_decay": 0.01,
        "lr": LEARNING_RATE * 0.5,
    },
    {
        "params": transformer_params_no_decay,
        "weight_decay": 0.0,
        "lr": LEARNING_RATE * 0.5,
    },
    {
        "params": custom_params_decay,
        "weight_decay": 0.01,
        "lr": LEARNING_RATE,
    },
    {
        "params": custom_params_no_decay,
        "weight_decay": 0.0,
        "lr": LEARNING_RATE,
    },
]

# Safety check: ensure no parameter appears in multiple groups
all_params_in_groups = []
for group in optimizer_grouped_parameters:
    all_params_in_groups.extend(id(p) for p in group["params"])
if len(all_params_in_groups) != len(set(all_params_in_groups)):
    raise RuntimeError("ERROR: Some parameters appear in multiple groups!")
print(f"Optimizer groups: {len(optimizer_grouped_parameters)}")
print(f"Transf decay: {len(transformer_params_decay)}, Transf no_decay: {len(transformer_params_no_decay)}")
print(f"Custom decay: {len(custom_params_decay)}, Custom no_decay: {len(custom_params_no_decay)}")

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)
criterion = MultiViewContrastiveLoss(temperature=TEMPERATURE, alpha=CONTRASTIVE_ALPHA)
scaler = GradScaler() if torch.cuda.is_available() else None

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

def compute_log_loss(y_true, y_pred_proba, eps=1e-15):
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        loss -= np.log(y_pred_proba[i, y_true[i]])
    return loss / n

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
training_history = []

for epoch in range(MAX_EPOCHS):
    model.train()
    total_train_loss = 0.0
    total_ce_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        (
            deberta_ids,
            deberta_mask,
            roberta_ids,
            roberta_mask,
            albert_ids,
            albert_mask,
            stylo_feats,
            labels,
        ) = [x.to(device) for x in batch]
        with autocast():
            logits, embeddings = model(
                input_ids_deberta=deberta_ids,
                attention_mask_deberta=deberta_mask,
                input_ids_roberta=roberta_ids,
                attention_mask_roberta=roberta_mask,
                input_ids_albert=albert_ids,
                attention_mask_albert=albert_mask,
                stylometric_features=stylo_feats,
                return_embeddings=True,
            )
            loss, ce_loss, cont_loss = criterion(logits, embeddings, labels)
            loss = loss / ACCUMULATION_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_train_loss += loss.item() * ACCUMULATION_STEPS
        total_ce_loss += ce_loss.item()
        total_contrastive_loss += (
            cont_loss.item() if isinstance(cont_loss, torch.Tensor) else cont_loss
        )
        num_batches += 1

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Handle last partial accumulation
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / num_batches
    avg_ce = total_ce_loss / num_batches
    avg_cont = total_contrastive_loss / num_batches

    # Validation
    model.eval()
    all_val_probs = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            (
                deberta_ids,
                deberta_mask,
                roberta_ids,
                roberta_mask,
                albert_ids,
                albert_mask,
                stylo_feats,
                labels,
            ) = [x.to(device) for x in batch]
            with autocast():
                logits, embeddings = model(
                    input_ids_deberta=deberta_ids,
                    attention_mask_deberta=deberta_mask,
                    input_ids_roberta=roberta_ids,
                    attention_mask_roberta=roberta_mask,
                    input_ids_albert=albert_ids,
                    attention_mask_albert=albert_mask,
                    stylometric_features=stylo_feats,
                    return_embeddings=True,
                )
                probs = torch.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    val_probs = np.vstack(all_val_probs)
    val_labels_np = np.concatenate(all_val_labels)
    log_loss_val = compute_log_loss(val_labels_np, val_probs)
    val_acc = np.mean(np.argmax(val_probs, axis=1) == val_labels_np)

    print(
        f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} (CE: {avg_ce:.4f}, Cont: {avg_cont:.4f}) | Val Loss: {log_loss_val:.4f} | Val Acc: {val_acc:.4f}"
    )

    training_history.append(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_log_loss": log_loss_val,
            "val_acc": val_acc,
        }
    )

    if log_loss_val < best_val_loss:
        best_val_loss = log_loss_val
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_multi_view_model.pt")
        np.save("./working/val_probs_best.npy", val_probs)
        print(f"  ✓ New best model saved! Val Log Loss: {log_loss_val:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val log loss: {best_val_loss:.4f}"
            )
            break

print(
    f"\nTraining complete! Best epoch: {best_epoch}, Best validation log loss: {best_val_loss:.6f}"
)

# ============================================================
# LOAD BEST MODEL AND FINAL VALIDATION METRIC
# ============================================================
print("\n" + "=" * 60)
print("LOADING BEST MODEL FOR INFERENCE")
print("=" * 60)

state_dict = torch.load("./working/best_multi_view_model.pt", map_location=device)
model_state = model.state_dict()
filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(filtered, strict=False)
model.eval()

print("Computing final validation metric...")
val_probs_final = np.load("./working/val_probs_best.npy")
final_log_loss = compute_log_loss(val_labels_np, val_probs_final)
print(f"Final Validation Log Loss: {final_log_loss:.6f}")

# ============================================================
# TEST INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("RUNNING TEST INFERENCE")
print("=" * 60)

test_dataset = TensorDataset(
    test_deberta["input_ids"],
    test_deberta["attention_mask"],
    test_roberta["input_ids"],
    test_roberta["attention_mask"],
    test_albert["input_ids"],
    test_albert["attention_mask"],
    torch.tensor(test_dense, dtype=torch.float32),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        (
            deberta_ids,
            deberta_mask,
            roberta_ids,
            roberta_mask,
            albert_ids,
            albert_mask,
            stylo_feats,
        ) = [x.to(device) for x in batch]
        with autocast():
            logits = model(
                input_ids_deberta=deberta_ids,
                attention_mask_deberta=deberta_mask,
                input_ids_roberta=roberta_ids,
                attention_mask_roberta=roberta_mask,
                input_ids_albert=albert_ids,
                attention_mask_albert=albert_mask,
                stylometric_features=stylo_feats,
                return_embeddings=False,
            )
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)
eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

print(f"Test predictions shape: {test_probs.shape}")
print(
    f"Test prediction stats: Mean: {test_probs.mean(axis=0)}, Std: {test_probs.std(axis=0)}"
)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION FILE")
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

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# CLEANUP
# ============================================================
print("\n" + "=" * 60)
print("CLEANING UP")
print("=" * 60)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {final_log_loss:.6f}")