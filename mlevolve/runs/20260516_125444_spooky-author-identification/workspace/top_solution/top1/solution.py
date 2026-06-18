"""
Merged solution: DeBERTa-v3-large fine-tuning + Handcrafted features + XGBoost + Logistic Regression + Ensemble
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
TEST_SIZE = 0.1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("./working", exist_ok=True)
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
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT — CRITICAL TO AVOID INDEX_BUG
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
assert len(set(train_idx) & set(val_idx)) == 0, "CRITICAL: Data leakage detected!"

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)
print(f"Train distribution: {np.bincount(y_train_labels)}")
print(f"Val distribution: {np.bincount(y_val_labels)}")

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================

def extract_stylometric_features(texts):
    archaic_words = set(
        [
            "thee",
            "thou",
            "thy",
            "thine",
            "hath",
            "doth",
            "wilt",
            "shalt",
            "canst",
            "didst",
            "art",
            "wert",
            "hence",
            "thence",
            "whence",
            "ere",
            "betwixt",
            "unto",
            "nay",
            "yea",
            "oft",
            "perchance",
            "methinks",
            "forsooth",
            "hark",
            "alas",
            "twas",
            "tis",
            "twill",
            "wherefore",
            "therein",
            "thereof",
            "thereupon",
            "herein",
            "thusly",
            "thenceforth",
            "wherewith",
            "whereunto",
            "aught",
            "naught",
            "ne'er",
            "o'er",
            "e'er",
            "dost",
            "doth",
            "hast",
            "saith",
            "dwell",
            "dwelt",
            "beheld",
            "beseech",
            "bethink",
            "betook",
            "bade",
            "borne",
            "clad",
            "durst",
            "gat",
            "hark",
            "hearken",
            "hie",
            "hither",
            "kine",
            "lest",
            "methought",
            "morn",
            "morrow",
            "nigh",
            "noble",
            "prithee",
            "slay",
            "slew",
            "slain",
            "smite",
            "smote",
            "smitten",
            "spake",
            "sped",
            "strode",
            "stricken",
            "sunder",
            "thrice",
            "throng",
            "trod",
            "trodden",
            "verily",
            "whence",
            "wheresoever",
            "wherewithal",
            "wight",
            "wist",
            "wot",
            "wrought",
            "ye",
            "yon",
            "yonder",
        ]
    )
    emotional_words = set(
        [
            "fear",
            "terror",
            "horror",
            "dread",
            "anguish",
            "agony",
            "despair",
            "grief",
            "sorrow",
            "woe",
            "mournful",
            "melancholy",
            "somber",
            "gloomy",
            "dreary",
            "bleak",
            "desolate",
            "forlorn",
            "hapless",
            "wretched",
            "miserable",
            "dismal",
            "solemn",
            "macabre",
            "ghastly",
            "hideous",
            "gruesome",
            "grotesque",
            "uncanny",
            "eerie",
            "weird",
            "strange",
            "mysterious",
            "supernatural",
            "uncarthly",
            "phantom",
            "spectral",
            "ghostly",
            "shadowy",
            "darkness",
            "obscure",
            "abyss",
            "chasm",
            "void",
            "infinite",
            "eternal",
            "ancient",
            "primeval",
            "terrifying",
            "appalling",
            "shocking",
            "startling",
            "sudden",
            "violent",
            "furious",
            "raging",
            "tempest",
            "turmoil",
            "chaos",
            "lament",
            "weep",
            "cry",
            "sob",
            "groan",
            "sigh",
            "moan",
            "shriek",
            "scream",
            "howl",
            "wail",
            "implore",
            "beseech",
            "pray",
            "supplicate",
            "entreat",
            "beg",
            "plead",
            "mortal",
            "mortality",
            "death",
            "dying",
            "deadly",
            "fatal",
        ]
    )
    lovecraft_words = set(
        [
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "nyarlathotep",
            "azathoth",
            "shub-niggurath",
            "yuggoth",
            "kadath",
            "leng",
            "innsmouth",
            "arkham",
            "dunwich",
            "miskatonic",
            "providence",
            "necronomicon",
            "cyclopean",
            "eldritch",
            "unmentionable",
            "indescribable",
            "unnamable",
            "non-euclidean",
            "antiquarian",
            "unfathomable",
            "ineffable",
            "blasphemous",
            "abominable",
            "ancient",
            "primordial",
            "immemorial",
            "alien",
            "cosmic",
            "ultimate",
            "infinite",
            "boundless",
            "dimensional",
            "otherworldly",
            "outre",
            "preternatural",
            "accursed",
            "nameless",
            "forbidden",
            "unspeakable",
            "unutterable",
            "night-gaunt",
            "shoggoth",
            "mi-go",
            "great old ones",
            "elder things",
            "gugs",
            "hounds of tindalos",
            "incantation",
            "evocation",
            "summon",
            "conjure",
            "portal",
            "monolith",
            "cryptic",
            "hieroglyphic",
            "cuneiform",
            "abyss",
            "star-spawned",
            "void",
            "immeasurable",
            "aeon",
            "blackness",
            "unguessed",
            "outer",
            "nightmare",
            "frenzied",
            "loathsome",
            "crawling",
            "tentacle",
            "mantis",
            "moldy",
            "vaulted",
            "subterrene",
            "gibbous",
            "moon-cursed",
            "bizarre",
            "hysterical",
            "maniacal",
            "grotesque",
            "festering",
            "putrid",
            "decadent",
            "cesspool",
        ]
    )
    sub_conjunctions = set(
        [
            "although",
            "though",
            "while",
            "whereas",
            "because",
            "since",
            "as",
            "if",
            "unless",
            "until",
            "after",
            "before",
            "when",
            "whenever",
            "where",
            "wherever",
            "that",
            "than",
            "so that",
            "in order that",
            "provided that",
            "even though",
            "even if",
            "rather than",
            "whether",
            "as if",
            "as though",
            "such that",
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
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
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
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "many",
            "some",
            "any",
            "no",
            "not",
            "only",
            "very",
            "just",
            "too",
            "so",
            "as",
            "than",
            "more",
            "most",
            "less",
            "least",
        ]
    )

    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0] * 30)
            continue
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        n_chars = len(text)
        n_words = len(words)
        n_sentences = len(sentences)
        avg_word_len = np.mean([len(w) for w in words]) if n_words > 0 else 0
        avg_sent_len = (
            np.mean([len(s.split()) for s in sentences]) if n_sentences > 0 else 0
        )

        n_upper = sum(1 for c in text if c.isupper())
        n_lower = sum(1 for c in text if c.islower())
        n_digits = sum(1 for c in text if c.isdigit())
        n_spaces = sum(1 for c in text if c.isspace())

        punct_counts = [text.count(p) for p in string.punctuation]
        total_punct = sum(punct_counts)

        char_diversity = len(set(text.lower())) / max(n_chars, 1)
        long_words = sum(1 for w in words if len(w) > 6)
        capitalized = sum(1 for w in words if w[0].isupper() and len(w) > 1)
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)

        sent_lens = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lens) if len(sent_lens) > 0 else 0
        sent_len_var = np.var(sent_lens) if len(sent_lens) > 0 else 0

        words_lower = [w.lower() for w in words]
        function_word_count = sum(1 for w in words_lower if w in function_words)
        function_word_ratio = function_word_count / max(n_words, 1)
        archaic_count = sum(1 for w in words_lower if w in archaic_words)
        archaic_ratio = archaic_count / max(n_words, 1)
        emotional_count = sum(1 for w in words_lower if w in emotional_words)
        emotional_ratio = emotional_count / max(n_words, 1)
        lovecraft_count = sum(1 for w in words_lower if w in lovecraft_words)
        lovecraft_ratio = lovecraft_count / max(n_words, 1)
        sub_conj_count = sum(1 for w in words_lower if w in sub_conjunctions)
        sub_conj_ratio = sub_conj_count / max(n_words, 1)

        feat = [
            n_chars,
            n_words,
            n_sentences,
            avg_word_len,
            avg_sent_len,
            n_upper / max(n_chars, 1),
            n_lower / max(n_chars, 1),
            n_digits / max(n_chars, 1),
            n_spaces / max(n_chars, 1),
            total_punct / max(n_chars, 1),
            n_chars / max(n_sentences, 1),
            n_words / max(n_sentences, 1),
            char_diversity,
            long_words / max(n_words, 1),
            capitalized / max(n_words, 1),
            all_caps / max(n_words, 1),
            sent_len_std,
            sent_len_var,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            sub_conj_ratio,
        ]
        punct_ratios = [p / max(total_punct, 1) for p in punct_counts[:12]]
        feat.extend(punct_ratios)
        if len(feat) < 30:
            feat.extend([0] * (30 - len(feat)))
        feat = feat[:30]
        features.append(feat)
    return np.array(features)

def create_readability_features(texts):
    def count_syllables(word):
        word = word.lower().strip(string.punctuation)
        if not word or len(word) <= 3:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0, 0, 0, 0])
            continue
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_words = len(words)
        n_sentences = len(sentences)
        syllables = sum(count_syllables(w) for w in words)
        if n_sentences > 0 and n_words > 0:
            flesch = (
                206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (syllables / n_words)
            )
        else:
            flesch = 0
        n_chars = len(text)
        if n_words > 0 and n_sentences > 0:
            ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sentences) - 21.43
        else:
            ari = 0
        avg_syllables = syllables / max(n_words, 1)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / max(n_words, 1)
        features.append([flesch, ari, avg_syllables, complex_ratio])
    return np.array(features)

def create_pos_tag_approximation(texts):
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "dom",
        "hood",
        "ship",
        "ism",
        "ist",
        "er",
        "or",
        "ee",
        "ing",
    ]
    verb_suffixes = ["ate", "ify", "ize", "ise", "en", "ed", "ing"]
    adj_suffixes = [
        "ous",
        "ious",
        "eous",
        "ive",
        "ative",
        "able",
        "ible",
        "ic",
        "ical",
        "al",
        "ful",
        "less",
        "y",
        "ish",
        "esque",
        "like",
        "ern",
    ]
    adv_suffixes = ["ly", "wards", "wise"]

    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        words = str(text).split()
        words_lower = [w.lower().strip(string.punctuation) for w in words if w.strip()]
        n_words = len(words_lower)
        if n_words == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        nouns = 0
        verbs = 0
        adjs = 0
        advs = 0
        content_words = 0
        for w in words_lower:
            if not w or len(w) < 3:
                continue
            if any(w.endswith(suf) for suf in noun_suffixes):
                nouns += 1
            if any(w.endswith(suf) for suf in verb_suffixes):
                verbs += 1
            if any(w.endswith(suf) for suf in adj_suffixes):
                adjs += 1
            if any(w.endswith(suf) for suf in adv_suffixes):
                advs += 1
            if len(w) > 3 or (
                len(w) > 2
                and not any(
                    w == fw
                    for fw in [
                        "the",
                        "and",
                        "for",
                        "are",
                        "but",
                        "not",
                        "you",
                        "all",
                        "can",
                        "had",
                        "her",
                        "was",
                        "one",
                        "our",
                        "out",
                        "has",
                        "had",
                        "did",
                        "get",
                        "got",
                        "may",
                        "say",
                        "she",
                        "his",
                        "its",
                        "now",
                        "how",
                        "man",
                        "new",
                        "old",
                        "way",
                        "who",
                        "boy",
                        "did",
                        "let",
                        "put",
                        "say",
                        "she",
                        "too",
                        "use",
                        "dad",
                        "key",
                        "etc",
                    ]
                )
            ):
                content_words += 1
        features.append(
            [
                nouns / max(n_words, 1),
                verbs / max(n_words, 1),
                adjs / max(n_words, 1),
                advs / max(n_words, 1),
                content_words / max(n_words, 1),
            ]
        )
    return np.array(features)

def extract_punctuation_sequence(text):
    if not isinstance(text, str):
        return ""
    return "".join([c for c in text if c in string.punctuation])

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

print(
    f"Dense handcrafted features: train {train_stylo_filtered.shape}, val {val_stylo_filtered.shape}, test {test_stylo_filtered.shape}"
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
print(
    f"Sparse features: train {train_sparse.shape}, val {val_sparse.shape}, test {test_sparse.shape}"
)

# ============================================================
# DEBERTA FINE-TUNING
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
class DebertaMeanPoolingForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.config = self.deberta.config
        hidden_size = self.deberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, hidden_size // 8)
        self.classifier_act = nn.GELU()
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier_out = nn.Linear(hidden_size // 8, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False):
        outputs = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        # Mean pooling weighted by attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        pooled_output = self.dropout(pooled_output)
        hidden = self.classifier(pooled_output)
        hidden = self.classifier_act(hidden)
        hidden = self.classifier_dropout(hidden)
        logits = self.classifier_out(hidden)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)

        if output_hidden_states:
            return type('Output', (), {
                'logits': logits,
                'loss': loss,
                'hidden_states': (hidden_states,),
            })()
        return type('Output', (), {
            'logits': logits,
            'loss': loss,
        })()

model = DebertaMeanPoolingForSequenceClassification(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    dropout=DROPOUT,
)
# No need to separately load base model for classifier init
# The custom model's Deberta backbone is already loaded from pretrained checkpoint
# The classifier layers will be randomly initialized and fine-tuned
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
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
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
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest DeBERTa model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
model.load_state_dict(best_model_state)
model = model.to(device)

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
                # Mean pooling weighted by attention mask
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(pooled_embeddings)
    return np.vstack(all_embeddings)

train_loader_no_labels = DataLoader(
    TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
val_loader_no_labels = DataLoader(
    TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
test_loader_no_labels = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
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
# DEBERTA FINAL PROBS
# ============================================================
print("\nGetting DeBERTa final probabilities...")
val_loader_eval = DataLoader(
    TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(y_val_labels, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
deberta_val_ll, deberta_val_acc, deberta_val_probs = evaluate_deberta(model, val_loader_eval)
# Convert to numpy if needed and ensure proper shape
if hasattr(deberta_val_probs, 'cpu'):
    deberta_val_probs = deberta_val_probs.cpu().numpy()
print(f"DeBERTa validation log loss: {deberta_val_ll:.4f}")

test_loader_eval = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
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

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
print(f"\nFinal Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
