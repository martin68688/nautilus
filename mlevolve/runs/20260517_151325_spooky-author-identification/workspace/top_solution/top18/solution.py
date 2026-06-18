import pandas as pd
import numpy as np
import os
import re
import string
import warnings
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import hstack
from scipy import stats
from collections import Counter

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
HIDDEN_SIZE = 256
DROPOUT = 0.1
MAX_LENGTH = 192
BATCH_SIZE = 32
NUM_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 1e-3
WEIGHT_DECAY = 0.01

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
print(f"Labels: {dict(zip(label_encoder.classes_, range(3)))}")

# ============================================================
# STRATIFIED SPLIT (NO INDEX_BUG)
# ============================================================
train_texts_full = train_df["text"].values
train_labels_full = train_df["author_encoded"].values

X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_texts_full,
    train_labels_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_labels_full,
)
X_test_texts = test_df["text"].values
test_ids = test_df["id"].values

print(
    f"Train: {len(X_train_texts)}, Val: {len(X_val_texts)}, Test: {len(X_test_texts)}"
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
train_features_list = []
val_features_list = []
test_features_list = []
feature_names = []

# 1. Syntactic Features
def extract_syntactic_features(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            results.append([0, 0, 0, 0, 0, 0])
            continue
        nested_punct = len(re.findall(r"[,;:\-—\(\)\[\]{}]", text))
        max_depth = 0
        current_depth = 0
        for char in text:
            if char in "([{":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ")]}":
                current_depth = max(0, current_depth - 1)
        clauses = len(re.findall(r"[,;:]", text)) + len(
            re.findall(
                r"\b(and|but|or|yet|for|nor|so|because|although|while|since|unless|if|when)\b",
                text.lower(),
            )
        )
        rel_clauses = len(re.findall(r"\b(which|who|whom|whose|that)\b", text.lower()))
        sub_conj = len(
            re.findall(
                r"\b(although|because|since|unless|while|whereas|after|before|when|if|though|until|as|whether)\b",
                text.lower(),
            )
        )
        words = len(text.split())
        complexity = words / (clauses + 1) if clauses > 0 else words
        results.append(
            [nested_punct, max_depth, clauses, rel_clauses, sub_conj, complexity]
        )
    return np.array(results)

print("Extracting syntactic features...")
train_syn = extract_syntactic_features(X_train_texts)
val_syn = extract_syntactic_features(X_val_texts)
test_syn = extract_syntactic_features(X_test_texts)
train_features_list.append(train_syn)
val_features_list.append(val_syn)
test_features_list.append(test_syn)
feature_names.extend(
    ["nested_punct", "max_depth", "clauses", "rel_clauses", "sub_conj", "complexity"]
)

# 2. Vocabulary Sophistication
def compute_zipf_divergence(text):
    words = text.lower().split()
    if len(words) == 0:
        return [0, 0, 0, 0]
    word_counts = Counter(words)
    total = sum(word_counts.values())
    freqs = np.array(sorted([c / total for c in word_counts.values()], reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    log_ranks = np.log(ranks[ranks > 0])
    log_freqs = np.log(freqs[freqs > 0])
    if len(log_ranks) > 2:
        slope, _, _, _, _ = stats.linregress(log_ranks, log_freqs)
    else:
        slope = -1.0
    ttr = len(word_counts) / total if total > 0 else 0
    hapax = sum(1 for c in word_counts.values() if c == 1) / total if total > 0 else 0
    core_vocab = {
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
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "as",
        "if",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "his",
        "her",
        "my",
        "your",
        "our",
        "we",
        "you",
        "i",
        "me",
        "mine",
        "myself",
    }
    long_words = [w for w in words if len(w) > 8]
    soph_words = [w for w in long_words if w not in core_vocab]
    soph_ratio = len(soph_words) / len(words) if len(words) > 0 else 0
    return [slope, ttr, hapax, soph_ratio]

def extract_vocabulary_features(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            results.append([-1.0, 0, 0, 0])
            continue
        results.append(compute_zipf_divergence(text))
    return np.array(results)

print("Extracting vocabulary features...")
train_vocab = extract_vocabulary_features(X_train_texts)
val_vocab = extract_vocabulary_features(X_val_texts)
test_vocab = extract_vocabulary_features(X_test_texts)
train_features_list.append(train_vocab)
val_features_list.append(val_vocab)
test_features_list.append(test_vocab)
feature_names.extend(["zipf_slope", "ttr", "hapax_ratio", "soph_vocab_ratio"])

# 3. Emotional Features
positive_words = set(
    [
        "love",
        "joy",
        "happy",
        "beautiful",
        "wonderful",
        "great",
        "splendid",
        "delight",
        "pleasure",
        "bliss",
        "ecstasy",
        "rapture",
        "glorious",
        "magnificent",
        "superb",
        "excellent",
        "perfect",
        "admire",
        "cherish",
        "tender",
        "gentle",
        "peaceful",
        "calm",
        "serene",
        "tranquil",
    ]
)
negative_words = set(
    [
        "horror",
        "terror",
        "fear",
        "dread",
        "anguish",
        "agony",
        "pain",
        "sorrow",
        "grief",
        "misery",
        "despair",
        "hopeless",
        "gloom",
        "dismal",
        "dreary",
        "lament",
        "mourn",
        "weep",
        "cry",
        "scream",
        "shriek",
        "howl",
        "groan",
        "moan",
        "sigh",
        "sob",
        "torment",
        "torture",
        "suffering",
        "ache",
        "hurt",
        "wound",
    ]
)
arousal_words = set(
    [
        "furious",
        "rage",
        "wrath",
        "frenzy",
        "frantic",
        "wild",
        "violent",
        "fierce",
        "intense",
        "passionate",
        "burning",
        "blazing",
        "feverish",
        "restless",
        "tremble",
        "shake",
        "quiver",
        "shudder",
        "throb",
        "pulse",
        "surge",
        "rush",
        "explode",
        "erupt",
        "burst",
    ]
)

def extract_emotional_features(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            results.append([0, 0, 0, 0, 0, 0])
            continue
        words = text.lower().split()
        if len(words) == 0:
            results.append([0, 0, 0, 0, 0, 0])
            continue
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        aro_count = sum(1 for w in words if w in arousal_words)
        emo_words = pos_count + neg_count + aro_count
        emo_intensity = emo_words / len(words)
        sent_vals = []
        negated = False
        for word in words:
            if word in {
                "not",
                "no",
                "never",
                "neither",
                "nor",
                "none",
                "nothing",
                "nowhere",
                "hardly",
                "scarcely",
                "barely",
                "doesn't",
                "don't",
                "didn't",
                "won't",
                "wouldn't",
                "couldn't",
                "shouldn't",
                "isn't",
                "aren't",
                "wasn't",
                "weren't",
                "haven't",
                "hasn't",
                "hadn't",
                "can't",
            }:
                negated = not negated
            elif word in positive_words:
                sent_vals.append(-1 if negated else 1)
                negated = False
            elif word in negative_words:
                sent_vals.append(1 if negated else -1)
                negated = False
        if len(sent_vals) == 0:
            sent_vals = [0, 0]
        mean_sent = np.mean(sent_vals)
        std_sent = np.std(sent_vals) if len(sent_vals) > 1 else 0
        max_sent = np.max(sent_vals)
        min_sent = np.min(sent_vals)
        results.append(
            [
                pos_count / len(words),
                neg_count / len(words),
                aro_count / len(words),
                mean_sent,
                std_sent,
                max_sent - min_sent,
            ]
        )
    return np.array(results)

print("Extracting emotional features...")
train_emo = extract_emotional_features(X_train_texts)
val_emo = extract_emotional_features(X_val_texts)
test_emo = extract_emotional_features(X_test_texts)
train_features_list.append(train_emo)
val_features_list.append(val_emo)
test_features_list.append(test_emo)
feature_names.extend(
    [
        "pos_ratio",
        "neg_ratio",
        "arousal_ratio",
        "sentiment_mean",
        "sentiment_std",
        "sentiment_range",
    ]
)

# 4. Narrative Style Markers
archaic_words = set(
    [
        "thee",
        "thou",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "dost",
        "art",
        "ere",
        "whilst",
        "hence",
        "thence",
        "whence",
        "wherefore",
        "thereof",
        "therein",
        "thereupon",
        "herewith",
        "hereby",
        "herein",
        "hereafter",
        "aforesaid",
        "aforementioned",
        "aforetime",
        "albeit",
        "anon",
        "bethink",
        "bethought",
        "beseech",
        "besought",
        "betimes",
        "betwixt",
        "bode",
        "boded",
        "boding",
        "brought",
        "champion",
        "charnel",
        "chimerical",
        "clime",
        "commence",
        "commenced",
        "commencing",
        "concerning",
        "conjuration",
        "conjure",
        "conjured",
        "conjuring",
        "countenance",
        "damask",
        "damned",
        "damp",
        "dark",
        "dash",
        "dead",
        "deathly",
        "deep",
        "denizen",
        "desolate",
        "desolation",
        "despair",
        "desperate",
    ]
)
gothic_words = set(
    [
        "ghastly",
        "ghostly",
        "grave",
        "gloom",
        "gloomy",
        "groan",
        "grotesque",
        "haunted",
        "hideous",
        "horrible",
        "horrid",
        "horror",
        "melancholy",
        "morbid",
        "mysterious",
        "mystic",
        "ominous",
        "pale",
        "pallid",
        "phantom",
        "portent",
        "portentous",
        "prodigious",
        "pulsation",
        "pulseless",
        "quaint",
        "quaver",
        "queer",
        "quiver",
        "quivering",
        "ragged",
        "raving",
        "recluse",
        "recoil",
        "recoiled",
        "recoiling",
        "remorse",
        "repent",
        "repentance",
        "repining",
        "repulsive",
        "resigned",
        "resignation",
        "resounding",
        "restless",
        "restlessness",
        "reverie",
    ]
)
lovecraft_words = set(
    [
        "abyss",
        "aeon",
        "amorphous",
        "antique",
        "antiquarian",
        "antiquity",
        "blasphemous",
        "blasphemy",
        "catacomb",
        "cataclysm",
        "cataclysmic",
        "cavern",
        "cavernous",
        "chthonic",
        "cosmic",
        "cyclopean",
        "cypress",
        "daemon",
        "daemonic",
        "decadent",
        "decay",
        "decayed",
        "decaying",
        "demon",
        "demonic",
        "dimension",
        "dimensional",
        "disembodied",
        "dread",
        "dreadful",
        "eldritch",
        "eternal",
        "eternity",
        "evil",
        "fathomless",
        "foetid",
        "formless",
        "fungoid",
        "gibbering",
        "gigantic",
        "immortal",
        "immortality",
        "indescribable",
        "ineffable",
        "infernal",
        "infinite",
        "infinitude",
        "inmost",
        "inscrutable",
        "interstellar",
        "iridescent",
        "labyrinthine",
    ]
)
latinate_suffixes = [
    "tion",
    "sion",
    "ment",
    "ance",
    "ence",
    "ity",
    "ical",
    "ial",
    "ous",
    "ive",
    "ate",
    "ify",
    "ize",
    "ure",
    "tude",
    "able",
    "ible",
    "al",
    "ic",
    "id",
    "ile",
    "ine",
    "ory",
]

def extract_style_markers(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            results.append([0, 0, 0, 0, 0, 0])
            continue
        words = text.lower().split()
        if len(words) == 0:
            results.append([0, 0, 0, 0, 0, 0])
            continue
        archaic_count = sum(1 for w in words if w in archaic_words)
        gothic_count = sum(1 for w in words if w in gothic_words)
        lovecraft_count = sum(1 for w in words if w in lovecraft_words)
        latinate_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in latinate_suffixes)
        )
        latinate_ratio = latinate_count / len(words)
        heavy_punct = len(re.findall(r"[!?—…;]", text))
        heavy_punct_ratio = heavy_punct / len(text) if len(text) > 0 else 0
        first_person = len(re.findall(r"\b(i|me|my|mine|myself)\b", text.lower()))
        first_person_ratio = first_person / len(words)
        results.append(
            [
                archaic_count / len(words),
                gothic_count / len(words),
                lovecraft_count / len(words),
                latinate_ratio,
                heavy_punct_ratio,
                first_person_ratio,
            ]
        )
    return np.array(results)

print("Extracting style markers...")
train_style = extract_style_markers(X_train_texts)
val_style = extract_style_markers(X_val_texts)
test_style = extract_style_markers(X_test_texts)
train_features_list.append(train_style)
val_features_list.append(val_style)
test_features_list.append(test_style)
feature_names.extend(
    [
        "archaic_ratio",
        "gothic_ratio",
        "lovecraft_ratio",
        "latinate_ratio",
        "heavy_punct_ratio",
        "first_person_ratio",
    ]
)

# 5. POS Approximation
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
verb_suffixes = ["ate", "ize", "ify", "en", "ing", "ed", "ly", "er", "est"]
adj_suffixes = ["ous", "ive", "able", "ible", "al", "ic", "ful", "less", "ish", "like"]
adv_suffixes = ["ly", "wards", "wise"]

def extract_pos_approximation(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            results.append([0, 0, 0, 0, 0])
            continue
        words = text.lower().split()
        if len(words) == 0:
            results.append([0, 0, 0, 0, 0])
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
        content_words = noun_count + verb_count + adj_count + adv_count
        content_ratio = content_words / len(words)
        results.append(
            [
                noun_count / len(words),
                verb_count / len(words),
                adj_count / len(words),
                adv_count / len(words),
                content_ratio,
            ]
        )
    return np.array(results)

print("Extracting POS features...")
train_pos = extract_pos_approximation(X_train_texts)
val_pos = extract_pos_approximation(X_val_texts)
test_pos = extract_pos_approximation(X_test_texts)
train_features_list.append(train_pos)
val_features_list.append(val_pos)
test_features_list.append(test_pos)
feature_names.extend(
    ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "content_word_ratio"]
)

# 6. Basic Stats
def extract_basic_stats(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            results.append([0, 0, 0, 0, 0, 0])
            continue
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        char_count = len(text)
        word_count = len(words)
        sent_count = len(sentences) if sentences else 1
        avg_word_len = char_count / word_count if word_count > 0 else 0
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0
        word_lengths = [len(w) for w in words]
        std_word_len = np.std(word_lengths) if len(word_lengths) > 0 else 0
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / char_count if char_count > 0 else 0
        results.append(
            [
                char_count,
                word_count,
                sent_count,
                avg_word_len,
                avg_sent_len,
                std_word_len,
            ]
        )
    return np.array(results)

print("Extracting basic stats...")
train_basic = extract_basic_stats(X_train_texts)
val_basic = extract_basic_stats(X_val_texts)
test_basic = extract_basic_stats(X_test_texts)
train_features_list.append(train_basic)
val_features_list.append(val_basic)
test_features_list.append(test_basic)
feature_names.extend(
    [
        "char_count",
        "word_count",
        "sent_count",
        "avg_word_len",
        "avg_sent_len",
        "std_word_len",
    ]
)

# Combine and scale handcrafted features
print("Combining handcrafted features...")
train_handcrafted = np.hstack(train_features_list)
val_handcrafted = np.hstack(val_features_list)
test_handcrafted = np.hstack(test_features_list)

dense_scaler = StandardScaler()
train_handcrafted_scaled = dense_scaler.fit_transform(train_handcrafted)
val_handcrafted_scaled = dense_scaler.transform(val_handcrafted)
test_handcrafted_scaled = dense_scaler.transform(test_handcrafted)

train_handcrafted_scaled = np.nan_to_num(
    train_handcrafted_scaled, nan=0.0, posinf=0.0, neginf=0.0
)
val_handcrafted_scaled = np.nan_to_num(
    val_handcrafted_scaled, nan=0.0, posinf=0.0, neginf=0.0
)
test_handcrafted_scaled = np.nan_to_num(
    test_handcrafted_scaled, nan=0.0, posinf=0.0, neginf=0.0
)

variance_selector = VarianceThreshold(threshold=0.001)
train_handcrafted_filtered = variance_selector.fit_transform(train_handcrafted_scaled)
val_handcrafted_filtered = variance_selector.transform(val_handcrafted_scaled)
test_handcrafted_filtered = variance_selector.transform(test_handcrafted_scaled)

# N-gram features
print("Extracting n-gram features...")
char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vectorizer_short.fit_transform(X_train_texts)
val_char_short = char_vectorizer_short.transform(X_val_texts)
test_char_short = char_vectorizer_short.transform(X_test_texts)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_train_texts)
val_char_med = char_vectorizer_med.transform(X_val_texts)
test_char_med = char_vectorizer_med.transform(X_test_texts)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=1500,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_train_texts)
val_char_long = char_vectorizer_long.transform(X_val_texts)
test_char_long = char_vectorizer_long.transform(X_test_texts)

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
test_word = word_vectorizer.transform(X_test_texts)

def extract_punctuation_sequence(text):
    return (
        "".join([c for c in text if c in string.punctuation])
        if isinstance(text, str)
        else ""
    )

punct_sequences_train = [extract_punctuation_sequence(str(t)) for t in X_train_texts]
punct_sequences_val = [extract_punctuation_sequence(str(t)) for t in X_val_texts]
punct_sequences_test = [extract_punctuation_sequence(str(t)) for t in X_test_texts]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=300, min_df=2
)
train_punct = punct_vectorizer.fit_transform(punct_sequences_train)
val_punct = punct_vectorizer.transform(punct_sequences_val)
test_punct = punct_vectorizer.transform(punct_sequences_test)

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()

print(f"Handcrafted features (filtered): {train_handcrafted_filtered.shape[1]}")
print(f"Sparse features: {train_sparse.shape[1]}")

# ============================================================
# MODEL DESIGN: Simple Transformer + Features
# ============================================================
MODEL_NAME = "microsoft/deberta-v3-small"

class HybridTransformerModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_AUTHORS, dropout=DROPOUT, num_handcrafted=32):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=False, output_attentions=False)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.feature_proj = nn.Linear(num_handcrafted, 64)
        # Attention-weighted pooling: learnable query vector
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attn_proj = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(HIDDEN_SIZE // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, handcrafted_features):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        # Attention-weighted pooling
        batch_size, seq_len, _ = token_hidden.shape
        # Compute attention scores: dot product with learnable query
        query = self.attn_query.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        attn_scores = torch.bmm(query, token_hidden.transpose(1, 2))  # (batch, 1, seq_len)
        attn_scores = attn_scores.squeeze(1)  # (batch, seq_len)
        # Apply GELU activation before softmax
        attn_scores = torch.nn.functional.gelu(attn_scores)
        # Mask padding tokens
        attn_mask = attention_mask.float()  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        # Weighted sum
        attn_output = (token_hidden * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        attn_output = self.dropout(attn_output)
        feat_proj = self.feature_proj(handcrafted_features)
        combined = torch.cat([attn_output, feat_proj], dim=1)
        logits = self.classifier(combined)
        return logits

print("Preparing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(tokenizer, texts, max_length=MAX_LENGTH):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

print("Tokenizing texts...")
train_enc = encode_texts(tokenizer, X_train_texts)
val_enc = encode_texts(tokenizer, X_val_texts)
test_enc = encode_texts(tokenizer, X_test_texts)

train_labels_tensor = torch.tensor(y_train_labels, dtype=torch.long)
val_labels_tensor = torch.tensor(y_val_labels, dtype=torch.long)

# Convert handcrafted features to tensors
train_handcrafted_tensor = torch.tensor(train_handcrafted_filtered, dtype=torch.float32)
val_handcrafted_tensor = torch.tensor(val_handcrafted_filtered, dtype=torch.float32)
test_handcrafted_tensor = torch.tensor(test_handcrafted_filtered, dtype=torch.float32)

class HybridDataset(Dataset):
    def __init__(self, enc, handcrafted, labels=None):
        self.enc = enc
        self.handcrafted = handcrafted
        self.labels = labels

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "handcrafted_features": self.handcrafted[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

train_dataset = HybridDataset(train_enc, train_handcrafted_tensor, train_labels_tensor)
val_dataset = HybridDataset(val_enc, val_handcrafted_tensor, val_labels_tensor)
test_dataset = HybridDataset(test_enc, test_handcrafted_tensor)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# ============================================================
# INITIALIZE MODEL
# ============================================================
num_handcrafted = train_handcrafted_filtered.shape[1]
print(f"Number of handcrafted features: {num_handcrafted}")

print("Initializing Hybrid Transformer Model...")
model = HybridTransformerModel(num_labels=NUM_AUTHORS, dropout=DROPOUT, num_handcrafted=num_handcrafted)
model.to(device)

# Two-stage training setup
STAGE_1_EPOCHS = 2  # First 2 epochs: freeze backbone, train only feature_proj and classifier
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2_BACKBONE = 2e-6
LEARNING_RATE_STAGE2_HEAD = 1e-3

# Initially freeze transformer backbone
for param in model.transformer.parameters():
    param.requires_grad = False

# Optimizer for Stage 1: only feature_proj and classifier
param_groups_stage1 = [
    {'params': model.feature_proj.parameters(), 'lr': LEARNING_RATE_STAGE1, 'weight_decay': WEIGHT_DECAY * 0.5},
    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_STAGE1, 'weight_decay': WEIGHT_DECAY * 0.5},
]
optimizer = AdamW(param_groups_stage1, eps=1e-8)
total_steps_stage1 = len(train_loader) * STAGE_1_EPOCHS
warmup_steps_stage1 = int(0.1 * total_steps_stage1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps_stage1, num_training_steps=total_steps_stage1
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler() if torch.cuda.is_available() else None
current_stage = 1

# ============================================================
# METRIC
# ============================================================
def compute_multiclass_log_loss(y_true, y_pred_proba):
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
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("TRAINING HYBRID TRANSFORMER MODEL")
print("=" * 60)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Switch to Stage 2 after STAGE_1_EPOCHS
    if epoch == STAGE_1_EPOCHS and current_stage == 1:
        print(f"\n--- Switching to Stage 2: Unfreezing transformer backbone (LR={LEARNING_RATE_STAGE2_BACKBONE}) ---\n")
        for param in model.transformer.parameters():
            param.requires_grad = True
        # Recreate optimizer with both backbone and head parameter groups
        param_groups_stage2 = [
            {'params': model.transformer.parameters(), 'lr': LEARNING_RATE_STAGE2_BACKBONE, 'weight_decay': WEIGHT_DECAY},
            {'params': model.feature_proj.parameters(), 'lr': LEARNING_RATE_STAGE2_HEAD, 'weight_decay': WEIGHT_DECAY * 0.5},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_STAGE2_HEAD, 'weight_decay': WEIGHT_DECAY * 0.5},
        ]
        optimizer = AdamW(param_groups_stage2, eps=1e-8)
        remaining_epochs = NUM_EPOCHS - STAGE_1_EPOCHS
        total_steps_stage2 = len(train_loader) * remaining_epochs
        warmup_steps_stage2 = int(0.1 * total_steps_stage2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps_stage2, num_training_steps=total_steps_stage2
        )
        current_stage = 2

    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        handcrafted_features = batch["handcrafted_features"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(input_ids, attention_mask, handcrafted_features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, handcrafted_features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    model.eval()
    val_loss_total = 0.0
    val_num_batches = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            handcrafted_features = batch["handcrafted_features"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask, handcrafted_features)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
            val_loss_total += loss.item()
            val_num_batches += 1
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
    avg_val_loss = val_loss_total / val_num_batches
    val_preds = np.vstack(all_val_preds)
    val_labels = np.concatenate(all_val_labels)
    val_logloss = compute_multiclass_log_loss(val_labels, val_preds)
    val_acc = np.mean(np.argmax(val_preds, axis=1) == val_labels)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            },
            "./working/best_model.pt",
        )
        print(f"  → New best model saved! Val LogLoss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val log loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest model: epoch {best_epoch}, val log loss: {best_val_loss:.4f}")

# ============================================================
# INFERENCE
# ============================================================
print("\nLoading best model for inference...")
checkpoint = torch.load("./working/best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Generating test predictions...")
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        handcrafted_features = batch["handcrafted_features"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask, handcrafted_features)
            probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())

test_preds = np.vstack(all_test_preds)
eps = 1e-15
test_preds = np.clip(test_preds, eps, 1 - eps)
row_sums = test_preds.sum(axis=1, keepdims=True)
test_preds = test_preds / row_sums
test_preds = np.clip(test_preds, eps, 1 - eps)

# ============================================================
# VALIDATION METRIC (final)
# ============================================================
all_val_preds_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        handcrafted_features = batch["handcrafted_features"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask, handcrafted_features)
            probs = torch.softmax(logits, dim=1)
        all_val_preds_final.append(probs.cpu().numpy())
val_preds_final = np.vstack(all_val_preds_final)
final_val_logloss = compute_multiclass_log_loss(y_val_labels, val_preds_final)

# ============================================================
# SUBMISSION
# ============================================================
print("Creating submission file...")
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# ============================================================
# CLEANUP
# ============================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nFinal Validation Score: {final_val_logloss:.6f}")