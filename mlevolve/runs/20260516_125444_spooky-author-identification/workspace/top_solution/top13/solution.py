import pandas as pd
import numpy as np
import os
import re
import string
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
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
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 16
WARMUP_RATIO = 0.1
PATIENCE = 7
DROPOUT = 0.2
ACCUMULATION_STEPS = 2
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1
HANDCRAFTED_DIM = None  # Will be computed dynamically after feature extraction

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(f"Label encoding: {dict(zip(label_encoder.classes_, range(NUM_AUTHORS)))}")

# ============================================================
# 2. STRATIFIED SPLIT (using numpy indexing to avoid INDEX_BUG)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

X_train_texts = train_df["text"].values[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_train_labels = y_train_full[train_idx]
y_val_labels = y_train_full[val_idx]

X_test_texts = test_df["text"].values

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(X_test_texts)}"
)

# ============================================================
# 3. FEATURE ENGINEERING FUNCTIONS
# ============================================================

FUNCTION_WORDS = set(
    [
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "that",
        "it",
        "was",
        "i",
        "he",
        "with",
        "for",
        "had",
        "is",
        "his",
        "as",
        "on",
        "not",
        "at",
        "by",
        "but",
        "from",
        "my",
        "which",
        "be",
        "this",
        "are",
        "or",
        "an",
        "have",
        "were",
        "me",
        "all",
        "so",
        "no",
        "she",
        "her",
        "their",
        "been",
        "its",
        "they",
        "we",
        "who",
        "do",
        "if",
        "will",
        "would",
        "can",
        "up",
    ]
)

def extract_stylometric_features(texts):
    features_list = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features_list.append(np.zeros(30))
            continue
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        text_len = len(text_str)
        word_count = len(words) if words else 1
        sent_count = len(sentences) if sentences else 1
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = word_count / sent_count
        upper_ratio = sum(1 for c in text_str if c.isupper()) / text_len
        lower_ratio = sum(1 for c in text_str if c.islower()) / text_len
        digit_ratio = sum(1 for c in text_str if c.isdigit()) / text_len
        whitespace_ratio = sum(1 for c in text_str if c.isspace()) / text_len
        punct_ratios = []
        for p in string.punctuation[:12]:
            punct_ratios.append(text_str.count(p) / text_len)
        char_diversity = len(set(text_str.lower())) / text_len if text_len > 0 else 0
        long_words = (
            sum(1 for w in words if len(w) >= 7) / word_count if word_count > 0 else 0
        )
        capitalized = (
            sum(1 for w in words if w[0].isupper()) / word_count
            if word_count > 0
            else 0
        )
        all_caps = (
            sum(1 for w in words if len(w) >= 2 and w.isupper()) / word_count
            if word_count > 0
            else 0
        )
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        words_lower = [w.lower().strip(string.punctuation) for w in words]
        words_lower = [w for w in words_lower if w]
        function_word_ratio = (
            sum(1 for w in words_lower if w in FUNCTION_WORDS) / len(words_lower)
            if words_lower
            else 0
        )
        archaic_ratio = (
            sum(
                1
                for w in words_lower
                if w
                in [
                    "thee",
                    "thou",
                    "thy",
                    "thine",
                    "hath",
                    "doth",
                    "dost",
                    "art",
                    "wilt",
                    "canst",
                    "shalt",
                    "didst",
                    "wert",
                    "whence",
                    "thence",
                    "hence",
                    "therein",
                    "thereof",
                    "thereto",
                    "wherefore",
                    "mayest",
                    "wouldst",
                    "couldst",
                    "mightst",
                ]
            )
            / len(words_lower)
            if words_lower
            else 0
        )
        emotional_ratio = (
            sum(
                1
                for w in words_lower
                if w
                in [
                    "fear",
                    "terror",
                    "horror",
                    "dread",
                    "awful",
                    "terrible",
                    "frightful",
                    "ghastly",
                    "dismal",
                    "weird",
                    "strange",
                    "mysterious",
                    "shadow",
                    "dark",
                    "gloom",
                    "death",
                    "night",
                    "soul",
                    "spirit",
                    "ghost",
                    "corpse",
                    "phantom",
                    "spectre",
                    "demon",
                    "devil",
                    "witch",
                    "curse",
                    "haunt",
                    "grave",
                    "coffin",
                    "agony",
                    "anguish",
                    "despair",
                    "madness",
                    "insanity",
                    "frenzy",
                    "passion",
                    "revenge",
                    "wrath",
                    "wrathful",
                ]
            )
            / len(words_lower)
            if words_lower
            else 0
        )
        lovecraft_ratio = (
            sum(
                1
                for w in words_lower
                if w
                in [
                    "cthulhu",
                    "r'lyeh",
                    "yog-sothoth",
                    "nyarlathotep",
                    "azathoth",
                    "shoggoth",
                    "necronomicon",
                    "eldritch",
                    "cyclopean",
                    "non-euclidean",
                    "squamous",
                    "rugose",
                    "gelatinous",
                    "ichor",
                    "fungus",
                    "tentacle",
                    "void",
                    "abyss",
                    "cosmos",
                    "aeon",
                    "primordial",
                    "unnameable",
                    "indescribable",
                    "blasphemous",
                    "cryptic",
                    "antediluvian",
                    "alien",
                    "infernal",
                    "accursed",
                    "loathsome",
                    "putrid",
                    "unwholesome",
                    "vortex",
                    "dimension",
                    "cult",
                    "monolith",
                    "basalt",
                    "pit",
                    "cave",
                    "cavern",
                    "cyclopean",
                ]
            )
            / len(words_lower)
            if words_lower
            else 0
        )
        sub_conj = [
            "although",
            "because",
            "since",
            "unless",
            "while",
            "after",
            "before",
            "when",
            "where",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "if",
            "though",
            "as",
            "until",
            "once",
            "whether",
        ]
        sub_conj_ratio = (
            sum(1 for w in words_lower if w in sub_conj) / len(words_lower)
            if words_lower
            else 0
        )
        features = (
            [
                text_len,
                word_count,
                sent_count,
                avg_word_len,
                avg_sent_len,
                upper_ratio,
                lower_ratio,
                digit_ratio,
                whitespace_ratio,
            ]
            + punct_ratios
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
        features_list.append(features[:30])
    return np.array(features_list)

def create_readability_features(texts):
    features_list = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features_list.append(np.zeros(4))
            continue
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_words = len(words) if words else 1
        total_sentences = len(sentences) if sentences else 1
        syllables = 0
        for w in words:
            w_clean = w.lower().strip(string.punctuation)
            if not w_clean:
                continue
            vowel_groups = len(re.findall(r"[aeiouy]+", w_clean))
            if w_clean.endswith("e") and not w_clean.endswith("le"):
                vowel_groups = max(1, vowel_groups - 1)
            if vowel_groups == 0:
                vowel_groups = 1
            syllables += vowel_groups
        avg_syllables = syllables / total_words
        flesch = (
            206.835 - 1.015 * (total_words / total_sentences) - 84.6 * avg_syllables
        )
        total_chars = sum(len(w) for w in words)
        ari = (
            4.71 * (total_chars / total_words)
            + 0.5 * (total_words / total_sentences)
            - 21.43
        )
        complex_words = 0
        for w in words:
            w_clean = w.lower().strip(string.punctuation)
            if not w_clean:
                continue
            vowel_groups = len(re.findall(r"[aeiouy]+", w_clean))
            if vowel_groups >= 3:
                complex_words += 1
        complex_word_ratio = complex_words / total_words
        features_list.append([flesch, ari, avg_syllables, complex_word_ratio])
    return np.array(features_list)

def create_pos_tag_approximation(texts):
    features_list = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features_list.append(np.zeros(5))
            continue
        text_str = str(text)
        words = text_str.split()
        total_words = len(words) if words else 1
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
            "ist",
            "ism",
            "er",
            "or",
            "ian",
        ]
        noun_suffix_ratio = (
            sum(1 for w in words if any(w.lower().endswith(s) for s in noun_suffixes))
            / total_words
        )
        verb_suffixes = ["ed", "ing", "ate", "ize", "ify", "en", "ish"]
        verb_suffix_ratio = (
            sum(1 for w in words if any(w.lower().endswith(s) for s in verb_suffixes))
            / total_words
        )
        adj_suffixes = [
            "able",
            "ible",
            "ful",
            "less",
            "ous",
            "ive",
            "al",
            "ic",
            "ical",
            "ish",
            "like",
            "ly",
            "ward",
            "ern",
        ]
        adj_suffix_ratio = (
            sum(1 for w in words if any(w.lower().endswith(s) for s in adj_suffixes))
            / total_words
        )
        adv_suffix_ratio = (
            sum(1 for w in words if w.lower().endswith("ly")) / total_words
        )
        content_word_ratio = (
            1
            - sum(
                1
                for w in words
                if w.lower().strip(string.punctuation) in FUNCTION_WORDS
            )
            / total_words
            if total_words > 0
            else 0
        )
        features_list.append(
            [
                noun_suffix_ratio,
                verb_suffix_ratio,
                adj_suffix_ratio,
                adv_suffix_ratio,
                content_word_ratio,
            ]
        )
    return np.array(features_list)

# ============================================================
# 4. APPLY FEATURE ENGINEERING
# ============================================================
print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(X_test_texts)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)
print(
    f"Stylometric features: {train_stylo_filtered.shape[1]} (after variance filtering)"
)

print("Extracting readability features...")
train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(X_test_texts)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

print("Extracting POS approximation features...")
train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(X_test_texts)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# Dynamically compute the actual handcrafted feature dimension after preprocessing
actual_handcrafted_dim = train_stylo_filtered.shape[1] + train_read_scaled.shape[1] + train_pos_scaled.shape[1]
print(f"Actual handcrafted feature dimension: {actual_handcrafted_dim}")
HANDCRAFTED_DIM = actual_handcrafted_dim

# ============================================================
# 5. N-GRAM FEATURES
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
test_char_short = char_vectorizer_short.transform(X_test_texts)

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
test_char_med = char_vectorizer_med.transform(X_test_texts)

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
    return "".join([c for c in str(text) if c in string.punctuation]) if text else ""

# Fit punctuation vectorizer on TRAIN only to prevent data leakage
train_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_train_texts]
val_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_val_texts]
test_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_test_texts]

punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
train_punct = punct_vectorizer.fit_transform(train_punct_sequences)
val_punct = punct_vectorizer.transform(val_punct_sequences)
test_punct = punct_vectorizer.transform(test_punct_sequences)

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
# 6. CONSOLIDATE FEATURES
# ============================================================
train_features_dict = {
    "stylo_filtered": train_stylo_filtered,
    "read_scaled": train_read_scaled,
    "pos_scaled": train_pos_scaled,
    "sparse": train_sparse,
    "texts": X_train_texts,
}
val_features_dict = {
    "stylo_filtered": val_stylo_filtered,
    "read_scaled": val_read_scaled,
    "pos_scaled": val_pos_scaled,
    "sparse": val_sparse,
    "texts": X_val_texts,
}
test_features_dict = {
    "stylo_filtered": test_stylo_filtered,
    "read_scaled": test_read_scaled,
    "pos_scaled": test_pos_scaled,
    "sparse": test_sparse,
    "texts": X_test_texts,
}

# ============================================================
# 7. MODEL ARCHITECTURE
# ============================================================
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = self.head_dim ** -0.5

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.shape

        Q = self.query(hidden_states)  # (B, S, H)
        K = self.key(hidden_states)    # (B, S, H)
        V = self.value(hidden_states)  # (B, S, H)

        # Reshape for multi-head: (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, S, S)

        # Apply attention mask: mask out padding tokens
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        attn_scores = attn_scores.masked_fill(extended_mask == 0, -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, S, S)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, S, head_dim)

        # Reshape back: (B, num_heads, S, head_dim) -> (B, S, num_heads*head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.out_proj(attn_output)

        # Pool: take the [CLS] token (first token) representation after attention
        pooled = attn_output[:, 0, :]  # (B, H)
        return pooled, attn_weights

class MultiScaleTransformer(nn.Module):
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        config.output_hidden_states = True  # Enable hidden states output
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Multi-head attention pooling instance (shared across scales)
        self.attention_pooling = MultiHeadAttentionPooling(self.hidden_size, num_heads=8)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple of (layer_0, ..., layer_last)

        # Extract layers: -2, -4, -6, -8, -10 and last layer (index -1)
        layer_indices = [-1, -2, -4, -6, -8, -10]
        multi_scale_features = []

        for idx in layer_indices:
            # Handle negative indices relative to num_layers
            actual_idx = idx if idx >= 0 else self.num_layers + idx + 1
            actual_idx = max(0, min(actual_idx, self.num_layers))
            layer_hidden = hidden_states[actual_idx]  # (B, S, H)

            # Apply multi-head attention pooling to get sample-specific weighted representation
            pooled, _ = self.attention_pooling(layer_hidden, attention_mask)
            multi_scale_features.append(pooled)

        # Concatenate multi-scale features
        combined = torch.cat(multi_scale_features, dim=-1)  # (B, 6 * H)
        return combined, outputs.last_hidden_state  # also return last_hidden for backward compat

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=6*768, hidden_dim=256, num_classes=3, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # 128
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)  # 128 -> 3

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.gelu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x

class EnhancedDeBERTaClassifier(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        num_classes=3,
        dropout=0.2,
    ):
        super().__init__()
        self.multi_scale = MultiScaleTransformer(model_name, dropout=dropout)
        self.hidden_size = self.multi_scale.hidden_size

        # Input dimension: 6 * hidden_size (from 6 layers)
        input_dim = 6 * self.hidden_size
        hidden_dim = 256
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.apply(self._init_weights)

        # Disable transformer weight initialization (it's pretrained)
        # Only initialize newly added layers
        for name, param in self.named_parameters():
            if "transformer" not in name:
                pass  # Let _init_weights handle it

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, return_embeds=False):
        combined, last_hidden = self.multi_scale(input_ids, attention_mask)
        logits = self.classifier(combined)
        if return_embeds:
            return logits, combined
        return logits

# ============================================================
# 8. PREPARE DATA FOR TRAINING
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    list(X_test_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

train_handcrafted = np.hstack(
    [
        train_features_dict["stylo_filtered"],
        train_features_dict["read_scaled"],
        train_features_dict["pos_scaled"],
    ]
)
val_handcrafted = np.hstack(
    [
        val_features_dict["stylo_filtered"],
        val_features_dict["read_scaled"],
        val_features_dict["pos_scaled"],
    ]
)
test_handcrafted = np.hstack(
    [
        test_features_dict["stylo_filtered"],
        test_features_dict["read_scaled"],
        test_features_dict["pos_scaled"],
    ]
)

train_handcrafted = torch.tensor(train_handcrafted, dtype=torch.float32)
val_handcrafted = torch.tensor(val_handcrafted, dtype=torch.float32)
test_handcrafted = torch.tensor(test_handcrafted, dtype=torch.float32)
train_labels = torch.tensor(y_train_labels, dtype=torch.long)
val_labels = torch.tensor(y_val_labels, dtype=torch.long)

class AuthorDataset(Dataset):
    def __init__(self, encodings, handcrafted, labels=None):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.handcrafted = handcrafted
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "handcrafted": self.handcrafted[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

train_dataset = AuthorDataset(train_encodings, train_handcrafted, train_labels)
val_dataset = AuthorDataset(val_encodings, val_handcrafted, val_labels)
test_dataset = AuthorDataset(test_encodings, test_handcrafted)

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
    f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
)

# ============================================================
# 9. BUILD MODEL
# ============================================================
model = EnhancedDeBERTaClassifier(
    model_name=MODEL_NAME,
    num_classes=NUM_AUTHORS,
    dropout=DROPOUT,
)
model.to(device)
print(f"Model created with simple DeBERTa-v3 classifier")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Optimizer with layer-wise learning rates
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
param_groups = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    lr_mult = 1.0
    if "transformer" in name:
        lr_mult = 0.5
    elif "classifier" in name:
        lr_mult = 2.0
    if any(nd in name for nd in no_decay):
        param_groups.append(
            {"params": param, "lr": LEARNING_RATE * lr_mult, "weight_decay": 0.0}
        )
    else:
        param_groups.append(
            {
                "params": param,
                "lr": LEARNING_RATE * lr_mult,
                "weight_decay": WEIGHT_DECAY,
            }
        )

optimizer = AdamW(param_groups, lr=LEARNING_RATE, eps=1e-8, betas=(0.9, 0.999))

total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
warmup_steps = int(WARMUP_RATIO * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# After warmup, switch to CosineAnnealingWarmRestarts
import math
T_0 = 8  # Number of epochs for first restart cycle
T_mult = 2  # Factor to increase cycle length after restart
eta_min = 1e-6  # Minimum learning rate

# We'll apply warmup manually for the first warmup_steps then switch
warmup_completed = False
scheduler_restarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_0 * len(train_loader) // ACCUMULATION_STEPS, T_mult=T_mult, eta_min=eta_min
)
scaler = GradScaler() if torch.cuda.is_available() else None

# ============================================================
# 10. EVALUATION METRIC
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

def evaluate(model, loader, criterion=None):
    model.eval()
    all_logits = []
    all_labels = []
    all_embeddings = []
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits, cls_embeds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_embeds=True,
                )
                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_embeddings.append(cls_embeds.cpu().numpy())
    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    all_embeddings = np.vstack(all_embeddings)
    y_pred_proba = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
    log_loss = compute_log_loss(all_labels, y_pred_proba)
    accuracy = np.mean(np.argmax(y_pred_proba, axis=1) == all_labels)
    stats = {"log_loss": log_loss, "accuracy": accuracy}
    if num_batches > 0:
        stats["ce_loss"] = total_loss / num_batches
    return stats, y_pred_proba, all_embeddings

# ============================================================
# 11. TRAINING LOOP
# ============================================================
print("=" * 60)
print("TRAINING SIMPLE ROBERTA MODEL")
print("=" * 60)

best_val_logloss = float("inf")
best_epoch = 0
patience_counter = 0
global_step = 0

# Store validation predictions for LightGBM training
best_val_embeddings = None
best_val_probs = None

warmup_completed = False

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeds=True,
            )
            loss = criterion(logits, labels) / ACCUMULATION_STEPS
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * ACCUMULATION_STEPS
        num_batches += 1
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            # Warmup then cosine restarts
            if global_step < warmup_steps:
                scheduler_warmup.step()
            else:
                if not warmup_completed:
                    warmup_completed = True
                    # Reset optimizer learning rates to initial values before starting cosine restarts
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = LEARNING_RATE * (2.0 if 'classifier' in str(id(param_group)) else (0.5 if 'transformer' in str(id(param_group)) else 1.0))
                scheduler_restarts.step()
            optimizer.zero_grad()
            global_step += 1
    # Handle remaining accumulation
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        # Warmup then cosine restarts
        if global_step < warmup_steps:
            scheduler_warmup.step()
        else:
            if not warmup_completed:
                warmup_completed = True
            scheduler_restarts.step()
        optimizer.zero_grad()
        global_step += 1

    avg_loss = total_loss / num_batches
    val_stats, val_probs, val_embeddings = evaluate(model, val_loader, criterion)
    val_logloss = val_stats["log_loss"]
    val_acc = val_stats["accuracy"]
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        best_val_embeddings = val_embeddings.copy()
        best_val_probs = val_probs.copy()
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_roberta_model.pt")
        print(f"  → New best model saved (log_loss: {val_logloss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping triggered. Best epoch: {best_epoch}, Best val log_loss: {best_val_logloss:.4f}"
            )
            break

print(
    f"\nBest model from epoch {best_epoch} with validation log_loss: {best_val_logloss:.4f}"
)

# ============================================================
# 12. LOAD BEST MODEL AND EVALUATE
# ============================================================
state_dict = torch.load(f"{WORKING_DIR}/best_roberta_model.pt", map_location=device)
model_state = model.state_dict()
filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
unexpected_keys = [k for k in state_dict if k not in model_state]
missing_keys = [k for k in model_state if k not in state_dict]
if unexpected_keys:
    print(f"  Warning: {len(unexpected_keys)} unexpected keys in checkpoint (e.g., {unexpected_keys[:3]})")
if missing_keys:
    print(f"  Warning: {len(missing_keys)} missing keys in checkpoint (e.g., {missing_keys[:3]}) - these will use random init")
model.load_state_dict(filtered, strict=False)

# Re-run evaluation to get best val embeddings and probabilities
val_stats, best_val_probs, best_val_embeddings = evaluate(model, val_loader, criterion)
final_val_logloss = val_stats["log_loss"]
print(f"Best Model Validation LogLoss: {final_val_logloss:.6f}")

# ============================================================
# 13. TEST-TIME AUGMENTATION (TTA) FOR TEST SET INFERENCE
# ============================================================
print("Generating RoBERTa test embeddings and predictions with TTA...")

def apply_test_augmentations(text):
    """Apply random punctuation replacement (10%), function word dropout (10%),
    and character-level noise (1% drop/replace) for test-time augmentation."""
    import random
    text = str(text)
    chars = list(text)

    # 1. Random punctuation replacement (10% of punctuation chars)
    punct_chars = set(string.punctuation)
    punct_indices = [i for i, c in enumerate(chars) if c in punct_chars]
    for idx in punct_indices:
        if random.random() < 0.10:
            chars[idx] = random.choice(string.punctuation)

    # 2. Random function word dropout (10% probability per function word occurrence)
    words = ''.join(chars).split()
    new_words = []
    for w in words:
        w_clean = w.lower().strip(string.punctuation)
        if w_clean in FUNCTION_WORDS and random.random() < 0.10:
            continue  # Drop the function word
        new_words.append(w)
    augmented_text = ' '.join(new_words)

    # 3. Character-level noise: 1% character drop or replace
    chars_aug = list(augmented_text)
    char_indices = list(range(len(chars_aug)))
    target_count = max(1, int(len(chars_aug) * 0.01))
    indices_to_modify = random.sample(char_indices, min(target_count, len(char_indices)))
    for idx in indices_to_modify:
        if random.random() < 0.5:
            # Drop character
            chars_aug[idx] = ''
        else:
            # Replace with a random printable character (same type if possible)
            orig = chars_aug[idx]
            if orig.isalpha():
                chars_aug[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            elif orig.isdigit():
                chars_aug[idx] = random.choice('0123456789')
            else:
                chars_aug[idx] = random.choice(string.punctuation + ' ')
    augmented_text = ''.join(chars_aug)
    return augmented_text

# Store per-sample averaged embeddings and logits for LightGBM
NUM_TTA = 4

# We process in batches: for each test sample, generate NUM_TTA augmentations,
# tokenize and run model, then average predictions and embeddings.
all_test_logits_tta = []
all_test_embeddings_tta = []
all_test_tta_logits_list = []  # For per-augmentation tracking if needed

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids_orig = batch["input_ids"]
        attention_mask_orig = batch["attention_mask"]
        batch_size = input_ids_orig.size(0)

        # Get original predictions first (always included)
        input_ids_orig_dev = input_ids_orig.to(device)
        attention_mask_orig_dev = attention_mask_orig.to(device)
        with autocast():
            logits_orig, embeds_orig = model(
                input_ids=input_ids_orig_dev,
                attention_mask=attention_mask_orig_dev,
                return_embeds=True,
            )

        # Accumulate original + augmented predictions
        accumulated_logits = logits_orig.cpu()
        accumulated_embeds = embeds_orig.cpu()

        # Get the original texts for this batch to augment
        start_idx = batch_idx * test_loader.batch_size
        end_idx = min(start_idx + batch_size, len(X_test_texts))
        batch_texts = list(X_test_texts[start_idx:end_idx])

        # Generate augmented versions
        for aug_idx in range(NUM_TTA):
            augmented_texts = [apply_test_augmentations(t) for t in batch_texts]
            aug_encodings = tokenizer(
                list(augmented_texts),
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            aug_input_ids = aug_encodings["input_ids"].to(device)
            aug_attention_mask = aug_encodings["attention_mask"].to(device)
            with autocast():
                aug_logits, aug_embeds = model(
                    input_ids=aug_input_ids,
                    attention_mask=aug_attention_mask,
                    return_embeds=True,
                )
            accumulated_logits += aug_logits.cpu()
            accumulated_embeds += aug_embeds.cpu()

        # Average over original + NUM_TTA augmentations
        avg_logits = accumulated_logits / (NUM_TTA + 1)
        avg_embeds = accumulated_embeds / (NUM_TTA + 1)

        all_test_logits_tta.append(avg_logits.numpy())
        all_test_embeddings_tta.append(avg_embeds.numpy())

all_test_logits = np.vstack(all_test_logits_tta)
all_test_embeddings = np.vstack(all_test_embeddings_tta)
test_probs_roberta = np.exp(all_test_logits) / np.exp(all_test_logits).sum(
    axis=1, keepdims=True
)
print(f"TTA applied: original + {NUM_TTA} augmentations per sample")

# ============================================================
# 14. TRAIN LIGHTGBM META-CLASSIFIER
# ============================================================
print("\nTraining LightGBM meta-classifier...")

try:
    import lightgbm as lgb

    # Concatenate RoBERTa embeddings with handcrafted features for validation
    # Also add character n-gram features (n=3-6, max_features=2000) as per improvement plan
    from sklearn.feature_extraction.text import TfidfVectorizer
    char_ngram_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 6),
        max_features=2000,
        sublinear_tf=True,
        norm='l2',
        use_idf=True,
    )
    train_char_ngrams = char_ngram_vectorizer.fit_transform(X_train_texts)
    val_char_ngrams = char_ngram_vectorizer.transform(X_val_texts)
    test_char_ngrams = char_ngram_vectorizer.transform(X_test_texts)

    # Convert sparse n-gram features to dense numpy arrays and concatenate
    val_meta_features = np.hstack([
        best_val_embeddings,
        val_handcrafted.numpy(),
        val_char_ngrams.toarray(),
    ])
    test_meta_features = np.hstack([
        all_test_embeddings,
        test_handcrafted.numpy(),
        test_char_ngrams.toarray(),
    ])

    # Train one LightGBM model per class (OvR) with early stopping
    lgb_models = []
    val_meta_preds = np.zeros((len(val_meta_features), NUM_AUTHORS))

    for class_idx in range(NUM_AUTHORS):
        print(f"  Training LightGBM for class {label_encoder.classes_[class_idx]}...")
        y_binary = (y_val_labels == class_idx).astype(int)

        # Split meta features into train/val for LightGBM early stopping
        lgb_train_idx, lgb_val_idx = train_test_split(
            np.arange(len(val_meta_features)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_binary,
        )
        lgb_train_x = val_meta_features[lgb_train_idx]
        lgb_val_x = val_meta_features[lgb_val_idx]
        lgb_train_y = y_binary[lgb_train_idx]
        lgb_val_y = y_binary[lgb_val_idx]

        dtrain = lgb.Dataset(lgb_train_x, label=lgb_train_y)
        dval = lgb.Dataset(lgb_val_x, label=lgb_val_y, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }

        model_lgb = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        lgb_models.append(model_lgb)
        val_meta_preds[:, class_idx] = model_lgb.predict(val_meta_features)

    # Normalize validation meta predictions to probabilities
    val_meta_probs = np.exp(val_meta_preds) / np.exp(val_meta_preds).sum(axis=1, keepdims=True)

    # Get test predictions from LightGBM
    test_meta_preds = np.zeros((len(test_meta_features), NUM_AUTHORS))
    for class_idx, model_lgb in enumerate(lgb_models):
        test_meta_preds[:, class_idx] = model_lgb.predict(test_meta_features)

    test_meta_probs = np.exp(test_meta_preds) / np.exp(test_meta_preds).sum(axis=1, keepdims=True)

    # Ensemble: average RoBERTa and LightGBM predictions
    test_probs = (test_probs_roberta + test_meta_probs) / 2.0

    # Compute ensemble validation log-loss for verification
    ensemble_val_probs = (best_val_probs + val_meta_probs) / 2.0
    ensemble_val_logloss = compute_log_loss(y_val_labels, ensemble_val_probs)
    print(f"  RoBERTa only val log-loss: {final_val_logloss:.6f}")
    print(f"  LightGBM meta val log-loss: {compute_log_loss(y_val_labels, val_meta_probs):.6f}")
    print(f"  Ensemble val log-loss: {ensemble_val_logloss:.6f}")

except ImportError:
    print("LightGBM not installed. Using only RoBERTa predictions.")
    test_probs = test_probs_roberta

# ============================================================
# 15. SAVE SUBMISSION
# ============================================================
eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSubmission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# 16. FINAL OUTPUT
# ============================================================
if 'ensemble_val_logloss' in locals():
    print(f"Final Ensemble Validation Score: {ensemble_val_logloss:.6f}")
else:
    print(f"Final Validation Score: {final_val_logloss:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()