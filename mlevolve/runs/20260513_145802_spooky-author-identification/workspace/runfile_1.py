import os
os.sched_setaffinity(0, {10, 11, 20, 21, 22})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import re
import string
from collections import Counter
import pickle
import os
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


# ============================================================
# 2. TEXT CLEANING FUNCTION
# ============================================================
def clean_text(text):
    """Clean text while preserving stylistic elements"""
    if not isinstance(text, str):
        return ""
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


train_df["cleaned_text"] = train_df["text"].apply(clean_text)
test_df["cleaned_text"] = test_df["text"].apply(clean_text)

# ============================================================
# 3. STYLOMETRIC FEATURE ENGINEERING (FIT ON TRAIN ONLY)
# ============================================================


def extract_stylometric_features(
    text_series, fit_mode=True, word_stats=None, char_stats=None
):
    """Extract author style indicators"""

    features_list = []

    for text in text_series:
        features = {}

        # Basic counts
        features["char_count"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = len(re.split(r"[.!?]+", text)) - 1
        features["avg_word_length"] = features["char_count"] / max(
            features["word_count"], 1
        )
        features["avg_sentence_length"] = features["word_count"] / max(
            features["sentence_count"], 1
        )

        # Punctuation features (Poe/Lovecraft use different punctuation patterns)
        punct_counts = {
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "comma_count": text.count(","),
            "semicolon_count": text.count(";"),
            "colon_count": text.count(":"),
            "dash_count": text.count("-") + text.count("\u2014"),
            "quote_count": text.count('"') + text.count("'"),
            "parenthesis_count": text.count("(") + text.count(")"),
            "ellipsis_count": text.count("...") + text.count("\u2026"),
        }
        features.update(punct_counts)

        # Punctuation ratios
        features["punct_per_word"] = sum(punct_counts.values()) / max(
            features["word_count"], 1
        )
        features["comma_per_sentence"] = punct_counts["comma_count"] / max(
            features["sentence_count"], 1
        )

        # Capitalization patterns (Lovecraft uses more proper nouns)
        words = text.split()
        caps_words = sum(1 for w in words if w[0].isupper() if len(w) > 0)
        features["caps_proportion"] = caps_words / max(len(words), 1)
        features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)

        # Vocabulary richness (different authors have different lexicon sizes)
        unique_words = len(set(w.lower() for w in words))
        features["type_token_ratio"] = unique_words / max(len(words), 1)

        # Stopword ratio (Poe tends to use more articles/prepositions)
        stopwords_set = set(
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
                "it",
                "its",
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
                "their",
                "our",
                "its",
            ]
        )
        word_lower = [w.lower().strip(string.punctuation) for w in words]
        stopword_count = sum(1 for w in word_lower if w in stopwords_set)
        features["stopword_ratio"] = stopword_count / max(len(words), 1)

        # Character n-gram diversity (captures suffix/prefix patterns)
        # Lovecraft uses "-ian", "-ous", "un-" more frequently
        if fit_mode:
            char_2grams = Counter([text[i : i + 2] for i in range(len(text) - 1)])
            char_3grams = Counter([text[i : i + 3] for i in range(len(text) - 2)])
            features["char_2gram_diversity"] = len(char_2grams) / max(
                len(char_2grams.values()), 1
            )
            features["char_3gram_diversity"] = len(char_3grams) / max(
                len(char_3grams.values()), 1
            )
        else:
            features["char_2gram_diversity"] = 0
            features["char_3gram_diversity"] = 0

        # Special character patterns
        features["digit_count"] = sum(1 for c in text if c.isdigit())
        features["special_char_count"] = sum(
            1 for c in text if c in "!@#$%^&*_+=<>?/~`"
        )

        features_list.append(features)

    df_features = pd.DataFrame(features_list)

    # Fill NaN values
    df_features = df_features.fillna(0)

    return df_features


# Extract stylometric features
train_stylo = extract_stylometric_features(train_df["cleaned_text"], fit_mode=True)
test_stylo = extract_stylometric_features(test_df["cleaned_text"], fit_mode=False)

print(f"Stylometric features: {train_stylo.shape[1]}")

# ============================================================
# 4. N-GRAM FEATURE ENGINEERING (FIT ON TRAIN ONLY)
# ============================================================

# Character n-grams: captures syllable patterns, suffix usage
# [2,3,4,5] grams capture character-level stylistic patterns
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    min_df=3,
    max_df=0.95,
)

# Word n-grams: captures phrase patterns
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=8000,
    sublinear_tf=True,
    stop_words="english",
    min_df=2,
    max_df=0.95,
)

# Stopword n-grams: captures unique stopword patterns
# Different authors arrange function words differently
stopword_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=1000,
    token_pattern=r"\b[a-zA-Z]{1,4}\b",  # Short words (mostly stopwords)
    max_df=0.8,
    min_df=3,
)

# Fit on training data only
char_ngrams_train = char_vectorizer.fit_transform(train_df["cleaned_text"])
word_ngrams_train = word_vectorizer.fit_transform(train_df["cleaned_text"])
stopword_ngrams_train = stopword_vectorizer.fit_transform(train_df["cleaned_text"])

# Transform test data
char_ngrams_test = char_vectorizer.transform(test_df["cleaned_text"])
word_ngrams_test = word_vectorizer.transform(test_df["cleaned_text"])
stopword_ngrams_test = stopword_vectorizer.transform(test_df["cleaned_text"])

print(f"Char n-grams: {char_ngrams_train.shape[1]} features")
print(f"Word n-grams: {word_ngrams_train.shape[1]} features")
print(f"Stopword n-grams: {stopword_ngrams_train.shape[1]} features")

# ============================================================
# 5. DYNAMIC FEATURES: Author-specific vocabulary indicators
# ============================================================

# Lovecraft-specific word patterns (cosmic horror vocabulary)
lovecraft_words = [
    "eldritch",
    "cyclopean",
    "non",
    "euclidean",
    "cthulhu",
    "rlyeh",
    "yog",
    "sothoth",
    "nyarlathotep",
    "azathoth",
    "shoggoth",
    "necronomicon",
    "arkham",
    "insmouth",
    "dunwich",
    "miskatonic",
    "antediluvian",
    "prehuman",
    "unspeakable",
    "unnameable",
    "unutterable",
    "blasphemous",
    "nameless",
    "abnormal",
    "gibbous",
    "phosphorescent",
    "squamous",
    "rugose",
    "gelatinous",
    "iridescent",
    "loathsome",
    "ichor",
    "fungoid",
    "miasmal",
    "cloven",
    "swarthy",
    "fetor",
    "amphibious",
    "anthropoid",
    "cephalopod",
    "aeons",
    "lurker",
    "cadence",
    "daemon",
    "demon",
    "abyss",
    "chthonic",
    "outré",
    "monolith",
    "megalith",
    "labyrinthine",
    "maddening",
    "hideous",
    "swarm",
    "what",
]

# Poe-specific word patterns (gothic, psychological)
poe_words = [
    "nevermore",
    "chamber",
    "chambers",
    "raven",
    "usher",
    "amontillado",
    "tell",
    "tale",
    "heart",
    "drooping",
    "bleeding",
    "grotesque",
    "fantastic",
    "dread",
    "dreadful",
    "horror",
    "terror",
    "anguish",
    "agony",
    "despair",
    "melancholy",
    "ghastly",
    "spectral",
    "phantasm",
    "phantom",
    "sepulchre",
    "tomb",
    "vault",
    "crypt",
    "coffin",
    "pallid",
    "livid",
    "cadaverous",
    "dissimulation",
    "premeditated",
    "perverseness",
    "perception",
    "intellect",
    "simile",
    "arabesque",
    "grotesques",
    "berenice",
    "ligeia",
    "annabel",
    "visit",
    "visiter",
    "vertigo",
    "cataleptic",
    "trance",
    "oppressed",
]

# Shelley-specific word patterns (romantic, philosophical)
shelley_words = [
    "frankenstein",
    "creature",
    "monster",
    "daemon",
    "waldman",
    "krempe",
    "clerval",
    "elizabeth",
    "justine",
    "sublime",
    "magnificent",
    "celestial",
    "benevolent",
    "virtuous",
    "philanthropic",
    "philosophical",
    "metaphysical",
    "alpine",
    "glacier",
    "mountain",
    "sublimity",
    "countenance",
    "paternal",
    "filial",
    "domestic",
    "affection",
    "sympathy",
    "compassion",
    "benevolence",
    "ethereal",
    "immortal",
    "transitory",
    "mutability",
    "apocalypse",
    "catastrophe",
    "elemental",
    "irrevocable",
    "irretrievable",
    "luminous",
    "omnipotent",
]

# Use a set for fast lookup (detect variants)
all_author_words = {
    "EAP": set(w.lower() for w in poe_words),
    "HPL": set(w.lower() for w in lovecraft_words),
    "MWS": set(w.lower() for w in shelley_words),
}


def extract_author_vocabulary_features(text_series, author_sets):
    """Count word matches from each author's vocabulary"""
    features_list = []

    for text in text_series:
        features = {}
        words_lower = set(re.findall(r"\b[a-z]+\b", text.lower()))

        for author, vocab_set in author_sets.items():
            matches = len(words_lower & vocab_set)
            features[f"{author}_vocab_matches"] = matches

            # Also check for partial matches (stems)
            partial_matches = 0
            for w in words_lower:
                for v in vocab_set:
                    if v in w or w in v:
                        if w != v:
                            partial_matches += 1
                            break
            features[f"{author}_partial_matches"] = partial_matches

        features_list.append(features)

    return pd.DataFrame(features_list)


train_author_vocab = extract_author_vocabulary_features(
    train_df["cleaned_text"], all_author_words
)
test_author_vocab = extract_author_vocabulary_features(
    test_df["cleaned_text"], all_author_words
)

# ============================================================
# 6. SENTIMENT/EMOTIONAL FEATURES
# ============================================================

# Lexicon-based sentiment scores (authors have different emotional tones)
negative_words = set(
    [
        "dread",
        "terror",
        "horror",
        "fear",
        "fright",
        "panic",
        "alarm",
        "anguish",
        "agony",
        "suffering",
        "pain",
        "misery",
        "despair",
        "hopeless",
        "gloom",
        "darkness",
        "shadow",
        "ominous",
        "sinister",
        "malignant",
        "malevolent",
        "deadly",
        "morbid",
        "macabre",
        "hideous",
        "ghastly",
        "gruesome",
        "brutal",
        "violent",
        "cruel",
        "savage",
        "furious",
        "wrath",
        "hatred",
        "loathing",
        "disgust",
        "abhorrence",
        "revulsion",
        "detestable",
        "vile",
        "wretched",
        "pitiful",
        "lamentable",
        "mournful",
        "sorrowful",
        "weeping",
        "tears",
        "grief",
        "sadness",
        "melancholy",
        "mourning",
        "funereal",
        "sepulchral",
        "startling",
        "shocking",
        "appalling",
        "dreadful",
        "awful",
        "terrible",
        "frightful",
        "fearsome",
        "alarming",
        "chilling",
        "creepy",
        "eerie",
        "uncanny",
        "weird",
        "strange",
        "mysterious",
        "supernatural",
        "occult",
    ]
)

positive_words = set(
    [
        "beautiful",
        "sublime",
        "magnificent",
        "glorious",
        "splendid",
        "radiant",
        "lovely",
        "charming",
        "delightful",
        "joyful",
        "happy",
        "blissful",
        "ecstasy",
        "wonder",
        "marvel",
        "miracle",
        "divine",
        "heavenly",
        "ethereal",
        "celestial",
        "benevolent",
        "virtuous",
        "noble",
        "exalted",
        "pure",
        "innocent",
        "angelic",
        "tender",
        "gentle",
        "peaceful",
        "serene",
        "tranquil",
        "calm",
        "soothing",
        "comfort",
        "consolation",
        "solace",
        "hope",
        "faith",
        "trust",
        "love",
        "affection",
        "passion",
        "ardent",
        "devotion",
        "reverence",
        "adoration",
        "grace",
        "elegance",
        "refinement",
        "polish",
        "delicate",
        "sensitive",
        "kindness",
        "compassion",
        "sympathy",
        "empathy",
        "mercy",
        "forgiveness",
    ]
)


def extract_sentiment_features(text_series):
    """Extract sentiment-related features"""
    features_list = []

    for text in text_series:
        features = {}
        words_lower = re.findall(r"\b[a-z]+\b", text.lower())
        word_set = set(words_lower)

        negative_count = len(word_set & negative_words)
        positive_count = len(word_set & positive_words)
        total_sentiment_words = negative_count + positive_count

        features["negative_word_count"] = negative_count
        features["positive_word_count"] = positive_count
        features["sentiment_word_ratio"] = total_sentiment_words / max(
            len(words_lower), 1
        )
        features["sentiment_bias"] = (positive_count - negative_count) / max(
            total_sentiment_words, 1
        )

        # Emotional intensity (number of emotional words per sentence)
        features["emotional_intensity"] = total_sentiment_words / max(
            len(re.split(r"[.!?]+", text)) - 1, 1
        )

        features_list.append(features)

    return pd.DataFrame(features_list)


train_sentiment = extract_sentiment_features(train_df["cleaned_text"])
test_sentiment = extract_sentiment_features(test_df["cleaned_text"])

# ============================================================
# 7. COMBINE ALL FEATURES INTO SINGLE DATAFRAMES
# ============================================================

# Convert sparse matrices to dense for concatenation (manageable sizes)
X_char_train = char_ngrams_train.toarray()
X_word_train = word_ngrams_train.toarray()
X_stopword_train = stopword_ngrams_train.toarray()

X_char_test = char_ngrams_test.toarray()
X_word_test = word_ngrams_test.toarray()
X_stopword_test = stopword_ngrams_test.toarray()

# Combine all features
train_features = pd.DataFrame(
    X_char_train, columns=[f"char_{i}" for i in range(X_char_train.shape[1])]
)
test_features = pd.DataFrame(
    X_char_test, columns=[f"char_{i}" for i in range(X_char_test.shape[1])]
)

word_train_df = pd.DataFrame(
    X_word_train, columns=[f"word_{i}" for i in range(X_word_train.shape[1])]
)
word_test_df = pd.DataFrame(
    X_word_test, columns=[f"word_{i}" for i in range(X_word_test.shape[1])]
)

stopword_train_df = pd.DataFrame(
    X_stopword_train, columns=[f"stop_{i}" for i in range(X_stopword_train.shape[1])]
)
stopword_test_df = pd.DataFrame(
    X_stopword_test, columns=[f"stop_{i}" for i in range(X_stopword_test.shape[1])]
)

# Reset indices for concatenation
train_stylo.reset_index(drop=True, inplace=True)
train_author_vocab.reset_index(drop=True, inplace=True)
train_sentiment.reset_index(drop=True, inplace=True)

test_stylo.reset_index(drop=True, inplace=True)
test_author_vocab.reset_index(drop=True, inplace=True)
test_sentiment.reset_index(drop=True, inplace=True)

# Combine all features for training
X_train = pd.concat(
    [
        train_features,
        word_train_df,
        stopword_train_df,
        train_stylo,
        train_author_vocab,
        train_sentiment,
    ],
    axis=1,
)

# Combine all features for testing
X_test = pd.concat(
    [
        test_features,
        word_test_df,
        stopword_test_df,
        test_stylo,
        test_author_vocab,
        test_sentiment,
    ],
    axis=1,
)

print(f"Total feature dimensions: {X_train.shape}")

# ============================================================
# 8. HANDLE NAN VALUES AND INFINITY
# ============================================================

# Check for NaN/Inf and fill
X_train = X_train.replace([np.inf, -np.inf], 0)
X_test = X_test.replace([np.inf, -np.inf], 0)

# Fill remaining NaN with 0 (sparse features)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ============================================================
# 9. FEATURE SCALING (FIT ON TRAIN ONLY)
# ============================================================

# Identify non-sparse columns for scaling (density > 10%)
non_sparse_cols = []
for col in X_train.columns:
    density = np.count_nonzero(X_train[col]) / len(X_train)
    if density > 0.1:
        non_sparse_cols.append(col)

print(f"Scaling {len(non_sparse_cols)} non-sparse features")

# Scale non-sparse features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if len(non_sparse_cols) > 0:
    train_scaled = scaler.fit_transform(X_train[non_sparse_cols])
    test_scaled = scaler.transform(X_test[non_sparse_cols])

    X_train_scaled[non_sparse_cols] = train_scaled
    X_test_scaled[non_sparse_cols] = test_scaled

# Save scaler for later use
os.makedirs("./working", exist_ok=True)
with open("./working/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ============================================================
# 10. CREATE STRATIFIED FOLDS
# ============================================================

y = train_df["author"].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = []

for train_idx, val_idx in skf.split(X_train_scaled, y_encoded):
    fold_indices.append((train_idx, val_idx))

# Save fold indices for later use (save as pickle since numpy can't handle inhomogeneous arrays)
with open("./working/fold_indices.pkl", "wb") as f:
    pickle.dump(fold_indices, f)

# ============================================================
# 11. SAVE PREPARED DATA
# ============================================================

# Save features
X_train_scaled.to_parquet("./working/X_train.parquet", index=False)
X_test_scaled.to_parquet("./working/X_test.parquet", index=False)

# Save labels
pd.DataFrame({"label": y_encoded}).to_csv("./working/y_train.csv", index=False)
np.save("./working/label_encoder.npy", label_encoder.classes_)

# Save text data for potential BERT usage
train_df[["id", "cleaned_text", "author"]].to_parquet(
    "./working/train_text.parquet", index=False
)
test_df[["id", "cleaned_text"]].to_parquet("./working/test_text.parquet", index=False)

# Save test IDs for submission
with open("./working/test_ids.pkl", "wb") as f:
    pickle.dump(test_df["id"].values, f)

print(f"Feature engineering complete. Train shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")
print(f"Validation folds: {len(fold_indices)}")

print("\nFeature categories:")
print(f"  - Character n-grams: {train_features.shape[1]}")
print(f"  - Word n-grams: {word_train_df.shape[1]}")
print(f"  - Stopword n-grams: {stopword_train_df.shape[1]}")
print(f"  - Stylometric: {train_stylo.shape[1]}")
print(f"  - Author vocabulary: {train_author_vocab.shape[1]}")
print(f"  - Sentiment: {train_sentiment.shape[1]}")

# ============================================================
# 12. MODEL DESIGN
# ============================================================


class CrossAttentionFusion(nn.Module):
    """Cross-attention between text tokens and stylometric features"""

    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(num_features, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        features: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        text_hidden_states: (batch, seq_len, hidden_dim) - from BERT encoder
        features: (batch, num_features) - stylometric features
        attention_mask: (batch, seq_len) - padding mask
        """
        # Project features to hidden dimension and expand to sequence length
        feat_proj = self.feature_proj(features)  # (batch, hidden_dim)
        feat_expanded = feat_proj.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Cross-attention: text attends to features
        # Query: text tokens, Key/Value: feature representation
        attn_out, _ = self.cross_attn(
            query=text_hidden_states,
            key=feat_expanded,
            value=feat_expanded,
            key_padding_mask=None,  # single feature token
        )

        # Residual connection + layer norm
        fused = self.layer_norm(text_hidden_states + self.dropout(attn_out))
        return fused


class SpookyDualEncoder(nn.Module):
    """
    Dual-encoder architecture:
    1. DeBERTa-v3-large backbone (partially unfrozen)
    2. Stylometric feature encoder with cross-attention fusion
    """

    def __init__(
        self,
        num_authors: int = 3,
        num_features: int = 150,
        dropout_rate: float = 0.3,
        num_cross_heads: int = 4,
    ):
        super().__init__()

        # Load DeBERTa-v3-large config and base model (without classification head)
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        self.config.output_hidden_states = True

        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large", config=self.config
        )

        # CRITICAL: Partial unfreezing - freeze first 16 layers, unfreeze last 8
        for param in self.backbone.parameters():
            param.requires_grad = False

        for layer in self.backbone.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze the embedding layer's LayerNorm
        for param in self.backbone.embeddings.LayerNorm.parameters():
            param.requires_grad = True
        # DeBERTa-v3-large doesn't have absolute position embeddings (uses relative positions)
        if hasattr(self.backbone.embeddings, 'position_embeddings') and self.backbone.embeddings.position_embeddings is not None:
            for param in self.backbone.embeddings.position_embeddings.parameters():
                param.requires_grad = True

        hidden_size = self.config.hidden_size  # 1024

        # Cross-attention fusion module
        self.cross_fusion = CrossAttentionFusion(
            hidden_dim=hidden_size,
            num_features=num_features,
            num_heads=num_cross_heads,
            dropout=dropout_rate,
        )

        # Classification head with multi-layer structure
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_authors),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.cross_fusion, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, features=None):
        # Get transformer hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use all hidden states instead of just last layer
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Apply cross-attention fusion if features are provided
        if features is not None:
            hidden_states = self.cross_fusion(hidden_states, features, attention_mask)

        # Pooling: weighted average pooling with attention mask
        # Expand attention mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(
            dim=1
        ).clamp(min=1e-9)

        # Classification
        logits = self.classifier(pooled)
        return logits


# ============================================================
# 13. TRAINING AND EVALUATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load prepared data
X_train = pd.read_parquet("./working/X_train.parquet")
X_test = pd.read_parquet("./working/X_test.parquet")
# Ensure float32 for consistency (might already be float32 if saved correctly)
for col in X_train.select_dtypes(include=[np.float64]).columns:
    X_train[col] = X_train[col].astype(np.float32)
for col in X_test.select_dtypes(include=[np.float64]).columns:
    X_test[col] = X_test[col].astype(np.float32)
y_train = pd.read_csv("./working/y_train.csv")["label"].values
with open("./working/fold_indices.pkl", "rb") as f:
    fold_indices = pickle.load(f)
with open("./working/test_ids.pkl", "rb") as f:
    test_ids = pickle.load(f)

# Load text data
train_text = pd.read_parquet("./working/train_text.parquet")
test_text = pd.read_parquet("./working/test_text.parquet")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
max_length = 256


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    scheduler,
    warmup_steps,
    initial_lrs,
    epoch,
    total_epochs,
):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        features = batch[2].to(device)
        labels = batch[3].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # CosineAnnealingWarmRestarts: step each batch iteration
        if warmup_steps > 0 and (epoch * len(train_loader) + batch_idx) < warmup_steps:
            current_step = epoch * len(train_loader) + batch_idx
            for pg in optimizer.param_groups:
                pg["lr"] = initial_lrs[0] * (current_step / max(1, warmup_steps))
        else:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader):
    model.eval()
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            features = batch[2].to(device)
            labels = batch[3].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

            val_probs.append(probs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)

    # Probability clipping and normalization
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

    score = log_loss(val_labels, val_probs)
    return score, val_probs, val_labels


def predict(model, loader):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            features = batch[2].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_probs = np.clip(all_probs, 1e-15, 1 - 1e-15)
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)

    return all_probs


# ============================================================
# 5-FOLD CROSS-VALIDATION TRAINING
# ============================================================
num_features = X_train.shape[1]
num_epochs = 30
patience = 5
best_score = float("inf")
no_improve = 0
all_val_scores = []
test_probs_folds = []
num_workers = 2
batch_size = 16
os.makedirs("./working/models", exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*50}")

    # Split data
    train_text_fold = train_text.iloc[train_idx]["cleaned_text"].values
    val_text_fold = train_text.iloc[val_idx]["cleaned_text"].values
    train_features_fold = X_train.iloc[train_idx].values
    val_features_fold = X_train.iloc[val_idx].values
    train_labels_fold = y_train[train_idx]
    val_labels_fold = y_train[val_idx]

    # Tokenize
    train_encodings = tokenizer(
        train_text_fold.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        val_text_fold.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Create datasets
    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(train_features_fold, dtype=torch.float32),
        torch.tensor(train_labels_fold, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(val_features_fold, dtype=torch.float32),
        torch.tensor(val_labels_fold, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize model
    model = SpookyDualEncoder(
        num_authors=3, num_features=num_features, dropout_rate=0.3, num_cross_heads=4
    ).to(device)

    # Setup optimizer with differentiated learning rates
    backbone_params = []
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                backbone_params.append(
                    {"params": param, "lr": 2e-5, "weight_decay": 0.0}
                )
            else:
                backbone_params.append(
                    {"params": param, "lr": 2e-5, "weight_decay": 0.01}
                )

    fusion_params = list(model.cross_fusion.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = AdamW(
        [
            *backbone_params,
            {
                "params": fusion_params,
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
            {
                "params": classifier_params,
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
        ]
    )

    # Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    fold_best_score = float("inf")
    fold_no_improve = 0

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            scheduler,
            warmup_steps,
            initial_lrs,
            epoch,
            num_epochs,
        )

        # Validation
        val_score, _, _ = validate(model, val_loader)

        print(
            f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Log Loss: {val_score:.4f}"
        )

        # Early stopping
        if val_score < fold_best_score:
            fold_best_score = val_score
            fold_no_improve = 0
            torch.save(model.state_dict(), f"./working/models/fold_{fold}_best.pt")
        else:
            fold_no_improve += 1
            if fold_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    all_val_scores.append(fold_best_score)

    # Load best model and get predictions
    model.load_state_dict(torch.load(f"./working/models/fold_{fold}_best.pt"))

    # Validate with best model
    val_score, _, _ = validate(model, val_loader)
    print(f"Fold {fold + 1} Best Val Log Loss: {val_score:.4f}")

    # Test inference
    test_encodings = tokenizer(
        test_text["cleaned_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(X_test.values, dtype=torch.float32),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    fold_test_probs = predict(model, test_loader)
    test_probs_folds.append(fold_test_probs)

    # Clean up
    del model, train_loader, val_loader, train_dataset, val_dataset, test_loader
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# FINAL EVALUATION AND SUBMISSION
# ============================================================
mean_val_score = np.mean(all_val_scores)
std_val_score = np.std(all_val_scores)
print(f"\n{'='*50}")
print(f"Cross-Validation Results:")
print(f"Mean Validation Log Loss: {mean_val_score:.4f} (+/- {std_val_score:.4f})")

# Average test predictions across folds
test_probs = np.mean(test_probs_folds, axis=0)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# Create submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission.to_csv("./submission/submission_88ed11130ce84bbf8c8a9e0ffe3b1d63.csv", index=False)
print(f"\nSubmission saved to ./submission/submission_88ed11130ce84bbf8c8a9e0ffe3b1d63.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample submission:\n{submission.head()}")

# Final validation score
score = mean_val_score
print(f"\nFinal Validation Score: {score}")