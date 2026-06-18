import os
os.sched_setaffinity(0, {32, 28, 29, 30, 31})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# INITIAL DATA CLEANING
# ============================================================
train_df["text"] = train_df["text"].fillna("")
test_df["text"] = test_df["text"].fillna("")


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = text.strip()
    return text


train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
train_df["char_count"] = train_df["text"].str.len()
train_df["word_count"] = train_df["text"].str.split().str.len()
train_df["avg_word_length"] = train_df["char_count"] / (train_df["word_count"] + 1)
train_df["sentence_count"] = train_df["text"].str.count(r"[.!?]+")
train_df["avg_sentence_length"] = train_df["word_count"] / (
    train_df["sentence_count"] + 1
)

test_df["char_count"] = test_df["text"].str.len()
test_df["word_count"] = test_df["text"].str.split().str.len()
test_df["avg_word_length"] = test_df["char_count"] / (test_df["word_count"] + 1)
test_df["sentence_count"] = test_df["text"].str.count(r"[.!?]+")
test_df["avg_sentence_length"] = test_df["word_count"] / (test_df["sentence_count"] + 1)


def count_punctuation(text, punct_pattern):
    return len(re.findall(punct_pattern, text))


for df in [train_df, test_df]:
    df["period_count"] = df["text"].apply(lambda x: count_punctuation(x, r"\."))
    df["exclamation_count"] = df["text"].apply(lambda x: count_punctuation(x, r"!"))
    df["question_count"] = df["text"].apply(lambda x: count_punctuation(x, r"\?"))
    df["dash_count"] = df["text"].apply(lambda x: count_punctuation(x, r"—|–|-"))
    df["semicolon_count"] = df["text"].apply(lambda x: count_punctuation(x, r";"))
    df["colon_count"] = df["text"].apply(lambda x: count_punctuation(x, r":"))
    df["comma_count"] = df["text"].apply(lambda x: count_punctuation(x, r","))
    df["quote_count"] = df["text"].apply(lambda x: count_punctuation(x, r'"'))
    df["apostrophe_count"] = df["text"].apply(lambda x: count_punctuation(x, r"'"))
    df["paren_count"] = df["text"].apply(lambda x: count_punctuation(x, r"\(|\)"))
    df["ellipsis_count"] = df["text"].apply(lambda x: count_punctuation(x, r"\.{3,}"))

for df in [train_df, test_df]:
    df["punctuation_ratio"] = (
        df["period_count"]
        + df["comma_count"]
        + df["semicolon_count"]
        + df["colon_count"]
        + df["dash_count"]
        + df["exclamation_count"]
        + df["question_count"]
        + df["quote_count"]
    ) / (df["word_count"] + 1)

for df in [train_df, test_df]:
    df["capital_letters"] = df["text"].apply(lambda x: len(re.findall(r"[A-Z]", x)))
    df["capital_ratio"] = df["capital_letters"] / (df["char_count"] + 1)
    df["all_caps_words"] = df["text"].apply(
        lambda x: len(re.findall(r"\b[A-Z]{2,}\b", x))
    )
    df["proper_noun_ratio"] = df["text"].apply(
        lambda x: len(re.findall(r"(?<!\w\.\s)(?<![A-Za-z])[A-Z][a-z]+", x))
    ) / (df["word_count"] + 1)

archaic_words = [
    "thee",
    "thou",
    "thy",
    "thine",
    "hath",
    "doth",
    "dost",
    "canst",
    "wilt",
    "shall",
    "shalt",
    "art",
    "wert",
    "hast",
    "didst",
    "hadst",
    "couldst",
    "wouldst",
    "shouldst",
    "mightst",
    "cometh",
    "goeth",
    "maketh",
    "taketh",
    "giveth",
    "speaketh",
    "knoweth",
    "loveth",
    "saith",
    "dwell",
    "hither",
    "thither",
    "whither",
    "thence",
    "whence",
    "hence",
    "anon",
    "ere",
    "oft",
    "forsooth",
    "perchance",
    "perforce",
    "betimes",
    "methinks",
    "prithee",
    "wherefore",
    "therefor",
    "therewith",
    "herewith",
    "wherewith",
]
archaic_set = set(archaic_words)


def count_archaic_words(text):
    words = text.lower().split()
    return sum(1 for w in words if w in archaic_set)


for df in [train_df, test_df]:
    df["archaic_word_count"] = df["text"].apply(count_archaic_words)
    df["archaic_word_ratio"] = df["archaic_word_count"] / (df["word_count"] + 1)

horror_words_eap = [
    "death",
    "dead",
    "terror",
    "horror",
    "dread",
    "fear",
    "dark",
    "shadow",
    "ghost",
    "spectre",
    "phantom",
    "spirit",
    "soul",
    "gloom",
    "mournful",
    "melancholy",
    "madness",
    "insanity",
    "lunatic",
    "raving",
    "delirium",
    "ravenous",
    "vulture",
    "raven",
    "tomb",
    "sepulchre",
    "coffin",
    "grave",
    "desolate",
    "despair",
    "agonizing",
    "torment",
    "anguish",
    "woe",
    "weary",
    "dreary",
    "bleak",
    "pallid",
    "ghastly",
    "hideous",
    "grotesque",
]
horror_words_hpl = [
    "elder",
    "old",
    "ancient",
    "primordial",
    "eldritch",
    "cyclopean",
    "non-euclidean",
    "crawling",
    "chaos",
    "void",
    "abyss",
    "rlyeh",
    "cthulhu",
    "yog-sothoth",
    "nyarlathotep",
    "azathoth",
    "necronomicon",
    "unspeakable",
    "unnameable",
    "unutterable",
    "indescribable",
    "nameless",
    "cosmic",
    "infinite",
    "eternal",
    "immortal",
    "blasphemous",
    "blasphemy",
    "gate",
    "dimension",
    "entity",
    "being",
    "existence",
    "consciousness",
    "dream",
    "nightmare",
    "sleep",
    "slumber",
    "awakening",
    "revealed",
]
horror_words_mws = [
    "science",
    "scientific",
    "experiment",
    "discovery",
    "knowledge",
    "creation",
    "creature",
    "monster",
    "demon",
    "fiend",
    "daemon",
    "power",
    "nature",
    "natural",
    "philosophy",
    "principle",
    "theory",
    "life",
    "death",
    "immortal",
    "soul",
    "spirit",
    "vital",
    "spark",
    "lightning",
    "electric",
    "galvanism",
    "anatomy",
    "dissection",
    "corpse",
    "body",
    "limb",
    "form",
    "shape",
    "figure",
    "being",
]

for df in [train_df, test_df]:
    words_lower = df["text"].str.lower()
    df["horror_eap_count"] = words_lower.apply(
        lambda x: sum(1 for w in horror_words_eap if w in x)
    )
    df["horror_hpl_count"] = words_lower.apply(
        lambda x: sum(1 for w in horror_words_hpl if w in x)
    )
    df["horror_mws_count"] = words_lower.apply(
        lambda x: sum(1 for w in horror_words_mws if w in x)
    )
    df["horror_total"] = (
        df["horror_eap_count"] + df["horror_hpl_count"] + df["horror_mws_count"]
    )

try:
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
except:
    stop_words = set(
        [
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
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
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
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
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
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
        ]
    )

for df in [train_df, test_df]:
    df["stop_word_count"] = (
        df["text"]
        .str.lower()
        .str.split()
        .apply(lambda x: sum(1 for w in x if w in stop_words))
    )
    df["stop_word_ratio"] = df["stop_word_count"] / (df["word_count"] + 1)


def count_suffix(text, suffixes):
    words = text.lower().split()
    return sum(1 for w in words if any(w.endswith(s) for s in suffixes))


for df in [train_df, test_df]:
    df["adjective_like"] = df["text"].apply(
        lambda x: count_suffix(
            x, ["ous", "ful", "less", "ive", "able", "ible", "al", "ic", "ish"]
        )
    )
    df["adverb_like"] = df["text"].apply(
        lambda x: count_suffix(x, ["ly", "ward", "wise"])
    )
    df["verb_past"] = df["text"].apply(lambda x: count_suffix(x, ["ed", "t", "en"]))
    df["verb_ing"] = df["text"].apply(lambda x: count_suffix(x, ["ing"]))
    df["suffix_ratio"] = (
        df["adjective_like"] + df["adverb_like"] + df["verb_past"] + df["verb_ing"]
    ) / (df["word_count"] + 1)

first_person_words = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}
third_person_words = {
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
}

for df in [train_df, test_df]:
    words_lower = df["text"].str.lower().str.split()
    df["first_person_count"] = words_lower.apply(
        lambda x: sum(1 for w in x if w in first_person_words)
    )
    df["third_person_count"] = words_lower.apply(
        lambda x: sum(1 for w in x if w in third_person_words)
    )
    df["person_ratio"] = (df["first_person_count"] + 1) / (df["third_person_count"] + 1)

for df in [train_df, test_df]:
    df["unique_words"] = df["text"].str.lower().str.split().apply(lambda x: len(set(x)))
    df["type_token_ratio"] = df["unique_words"] / (df["word_count"] + 1)
    df["long_words"] = (
        df["text"].str.split().apply(lambda x: sum(1 for w in x if len(w) > 8))
    )
    df["long_word_ratio"] = df["long_words"] / (df["word_count"] + 1)

# ============================================================
# TF-IDF N-GRAM FEATURES
# ============================================================
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
)
tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    sublinear_tf=True,
    strip_accents="unicode",
    min_df=5,
    max_df=0.95,
)

n_components = 50
svd_char = TruncatedSVD(n_components=n_components, random_state=42)
svd_word = TruncatedSVD(n_components=n_components, random_state=42)

scaler = StandardScaler()

# ============================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION SETUP
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Full training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Fit TF-IDF and SVD on FULL TRAINING SET (no leakage for feature engineering)
train_char_features = tfidf_char.fit_transform(train_df["text"])
train_word_features = tfidf_word.fit_transform(train_df["text"])

test_char_features = tfidf_char.transform(test_df["text"])
test_word_features = tfidf_word.transform(test_df["text"])

train_char_svd = svd_char.fit_transform(train_char_features)
train_word_svd = svd_word.fit_transform(train_word_features)

test_char_svd = svd_char.transform(test_char_features)
test_word_svd = svd_word.transform(test_word_features)

for i in range(n_components):
    train_df[f"char_svd_{i}"] = train_char_svd[:, i]
    test_df[f"char_svd_{i}"] = test_char_svd[:, i]
    train_df[f"word_svd_{i}"] = train_word_svd[:, i]
    test_df[f"word_svd_{i}"] = test_word_svd[:, i]

feature_columns = [col for col in train_df.columns if col not in ["id", "text", "author"]]
scaler_full = StandardScaler()
train_df[feature_columns] = scaler_full.fit_transform(train_df[feature_columns])
test_df[feature_columns] = scaler_full.transform(test_df[feature_columns])

print(f"Number of features: {len(feature_columns)}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# VOCABULARY LOOKUP FOR AUXILIARY FEATURES
# ============================================================
archaic_words_set = set(archaic_words)
horror_eap_set = set(horror_words_eap)
horror_hpl_set = set(horror_words_hpl)
horror_mws_set = set(horror_words_mws)


def compute_vocab_densities(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0, 0.0, 0.0, 0.0
    words = text.lower().split()
    word_count = len(words) if len(words) > 0 else 1
    archaic = sum(1 for w in words if w in archaic_words_set) / word_count
    eap_horror = sum(1 for w in words if w in horror_eap_set) / word_count
    hpl_horror = sum(1 for w in words if w in horror_hpl_set) / word_count
    mws_horror = sum(1 for w in words if w in horror_mws_set) / word_count
    return archaic, eap_horror, hpl_horror, mws_horror


# ============================================================
# DATASET WITH AUXILIARY FEATURES
# ============================================================
class SpookyDataset(Dataset):
    def __init__(
        self, texts, labels=None, tokenizer=None, max_length=256, compute_aux=False
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compute_aux = compute_aux
        if compute_aux:
            self.char_maps = []
            self.archaic_densities = []
            self.eap_horror_densities = []
            self.hpl_horror_densities = []
            self.mws_horror_densities = []
            for text in texts:
                char_ids = []
                for ch in str(text)[:200]:
                    char_ids.append(ord(ch) % 100 + 1)
                max_char_len = 200
                if len(char_ids) < max_char_len:
                    char_ids = char_ids + [0] * (max_char_len - len(char_ids))
                else:
                    char_ids = char_ids[:max_char_len]
                self.char_maps.append(torch.tensor(char_ids, dtype=torch.long))
                archaic, eap, hpl, mws = compute_vocab_densities(text)
                self.archaic_densities.append(archaic)
                self.eap_horror_densities.append(eap)
                self.hpl_horror_densities.append(hpl)
                self.mws_horror_densities.append(mws)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
        }
        if self.compute_aux:
            item["char_ids"] = self.char_maps[idx]
            item["archaic_density"] = torch.tensor(
                self.archaic_densities[idx], dtype=torch.float
            )
            item["horror_eap_density"] = torch.tensor(
                self.eap_horror_densities[idx], dtype=torch.float
            )
            item["horror_hpl_density"] = torch.tensor(
                self.hpl_horror_densities[idx], dtype=torch.float
            )
            item["horror_mws_density"] = torch.tensor(
                self.mws_horror_densities[idx], dtype=torch.float
            )
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class PunctuationPatternEncoder(nn.Module):
    def __init__(
        self,
        char_vocab_size=101,
        char_embed_dim=32,
        num_filters=64,
        kernel_sizes=[2, 3, 4],
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=0
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )
        self.projection = nn.Linear(len(kernel_sizes) * num_filters, 64)

    def forward(self, char_ids):
        embedded = self.char_embedding(char_ids)
        embedded = embedded.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)
            conv_out = torch.relu(conv_out)
            conv_out, _ = conv_out.max(dim=-1)
            conv_outputs.append(conv_out)
        concatenated = torch.cat(conv_outputs, dim=1)
        out = self.projection(concatenated)
        return out


class ArchaicVocabularyDetector(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.archaic_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.horror_eap_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.horror_hpl_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.horror_mws_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.density_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 64)
        )
        self.fusion = nn.Linear(embed_dim + 64, 64)

    def forward(
        self,
        archaic_density,
        horror_eap_density,
        horror_hpl_density,
        horror_mws_density,
    ):
        density_features = torch.stack(
            [
                archaic_density,
                horror_eap_density,
                horror_hpl_density,
                horror_mws_density,
            ],
            dim=1,
        )
        density_encoded = self.density_encoder(density_features)
        batch_size = density_features.size(0)
        vocab_embed = torch.cat(
            [
                self.archaic_embed.expand(batch_size, -1),
                self.horror_eap_embed.expand(batch_size, -1),
                self.horror_hpl_embed.expand(batch_size, -1),
                self.horror_mws_embed.expand(batch_size, -1),
            ],
            dim=1,
        )
        weights = torch.stack(
            [
                archaic_density,
                horror_eap_density,
                horror_hpl_density,
                horror_mws_density,
            ],
            dim=-1,
        ).unsqueeze(1)
        vocab_embed = vocab_embed.view(batch_size, 4, -1)
        weighted_vocab = torch.bmm(weights, vocab_embed).squeeze(1)
        combined = torch.cat([weighted_vocab, density_encoded], dim=-1)
        out = self.fusion(combined)
        return out


class MultiLayerAttentionPooling(nn.Module):
    def __init__(self, hidden_size, projection_dim=256, num_layers=4):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.query = nn.Parameter(torch.randn(1, 1, projection_dim) * 0.02)
        self.scale = projection_dim ** -0.5

    def forward(self, hidden_states_list, attention_mask):
        # hidden_states_list: list of layer outputs, each (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len) - 1 for tokens, 0 for padding
        batch_size = hidden_states_list[0].size(0)
        seq_len = hidden_states_list[0].size(1)
        device = hidden_states_list[0].device

        # Stack and project: (batch, num_layers * seq_len, hidden) -> (batch, num_layers * seq_len, proj_dim)
        stacked = torch.stack(hidden_states_list, dim=1)  # (batch, num_layers, seq_len, hidden)
        num_layers = stacked.size(1)
        stacked = stacked.view(batch_size, num_layers * seq_len, -1)  # (batch, num_layers*seq_len, hidden)
        projected = self.projection(stacked)  # (batch, num_layers*seq_len, proj_dim)

        # Compute attention weights
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, proj_dim)
        attn_scores = torch.bmm(query, projected.transpose(1, 2)) * self.scale  # (batch, 1, num_layers*seq_len)

        # Create extended attention mask
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        extended_mask = extended_mask.expand(-1, -1, num_layers, -1)  # (batch, 1, num_layers, seq_len)
        extended_mask = extended_mask.reshape(batch_size, 1, num_layers * seq_len)
        attn_scores = attn_scores.masked_fill(extended_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, 1, num_layers*seq_len)

        # Weighted sum
        pooled = torch.bmm(attn_weights, projected).squeeze(1)  # (batch, proj_dim)
        return pooled


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim=256, aux_dim=128, fusion_dim=256, num_heads=4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.aux_proj = nn.Linear(aux_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.output_proj = nn.Linear(fusion_dim, 256)

    def forward(self, text_repr, aux_features):
        # text_repr: (batch, text_dim) - query
        # aux_features: (batch, aux_dim) - key/value
        query = self.text_proj(text_repr).unsqueeze(1)  # (batch, 1, fusion_dim)
        key_value = self.aux_proj(aux_features).unsqueeze(1)  # (batch, 1, fusion_dim)
        fused, _ = self.cross_attn(query, key_value, key_value)
        out = self.output_proj(fused.squeeze(1))
        return out


class StyleAwareClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.punctuation_encoder = PunctuationPatternEncoder()
        self.vocabulary_detector = ArchaicVocabularyDetector()
        self.aux_projection = nn.Sequential(
            nn.Linear(64 + 64, 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.pooling = MultiLayerAttentionPooling(
            hidden_size=hidden_size, projection_dim=256, num_layers=4
        )
        self.cross_attn_fusion = CrossAttentionFusion(
            text_dim=256, aux_dim=128, fusion_dim=256, num_heads=4
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_authors),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        char_ids=None,
        archaic_density=None,
        horror_eap_density=None,
        horror_hpl_density=None,
        horror_mws_density=None,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        # Use last 4 layers: layers 20,21,22,23 (indices -4, -3, -2, -1)
        last_layers = [hidden_states[-4], hidden_states[-3], hidden_states[-2], hidden_states[-1]]
        text_repr = self.pooling(last_layers, attention_mask)
        if char_ids is not None:
            punct_features = self.punctuation_encoder(char_ids)
        else:
            punct_features = torch.zeros(
                text_repr.size(0), 64, device=text_repr.device
            )
        if archaic_density is not None:
            vocab_features = self.vocabulary_detector(
                archaic_density,
                horror_eap_density,
                horror_hpl_density,
                horror_mws_density,
            )
        else:
            vocab_features = torch.zeros(
                text_repr.size(0), 64, device=text_repr.device
            )
        aux_features = self.aux_projection(
            torch.cat([punct_features, vocab_features], dim=-1)
        )
        # Fuse using cross-attention: text_repr as Query, aux_features as Key/Value
        fused = self.cross_attn_fusion(text_repr, aux_features)
        logits = self.classifier(fused)
        return logits


# ============================================================
# 5-FOLD CROSS-VALIDATION WITH OneCycleLR
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
max_length = 256

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

batch_size = 8
num_epochs = 30
patience = 3

# Prepare test dataset once
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length, compute_aux=True)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# Store fold predictions
fold_test_probs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["author"])):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*50}")

    train_texts_final = train_texts_orig[train_idx]
    train_labels_final = train_labels_orig[train_idx]
    val_texts_final = train_texts_orig[val_idx]
    val_labels_final = train_labels_orig[val_idx]

    train_dataset = SpookyDataset(
        train_texts_final, train_labels_final, tokenizer, max_length, compute_aux=True
    )
    val_dataset = SpookyDataset(
        val_texts_final, val_labels_final, tokenizer, max_length, compute_aux=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model for each fold
    model = StyleAwareClassifier(num_authors=3, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    backbone_unfrozen_params = []
    for layer in model.backbone.encoder.layer[-4:]:
        for name, param in layer.named_parameters():
            if "bias" not in name and "LayerNorm" not in name:
                backbone_unfrozen_params.append(param)

    aux_params = (
        list(model.punctuation_encoder.parameters())
        + list(model.vocabulary_detector.parameters())
        + list(model.aux_projection.parameters())
    )

    head_params = list(model.classifier.parameters())

    # New pooling module parameters
    pooling_params = list(model.pooling.parameters())
    cross_attn_params = list(model.cross_attn_fusion.parameters())

    norm_params = []
    for layer in model.backbone.encoder.layer[-4:]:
        for name, param in layer.named_parameters():
            if "LayerNorm" in name:
                norm_params.append(param)

    optimizer = AdamW(
        [
            {
                "params": backbone_unfrozen_params,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {"params": aux_params, "lr": 1e-4, "weight_decay": 0.01, "betas": (0.9, 0.999)},
            {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
            {"params": pooling_params, "lr": 1e-4, "weight_decay": 0.01, "betas": (0.9, 0.999)},
            {"params": cross_attn_params, "lr": 1e-4, "weight_decay": 0.01, "betas": (0.9, 0.999)},
            {"params": norm_params, "lr": 5e-5, "weight_decay": 0.0, "betas": (0.9, 0.999)},
        ]
    )

    print(f"Fold {fold+1} - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Fold {fold+1} - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-5, 1e-4, 5e-5, 1e-4, 1e-4, 5e-5],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Training loop for this fold
    best_val_score = float("inf")
    epochs_no_improve = 0
    scaler_grad = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            char_ids = batch["char_ids"].to(device)
            archaic_density = batch["archaic_density"].to(device)
            horror_eap_density = batch["horror_eap_density"].to(device)
            horror_hpl_density = batch["horror_hpl_density"].to(device)
            horror_mws_density = batch["horror_mws_density"].to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(
                    input_ids,
                    attention_mask,
                    char_ids=char_ids,
                    archaic_density=archaic_density,
                    horror_eap_density=horror_eap_density,
                    horror_hpl_density=horror_hpl_density,
                    horror_mws_density=horror_mws_density,
                )
                loss = criterion(logits, labels)

            scaler_grad.scale(loss).backward()
            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()
            scheduler.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                char_ids = batch["char_ids"].to(device)
                archaic_density = batch["archaic_density"].to(device)
                horror_eap_density = batch["horror_eap_density"].to(device)
                horror_hpl_density = batch["horror_hpl_density"].to(device)
                horror_mws_density = batch["horror_mws_density"].to(device)

                with autocast():
                    logits = model(
                        input_ids,
                        attention_mask,
                        char_ids=char_ids,
                        archaic_density=archaic_density,
                        horror_eap_density=horror_eap_density,
                        horror_hpl_density=horror_hpl_density,
                        horror_mws_density=horror_mws_density,
                    )
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)

                total_val_loss += loss.item()
                num_val_batches += 1
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / num_val_batches
        val_probs = np.concatenate(all_val_probs, axis=0)
        val_true = np.concatenate(all_val_labels, axis=0)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
        val_score = log_loss(val_true, val_probs_clipped)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Fold {fold+1} | Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"./working/best_model_fold{fold+1}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Fold {fold+1} - Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model for this fold and predict on test
    model.load_state_dict(torch.load(f"./working/best_model_fold{fold+1}.pt"))
    model.eval()

    fold_test_probs_fold = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            archaic_density = batch["archaic_density"].to(device)
            horror_eap_density = batch["horror_eap_density"].to(device)
            horror_hpl_density = batch["horror_hpl_density"].to(device)
            horror_mws_density = batch["horror_mws_density"].to(device)

            with autocast():
                logits = model(
                    input_ids,
                    attention_mask,
                    char_ids=char_ids,
                    archaic_density=archaic_density,
                    horror_eap_density=horror_eap_density,
                    horror_hpl_density=horror_hpl_density,
                    horror_mws_density=horror_mws_density,
                )
                probs = torch.softmax(logits, dim=1)
            fold_test_probs_fold.append(probs.cpu().numpy())

    fold_test_probs.append(np.concatenate(fold_test_probs_fold, axis=0))
    print(f"Fold {fold+1} - Best Val LogLoss: {best_val_score:.4f}")

# ============================================================
# ENSEMBLE PREDICTIONS ACROSS FOLDS
# ============================================================
test_probs = np.mean(fold_test_probs, axis=0)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission_66c7d03465ad4b33b1f6f78ca5b49eb3.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"5-fold CV Ensemble complete. Average of 5 fold predictions.")