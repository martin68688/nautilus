import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import re
import string
from collections import Counter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
all_text = pd.concat([train_df["text"], test_df["text"]], ignore_index=True)


# ============================================================
# 1. STYLOMETRIC FEATURES
# ============================================================
def extract_stylometric_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features["word_count"] = len(words)
    features["char_count"] = len(text)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["sentence_count"] = len(sentences)
    features["avg_sentence_length"] = (
        np.mean([len(s.split()) for s in sentences]) if sentences else 0
    )
    features["char_per_word"] = features["char_count"] / max(features["word_count"], 1)
    punct_count = sum(1 for c in text if c in string.punctuation)
    features["punctuation_density"] = punct_count / max(len(text), 1)
    for punct in [".", "!", "?", ",", ";", ":", "-", '"', "'", "(", ")"]:
        features[f"punct_{punct}_count"] = text.count(punct)
    features["capital_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features["start_capital_ratio"] = sum(
        1 for s in sentences if s and s[0].isupper()
    ) / max(len(sentences), 1)
    features["digit_count"] = sum(1 for c in text if c.isdigit())
    features["special_char_count"] = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    )
    unique_words = set(w.lower() for w in words)
    features["unique_word_ratio"] = len(unique_words) / max(len(words), 1)
    features["lexical_diversity"] = len(unique_words) / max(features["word_count"], 1)
    syllables = sum(1 for w in words for c in w.lower() if c in "aeiou")
    features["syllable_count"] = syllables
    features["flesch_score"] = (
        206.835
        - 1.015 * features["avg_sentence_length"]
        - 84.6 * (syllables / max(features["word_count"], 1))
    )
    stopwords = set(
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
            "have",
            "has",
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
            "shall",
            "can",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "than",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
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
            "mine",
            "yours",
            "hers",
            "theirs",
            "ours",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
        ]
    )
    stopword_count = sum(1 for w in words if w.lower() in stopwords)
    features["stopword_ratio"] = stopword_count / max(len(words), 1)
    function_words = [
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
        "how",
        "what",
        "which",
        "who",
        "whom",
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
        "hers",
        "its",
        "ours",
        "theirs",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
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
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "only",
        "just",
        "very",
        "too",
        "quite",
        "rather",
        "some",
        "any",
        "all",
        "each",
        "every",
        "both",
        "few",
        "several",
        "many",
        "much",
        "more",
        "most",
        "other",
        "another",
    ]
    fw_counts = Counter()
    for w in words:
        w_lower = w.lower()
        if w_lower in function_words:
            fw_counts[w_lower] += 1
    for fw in function_words:
        features[f"fw_{fw}"] = fw_counts.get(fw, 0) / max(len(words), 1)
    return pd.Series(features)


print("Extracting stylometric features...")
stylometric_features = all_text.apply(extract_stylometric_features)
print(f"Created {stylometric_features.shape[1]} stylometric features")

# ============================================================
# 2. EMOTION / AUTHOR KEYWORD FEATURES
# ============================================================
eap_keywords = set(
    [
        "nevermore",
        "raven",
        "chamber",
        "shadow",
        "dream",
        "death",
        "darkness",
        "soul",
        "spirit",
        "horror",
        "terror",
        "madness",
        "sepulchre",
        "ghastly",
        "pallid",
        "dreary",
        "weary",
        "fantastic",
        "grotesque",
        "wild",
        "strange",
        "mystery",
        "secret",
        "ancient",
        "old",
        "eye",
        "heart",
        "blood",
        "pale",
        "white",
        "black",
        "night",
        "ghost",
        "spectre",
        "dread",
        "evil",
        "wretched",
        "miserable",
        "desolate",
        "gloomy",
        "melancholy",
        "fear",
        "fright",
        "alarm",
        "agony",
        "anguish",
    ]
)
hpl_keywords = set(
    [
        "cthulhu",
        "elder",
        "great",
        "old",
        "one",
        "ancient",
        "cosmic",
        "unknown",
        "unnameable",
        "unspeakable",
        "cyclopean",
        "non-euclidean",
        "yog-sothoth",
        "nyarlathotep",
        "azathoth",
        "necronomicon",
        "arkham",
        "innsmouth",
        "dunwich",
        "r'lyeh",
        "miskatonic",
        "providence",
        "abnormal",
        "blasphemous",
        "frightful",
        "hideous",
        "loathsome",
        "monstrous",
        "nameless",
        "eldritch",
        "primordial",
        "profound",
        "terrible",
        "accursed",
        "daemonic",
        "fathomless",
        "gibbous",
        "indescribable",
        "morbid",
        "noisome",
        "putrid",
        "squamous",
        "viscous",
        "feverish",
        "nightmare",
        "shrieking",
        "whispering",
        "ruin",
        "decay",
    ]
)
mws_keywords = set(
    [
        "frankenstein",
        "creature",
        "monster",
        "soul",
        "spirit",
        "nature",
        "life",
        "death",
        "love",
        "hate",
        "fear",
        "hope",
        "despair",
        "knowledge",
        "science",
        "creation",
        "light",
        "dark",
        "mountain",
        "valley",
        "river",
        "lake",
        "forest",
        "sky",
        "human",
        "being",
        "friend",
        "father",
        "brother",
        "sister",
        "child",
        "man",
        "woman",
        "gentle",
        "kind",
        "beautiful",
        "sublime",
        "magnificent",
        "terrible",
        "sorrow",
        "grief",
        "misery",
        "wretched",
        "unhappy",
        "dear",
        "beloved",
        "switzerland",
        "geneva",
        "ingolstadt",
        "oratory",
        "eloquence",
    ]
)


def extract_emotion_features(text):
    features = {}
    words_lower = set(text.lower().split())
    total_words = len(text.split())
    features["eap_keyword_ratio"] = sum(
        1 for w in words_lower if w in eap_keywords
    ) / max(total_words, 1)
    features["hpl_keyword_ratio"] = sum(
        1 for w in words_lower if w in hpl_keywords
    ) / max(total_words, 1)
    features["mws_keyword_ratio"] = sum(
        1 for w in words_lower if w in mws_keywords
    ) / max(total_words, 1)
    positive_words = {
        "beautiful",
        "wonderful",
        "happy",
        "joy",
        "love",
        "dear",
        "gentle",
        "kind",
        "sweet",
        "pleasant",
        "delightful",
        "charming",
        "bright",
        "hope",
        "bliss",
        "peace",
        "calm",
        "pleasure",
        "happiness",
        "lovely",
        "nice",
        "fine",
        "good",
        "great",
        "excellent",
    }
    negative_words = {
        "horror",
        "terror",
        "dread",
        "fear",
        "pain",
        "sorrow",
        "grief",
        "misery",
        "anguish",
        "agony",
        "despair",
        "woe",
        "mourning",
        "death",
        "darkness",
        "evil",
        "hate",
        "cruel",
        "terrible",
        "hideous",
        "ghastly",
        "monstrous",
        "wicked",
        "fearful",
        "dreadful",
        "horrible",
        "awful",
        "sad",
        "gloomy",
        "melancholy",
    }
    features["positive_word_ratio"] = sum(
        1 for w in words_lower if w in positive_words
    ) / max(total_words, 1)
    features["negative_word_ratio"] = sum(
        1 for w in words_lower if w in negative_words
    ) / max(total_words, 1)
    features["emotional_balance"] = (
        features["positive_word_ratio"] - features["negative_word_ratio"]
    )
    first_person = {
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
    third_person = {
        "he",
        "she",
        "it",
        "they",
        "him",
        "her",
        "them",
        "his",
        "its",
        "their",
        "theirs",
    }
    first_person_count = sum(1 for w in text.lower().split() if w in first_person)
    third_person_count = sum(1 for w in text.lower().split() if w in third_person)
    features["first_person_ratio"] = first_person_count / max(total_words, 1)
    features["third_person_ratio"] = third_person_count / max(total_words, 1)
    features["person_ratio"] = (first_person_count - third_person_count) / max(
        first_person_count + third_person_count, 1
    )
    past_tense = {
        "was",
        "were",
        "had",
        "did",
        "said",
        "went",
        "came",
        "saw",
        "knew",
        "thought",
        "felt",
        "became",
        "began",
        "made",
        "took",
        "gave",
        "found",
        "left",
        "seemed",
    }
    present_tense = {
        "is",
        "are",
        "has",
        "do",
        "say",
        "go",
        "come",
        "see",
        "know",
        "think",
        "feel",
        "become",
        "begin",
        "make",
        "take",
        "give",
        "find",
        "leave",
        "seem",
    }
    features["past_tense_ratio"] = sum(
        1 for w in text.lower().split() if w in past_tense
    ) / max(total_words, 1)
    features["present_tense_ratio"] = sum(
        1 for w in text.lower().split() if w in present_tense
    ) / max(total_words, 1)
    return pd.Series(features)


print("Extracting emotion features...")
emotion_features = all_text.apply(extract_emotion_features)
print(f"Created {emotion_features.shape[1]} emotion features")

# ============================================================
# 3. N-GRAM FEATURES
# ============================================================
print("Extracting n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    max_df=0.95,
    min_df=3,
)
char_features = char_vectorizer.fit_transform(train_df["text"])
char_features_test = char_vectorizer.transform(test_df["text"])
char_features_df = pd.DataFrame(
    char_features.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_features.shape[1])],
)
char_features_test_df = pd.DataFrame(
    char_features_test.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_features_test.shape[1])],
)
print(f"Created {char_features_df.shape[1]} character n-gram features")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    sublinear_tf=True,
    max_df=0.85,
    min_df=2,
    stop_words="english",
)
word_features = word_vectorizer.fit_transform(train_df["text"])
word_features_test = word_vectorizer.transform(test_df["text"])
word_features_df = pd.DataFrame(
    word_features.toarray(),
    columns=[f"word_ngram_{i}" for i in range(word_features.shape[1])],
)
word_features_test_df = pd.DataFrame(
    word_features_test.toarray(),
    columns=[f"word_ngram_{i}" for i in range(word_features_test.shape[1])],
)
print(f"Created {word_features_df.shape[1]} word n-gram features")

# ============================================================
# 4. POS-LIKE FEATURES
# ============================================================
print("Extracting POS-like features...")


def extract_pos_features(text):
    features = {}
    words = text.split()
    total = max(len(words), 1)
    articles = {"the", "a", "an"}
    features["article_ratio"] = sum(1 for w in words if w.lower() in articles) / total
    prepositions = {
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "to",
        "for",
        "of",
        "through",
        "across",
        "between",
        "under",
        "over",
        "before",
        "after",
        "above",
        "below",
        "without",
        "within",
        "upon",
        "into",
        "toward",
        "around",
        "about",
    }
    features["preposition_ratio"] = (
        sum(1 for w in words if w.lower() in prepositions) / total
    )
    conjunctions = {
        "and",
        "but",
        "or",
        "nor",
        "yet",
        "so",
        "for",
        "because",
        "although",
        "while",
        "when",
        "where",
        "if",
        "unless",
        "since",
        "after",
        "before",
    }
    features["conjunction_ratio"] = (
        sum(1 for w in words if w.lower() in conjunctions) / total
    )
    pronouns = {
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
    }
    features["pronoun_ratio"] = sum(1 for w in words if w.lower() in pronouns) / total
    modals = {
        "can",
        "could",
        "may",
        "might",
        "shall",
        "should",
        "will",
        "would",
        "must",
    }
    features["modal_verb_ratio"] = sum(1 for w in words if w.lower() in modals) / total
    comparatives = {
        "more",
        "less",
        "better",
        "worse",
        "greater",
        "lesser",
        "higher",
        "lower",
    }
    superlatives = {"most", "least", "best", "worst", "greatest", "highest"}
    features["comparative_ratio"] = (
        sum(1 for w in words if w.lower() in comparatives) / total
    )
    features["superlative_ratio"] = (
        sum(1 for w in words if w.lower() in superlatives) / total
    )
    negatives = {"not", "no", "never", "nothing", "none", "neither", "nor", "nowhere"}
    features["negation_ratio"] = sum(1 for w in words if w.lower() in negatives) / total
    return pd.Series(features)


pos_features = all_text.apply(extract_pos_features)
print(f"Created {pos_features.shape[1]} POS-like features")

# ============================================================
# 5. COMBINE ALL FEATURES (TRAIN)
# ============================================================
print("Combining all features...")
stylometric_train = stylometric_features.iloc[: len(train_df)].reset_index(drop=True)
stylometric_test = stylometric_features.iloc[len(train_df) :].reset_index(drop=True)
emotion_train = emotion_features.iloc[: len(train_df)].reset_index(drop=True)
emotion_test = emotion_features.iloc[len(train_df) :].reset_index(drop=True)
pos_train = pos_features.iloc[: len(train_df)].reset_index(drop=True)
pos_test = pos_features.iloc[len(train_df) :].reset_index(drop=True)

X_train = pd.concat(
    [
        stylometric_train,
        emotion_train,
        char_features_df,
        word_features_df,
        pos_train,
    ],
    axis=1,
)
X_train = X_train.fillna(0)

X_test = pd.concat(
    [
        stylometric_test,
        emotion_test,
        char_features_test_df,
        word_features_test_df,
        pos_test,
    ],
    axis=1,
)
X_test = X_test.fillna(0)

print(f"Total features: {X_train.shape[1]}")
print(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

# ============================================================
# 6. SPLIT AND SCALE
# ============================================================
y_train = train_df["author"].values
train_ids = train_df["id"].values
test_ids = test_df["id"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(splitter.split(X_train_scaled, y_train_encoded))

X_train_final = X_train_scaled.iloc[train_idx].reset_index(drop=True)
X_val = X_train_scaled.iloc[val_idx].reset_index(drop=True)
y_train_final = y_train_encoded[train_idx]
y_val = y_train_encoded[val_idx]
train_ids_final = train_ids[train_idx]
val_ids_final = train_ids[val_idx]

num_stylometric_features = X_train_final.shape[1]
print(
    f"Train samples: {len(X_train_final)}, Val samples: {len(X_val)}, Test samples: {len(X_test_scaled)}"
)
print(f"Number of stylometric features: {num_stylometric_features}")

# ============================================================
# 7. PREPARE TEXT DATA FOR TRANSFORMER
# ============================================================
text_dict = dict(zip(train_df["id"], train_df["text"]))
train_texts = np.array([text_dict[id_] for id_ in train_ids_final])
val_texts = np.array([text_dict[id_] for id_ in val_ids_final])
test_texts = test_df["text"].values

# ============================================================
# 8. DATASET AND DATALOADER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, features=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.features = features
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = SpookyDataset(train_texts, y_train_final, X_train_final.values)
val_dataset = SpookyDataset(val_texts, y_val, X_val.values)
test_dataset = SpookyDataset(test_texts, features=X_test_scaled.values)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)


# ============================================================
# 9. MODEL DEFINITION (SpookyClassifier with Cross-Attention)
# ============================================================
class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=None, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size  # 1024 for deberta-v3-large
        if num_features > 0:
            # Feature projection: map engineered features to 128-dim per head (2 heads -> 256 total)
            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            # Multi-head cross-attention: CLS token as query, projected features as key/value
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,  # 1024
                num_heads=2,
                kdim=256,
                vdim=256,
                dropout=dropout_rate,
                batch_first=True,
            )
            # After cross-attention: concatenate attention output with original CLS -> 1024+256=1280
            self.post_attn_mlp = nn.Sequential(
                nn.Linear(hidden_size + 256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.head = nn.Linear(256, num_authors)
        else:
            self.feature_proj = None
            self.cross_attn = None
            self.post_attn_mlp = None
            self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_pool = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        if self.feature_proj is not None and features is not None:
            # Project features
            feat_embed = self.feature_proj(features)  # (B, 256)
            # Cross-attention: CLS token as query, projected features as key/value
            # Need to add sequence dim: (B, 1, 1024) for query, (B, 1, 256) for key/value
            cls_query = cls_pool.unsqueeze(1)  # (B, 1, 1024)
            feat_key_value = feat_embed.unsqueeze(1)  # (B, 1, 256)
            attn_out, _ = self.cross_attn(
                query=cls_query,
                key=feat_key_value,
                value=feat_key_value,
                need_weights=False,
            )  # (B, 1, 1024)
            attn_out = attn_out.squeeze(1)  # (B, 1024)
            # Concatenate attention output with original CLS
            combined = torch.cat([attn_out, feat_embed], dim=1)  # (B, 1024+256=1280)
            # Pass through 2-layer MLP (896->256 in plan description, 1280->256 here)
            combined = self.post_attn_mlp(combined)  # (B, 256)
        else:
            combined = cls_pool
        logits = self.head(combined)
        return logits


model = SpookyClassifier(
    num_authors=3, num_features=num_stylometric_features if num_stylometric_features > 0 else None, dropout_rate=0.3
)
model.to(device)

# ============================================================
# 10. OPTIMIZER, SCHEDULER, CRITERION (with curriculum smoothing)
# ============================================================
backbone_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for n, p in layer.named_parameters():
        if "bias" not in n and "LayerNorm" not in n:
            backbone_params.append(p)
head_params = list(model.head.parameters()) + (
    list(model.feature_proj.parameters()) if model.feature_proj else []
) + (
    list(model.cross_attn.parameters()) if model.cross_attn else []
) + (
    list(model.post_attn_mlp.parameters()) if model.post_attn_mlp else []
)

optimizer = AdamW(
    [
        {
            "params": backbone_params,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ]
)

num_epochs = 20
total_steps = len(train_loader) * num_epochs
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=len(train_loader) * 2, T_mult=1, eta_min=1e-7
)
initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

# Curriculum label smoothing: decays from 0.15 to 0.05 over first 5 epochs
initial_smoothing = 0.15
final_smoothing = 0.05
curriculum_epochs = 5
scaler = GradScaler()

# ============================================================
# 11. TRAINING WITH GRADIENT ACCUMULATION AND CURRICULUM SMOOTHING
# ============================================================
print("Starting training with cross-validation...")

# ============================================================
# NEW: K-FOLD CROSS VALIDATION
# ============================================================
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_test_probs = []

for fold, (train_fold_idx, val_fold_idx) in enumerate(kfold.split(X_train_scaled, y_train_encoded)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/3")
    print(f"{'='*50}")

    # Split data for this fold
    X_train_fold = X_train_scaled.iloc[train_fold_idx].reset_index(drop=True)
    X_val_fold = X_train_scaled.iloc[val_fold_idx].reset_index(drop=True)
    y_train_fold = y_train_encoded[train_fold_idx]
    y_val_fold = y_train_encoded[val_fold_idx]
    train_ids_fold = train_ids[train_fold_idx]
    val_ids_fold = train_ids[val_fold_idx]

    text_dict = dict(zip(train_df["id"], train_df["text"]))
    train_texts_fold = np.array([text_dict[id_] for id_ in train_ids_fold])
    val_texts_fold = np.array([text_dict[id_] for id_ in val_ids_fold])

    train_dataset = SpookyDataset(train_texts_fold, y_train_fold, X_train_fold.values)
    val_dataset = SpookyDataset(val_texts_fold, y_val_fold, X_val_fold.values)
    test_dataset = SpookyDataset(test_texts, features=X_test_scaled.values)

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize new model for each fold
    model = SpookyClassifier(
        num_authors=3, num_features=num_stylometric_features if num_stylometric_features > 0 else None, dropout_rate=0.3
    )
    model.to(device)

    # Optimizer
    backbone_params = []
    for layer in model.backbone.deberta.encoder.layer[-8:]:
        for n, p in layer.named_parameters():
            if "bias" not in n and "LayerNorm" not in n:
                backbone_params.append(p)
    head_params = list(model.head.parameters()) + (
        list(model.feature_proj.parameters()) if model.feature_proj else []
    ) + (
        list(model.cross_attn.parameters()) if model.cross_attn else []
    ) + (
        list(model.post_attn_mlp.parameters()) if model.post_attn_mlp else []
    )

    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
        ]
    )

    num_epochs = 20
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 2, T_mult=1, eta_min=1e-7
    )

    # Gradient accumulation steps (effective batch size 32)
    accumulation_steps = 2

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        # Curriculum label smoothing
        if epoch < curriculum_epochs:
            smoothing = initial_smoothing - (initial_smoothing - final_smoothing) * (epoch / curriculum_epochs)
        else:
            smoothing = final_smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_step = epoch * len(train_loader) + batch_idx
            scheduler.step(epoch + current_step / len(train_loader))
            total_loss += loss.item() * accumulation_steps

        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch.get("features", None)
                if features is not None:
                    features = features.to(device)
                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        val_loss = log_loss(y_val_fold, val_probs)
        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Log Loss: {val_loss:.4f} (smoothing={smoothing:.3f})"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Fold {fold+1} best val log loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    model.to(device)

    # Test inference for this fold
    model.eval()
    fold_test_probs_fold = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs_fold.append(probs.cpu().numpy())
    fold_test_probs_fold = np.concatenate(fold_test_probs_fold)
    fold_test_probs.append(fold_test_probs_fold)

    # Clean up to save memory
    del model
    torch.cuda.empty_cache()

# Average test predictions across folds
test_probs = np.mean(fold_test_probs, axis=0)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# ============================================================
# 12. SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")