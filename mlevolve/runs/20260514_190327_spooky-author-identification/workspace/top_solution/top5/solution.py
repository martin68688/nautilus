import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import warnings
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author_encoded"] = le.fit_transform(train_set["author"])
val_set["author_encoded"] = le.transform(val_set["author"])

print(f"Classes: {le.classes_}")
print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
print(f"Test size: {len(test_df)}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================


def extract_basic_features(texts):
    """Extract basic text features"""
    features = []
    for text in texts:
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s for s in sentences if s.strip()]

        n_chars = len(text_str)
        n_words = len(words)
        n_sentences = max(len(sentences), 1)
        n_unique_words = len(set(w.lower() for w in words))

        avg_word_len = n_chars / max(n_words, 1)
        avg_sentence_len = n_words / n_sentences

        n_exclamation = text_str.count("!")
        n_question = text_str.count("?")
        n_period = text_str.count(".")
        n_comma = text_str.count(",")
        n_semicolon = text_str.count(";")
        n_colon = text_str.count(":")
        n_quote = text_str.count('"') + text_str.count("'")
        n_dash = text_str.count("-") + text_str.count("—")
        n_punct = (
            n_exclamation + n_question + n_period + n_comma + n_semicolon + n_colon
        )

        n_capitalized = sum(1 for w in words if w and w[0].isupper())
        pct_capitalized = n_capitalized / max(n_words, 1)

        short_words = sum(1 for w in words if len(w) <= 3)
        medium_words = sum(1 for w in words if 4 <= len(w) <= 7)
        long_words = sum(1 for w in words if len(w) >= 8)

        lexical_diversity = n_unique_words / max(n_words, 1)

        stop_words = {
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
            "has",
            "have",
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
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "not",
            "no",
            "nor",
            "so",
        }
        n_stop_words = sum(1 for w in words if w.lower() in stop_words)
        stop_word_ratio = n_stop_words / max(n_words, 1)

        feats = [
            n_chars,
            n_words,
            n_sentences,
            n_unique_words,
            avg_word_len,
            avg_sentence_len,
            n_exclamation,
            n_question,
            n_period,
            n_comma,
            n_semicolon,
            n_colon,
            n_quote,
            n_dash,
            n_punct,
            n_capitalized,
            pct_capitalized,
            short_words,
            medium_words,
            long_words,
            short_words / max(n_words, 1),
            medium_words / max(n_words, 1),
            long_words / max(n_words, 1),
            lexical_diversity,
            stop_word_ratio,
        ]
        features.append(feats)

    return np.array(features)


def extract_stylistic_features(texts):
    """Extract stylistic writing features"""
    features = []
    archaic_words = {
        "thou",
        "thee",
        "thy",
        "thine",
        "doth",
        "hath",
        "art",
        "wilt",
        "shalt",
        "canst",
        "didst",
        "hast",
        "whence",
        "thence",
        "hither",
        "whither",
        "ere",
        "o'er",
        "ne'er",
        "oft",
        "perchance",
        "forsooth",
        "alas",
        "aye",
        "nay",
        "yea",
        "methinks",
        "prithee",
        "anon",
    }
    horror_words = {
        "dark",
        "shadow",
        "fear",
        "dread",
        "terror",
        "horror",
        "ghost",
        "phantom",
        "spectre",
        "demon",
        "devil",
        "hell",
        "death",
        "dead",
        "corpse",
        "grave",
        "tomb",
        "scream",
        "shriek",
        "howl",
        "gloom",
        "mist",
        "fog",
        "night",
        "moon",
        "ancient",
        "eldritch",
        "cosmic",
        "abyss",
        "void",
        "chaos",
        "monster",
        "creature",
        "beast",
        "evil",
    }

    for text in texts:
        text_str = str(text).lower()
        words = text_str.split()

        n_archaic = sum(1 for w in words if w in archaic_words)
        n_horror = sum(1 for w in words if w in horror_words)

        first_person = sum(
            1
            for w in words
            if w
            in {
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
        )
        third_person = sum(
            1
            for w in words
            if w
            in {
                "he",
                "him",
                "his",
                "himself",
                "she",
                "her",
                "hers",
                "herself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
            }
        )

        conjunctions = sum(
            1
            for w in words
            if w
            in {
                "and",
                "but",
                "or",
                "nor",
                "for",
                "yet",
                "so",
                "although",
                "because",
                "since",
                "while",
                "though",
                "unless",
            }
        )
        dialogue_markers = sum(
            1
            for w in words
            if w
            in {
                "said",
                "cried",
                "exclaimed",
                "whispered",
                "shouted",
                "asked",
                "replied",
                "answered",
                "murmured",
            }
        )
        emphatic_words = sum(
            1
            for w in words
            if w
            in {
                "very",
                "quite",
                "extremely",
                "absolutely",
                "utterly",
                "totally",
                "completely",
                "entirely",
                "highly",
            }
        )
        negation = sum(
            1
            for w in words
            if w
            in {
                "not",
                "no",
                "never",
                "nothing",
                "none",
                "neither",
                "nor",
                "nowhere",
                "nobody",
                "cannot",
                "can't",
                "don't",
                "doesn't",
            }
        )
        adjective_endings = sum(
            1
            for w in words
            if len(w) > 3
            and (
                w.endswith("ful")
                or w.endswith("ous")
                or w.endswith("ive")
                or w.endswith("able")
                or w.endswith("ible")
                or w.endswith("al")
            )
        )
        adverb_ly = sum(1 for w in words if len(w) > 4 and w.endswith("ly"))

        total_trigrams = 0
        unique_trigrams = set()
        for word in words:
            for i in range(len(word) - 2):
                trigram = word[i : i + 3]
                unique_trigrams.add(trigram)
                total_trigrams += 1
        trigram_diversity = len(unique_trigrams) / max(total_trigrams, 1)

        features.append(
            [
                n_archaic,
                n_horror,
                first_person,
                third_person,
                conjunctions,
                dialogue_markers,
                emphatic_words,
                negation,
                adjective_endings,
                adverb_ly,
                trigram_diversity,
            ]
        )

    return np.array(features)


def extract_sentiment_features(texts):
    """Extract simple positive/negative sentiment indicators based on word lists"""
    positive_words = {
        "beautiful",
        "wonderful",
        "happy",
        "joy",
        "love",
        "hope",
        "bright",
        "light",
        "peace",
        "pleasure",
        "delight",
        "glad",
        "cheerful",
        "bliss",
        "splendid",
        "magnificent",
        "excellent",
        "pleasant",
        "sweet",
        "calm",
        "gentle",
        "tender",
        "kind",
        "generous",
        "noble",
        "brave",
        "glorious",
    }
    negative_words = {
        "terrible",
        "awful",
        "horrible",
        "dreadful",
        "painful",
        "sad",
        "grief",
        "sorrow",
        "misery",
        "suffering",
        "agony",
        "despair",
        "anger",
        "rage",
        "fury",
        "hate",
        "cruel",
        "wicked",
        "vile",
        "fearful",
        "frightful",
        "ghastly",
        "hideous",
        "loathsome",
        "disgusting",
        "revolting",
        "gloomy",
        "dreary",
        "bleak",
        "somber",
    }

    features = []
    for text in texts:
        text_str = str(text).lower()
        words = set(text_str.split())
        n_pos = sum(1 for w in words if w in positive_words)
        n_neg = sum(1 for w in words if w in negative_words)
        total = n_pos + n_neg
        sentiment_ratio = (n_pos - n_neg) / total if total > 0 else 0.0
        features.append([n_pos, n_neg, sentiment_ratio])

    return np.array(features)


def extract_part_of_speech_patterns(texts):
    """Extract POS-like patterns using word endings as proxy"""
    features = []
    for text in texts:
        text_str = str(text)
        words = text_str.split()
        ing_words = sum(1 for w in words if len(w) > 4 and w.lower().endswith("ing"))
        ed_words = sum(1 for w in words if len(w) > 3 and w.lower().endswith("ed"))
        tion_words = sum(1 for w in words if len(w) > 5 and w.lower().endswith("tion"))
        ness_words = sum(1 for w in words if len(w) > 4 and w.lower().endswith("ness"))
        ment_words = sum(1 for w in words if len(w) > 4 and w.lower().endswith("ment"))
        determiners = sum(
            1
            for w in words
            if w.lower() in {"the", "a", "an", "this", "that", "these", "those"}
        )
        prepositions = sum(
            1
            for w in words
            if w.lower()
            in {
                "in",
                "on",
                "at",
                "to",
                "from",
                "by",
                "with",
                "about",
                "against",
                "between",
                "under",
                "over",
                "above",
                "below",
                "through",
                "during",
                "before",
                "after",
                "of",
                "for",
            }
        )
        articles = sum(1 for w in words if w.lower() in {"the", "a", "an"})
        features.append(
            [
                ing_words,
                ed_words,
                tion_words,
                ness_words,
                ment_words,
                determiners,
                prepositions,
                articles,
            ]
        )

    return np.array(features)


# ============================================================
# APPLY FEATURE ENGINEERING
# ============================================================

train_texts_feat = train_set["text"].values
val_texts_feat = val_set["text"].values
test_texts_feat = test_df["text"].values
print("Extracting basic features on train...")
basic_features_train = extract_basic_features(train_texts_feat)

print("Extracting stylistic features on train...")
stylistic_features_train = extract_stylistic_features(train_texts_feat)

print("Extracting sentiment features on train...")
sentiment_features_train = extract_sentiment_features(train_texts_feat)

print("Extracting POS pattern features on train...")
pos_features_train = extract_part_of_speech_patterns(train_texts_feat)

print("Extracting basic features on val...")
basic_features_val = extract_basic_features(val_texts_feat)

print("Extracting stylistic features on val...")
stylistic_features_val = extract_stylistic_features(val_texts_feat)

print("Extracting sentiment features on val...")
sentiment_features_val = extract_sentiment_features(val_texts_feat)

print("Extracting POS pattern features on val...")
pos_features_val = extract_part_of_speech_patterns(val_texts_feat)

print("Extracting basic features on test...")
basic_features_test = extract_basic_features(test_texts_feat)

print("Extracting stylistic features on test...")
stylistic_features_test = extract_stylistic_features(test_texts_feat)

print("Extracting sentiment features on test...")
sentiment_features_test = extract_sentiment_features(test_texts_feat)

print("Extracting POS pattern features on test...")
pos_features_test = extract_part_of_speech_patterns(test_texts_feat)

print("Extracting character n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
)
char_tfidf = char_vectorizer.fit_transform(train_texts_feat)
val_char_tfidf = char_vectorizer.transform(val_texts_feat)
test_char_tfidf = char_vectorizer.transform(test_texts_feat)

print("Extracting word n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=500,
    sublinear_tf=True,
    stop_words="english",
    strip_accents="unicode",
)
word_tfidf = word_vectorizer.fit_transform(train_texts_feat)
val_word_tfidf = word_vectorizer.transform(val_texts_feat)
test_word_tfidf = word_vectorizer.transform(test_texts_feat)

# Stack numerical features
num_feature_names = [
    "n_chars",
    "n_words",
    "n_sentences",
    "n_unique_words",
    "avg_word_len",
    "avg_sentence_len",
    "n_exclamation",
    "n_question",
    "n_period",
    "n_comma",
    "n_semicolon",
    "n_colon",
    "n_quote",
    "n_dash",
    "n_punct",
    "n_capitalized",
    "pct_capitalized",
    "short_words",
    "medium_words",
    "long_words",
    "pct_short_words",
    "pct_medium_words",
    "pct_long_words",
    "lexical_diversity",
    "stop_word_ratio",
    "n_archaic",
    "n_horror",
    "first_person",
    "third_person",
    "conjunctions",
    "dialogue_markers",
    "emphatic_words",
    "negation",
    "adjective_endings",
    "adverb_ly",
    "trigram_diversity",
    "n_pos",
    "n_neg",
    "sentiment_ratio",
    "ing_words",
    "ed_words",
    "tion_words",
    "ness_words",
    "ment_words",
    "determiners",
    "prepositions",
    "articles",
]

train_num = np.hstack(
    [basic_features_train, stylistic_features_train, sentiment_features_train, pos_features_train]
)
val_num = np.hstack(
    [basic_features_val, stylistic_features_val, sentiment_features_val, pos_features_val]
)
test_num = np.hstack(
    [basic_features_test, stylistic_features_test, sentiment_features_test, pos_features_test]
)
print(f"Train numerical features shape: {train_num.shape}")
print(f"Val numerical features shape: {val_num.shape}")
print(f"Test numerical features shape: {test_num.shape}")

scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num)
val_num_scaled = scaler.transform(val_num)
test_num_scaled = scaler.transform(test_num)

train_num_sparse = csr_matrix(train_num_scaled)
val_num_sparse = csr_matrix(val_num_scaled)
test_num_sparse = csr_matrix(test_num_scaled)

train_features_all = hstack([train_num_sparse, char_tfidf, word_tfidf])
val_features_all = hstack([val_num_sparse, val_char_tfidf, val_word_tfidf])
test_features_all = hstack([test_num_sparse, test_char_tfidf, test_word_tfidf])

print(f"Final train feature shape: {train_features_all.shape}")
print(f"Final val feature shape: {val_features_all.shape}")
print(f"Final test feature shape: {test_features_all.shape}")

train_targets = train_set["author_encoded"].values
val_targets = val_set["author_encoded"].values

# Save processed features
save_npz("./working/train_features.npz", train_features_all)
save_npz("./working/val_features.npz", val_features_all)
save_npz("./working/test_features.npz", test_features_all)
np.save("./working/train_targets.npy", train_targets)
np.save("./working/val_targets.npy", val_targets)
joblib.dump(scaler, "./working/scaler.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
joblib.dump(le, "./working/label_encoder.pkl")

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    """Custom classifier with multi-head attention pooling over last 4 layers and handcrafted feature integration"""
    def __init__(self, base_model, hidden_size=1024, num_attention_heads=8, num_labels=3):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # Multi-head attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=0.1,
        )
        # Learned query for attention pooling
        self.pooling_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Layer normalization for residual connection
        self.norm = nn.LayerNorm(hidden_size)

        # Projection for handcrafted features (1047 -> 256 -> 128)
        self.feature_proj = nn.Sequential(
            nn.Linear(1047, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 128, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, features=None):
        # Get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)

        # Extract last 4 layers
        last_4_layers = hidden_states[-4:]  # list of 4 tensors each (batch, seq_len, hidden_size)

        # Mean pool each layer using attention_mask
        pooled_layers = []
        for layer_hidden in last_4_layers:
            # Expand attention_mask to (batch, seq_len, hidden_size)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            mask_expanded = mask_expanded.expand_as(layer_hidden)
            # Sum over seq_len and divide by sum of mask
            sum_hidden = (layer_hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
            pooled_layers.append(pooled)

        # Stack pooled representations: (batch, 4, hidden_size)
        pooled_stack = torch.stack(pooled_layers, dim=1)  # (batch, 4, hidden_size)

        # Multi-head attention pooling
        # Query: (batch, 1, hidden_size), Keys and Values: (batch, 4, hidden_size)
        query = self.pooling_query.expand(pooled_stack.size(0), -1, -1)  # (batch, 1, hidden_size)
        attn_output, _ = self.attention_pooling(query, pooled_stack, pooled_stack)
        # attn_output: (batch, 1, hidden_size)
        attn_output = attn_output.squeeze(1)  # (batch, hidden_size)

        # Residual connection with global mean pool of last layer
        global_pool_last = pooled_layers[-1]  # (batch, hidden_size)
        pooled_output = self.norm(attn_output + global_pool_last)

        # Process handcrafted features if provided
        if features is not None:
            feature_emb = self.feature_proj(features)
            # Concatenate with transformer output
            combined = torch.cat([pooled_output, feature_emb], dim=1)
        else:
            combined = pooled_output

        # Final classification
        logits = self.classifier(combined)

        return logits

base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=3,
    output_hidden_states=True,
    output_attentions=False,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)

hidden_size = base_model.config.hidden_size
model = SpookyAuthorClassifier(base_model, hidden_size=hidden_size, num_attention_heads=8, num_labels=3)

# Freeze first 1/3 of the encoder layers
num_layers = len(base_model.deberta.encoder.layer)
freeze_up_to = int(num_layers * 0.333)
for i, layer in enumerate(base_model.deberta.encoder.layer):
    if i < freeze_up_to:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True

# Unfreeze all custom head parameters
for param in model.attention_pooling.parameters():
    param.requires_grad = True
model.pooling_query.requires_grad = True
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.feature_proj.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Separate parameters for differential learning rates
backbone_params = []
custom_head_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if "base_model" in name:
            backbone_params.append(param)
        else:
            custom_head_params.append(param)

optimizer = AdamW(
    [
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": custom_head_params, "lr": 5e-5, "weight_decay": 0.01},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Unfrozen backbone params: {sum(p.numel() for p in backbone_params):,}")
print(f"Custom head params: {sum(p.numel() for p in custom_head_params):,}")
print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = features

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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx].flatten(), dtype=torch.float)
        return item


author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

# Convert sparse features to dense for model input
train_features_array = train_features_all.toarray() if hasattr(train_features_all, 'toarray') else train_features_all
val_features_array = val_features_all.toarray() if hasattr(val_features_all, 'toarray') else val_features_all
test_features_array = test_features_all.toarray() if hasattr(test_features_all, 'toarray') else test_features_all

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length, features=train_features_array
)
val_dataset = SpookyDataset(val_texts_final, val_labels_final, tokenizer, max_length, features=val_features_array)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length, features=test_features_array)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
num_epochs = 25
patience = 8
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

total_steps = len(train_loader) * num_epochs
warmup_frac = 0.1
warmup_steps = int(warmup_frac * total_steps)
cosine_steps = total_steps - warmup_steps

# Proper scheduler: linear warmup + cosine annealing
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=2e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# SWA setup
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
swa_start_epoch = 20
swa_cycle_length = 5
swa_scheduler = SWALR(optimizer, swa_lr=2e-5, anneal_epochs=3)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features = batch.get("features", None)
        if features is not None:
            features = features.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        # Step scheduler once per batch after optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    # SWA update after epoch
    if epoch + 1 >= swa_start_epoch and (epoch + 1 - swa_start_epoch) % swa_cycle_length == 0:
        swa_model.update_parameters(model)
        swa_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
    else:
        current_lr = optimizer.param_groups[0]["lr"]

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
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
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

    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE (use SWA model)
# ============================================================
# Update SWA with final model if SWA was active
if num_epochs >= swa_start_epoch:
    swa_model.update_parameters(model)
    # Load SWA averages into model for evaluation
    model.load_state_dict(swa_model.module.state_dict())
else:
    model.load_state_dict(torch.load("./working/best_model.pt"))

model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features = batch.get("features", None)
        if features is not None:
            features = features.to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_true, val_probs_clipped)

# ============================================================
# TEST INFERENCE
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch.get("features", None)
        if features is not None:
            features = features.to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

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

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")