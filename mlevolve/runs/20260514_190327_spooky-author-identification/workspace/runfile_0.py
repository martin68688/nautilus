import os
os.sched_setaffinity(0, {104, 41, 42, 106, 112})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


# ============================================================
# Custom Model with Multi-Layer Pooling + Handcrafted Features
# ============================================================
class SpookyAuthorClassifier(nn.Module):
    def __init__(self, base_model_name="microsoft/deberta-v3-large", num_labels=3,
                 handcrafted_feature_dim=1047, projection_dim=256, dropout_rate=0.2):
        super().__init__()
        # Load backbone with hidden states enabled
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # Remove the default classifier head (we'll build our own)
        self.deberta.config.num_labels = num_labels
        hidden_size = self.deberta.config.hidden_size

        # Freeze first 1/3 of encoder layers
        num_layers = len(self.deberta.deberta.encoder.layer)
        freeze_up_to = int(num_layers * 0.333)
        for i, layer in enumerate(self.deberta.deberta.encoder.layer):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # Projection layer for combined features
        combined_dim = hidden_size + handcrafted_feature_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Final classifier head
        self.classifier = nn.Linear(projection_dim, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids=None, attention_mask=None, handcrafted_features=None):
        # Get backbone outputs with hidden states
        outputs = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last 4 hidden layers and average pool each
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
        last_4 = hidden_states[-4:]  # Last 4 layers

        # Mean pooling for each layer, weighted by attention_mask
        pooled_layers = []
        for layer_hidden in last_4:
            # Expand attention_mask for broadcasting: (batch, seq_len) -> (batch, seq_len, hidden_size)
            mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
            # Sum hidden states where mask=1, divide by sum of mask
            sum_hidden = (layer_hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
            pooled_layers.append(pooled)

        # Average the pooled representations from last 4 layers
        pooled_output = torch.stack(pooled_layers, dim=0).mean(dim=0)

        # If handcrafted features are provided, concatenate them
        if handcrafted_features is not None:
            # Ensure handcrafted_features is on same device as pooled_output
            if handcrafted_features.device != pooled_output.device:
                handcrafted_features = handcrafted_features.to(pooled_output.device)
            combined = torch.cat([pooled_output, handcrafted_features], dim=1)
        else:
            combined = pooled_output

        # Project and classify
        projected = self.projection(combined)
        logits = self.classifier(projected)

        return logits


# Initialize custom model
model = SpookyAuthorClassifier(
    base_model_name="microsoft/deberta-v3-large",
    num_labels=3,
    handcrafted_feature_dim=train_features_all.shape[1],
    projection_dim=256,
    dropout_rate=0.2,
)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Separate parameters for differential learning rates
unfrozen_backbone_params = []
classifier_projection_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if "deberta" in name:
            unfrozen_backbone_params.append(param)
        else:
            classifier_projection_params.append(param)

optimizer = AdamW(
    [
        {"params": unfrozen_backbone_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": classifier_projection_params, "lr": 5e-5, "weight_decay": 0.01},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Unfrozen backbone params: {sum(p.numel() for p in unfrozen_backbone_params):,}")
print(f"Classifier+Projection params: {sum(p.numel() for p in classifier_projection_params):,}")
print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, handcrafted_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.handcrafted_features = handcrafted_features

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
        if self.handcrafted_features is not None:
            item["handcrafted_features"] = torch.tensor(
                self.handcrafted_features[idx], dtype=torch.float32
            )
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

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length,
    handcrafted_features=train_features_all.toarray()
)
val_dataset = SpookyDataset(
    val_texts_final, val_labels_final, tokenizer, max_length,
    handcrafted_features=val_features_all.toarray()
)
test_dataset = SpookyDataset(
    test_texts, None, tokenizer, max_length,
    handcrafted_features=test_features_all.toarray()
)

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
num_epochs = 30
patience = 5
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
)

initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
warmup_lrs = initial_lrs.copy()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        handcrafted_features = batch["handcrafted_features"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                handcrafted_features=handcrafted_features
            )
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = warmup_lrs[i] * warmup_factor
        else:
            scheduler.step(epoch + current_step / len(train_loader))

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
            handcrafted_features = batch["handcrafted_features"].to(device)

            with torch.no_grad():
                with autocast():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        handcrafted_features=handcrafted_features
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
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model_86d5aad792fd4cc6a5d1b5ec446eb206.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model_86d5aad792fd4cc6a5d1b5ec446eb206.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        handcrafted_features = batch["handcrafted_features"].to(device)
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                handcrafted_features=handcrafted_features
            )
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
        handcrafted_features = batch["handcrafted_features"].to(device)
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                handcrafted_features=handcrafted_features
            )
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

submission.to_csv("./submission/submission_86d5aad792fd4cc6a5d1b5ec446eb206.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")