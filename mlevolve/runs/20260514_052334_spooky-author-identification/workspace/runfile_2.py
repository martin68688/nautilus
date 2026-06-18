import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9, 10, 11, 12, 13})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# =============================================================================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Combine for consistent feature engineering
all_text = pd.concat([train_df["text"], test_df["text"]], axis=0).reset_index(drop=True)


def extract_basic_features(text_series):
    features = pd.DataFrame()
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.count("[.!?]") + 1
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["dash_count"] = text_series.str.count("—") + text_series.str.count("-")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["parentheses_count"] = text_series.str.count(
        r"\("
    ) + text_series.str.count(r"\)")
    features["punctuation_density"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["comma_count"]
        + features["semicolon_count"]
        + features["colon_count"]
    ) / (features["char_count"] + 1)
    features["capital_word_ratio"] = text_series.apply(
        lambda x: sum(1 for word in str(x).split() if word[0].isupper())
        / (len(str(x).split()) + 1)
    )
    features["all_caps_word_ratio"] = text_series.apply(
        lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 1)
        / (len(str(x).split()) + 1)
    )
    features["unique_word_ratio"] = text_series.apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )
    return features


def extract_readability_features(text_series):
    features = pd.DataFrame()

    def count_syllables(word):
        word = str(word).lower()
        if len(word) <= 3:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(1, count)

    features["word_count"] = text_series.str.split().str.len()
    features["sentence_count"] = text_series.str.count("[.!?]") + 1
    features["avg_sentence_len"] = features["word_count"] / (features["sentence_count"] + 1)
    features["avg_syllables_per_word"] = text_series.apply(
        lambda x: (
            np.mean([count_syllables(w) for w in str(x).split()])
            if len(str(x).split()) > 0
            else 0
        )
    )
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["avg_sentence_len"]
        - 84.6 * features["avg_syllables_per_word"]
    )
    features["long_word_ratio"] = text_series.apply(
        lambda x: sum(1 for word in str(x).split() if len(word) > 8)
        / (len(str(x).split()) + 1)
    )
    features["very_long_word_ratio"] = text_series.apply(
        lambda x: sum(1 for word in str(x).split() if len(word) > 12)
        / (len(str(x).split()) + 1)
    )
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
        "by",
        "with",
        "from",
        "as",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "shall",
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
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "very",
        "just",
        "also",
        "too",
        "only",
        "more",
        "most",
        "some",
        "any",
        "each",
        "every",
        "all",
        "both",
        "few",
        "many",
        "much",
    }
    features["stop_word_ratio"] = text_series.apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in stop_words)
        / (len(str(x).split()) + 1)
    )
    return features


def extract_pos_patterns(text_series):
    features = pd.DataFrame()

    def count_pos_patterns(text):
        words = str(text).split()
        ing_words = sum(1 for w in words if w.endswith("ing"))
        ed_words = sum(1 for w in words if w.endswith("ed"))
        ly_words = sum(1 for w in words if w.endswith("ly"))
        tion_words = sum(1 for w in words if w.endswith("tion"))
        sion_words = sum(1 for w in words if w.endswith("sion"))
        ment_words = sum(1 for w in words if w.endswith("ment"))
        ness_words = sum(1 for w in words if w.endswith("ness"))
        ful_words = sum(1 for w in words if w.endswith("ful"))
        less_words = sum(1 for w in words if w.endswith("less"))
        able_words = sum(1 for w in words if w.endswith("able"))
        ible_words = sum(1 for w in words if w.endswith("ible"))
        ous_words = sum(1 for w in words if w.endswith("ous"))
        return [
            ing_words,
            ed_words,
            ly_words,
            tion_words,
            sion_words,
            ment_words,
            ness_words,
            ful_words,
            less_words,
            able_words,
            ible_words,
            ous_words,
        ]

    pos_counts = text_series.apply(count_pos_patterns)
    pos_df = pd.DataFrame(
        pos_counts.tolist(),
        columns=[
            "ing_suffix",
            "ed_suffix",
            "ly_suffix",
            "tion_suffix",
            "sion_suffix",
            "ment_suffix",
            "ness_suffix",
            "ful_suffix",
            "less_suffix",
            "able_suffix",
            "ible_suffix",
            "ous_suffix",
        ],
    )
    word_counts = text_series.str.split().str.len()
    for col in pos_df.columns:
        pos_df[col] = pos_df[col] / (word_counts + 1)
    return pos_df


def extract_authorial_signature(text_series):
    features = pd.DataFrame()
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "cryptic",
        "noisome",
        "squamous",
        "ichor",
        "cacodaemonial",
        "gibbous",
        "charnel",
        "sepulchral",
        "fungoid",
        "nameless",
        "unnameable",
        "unspeakable",
        "abyss",
        "aeon",
        "primordial",
        "antediluvian",
        "antediluvial",
        "prehuman",
        "nonhuman",
        "non-euclidean",
        "extra-dimensional",
        "cosmic",
        "gulf",
        "void",
        "madness",
        "nightmare",
        "ghoul",
        "demon",
        "monstrous",
        "loathsome",
        "hideous",
        "frightful",
        "dread",
        "fearful",
        "horrible",
        "terrible",
        "awful",
        "appalling",
        "uncanny",
        "weird",
        "strange",
        "bizarre",
        "queer",
        "indescribable",
        "ineffable",
        "innominate",
        "inconceivable",
    }
    poe_words = {
        "nevermore",
        "chamber",
        "raven",
        "tintinnabulation",
        "ghastly",
        "pallid",
        "drapery",
        "ebony",
        "plutonian",
        "belfry",
        "baleful",
        "malediction",
        "phantasm",
        "spectral",
        "gloom",
        "melancholy",
        "dreary",
        "bleak",
        "desolate",
        "forlorn",
        "weary",
        "fatigued",
        "exhausted",
        "oppressed",
        "burdened",
        "agony",
        "anguish",
        "torment",
        "torture",
        "suffering",
        "mad",
        "insane",
        "lunatic",
        "deranged",
        "delirious",
        "supernatural",
        "occult",
        "mystic",
        "enigmatic",
        "inscrutable",
    }
    shelley_words = {
        "frankenstein",
        "monster",
        "creature",
        "daemon",
        "wretch",
        "victor",
        "geneva",
        "ingolstadt",
        "alpine",
        "glacier",
        "sublime",
        "majestic",
        "picturesque",
        "romantic",
        "sentimental",
        "passion",
        "affection",
        "emotion",
        "sentiment",
        "feeling",
        "virtue",
        "vice",
        "moral",
        "immoral",
        "virtuous",
        "nature",
        "natural",
        "artificial",
        "unnatural",
        "science",
        "knowledge",
        "wisdom",
        "ignorance",
        "curiosity",
        "discovery",
        "creation",
        "creator",
        "life",
        "death",
        "soul",
        "spirit",
        "immortal",
        "eternal",
        "infinite",
    }

    def count_author_vocab(text, word_set):
        words = set(str(text).lower().split())
        return len(words.intersection(word_set))

    features["lovecraft_vocab_count"] = text_series.apply(
        lambda x: count_author_vocab(x, lovecraft_words)
    )
    features["poe_vocab_count"] = text_series.apply(
        lambda x: count_author_vocab(x, poe_words)
    )
    features["shelley_vocab_count"] = text_series.apply(
        lambda x: count_author_vocab(x, shelley_words)
    )
    archaic_words = {
        "thou",
        "thee",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "art",
        "wilt",
        "shalt",
        "dost",
        "didst",
        "hast",
        "hadst",
        "canst",
        "couldst",
        "wouldst",
        "shouldst",
        "mightst",
        "mayst",
        "wherefore",
        "henceforth",
        "thenceforth",
        "herewith",
        "therewith",
        "wherein",
        "wherewith",
        "whereby",
        "perchance",
        "peradventure",
        "forsooth",
        "methinks",
        "prithee",
        "anon",
        "ere",
        "betwixt",
        "unto",
        "nay",
    }
    features["archaic_word_count"] = text_series.apply(
        lambda x: count_author_vocab(x, archaic_words)
    )
    features["archaic_ratio"] = features["archaic_word_count"] / (
        text_series.str.split().str.len() + 1
    )
    return features


def extract_sentence_complexity(text_series):
    features = pd.DataFrame()
    subordinating_conjunctions = {
        "although",
        "though",
        "while",
        "because",
        "since",
        "after",
        "before",
        "until",
        "unless",
        "provided",
        "if",
        "when",
        "where",
        "whereas",
        "even",
        "as",
        "so",
        "that",
        "which",
        "who",
        "whom",
        "whose",
    }
    features["subordinate_clause_markers"] = text_series.apply(
        lambda x: sum(
            1 for w in str(x).lower().split() if w in subordinating_conjunctions
        )
    )
    features["subordinate_density"] = features["subordinate_clause_markers"] / (
        text_series.str.split().str.len() + 1
    )
    relative_pronouns = {"which", "that", "who", "whom", "whose", "where", "when"}
    features["relative_clause_markers"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in relative_pronouns)
    )
    conjunctions = {"and", "but", "or", "nor", "for", "yet", "so"}
    features["conjunction_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in conjunctions)
    )
    features["conjunction_ratio"] = features["conjunction_count"] / (
        text_series.str.split().str.len() + 1
    )
    features["sentence_start_the"] = text_series.str.match(r"^[Tt]he\b").astype(int)
    features["sentence_start_it"] = text_series.str.match(r"^[Ii]t\b").astype(int)
    return features


def extract_ngram_features_fit(text_series, max_features=50):
    """Fit vectorizer on training data only."""
    ngram_features = None
    vectorizers = []
    for n in [2, 3]:
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(n, n),
            max_features=max_features,
            lowercase=True,
            binary=True,
        )
        ngram_matrix = vectorizer.fit_transform(text_series)
        ngram_df = pd.DataFrame(
            ngram_matrix.toarray(),
            columns=[
                f"char_ngram_{n}_{feat}" for feat in vectorizer.get_feature_names_out()
            ],
        )
        if ngram_features is None:
            ngram_features = ngram_df
        else:
            ngram_features = pd.concat([ngram_features, ngram_df], axis=1)
        vectorizers.append(vectorizer)
    return ngram_features, vectorizers

def extract_ngram_features_transform(text_series, vectorizers):
    """Transform using fitted vectorizers."""
    ngram_features = None
    for n, vectorizer in zip([2, 3], vectorizers):
        ngram_matrix = vectorizer.transform(text_series)
        ngram_df = pd.DataFrame(
            ngram_matrix.toarray(),
            columns=[
                f"char_ngram_{n}_{feat}" for feat in vectorizer.get_feature_names_out()
            ],
        )
        if ngram_features is None:
            ngram_features = ngram_df
        else:
            ngram_features = pd.concat([ngram_features, ngram_df], axis=1)
    return ngram_features


print("Extracting features (on train only for n-grams)...")
basic_features = extract_basic_features(all_text)
readability_features = extract_readability_features(all_text)
pos_features = extract_pos_patterns(all_text)
authorial_features = extract_authorial_signature(all_text)
complexity_features = extract_sentence_complexity(all_text)

# Fit n-gram vectorizers on TRAIN only to prevent data leakage
train_text_only = all_text.iloc[: len(train_df)]
ngram_train, ngram_vectorizers = extract_ngram_features_fit(train_text_only, max_features=50)
test_text_only = all_text.iloc[len(train_df):]
ngram_test = extract_ngram_features_transform(test_text_only, ngram_vectorizers)

all_features_train = pd.concat(
    [
        basic_features.iloc[:len(train_df)].reset_index(drop=True),
        readability_features.iloc[:len(train_df)].reset_index(drop=True),
        pos_features.iloc[:len(train_df)].reset_index(drop=True),
        authorial_features.iloc[:len(train_df)].reset_index(drop=True),
        complexity_features.iloc[:len(train_df)].reset_index(drop=True),
        ngram_train.reset_index(drop=True),
    ],
    axis=1,
)
all_features_test = pd.concat(
    [
        basic_features.iloc[len(train_df):].reset_index(drop=True),
        readability_features.iloc[len(train_df):].reset_index(drop=True),
        pos_features.iloc[len(train_df):].reset_index(drop=True),
        authorial_features.iloc[len(train_df):].reset_index(drop=True),
        complexity_features.iloc[len(train_df):].reset_index(drop=True),
        ngram_test.reset_index(drop=True),
    ],
    axis=1,
)

all_features = pd.concat([all_features_train, all_features_test], axis=0).reset_index(drop=True)
all_features = all_features.replace([np.inf, -np.inf], 0)
all_features = all_features.fillna(0)

train_features = all_features.iloc[: len(train_df)].copy()
test_features = all_features.iloc[len(train_df) :].copy()

train_features["text"] = train_df["text"].values
test_features["text"] = test_df["text"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df["author"])

numerical_cols = [c for c in all_features.columns]
scaler = StandardScaler()
train_features[numerical_cols] = scaler.fit_transform(train_features[numerical_cols])
test_features[numerical_cols] = scaler.transform(test_features[numerical_cols])

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")
print(f"Number of numerical features: {len(numerical_cols)}")


# =============================================================================
# MODEL DEFINITION (SpookyClassifier from template)
# =============================================================================


class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=150, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # CRITICAL: Partial unfreezing - freeze first 16 layers, unfreeze last 8
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size  # 1024
        if num_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.head = nn.Linear(hidden_size + 64, num_authors)
        else:
            self.feature_proj = None
            self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_pool = outputs.last_hidden_state[:, 0, :]
        if self.feature_proj is not None and features is not None:
            feat_embed = self.feature_proj(features)
            combined = torch.cat([cls_pool, feat_embed], dim=1)
        else:
            combined = cls_pool
        logits = self.head(combined)
        return logits


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
max_length = 256

X_stylo = train_features[numerical_cols].values.astype(np.float32)
X_test_stylo = test_features[numerical_cols].values.astype(np.float32)
train_texts = train_features["text"].values
test_texts = test_features["text"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
all_test_probs = []

num_epochs = 30
patience = 5
batch_size = 16

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"\n=== Fold {fold+1}/5 ===")

    X_train_text = train_texts[train_idx]
    X_val_text = train_texts[val_idx]
    X_train_stylo = X_stylo[train_idx]
    X_val_stylo = X_stylo[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = SpookyClassifier(
        num_authors=3, num_features=X_stylo.shape[1], dropout_rate=0.3
    )
    model.to(device)

    # CRITICAL: Differentiated learning rates
    backbone_params = [
        p
        for layer in model.backbone.deberta.encoder.layer[-8:]
        for n, p in layer.named_parameters()
        if "bias" not in n and "LayerNorm" not in n
    ]
    head_params = list(model.head.parameters()) + (
        list(model.feature_proj.parameters()) if model.feature_proj else []
    )
    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
        ]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    def collate_fn(batch):
        texts, stylo, labels = zip(*batch)
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "features": torch.tensor(np.stack(stylo), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_dataset = list(zip(X_train_text, X_train_stylo, y_train))
    val_dataset = list(zip(X_val_text, X_val_stylo, y_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    best_val_loss = float("inf")
    best_model_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = initial_lrs[i] * (current_step / max(1, warmup_steps))
            else:
                scheduler.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_probs = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)

                val_loss += loss.item()
                val_probs.append(probs.cpu().numpy())
                val_true.append(labels.cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

        score = log_loss(val_true, val_probs)
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Log Loss: {score:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    model.eval()

    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
    val_probs = np.concatenate(val_probs)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    fold_score = log_loss(val_true, val_probs)
    fold_scores.append(fold_score)

    test_dataset = list(zip(test_texts, X_test_stylo, [0] * len(test_texts)))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            test_probs.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs)
    all_test_probs.append(test_probs)

    print(f"Fold {fold+1} Log Loss: {fold_score:.4f}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
print(f"\n=== 5-Fold Cross-Validation Results ===")
print(f"Mean Log Loss: {mean_score:.4f} ± {std_score:.4f}")

final_test_probs = np.mean(all_test_probs, axis=0)
# Ensure no NaN values in test predictions
if np.isnan(final_test_probs).any():
    final_test_probs = np.nan_to_num(final_test_probs, nan=1.0 / final_test_probs.shape[1])
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_8b49f017af034343b37958c15562811c.csv", index=False)

print(f"\nSubmission saved to ./submission/submission_8b49f017af034343b37958c15562811c.csv")
score = mean_score
print(f"Final Validation Score: {score}")