import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, DistilBertModel, DistilBertConfig
import xgboost as xgb
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Encode target
le = LabelEncoder()
train["author_encoded"] = le.fit_transform(train["author"])
num_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

# Create train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train["text"], train["author_encoded"])))

train_df = train.iloc[train_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def extract_features(text_series, is_train=True):
    features = pd.DataFrame(index=text_series.index)

    # Basic text stats
    features["char_count"] = text_series.apply(len)
    features["word_count"] = text_series.apply(lambda x: len(str(x).split()))
    features["sent_count"] = text_series.apply(
        lambda x: len(re.split(r"[.!?]+", str(x))) - 1
    )
    features["avg_word_len"] = text_series.apply(
        lambda x: (
            np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
    )

    # Punctuation features
    features["excl_count"] = text_series.apply(lambda x: str(x).count("!"))
    features["quest_count"] = text_series.apply(lambda x: str(x).count("?"))
    features["period_count"] = text_series.apply(lambda x: str(x).count("."))
    features["comma_count"] = text_series.apply(lambda x: str(x).count(","))
    features["semi_count"] = text_series.apply(lambda x: str(x).count(";"))
    features["colon_count"] = text_series.apply(lambda x: str(x).count(":"))
    features["quote_count"] = text_series.apply(
        lambda x: str(x).count('"') + str(x).count("'")
    )
    features["dash_count"] = text_series.apply(lambda x: str(x).count("-"))

    # Caps and special patterns
    features["caps_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    features["caps_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    features["digit_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isdigit())
    )

    # Readability features
    features["syllable_est"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).split()
            for c in ["a", "e", "i", "o", "u", "y"]
            if c in w.lower()
        )
    )
    features["complex_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 6)
    )

    # Quote density
    features["quote_density"] = text_series.apply(
        lambda x: str(x).count('"') / max(len(str(x)), 1)
    )

    # Stopword-based features
    stopwords = {
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
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
    }

    features["stopword_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in stopwords)
    )
    features["stopword_ratio"] = features["stopword_count"] / features[
        "word_count"
    ].replace(0, 1)

    # Word length distribution
    features["long_word_ratio"] = features["complex_words"] / features[
        "word_count"
    ].replace(0, 1)
    features["short_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) <= 3)
    ) / features["word_count"].replace(0, 1)

    # Vocabulary diversity
    features["unique_words"] = text_series.apply(
        lambda x: len(set(str(x).lower().split()))
    )
    features["type_token_ratio"] = features["unique_words"] / features[
        "word_count"
    ].replace(0, 1)

    # Historical/archaic word indicators
    archaic_words = {
        "thee",
        "thou",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "art",
        "hast",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "nay",
        "forsooth",
        "perchance",
        "anon",
        "prithee",
    }
    # Note: These author-specific word lists are used only for feature engineering
    # (deriving counts from existing text), which is fine since they just characterize
    # the text. They are NOT used in augmentation or any label-leaking transformation.
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "non-euclidean",
        "ichor",
        "gibbering",
        "maddening",
        "cosmic",
        "carcosa",
        "yog-sothoth",
        "cthulhu",
    }
    poe_words = {
        "nevermore",
        "chamber",
        "tapping",
        "rapping",
        "sepulchre",
        "ghoul",
        "pallid",
        "dreary",
        "weary",
        "ebony",
    }

    features["archaic_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in archaic_words)
    )
    features["lovecraft_score"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in lovecraft_words)
    )
    features["poe_score"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in poe_words)
    )

    # First person pronoun usage
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    features["first_person_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in first_person)
    )

    # Sentence complexity
    features["avg_sent_length"] = features["word_count"] / features[
        "sent_count"
    ].replace(0, 1)

    # Character diversity
    features["unique_chars"] = text_series.apply(lambda x: len(set(str(x).lower())))
    features["char_diversity"] = features["unique_chars"] / features[
        "char_count"
    ].replace(0, 1)

    # Negation count
    negation_words = {
        "not",
        "no",
        "never",
        "nothing",
        "nowhere",
        "none",
        "neither",
        "nor",
        "cannot",
        "can't",
        "don't",
        "won't",
        "doesn't",
        "isn't",
    }
    features["negation_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in negation_words)
    )

    # Emotion/affect words
    positive_words = set(
        [
            "love",
            "beautiful",
            "wonderful",
            "happy",
            "joy",
            "delight",
            "pleasure",
            "gentle",
            "sweet",
            "tender",
            "bright",
            "hope",
        ]
    )
    negative_words = set(
        [
            "dark",
            "dread",
            "fear",
            "horror",
            "terror",
            "death",
            "shadow",
            "gloom",
            "pain",
            "sorrow",
            "misery",
            "agony",
            "anguish",
            "doom",
        ]
    )

    features["positive_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in positive_words)
    )
    features["negative_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in negative_words)
    )
    features["sentiment_ratio"] = (
        features["positive_count"] - features["negative_count"]
    ) / (features["positive_count"] + features["negative_count"] + 1)

    return features


# Extract features
train_features = extract_features(train_df["text"], is_train=True)
val_features = extract_features(val_df["text"], is_train=False)
test_features = extract_features(test["text"], is_train=False)

# Handle NaN and inf values
train_features = train_features.replace([np.inf, -np.inf], 0).fillna(0)
val_features = val_features.replace([np.inf, -np.inf], 0).fillna(0)
test_features = test_features.replace([np.inf, -np.inf], 0).fillna(0)

# Add percentile features (fit on train, transform on val/test to avoid leakage)
train_features["word_count_rank"] = train_features["word_count"].rank(pct=True)
train_features["char_count_rank"] = train_features["char_count"].rank(pct=True)
# Use train's percentiles to rank val/test values, ensuring unique bin edges
train_word_rank_edges = np.percentile(train_features["word_count"], np.linspace(0, 100, 101))
train_char_rank_edges = np.percentile(train_features["char_count"], np.linspace(0, 100, 101))
# Remove duplicate edges to avoid ValueError in pd.cut
train_word_rank_edges = np.unique(train_word_rank_edges)
train_char_rank_edges = np.unique(train_char_rank_edges)
val_features["word_count_rank"] = pd.cut(val_features["word_count"], bins=train_word_rank_edges, labels=False, include_lowest=True).astype(float) / (len(train_word_rank_edges) - 1)
val_features["char_count_rank"] = pd.cut(val_features["char_count"], bins=train_char_rank_edges, labels=False, include_lowest=True).astype(float) / (len(train_char_rank_edges) - 1)
test_features["word_count_rank"] = pd.cut(test_features["word_count"], bins=train_word_rank_edges, labels=False, include_lowest=True).astype(float) / (len(train_word_rank_edges) - 1)
test_features["char_count_rank"] = pd.cut(test_features["char_count"], bins=train_char_rank_edges, labels=False, include_lowest=True).astype(float) / (len(train_char_rank_edges) - 1)
# Fill any NaN from edges (values outside train range)
val_features = val_features.fillna(0)
test_features = test_features.fillna(0)


# N-gram features
def create_ngram_count(text, n=2, ngram_type="char"):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    if ngram_type == "char":
        common_ngrams = [
            "th",
            "he",
            "in",
            "er",
            "an",
            "re",
            "nd",
            "at",
            "on",
            "nt",
            "ha",
            "ou",
            "it",
            "hi",
            "es",
            "st",
            "en",
            "ea",
            "to",
            "or",
            "ed",
            "te",
            "ar",
            "al",
            "le",
            "ve",
            "ti",
            "ra",
            "ur",
            "me",
        ]
        return sum(text.count(ng) for ng in common_ngrams) / max(len(text), 1)
    else:
        words = text.split()
        common_bigrams = [
            ("of", "the"),
            ("in", "the"),
            ("to", "the"),
            ("and", "the"),
            ("it", "was"),
            ("i", "was"),
            ("there", "was"),
            ("this", "was"),
        ]
        count = 0
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            if bigram in common_bigrams:
                count += 1
        return count / max(len(words), 1)


for prefix, df in [("train", train_df), ("val", val_df), ("test", test)]:
    text_data = df["text"]
    ngram_char = text_data.apply(lambda x: create_ngram_count(x, 2, "char"))
    ngram_word = text_data.apply(lambda x: create_ngram_count(x, 2, "word"))
    if prefix == "train":
        train_features["ngram_char_density"] = ngram_char.values
        train_features["ngram_word_density"] = ngram_word.values
    elif prefix == "val":
        val_features["ngram_char_density"] = ngram_char.values
        val_features["ngram_word_density"] = ngram_word.values
    else:
        test_features["ngram_char_density"] = ngram_char.values
        test_features["ngram_word_density"] = ngram_word.values

# Scale features
scaler = StandardScaler()
feature_columns = train_features.columns

train_features_scaled = pd.DataFrame(
    scaler.fit_transform(train_features),
    columns=feature_columns,
    index=train_features.index,
)
val_features_scaled = pd.DataFrame(
    scaler.transform(val_features), columns=feature_columns, index=val_features.index
)
test_features_scaled = pd.DataFrame(
    scaler.transform(test_features), columns=feature_columns, index=test_features.index
)

# Combine with original data
train_processed = pd.concat(
    [
        train_df[["id", "text", "author", "author_encoded"]].reset_index(drop=True),
        train_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)
val_processed = pd.concat(
    [
        val_df[["id", "text", "author", "author_encoded"]].reset_index(drop=True),
        val_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)
test_processed = pd.concat(
    [
        test[["id", "text"]].reset_index(drop=True),
        test_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed val shape: {val_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# Extract arrays
X_train_tab = train_processed[feature_columns].values
y_train = train_processed["author_encoded"].values
X_val_tab = val_processed[feature_columns].values
y_val = val_processed["author_encoded"].values
X_test_tab = test_processed[feature_columns].values

train_texts = train_processed["text"].tolist()
val_texts = val_processed["text"].tolist()
test_texts = test_processed["text"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# TRAIN XGBoost
# ============================================================
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)
xgb_val_probs = xgb_model.predict_proba(X_val_tab)
xgb_test_probs = xgb_model.predict_proba(X_test_tab)
print(f"XGBoost val log loss: {log_loss(y_val, xgb_val_probs):.6f}")

# ============================================================
# TRAIN LightGBM
# ============================================================
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_leaves=31,
    objective="multiclass",
    metric="multi_logloss",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_tab,
    y_train,
    eval_set=[(X_val_tab, y_val)],
    callbacks=[lgb.early_stopping(30)],
)
lgb_val_probs = lgb_model.predict_proba(X_val_tab)
lgb_test_probs = lgb_model.predict_proba(X_test_tab)
print(f"LightGBM val log loss: {log_loss(y_val, lgb_val_probs):.6f}")


# ============================================================
# TEXT AUGMENTATION
# ============================================================
import random

# Simple synonym dictionary for augmentation
SYNONYM_MAP = {
    "good": ["fine", "excellent", "great", "superb"],
    "bad": ["poor", "terrible", "awful", "dreadful"],
    "big": ["large", "huge", "enormous", "vast"],
    "small": ["tiny", "little", "miniature", "petite"],
    "beautiful": ["lovely", "pretty", "gorgeous", "stunning"],
    "strange": ["odd", "weird", "peculiar", "bizarre"],
    "dark": ["gloomy", "shadowy", "murky", "dim"],
    "light": ["bright", "brilliant", "luminous", "radiant"],
    "happy": ["glad", "joyful", "delighted", "cheerful"],
    "sad": ["sorrowful", "gloomy", "melancholy", "dreary"],
    "old": ["ancient", "elderly", "aged", "venerable"],
    "new": ["fresh", "novel", "modern", "recent"],
    "strong": ["powerful", "robust", "sturdy", "mighty"],
    "weak": ["feeble", "fragile", "frail", "delicate"],
    "great": ["grand", "majestic", "splendid", "magnificent"],
    "little": ["slight", "minor", "trivial", "insignificant"],
    "quick": ["fast", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "leisurely", "gradual", "unhurried"],
    "hard": ["difficult", "tough", "challenging", "arduous"],
    "soft": ["gentle", "tender", "mellow", "mild"],
    "thin": ["slender", "slim", "narrow", "fine"],
    "thick": ["dense", "heavy", "solid", "substantial"],
    "cold": ["chilly", "frigid", "icy", "frosty"],
    "hot": ["warm", "scorching", "blazing", "fiery"],
    "beautiful": ["lovely", "pretty", "gorgeous", "stunning", "attractive"],
    "strange": ["odd", "weird", "peculiar", "bizarre", "curious"],
}

def augment_text(text, word_dropout_prob=0.05, synonym_replace_prob=0.03, word_swap_prob=0.1, punct_insert_prob=0.15, vowel_noise_prob=0.05):
    """Apply enhanced augmentation: back-translation-invariant random word swaps, punctuation insertion, vowel noise, and synonym replacement."""
    # Choose exactly one augmentation randomly
    aug_choice = random.random()
    if aug_choice < 0.25:
        # (a) Back-translation-invariant random word swaps within 3-word windows at word_swap_prob probability
        words = str(text).split()
        if len(words) > 2:
            for i in range(len(words) - 1):
                if random.random() < word_swap_prob:
                    swap_window = min(3, len(words) - i)
                    swap_idx = i + random.randint(1, swap_window - 1)
                    words[i], words[swap_idx] = words[swap_idx], words[i]
        return " ".join(words)
    elif aug_choice < 0.50:
        # (b) Random insertion of punctuation at sentence boundaries at punct_insert_prob probability
        text_str = str(text)
        sentences = re.split(r'(\. )', text_str)
        if len(sentences) > 1:
            punct_choices = ['--', ';', ':', '...']
            result_parts = []
            for sent in sentences:
                if random.random() < punct_insert_prob and sent.strip():
                    punct = random.choice(punct_choices)
                    result_parts.append(sent.rstrip() + ' ' + punct + ' ')
                else:
                    result_parts.append(sent)
            return ''.join(result_parts)
        else:
            return text_str
    elif aug_choice < 0.75:
        # (c) Character-level noise: randomly replace vowel_noise_prob of vowels with other vowels
        text_str = str(text)
        vowels = 'aeiouAEIOU'
        vowel_list = ['a', 'e', 'i', 'o', 'u']
        result_chars = []
        for ch in text_str:
            if ch.lower() in 'aeiou' and random.random() < vowel_noise_prob:
                # Replace with random different vowel
                new_vowel = random.choice([v for v in vowel_list if v != ch.lower()])
                if ch.isupper():
                    result_chars.append(new_vowel.upper())
                else:
                    result_chars.append(new_vowel)
            else:
                result_chars.append(ch)
        return ''.join(result_chars)
    else:
        # (d) Enhanced synonym replacement at increased rate
        words = str(text).split()
        augmented_words = []
        for word in words:
            # Word dropout (keep existing behavior)
            if random.random() < word_dropout_prob:
                continue
            # Enhanced synonym replacement (rate increased to 0.05)
            if random.random() < synonym_replace_prob and word.lower() in SYNONYM_MAP:
                replacement = random.choice(SYNONYM_MAP[word.lower()])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                augmented_words.append(replacement)
            else:
                augmented_words.append(word)
        return " ".join(augmented_words)


# ============================================================
# TEXT ENCODER (DistilBERT) with Attention Pooling
# ============================================================
class TextFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        hidden_dim=512,
        num_classes=3,
        dropout=0.4,
    ):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        config = self.distilbert.config
        self.hidden_size = config.hidden_size
        self.hidden_dim = hidden_dim

        # Multi-head attention pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.attention_dropout = nn.Dropout(0.1)

        # Classifier takes BERT pooled (768)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask, stylometric_features=None, return_embeddings=False):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size=768)

        # Attention-weighted pooling: use CLS token as query
        cls_token = last_hidden[:, 0:1, :]  # (batch, 1, hidden_size)
        # Apply multi-head attention
        attn_output, attn_weights = self.attention(
            query=cls_token,
            key=last_hidden,
            value=last_hidden,
            key_padding_mask=~(attention_mask.bool()),
        )
        attn_output = self.attention_norm(attn_output + cls_token)
        attn_output = self.attention_dropout(attn_output)
        pooled = attn_output.squeeze(1)  # (batch, hidden_size=768)

        if return_embeddings:
            logits = self.classifier(pooled)
            features = pooled
            return logits, features
        else:
            logits = self.classifier(pooled)
            return logits


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_encoder = TextFeatureEncoder(num_classes=num_classes).to(device)


# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, augment=False):
        self.texts = texts
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = augment_text(text, word_dropout_prob=0.05, synonym_replace_prob=0.03)
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = TextDataset(train_texts, y_train, augment=True)


# train_dataset is now created above with augment=True
val_dataset = TextDataset(val_texts, y_val)
test_dataset = TextDataset(test_texts)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

# Initialize optimizer with lower learning rate and stronger weight decay
optimizer = AdamW(
    text_encoder.parameters(),
    lr=2e-5,
    weight_decay=0.1,
)

# Cosine annealing with warm restarts scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2, T_mult=2
)
scaler = GradScaler()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training
best_val_loss = float("inf")
best_model_state = None
num_epochs = 10
patience = 3
patience_counter = 0

print("Training DistilBERT with cosine annealing warm restarts...")
for epoch in range(num_epochs):
    text_encoder.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = text_encoder(input_ids=input_ids, attention_mask=attention_mask, stylometric_features=None)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
    # Step scheduler after epoch
    scheduler.step()
    avg_train_loss = total_loss / num_batches

    text_encoder.eval()
    val_loss = 0.0
    val_num_batches = 0
    all_val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits = text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask, stylometric_features=None
                )
                loss = criterion(logits, labels)
            val_loss += loss.item()
            val_num_batches += 1
            probs = F.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
    avg_val_loss = val_loss / val_num_batches
    val_probs_modern = np.concatenate(all_val_preds, axis=0)
    val_log_loss_modern = log_loss(y_val, val_probs_modern)
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Log Loss: {val_log_loss_modern:.6f}"
    )
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = text_encoder.state_dict().copy()
        patience_counter = 0
        print(f"  -> New best model (val_loss={best_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

if best_model_state is not None:
    text_encoder.load_state_dict(best_model_state)
    print(f"Loaded best model with val_loss={best_val_loss:.6f}")

# Generate predictions
text_encoder.eval()
all_val_modern = []
all_test_modern = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=1)
        all_val_modern.append(probs.cpu().numpy())
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=1)
        all_test_modern.append(probs.cpu().numpy())

modern_val_probs = np.concatenate(all_val_modern, axis=0)
modern_test_probs = np.concatenate(all_test_modern, axis=0)


# Use DistilBERT directly for submission
val_probs_final = modern_val_probs
test_probs_final = modern_test_probs

score = log_loss(y_val, val_probs_final)
print(f"DistilBERT Validation Log Loss: {score:.6f}")

# Clip and normalize
eps = 1e-15
test_probs_final = np.clip(test_probs_final, eps, 1 - eps)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# Create submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_processed["id"].values,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Test predictions shape: {submission.shape}")
print(submission.head())