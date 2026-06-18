import pandas as pd
import numpy as np
import re
import os
import warnings
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# Download NLTK resources if not already present
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors distribution:\n{train_df['author'].value_counts()}")

# ============================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================


def clean_text(text):
    """Basic text cleaning while preserving stylistic elements"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text


def count_syllables(word):
    """Simple syllable counting heuristic"""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2:
        count += 1
    if count == 0:
        count = 1
    return count


def calculate_readability(text):
    """Calculate Flesch Reading Ease score"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    if len(sentences) == 0 or len(words) == 0:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words if w.isalpha())

    if num_sentences == 0 or num_words == 0:
        return 0.0

    # Flesch Reading Ease = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    score = (
        206.835
        - 1.015 * (num_words / num_sentences)
        - 84.6 * (num_syllables / num_words)
    )
    return max(0, min(100, score))


# ============================================================
# STYLOMETRIC FEATURE ENGINEERING
# ============================================================


def extract_stylometric_features(text_series, name_prefix=""):
    """Extract comprehensive stylometric features from text"""
    features = pd.DataFrame(index=text_series.index)

    # Basic text statistics
    features[f"{name_prefix}char_count"] = text_series.apply(len)
    features[f"{name_prefix}word_count"] = text_series.apply(
        lambda x: len(str(x).split())
    )
    features[f"{name_prefix}avg_word_length"] = features[f"{name_prefix}char_count"] / (
        features[f"{name_prefix}word_count"] + 1
    )

    # Sentence-level features
    features[f"{name_prefix}sentence_count"] = text_series.apply(
        lambda x: len(sent_tokenize(str(x)))
    )
    features[f"{name_prefix}avg_sentence_length"] = features[
        f"{name_prefix}word_count"
    ] / (features[f"{name_prefix}sentence_count"] + 1)

    # Punctuation features
    for punct in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "—"]:
        punct_count = text_series.apply(lambda x: str(x).count(punct))
        features[f"{name_prefix}punct_{punct}"] = punct_count
        features[f"{name_prefix}punct_{punct}_ratio"] = punct_count / (
            features[f"{name_prefix}word_count"] + 1
        )

    # Total punctuation count
    all_punct = text_series.apply(
        lambda x: sum(1 for c in str(x) if c in ".,!?;:\-\"'()—")
    )
    features[f"{name_prefix}total_punctuation"] = all_punct
    features[f"{name_prefix}punctuation_ratio"] = all_punct / (
        features[f"{name_prefix}word_count"] + 1
    )

    # Capitalization features
    features[f"{name_prefix}capitalized_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if w[0].isupper() if len(w) > 0)
    )
    features[f"{name_prefix}all_caps_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 1 and w.isupper())
    )

    # Vocabulary richness
    features[f"{name_prefix}unique_words"] = text_series.apply(
        lambda x: len(set(str(x).lower().split()))
    )
    word_counts = text_series.apply(lambda x: len(str(x).split()))
    features[f"{name_prefix}type_token_ratio"] = features[
        f"{name_prefix}unique_words"
    ] / (word_counts + 1)

    # Readability
    features[f"{name_prefix}flesch_reading_ease"] = text_series.apply(
        calculate_readability
    )

    # Syllable features
    features[f"{name_prefix}avg_syllables_per_word"] = text_series.apply(
        lambda x: (
            np.mean([count_syllables(w) for w in str(x).split() if w.isalpha()])
            if len(str(x).split()) > 0
            else 0
        )
    )

    # Word length distribution
    lengths = text_series.apply(lambda x: [len(w) for w in str(x).split()])
    features[f"{name_prefix}short_words_ratio"] = lengths.apply(
        lambda x: sum(1 for l in x if l <= 3) / (len(x) + 1)
    )
    features[f"{name_prefix}medium_words_ratio"] = lengths.apply(
        lambda x: sum(1 for l in x if 4 <= l <= 6) / (len(x) + 1)
    )
    features[f"{name_prefix}long_words_ratio"] = lengths.apply(
        lambda x: sum(1 for l in x if l >= 7) / (len(x) + 1)
    )

    # Quote usage
    features[f"{name_prefix}quotes_count"] = text_series.apply(
        lambda x: str(x).count('"') // 2 + str(x).count("'") // 2
    )

    # Special character features
    features[f"{name_prefix}ellipsis_count"] = text_series.apply(
        lambda x: str(x).count("...")
    )
    features[f"{name_prefix}exclamation_count"] = text_series.apply(
        lambda x: str(x).count("!")
    )
    features[f"{name_prefix}question_count"] = text_series.apply(
        lambda x: str(x).count("?")
    )

    return features


# ============================================================
# CONTENT-BASED FEATURE ENGINEERING (ON TRAINING DATA)
# ============================================================


def extract_content_features(train_texts, test_texts, max_features=500):
    """Extract TF-IDF and count-based features from text"""
    # Character n-grams (captures writing style at character level)
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
    )

    train_char_features = char_vectorizer.fit_transform(train_texts)
    test_char_features = char_vectorizer.transform(test_texts)

    char_feature_names = [
        f"char_ngram_{i}" for i in range(train_char_features.shape[1])
    ]

    train_char_df = pd.DataFrame(
        train_char_features.toarray(),
        columns=char_feature_names,
        index=train_texts.index,
    )

    test_char_df = pd.DataFrame(
        test_char_features.toarray(), columns=char_feature_names, index=test_texts.index
    )

    # Word n-grams (captures content and style)
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True,
        max_df=0.95,
        min_df=2,
    )

    train_word_features = word_vectorizer.fit_transform(train_texts)
    test_word_features = word_vectorizer.transform(test_texts)

    word_feature_names = [
        f"word_ngram_{i}" for i in range(train_word_features.shape[1])
    ]

    train_word_df = pd.DataFrame(
        train_word_features.toarray(),
        columns=word_feature_names,
        index=train_texts.index,
    )

    test_word_df = pd.DataFrame(
        test_word_features.toarray(), columns=word_feature_names, index=test_texts.index
    )

    return (
        train_char_df,
        test_char_df,
        train_word_df,
        test_word_df,
        char_vectorizer,
        word_vectorizer,
    )


# ============================================================
# PARTS-OF-SPEECH FEATURES (Optional - requires NLTK tagger)
# ============================================================


def extract_pos_features(text_series, name_prefix=""):
    """Extract part-of-speech distribution features"""
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    features = pd.DataFrame(index=text_series.index)

    pos_tags = ["CC", "DT", "IN", "JJ", "NN", "PRP", "RB", "VB", "WRB", "TO"]

    def get_pos_distribution(text):
        if not isinstance(text, str) or len(text) == 0:
            return {tag: 0 for tag in pos_tags}

        tokens = word_tokenize(text.lower())
        if len(tokens) == 0:
            return {tag: 0 for tag in pos_tags}

        tagged = nltk.pos_tag(tokens)
        tag_counts = Counter(tag for _, tag in tagged)

        total = len(tokens)
        return {tag: tag_counts.get(tag, 0) / total for tag in pos_tags}

    pos_distributions = text_series.apply(get_pos_distribution)

    for tag in pos_tags:
        features[f"{name_prefix}pos_{tag}"] = pos_distributions.apply(lambda x: x[tag])

    return features


# ============================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================


def engineer_features(train_df, test_df):
    """Complete feature engineering pipeline"""

    # Clean text
    train_df["text_clean"] = train_df["text"].apply(clean_text)
    test_df["text_clean"] = test_df["text"].apply(clean_text)

    print("Extracting stylometric features...")
    # Stylometric features
    train_stylo = extract_stylometric_features(train_df["text_clean"], name_prefix="")
    test_stylo = extract_stylometric_features(test_df["text_clean"], name_prefix="")

    print("Extracting content features...")
    # Content features (TF-IDF from training only)
    train_char_tfidf, test_char_tfidf, train_word_tfidf, test_word_tfidf, _, _ = (
        extract_content_features(
            train_df["text_clean"], test_df["text_clean"], max_features=500
        )
    )

    print("Extracting POS features...")
    # POS features
    try:
        train_pos = extract_pos_features(train_df["text_clean"])
        test_pos = extract_pos_features(test_df["text_clean"])
    except Exception as e:
        print(f"Warning: POS feature extraction failed: {e}")
        train_pos = pd.DataFrame()
        test_pos = pd.DataFrame()

    # Combine all features
    train_features = pd.concat(
        [
            train_stylo.reset_index(drop=True),
            train_char_tfidf.reset_index(drop=True),
            train_word_tfidf.reset_index(drop=True),
            train_pos.reset_index(drop=True),
        ],
        axis=1,
    )

    test_features = pd.concat(
        [
            test_stylo.reset_index(drop=True),
            test_char_tfidf.reset_index(drop=True),
            test_word_tfidf.reset_index(drop=True),
            test_pos.reset_index(drop=True),
        ],
        axis=1,
    )

    # Handle NaN values (fill with 0 for sparse features)
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)

    # Remove infinite values
    train_features = train_features.replace([np.inf, -np.inf], 0)
    test_features = test_features.replace([np.inf, -np.inf], 0)

    # Scale features (fit on training only)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    train_features = pd.DataFrame(
        train_features_scaled, columns=train_features.columns, index=train_df.index
    )

    test_features = pd.DataFrame(
        test_features_scaled, columns=test_features.columns, index=test_df.index
    )

    return train_features, test_features, scaler


# ============================================================
# TRAIN/VALIDATION SPLIT
# ============================================================
print("\n" + "=" * 50)
print("Creating Train/Validation Split")
print("=" * 50)

# Use StratifiedKFold to preserve class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

# Split the original dataframe
train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

# Encode labels
le = LabelEncoder()
train_labels = le.fit_transform(train_split["author"])
val_labels = le.transform(val_split["author"])

print(f"Train size: {len(train_split)}")
print(f"Validation size: {len(val_split)}")
print(f"Authors: {le.classes_}")
print(f"Train distribution: {np.bincount(train_labels)}")
print(f"Validation distribution: {np.bincount(val_labels)}")

# ============================================================
# APPLY FEATURE ENGINEERING (AFTER SPLIT TO PREVENT LEAKAGE)
# ============================================================
print("=" * 50)
print("Starting Feature Engineering Pipeline (after split)")
print("=" * 50)

# Feature engineering on train split only (fit on train, transform val and test)
# Clean text
train_split["text_clean"] = train_split["text"].apply(clean_text)
val_split["text_clean"] = val_split["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(train_split["text_clean"], name_prefix="")
val_stylo = extract_stylometric_features(val_split["text_clean"], name_prefix="")
test_stylo = extract_stylometric_features(test_df["text_clean"], name_prefix="")

print("Extracting content features...")
# Fit vectorizers on train only, transform val and test
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    lowercase=True,
)
train_char_tfidf = char_vectorizer.fit_transform(train_split["text_clean"])
val_char_tfidf = char_vectorizer.transform(val_split["text_clean"])
test_char_tfidf = char_vectorizer.transform(test_df["text_clean"])
char_feature_names = [f"char_ngram_{i}" for i in range(train_char_tfidf.shape[1])]
train_char_df = pd.DataFrame(train_char_tfidf.toarray(), columns=char_feature_names, index=train_split.index)
val_char_df = pd.DataFrame(val_char_tfidf.toarray(), columns=char_feature_names, index=val_split.index)
test_char_df = pd.DataFrame(test_char_tfidf.toarray(), columns=char_feature_names, index=test_df.index)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=500,
    sublinear_tf=True,
    stop_words="english",
    lowercase=True,
    max_df=0.95,
    min_df=2,
)
train_word_tfidf = word_vectorizer.fit_transform(train_split["text_clean"])
val_word_tfidf = word_vectorizer.transform(val_split["text_clean"])
test_word_tfidf = word_vectorizer.transform(test_df["text_clean"])
word_feature_names = [f"word_ngram_{i}" for i in range(train_word_tfidf.shape[1])]
train_word_df = pd.DataFrame(train_word_tfidf.toarray(), columns=word_feature_names, index=train_split.index)
val_word_df = pd.DataFrame(val_word_tfidf.toarray(), columns=word_feature_names, index=val_split.index)
test_word_df = pd.DataFrame(test_word_tfidf.toarray(), columns=word_feature_names, index=test_df.index)

print("Extracting POS features...")
try:
    train_pos = extract_pos_features(train_split["text_clean"])
    val_pos = extract_pos_features(val_split["text_clean"])
    test_pos = extract_pos_features(test_df["text_clean"])
except Exception as e:
    print(f"Warning: POS feature extraction failed: {e}")
    train_pos = pd.DataFrame()
    val_pos = pd.DataFrame()
    test_pos = pd.DataFrame()

# Combine all features for train
train_features = pd.concat(
    [train_stylo.reset_index(drop=True), train_char_df.reset_index(drop=True), train_word_df.reset_index(drop=True), train_pos.reset_index(drop=True)],
    axis=1,
)

val_features = pd.concat(
    [val_stylo.reset_index(drop=True), val_char_df.reset_index(drop=True), val_word_df.reset_index(drop=True), val_pos.reset_index(drop=True)],
    axis=1,
)

test_features = pd.concat(
    [test_stylo.reset_index(drop=True), test_char_df.reset_index(drop=True), test_word_df.reset_index(drop=True), test_pos.reset_index(drop=True)],
    axis=1,
)

# Handle NaN and infinite values
train_features = train_features.fillna(0).replace([np.inf, -np.inf], 0)
val_features = val_features.fillna(0).replace([np.inf, -np.inf], 0)
test_features = test_features.fillna(0).replace([np.inf, -np.inf], 0)

# Scale features (fit on train only)
scaler = StandardScaler()
train_features = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns, index=train_split.index)
val_features = pd.DataFrame(scaler.transform(val_features), columns=val_features.columns, index=val_split.index)
test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns, index=test_df.index)

print(f"\nFeature engineering complete!")
print(f"Train features shape: {train_features.shape}")
print(f"Val features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        cls_pool = hidden_states[:, 0, :]
        logits = self.head(cls_pool)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Collect backbone unfrozen params (last 8 layers)
backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

# Collect head params (only the new single linear layer)
head_params = list(model.head.parameters())

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_unfrozen_params):,}")
print(f"Head params: {sum(p.numel() for p in head_params):,}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        return item


# Get original texts for training
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Use previously computed integer position indices for train/validation split
train_indices = train_idx.tolist()
val_indices = val_idx.tolist()

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length
)
val_dataset = SpookyDataset(val_texts_final, val_labels_final, tokenizer, max_length)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length)

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
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Linear warmup, then cosine annealing
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

# Initialize scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
)

# Save initial LR for warmup scaling
initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        # Apply scheduler step per batch
        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = initial_lrs[0] * warmup_factor
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

            with autocast():
                logits = model(input_ids, attention_mask)
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
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
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
        with autocast():
            logits = model(input_ids, attention_mask)
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

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")