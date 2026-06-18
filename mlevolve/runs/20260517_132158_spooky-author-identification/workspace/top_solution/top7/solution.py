import numpy as np
import pandas as pd
import re
import os
import gc
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import log_loss
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# DATA PROCESSING AND FEATURE ENGINEERING
# =====================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors distribution:\n{train_df['author'].value_counts()}")

# =====================
# TEXT PREPROCESSING
# =====================

def clean_text(text):
    """Clean text while preserving author-specific patterns"""
    if not isinstance(text, str):
        return ""
    text = str(text)
    # Fix common abbreviations and contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'twas", " it was", text)
    # Preserve em-dashes and semicolons (stylistic markers)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Apply cleaning
train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# =====================
# BASIC TEXT FEATURES
# =====================

def extract_basic_features(text):
    """Extract basic text features useful for author identification"""
    features = {}
    text_str = str(text)
    words = text_str.split()
    sentences = re.split(r"[.!?]+", text_str)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Length features
    features["char_count"] = len(text_str)
    features["word_count"] = len(words)
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["avg_sentence_length"] = (
        features["word_count"] / features["sentence_count"]
    )

    # Vocabulary richness
    unique_words = set([w.lower() for w in words])
    features["unique_word_ratio"] = len(unique_words) / max(len(words), 1)

    # Punctuation features (stylistic markers)
    features["exclamation_count"] = text_str.count("!")
    features["question_count"] = text_str.count("?")
    features["period_count"] = text_str.count(".")
    features["comma_count"] = text_str.count(",")
    features["semicolon_count"] = text_str.count(";")
    features["colon_count"] = text_str.count(":")
    features["emdash_count"] = text_str.count("—") + text_str.count("--")
    features["quote_count"] = (
        text_str.count('"') + text_str.count('"') + text_str.count("'")
    )
    features["parenthesis_count"] = text_str.count("(") + text_str.count(")")

    # Punctuation ratios
    features["punctuation_ratio"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["period_count"]
        + features["comma_count"]
        + features["semicolon_count"]
        + features["colon_count"]
    ) / max(len(words), 1)

    # Special character patterns
    features["capital_letter_count"] = sum(1 for c in text_str if c.isupper())
    features["capital_ratio"] = features["capital_letter_count"] / max(len(text_str), 1)

    # Dialog markers (em-dash, quotation)
    features["dialog_intensity"] = (
        features["emdash_count"] + features["quote_count"]
    ) / max(features["sentence_count"], 1)

    # Stopword ratio (common vs rare words)
    common_stopwords = {
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
        "was",
        "were",
        "had",
        "have",
        "has",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "be",
        "this",
        "that",
        "it",
        "is",
    }
    word_lower = [w.lower() for w in words]
    stopword_count = sum(1 for w in word_lower if w in common_stopwords)
    features["stopword_ratio"] = stopword_count / max(len(words), 1)

    # Contraction and specific word patterns
    features["negation_count"] = len(
        re.findall(r"\b(not|never|nothing|no|none|nobody|nowhere)\b", text_str.lower())
    )
    features["first_person_count"] = len(
        re.findall(r"\b(I|me|my|mine|we|us|our|ours)\b", text_str)
    )
    features["past_tense_markers"] = len(
        re.findall(
            r"\b(was|were|had|did|been|became|seemed|appeared)\b", text_str.lower()
        )
    )

    return features

# Extract features for train and test
print("Extracting basic features...")
train_features_list = []
for text in train_df["text_clean"]:
    train_features_list.append(extract_basic_features(text))
train_features = pd.DataFrame(train_features_list)

test_features_list = []
for text in test_df["text_clean"]:
    test_features_list.append(extract_basic_features(text))
test_features = pd.DataFrame(test_features_list)

# =====================
# N-GRAM FEATURES (Character and Word)
# =====================

print("Creating n-gram features...")

# Character n-grams (catching author-specific patterns)
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=500, min_df=3, max_df=0.9
)
char_features_train = char_vectorizer.fit_transform(train_df["text_clean"])
char_features_test = char_vectorizer.transform(test_df["text_clean"])

char_feat_df_train = pd.DataFrame(
    char_features_train.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_features_train.shape[1])],
)
char_feat_df_test = pd.DataFrame(
    char_features_test.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_features_test.shape[1])],
)

# Word n-grams (capturing phrase patterns)
word_vectorizer = CountVectorizer(
    analyzer="word", ngram_range=(1, 3), max_features=500, min_df=3, max_df=0.8
)
word_features_train = word_vectorizer.fit_transform(train_df["text_clean"])
word_features_test = word_vectorizer.transform(test_df["text_clean"])

word_feat_df_train = pd.DataFrame(
    word_features_train.toarray(),
    columns=[f"word_ngram_{i}" for i in range(word_features_train.shape[1])],
)
word_feat_df_test = pd.DataFrame(
    word_features_test.toarray(),
    columns=[f"word_ngram_{i}" for i in range(word_features_test.shape[1])],
)

# =====================
# TF-IDF FEATURES
# =====================

print("Creating TF-IDF features...")

tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=1000,
    min_df=3,
    max_df=0.85,
    sublinear_tf=True,
)
tfidf_word_train = tfidf_word.fit_transform(train_df["text_clean"])
tfidf_word_test = tfidf_word.transform(test_df["text_clean"])

tfidf_word_df_train = pd.DataFrame(
    tfidf_word_train.toarray(),
    columns=[f"tfidf_word_{i}" for i in range(tfidf_word_train.shape[1])],
)
tfidf_word_df_test = pd.DataFrame(
    tfidf_word_test.toarray(),
    columns=[f"tfidf_word_{i}" for i in range(tfidf_word_test.shape[1])],
)

# =====================
# STYLOMETRIC FEATURES (Author-specific patterns)
# =====================

print("Creating stylometric features...")

def extract_stylometric_features(text):
    """Extract author-specific writing style features"""
    features = {}
    text_str = str(text)
    words = text_str.split()
    sentences = re.split(r"[.!?]+", text_str)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Hemingway vs. verbose metrics
    sentences_long = sum(1 for s in sentences if len(s.split()) > 20)
    features["long_sentence_ratio"] = sentences_long / max(len(sentences), 1)

    # Rare word usage (words > 8 characters)
    long_words = [w for w in words if len(w) > 8]
    features["rare_word_ratio"] = len(long_words) / max(len(words), 1)

    # Adjective/adverb markers (words ending in ly, ing, ed, ous, ive, al)
    features["ly_adverb_count"] = len(re.findall(r"\b\w+ly\b", text_str.lower()))
    features["ing_verb_count"] = len(re.findall(r"\b\w+ing\b", text_str.lower()))
    features["ed_past_count"] = len(re.findall(r"\b\w+ed\b", text_str.lower()))
    features["ous_adjective_count"] = len(re.findall(r"\b\w+ous\b", text_str.lower()))
    features["ive_adjective_count"] = len(re.findall(r"\b\w+ive\b", text_str.lower()))
    features["tion_noun_count"] = len(re.findall(r"\b\w+tion\b", text_str.lower()))

    # Sentence start patterns (author-specific)
    sentence_starts = []
    for s in sentences:
        if s:
            first_word = s.split()[0] if s.split() else ""
            sentence_starts.append(first_word.lower())

    # Common sentence starters
    common_starts = [
        "the",
        "i",
        "it",
        "he",
        "she",
        "they",
        "we",
        "this",
        "that",
        "there",
        "and",
        "but",
        "for",
        "so",
        "then",
        "yet",
        "thus",
        "hence",
        "now",
        "oh",
    ]
    features["common_start_ratio"] = sum(
        1 for s in sentence_starts if s in common_starts
    ) / max(len(sentence_starts), 1)

    # (Removed: author-specific marker words that cause data leakage)

    # Readability proxy (Flesch-Kincaid like)
    features["syllable_count"] = sum(
        max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words
    )
    features["syllables_per_word"] = features["syllable_count"] / max(len(words), 1)

    return features

# Extract stylometric features
stylo_features_train = pd.DataFrame(
    [extract_stylometric_features(t) for t in train_df["text_clean"]]
)
stylo_features_test = pd.DataFrame(
    [extract_stylometric_features(t) for t in test_df["text_clean"]]
)

# =====================
# ADDITIONAL TEXT QUALITY FEATURES
# =====================

print("Creating additional features...")

# Train additional features
train_df["text_length"] = train_df["text_clean"].str.len()
train_df["word_count"] = train_df["text_clean"].str.split().str.len()
train_df["avg_word_len"] = train_df["text_length"] / train_df["word_count"].clip(
    lower=1
)

# Test additional features
test_df["text_length"] = test_df["text_clean"].str.len()
test_df["word_count"] = test_df["text_clean"].str.split().str.len()
test_df["avg_word_len"] = test_df["text_length"] / test_df["word_count"].clip(lower=1)

# =====================
# COMBINE ALL FEATURES
# =====================

print("Combining all features...")

# Basic features
train_basic_final = train_features.reset_index(drop=True)
test_basic_final = test_features.reset_index(drop=True)

# Stylometric features
stylo_train_final = stylo_features_train.reset_index(drop=True)
stylo_test_final = stylo_features_test.reset_index(drop=True)

# Text quality features
train_quality = train_df[["text_length", "word_count", "avg_word_len"]].reset_index(
    drop=True
)
test_quality = test_df[["text_length", "word_count", "avg_word_len"]].reset_index(
    drop=True
)

# Combine engineered features (excluding n-grams for model input, keeping them separate)
train_engineered = pd.concat(
    [train_basic_final, stylo_train_final, train_quality], axis=1
)

test_engineered = pd.concat([test_basic_final, stylo_test_final, test_quality], axis=1)

# Handle any NaN or infinite values
train_engineered = train_engineered.replace([np.inf, -np.inf], np.nan).fillna(0)
test_engineered = test_engineered.replace([np.inf, -np.inf], np.nan).fillna(0)

# =====================
# HANDLE MISSING VALUES AND FINAL PROCESSING
# =====================

print(f"Train engineered features shape: {train_engineered.shape}")
print(f"Test engineered features shape: {test_engineered.shape}")

# =====================
# ENCODE TARGET
# =====================

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
author_mapping = dict(
    zip(label_encoder.classes_, [int(x) for x in label_encoder.transform(label_encoder.classes_)])
)
print(f"Author encoding: {author_mapping}")

# =====================
# CREATE TRAIN/VALIDATION/TEST SPLITS (AVOIDING INDEX_BUG)
# =====================

print("Creating stratified validation split...")

# Use StratifiedKFold for consistent splitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get train and validation indices (use fold 0 for validation)
train_idx, val_idx = next(skf.split(train_df, train_df["author_encoded"]))

# CORRECT split method (avoiding INDEX_BUG) - Option C: No reset_index, use .iloc
train_split = train_df.iloc[train_idx]
val_split = train_df.iloc[val_idx]

# Get engineered features for train and val
X_train_engineered = train_engineered.iloc[train_idx].reset_index(drop=True)
X_val_engineered = train_engineered.iloc[val_idx].reset_index(drop=True)

# Get n-gram features
X_train_char = char_feat_df_train.iloc[train_idx].reset_index(drop=True)
X_val_char = char_feat_df_train.iloc[val_idx].reset_index(drop=True)

X_train_word = word_feat_df_train.iloc[train_idx].reset_index(drop=True)
X_val_word = word_feat_df_train.iloc[val_idx].reset_index(drop=True)

X_train_tfidf = tfidf_word_df_train.iloc[train_idx].reset_index(drop=True)
X_val_tfidf = tfidf_word_df_train.iloc[val_idx].reset_index(drop=True)

# Get text and labels
y_train = train_split["author_encoded"].values
y_val = val_split["author_encoded"].values

train_texts = train_split["text_clean"].values
val_texts = val_split["text_clean"].values

# Test data remains complete
test_texts = test_df["text_clean"].values

# =====================
# VERIFY SPLITS
# =====================

# Verify no overlap
assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val indices overlap!"
print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
print(f"Train dist: {np.bincount(y_train)}")
print(f"Val dist: {np.bincount(y_val)}")

# =====================
# SAVE PROCESSED DATA
# =====================

print("Saving processed data...")

os.makedirs("./working", exist_ok=True)

# Save the original cleaned dataframes with splits
train_split.to_pickle("./working/train_split.pkl")
val_split.to_pickle("./working/val_split.pkl")
test_df.to_pickle("./working/test_df.pkl")

# Save engineered features
train_engineered.to_pickle("./working/train_engineered.pkl")
val_engineered = X_val_engineered.copy()
val_engineered.to_pickle("./working/val_engineered.pkl")
test_engineered.to_pickle("./working/test_engineered.pkl")

# Save n-gram features
X_train_char.to_pickle("./working/train_char_ngrams.pkl")
X_val_char.to_pickle("./working/val_char_ngrams.pkl")
char_feat_df_test.to_pickle("./working/test_char_ngrams.pkl")

X_train_word.to_pickle("./working/train_word_ngrams.pkl")
X_val_word.to_pickle("./working/val_word_ngrams.pkl")
word_feat_df_test.to_pickle("./working/test_word_ngrams.pkl")

X_train_tfidf.to_pickle("./working/train_tfidf.pkl")
X_val_tfidf.to_pickle("./working/val_tfidf.pkl")
tfidf_word_df_test.to_pickle("./working/test_tfidf.pkl")

# Save text arrays
np.save("./working/train_texts.npy", train_texts)
np.save("./working/val_texts.npy", val_texts)
np.save("./working/test_texts.npy", test_texts)

# Save labels
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)

# Save encoding information
with open("./working/author_encoding.json", "w") as f:
    json.dump(author_mapping, f)

# Save split indices for later use
np.save("./working/train_idx.npy", train_idx)
np.save("./working/val_idx.npy", val_idx)

# Save label encoder for later use
with open("./working/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save vectorizers for later use in inference
with open("./working/char_vectorizer.pkl", "wb") as f:
    pickle.dump(char_vectorizer, f)
with open("./working/word_vectorizer.pkl", "wb") as f:
    pickle.dump(word_vectorizer, f)
with open("./working/tfidf_word_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_word, f)

print("\n=== DATA PROCESSING COMPLETE ===")
print(f"Train texts shape: {train_texts.shape}")
print(f"Val texts shape: {val_texts.shape}")
print(f"Test texts shape: {test_texts.shape}")

# Clean up memory
gc.collect()

# =====================
# MODEL DESIGN: MultiPoolingDeBERTa
# =====================

class MultiPoolingDeBERTa(nn.Module):
    """
    DeBERTa-v3-large with multi-pooling strategy for author classification.
    Combines CLS pooling, mean pooling, and attention-weighted pooling
    to capture both sentence-level and token-level stylometric features.
    """

    def __init__(
        self, model_name="microsoft/deberta-v3-large", num_labels=3, dropout=0.15
    ):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size
        self.pool_norm = nn.LayerNorm(self.hidden_size)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Dropout(dropout * 0.8),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.hidden_size // 4, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.attention_weights:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        # CLS pooling
        cls_pool = last_hidden[:, 0, :]

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        mask_sum = mask_expanded.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1e-9)
        mean_pool = (last_hidden * mask_expanded).sum(dim=1) / mask_sum

        # Attention-weighted pooling
        attn_scores = self.attention_weights(last_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_pool = (last_hidden * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Normalize pooling outputs
        cls_pool = self.pool_norm(cls_pool)
        mean_pool = self.pool_norm(mean_pool)
        attn_pool = self.pool_norm(attn_pool)

        # Concatenate all pooling strategies
        combined_pool = torch.cat([cls_pool, mean_pool, attn_pool], dim=1)

        # Classification head
        logits = self.classifier(combined_pool)
        return logits

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.05, weight=None, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# =====================
# DATASET CLASS
# =====================

class AuthorDataset(Dataset):
    """Dataset for author classification with DeBERTa"""

    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# =====================
# TRAINING SETUP
# =====================

print("\n" + "=" * 60)
print("Training DeBERTa-v3-large with Multi-Pooling")
print("=" * 60)

# Configuration
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.1

# Initialize tokenizer
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded: {model_name}")

# Compute class weights for balanced training
class_counts = np.bincount(y_train)
class_weights = torch.tensor(
    [max(class_counts) / count for count in class_counts], dtype=torch.float32
).to(device)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights.cpu().numpy()}")

# Initialize model
model = MultiPoolingDeBERTa(model_name=model_name, num_labels=3, dropout=0.15)
model = model.to(device)

# Print model statistics
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Create datasets
train_dataset = AuthorDataset(train_texts, y_train, tokenizer, MAX_LENGTH)
val_dataset = AuthorDataset(val_texts, y_val, tokenizer, MAX_LENGTH)
test_dataset = AuthorDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
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

# Loss function with label smoothing
criterion = LabelSmoothingLoss(
    num_classes=3, smoothing=LABEL_SMOOTHING, weight=class_weights
)

# Optimizer with layer-wise decay
def get_optimizer_with_layerwise_decay(
    model, learning_rate=2e-5, weight_decay=0.01, layerwise_decay=0.95
):
    named_params = list(model.named_parameters())
    classifier_params = []
    attention_params = []
    pool_norm_params = []
    backbone_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        elif "attention_weights" in name:
            attention_params.append(param)
        elif "pool_norm" in name:
            pool_norm_params.append(param)
        else:
            backbone_params.append((name, param))

    optimizer_grouped_parameters = [
        {
            "params": classifier_params,
            "lr": learning_rate,
            "weight_decay": weight_decay * 0.5,
        },
        {
            "params": attention_params,
            "lr": learning_rate,
            "weight_decay": weight_decay * 0.5,
        },
        {
            "params": pool_norm_params,
            "lr": learning_rate * 0.8,
            "weight_decay": weight_decay,
        },
    ]

    layer_params = {}
    for name, param in backbone_params:
        if "encoder.layer." in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num not in layer_params:
                layer_params[layer_num] = []
            layer_params[layer_num].append(param)
        elif "embeddings" in name:
            optimizer_grouped_parameters.append(
                {
                    "params": param,
                    "lr": learning_rate * (layerwise_decay**12),
                    "weight_decay": weight_decay,
                }
            )
        else:
            optimizer_grouped_parameters.append(
                {
                    "params": param,
                    "lr": learning_rate * (layerwise_decay**12),
                    "weight_decay": weight_decay,
                }
            )

    num_layers = len(layer_params)
    for layer_num in sorted(layer_params.keys()):
        decay_factor = layerwise_decay ** (num_layers - 1 - layer_num)
        optimizer_grouped_parameters.append(
            {
                "params": layer_params[layer_num],
                "lr": learning_rate * decay_factor,
                "weight_decay": weight_decay,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    return optimizer

optimizer = get_optimizer_with_layerwise_decay(model, LEARNING_RATE, WEIGHT_DECAY)

# Learning rate scheduler
total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)

def get_scheduler_with_warmup(optimizer, num_training_steps, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_scheduler_with_warmup(optimizer, total_steps, warmup_steps)

# Mixed precision
scaler = GradScaler()

# Early stopping
best_val_loss = float("inf")
best_val_logloss = float("inf")
patience = 3
patience_counter = 0
best_model_state = None

# =====================
# TRAINING LOOP
# =====================
print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Total steps: {total_steps}")
print(
    f"Warmup steps: {warmup_steps}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}"
)

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                probs = F.softmax(logits, dim=-1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)

    # Compute validation log loss
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)

    # Clip probabilities to avoid extreme values in log loss
    val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
    # Normalize probabilities to sum to 1
    val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)

    val_logloss = log_loss(val_labels, val_preds)

    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {current_lr:.2e}"
    )

    # Early stopping check
    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ✓ New best model! Val LogLoss: {best_val_logloss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping triggered after {epoch+1} epochs")
            break

# =====================
# LOAD BEST MODEL AND EVALUATE
# =====================
print("\n" + "=" * 60)
print("Loading Best Model for Final Evaluation")
print("=" * 60)

model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

# Final validation evaluation
all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast():
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)

        all_val_preds.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_preds_final = np.concatenate(all_val_preds, axis=0)
val_labels_final = np.concatenate(all_val_labels, axis=0)

# Clip and normalize
val_preds_final = np.clip(val_preds_final, 1e-15, 1 - 1e-15)
val_preds_final = val_preds_final / val_preds_final.sum(axis=1, keepdims=True)

final_val_logloss = log_loss(val_labels_final, val_preds_final)
print(f"Final Validation LogLoss: {final_val_logloss:.6f}")

# =====================
# TEST INFERENCE
# =====================
print("\n" + "=" * 60)
print("Performing Test Inference")
print("=" * 60)

all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with autocast():
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)

        all_test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)

# Clip and normalize
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

print(f"Test predictions shape: {test_preds.shape}")

# =====================
# CREATE SUBMISSION FILE
# =====================
print("\n" + "=" * 60)
print("Creating Submission File")
print("=" * 60)

# Create submission directory if it doesn't exist
os.makedirs("./submission", exist_ok=True)

# Load test IDs
test_ids = test_df["id"].values

# Create submission dataframe
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

# Save submission file
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Submission columns: {submission_df.columns.tolist()}")

# Verify submission format
expected_columns = ["id", "EAP", "HPL", "MWS"]
assert (
    list(submission_df.columns) == expected_columns
), f"Column mismatch: {list(submission_df.columns)}"
assert len(submission_df) == len(
    test_df
), f"Row count mismatch: {len(submission_df)} vs {len(test_df)}"
print("✓ Submission format verified!")

# =====================
# CLEANUP
# =====================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")

print("\n" + "=" * 60)
print("Training and Evaluation Complete")
print("=" * 60)

# Final validation score (required for parser)
print(f"Final Validation Score: {final_val_logloss}")