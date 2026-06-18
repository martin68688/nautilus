import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
import string

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Create a copy to work with
train_data = train_df.copy()
test_data = test_df.copy()

# Combine for consistent feature engineering (but split before scaling!)
all_text = pd.concat([train_data["text"], test_data["text"]], ignore_index=True)
train_len = len(train_data)


# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    """Basic text cleaning for horror fiction text"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Fix spacing around punctuation
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Fix common OCR artifacts
    text = text.replace('"', "'")
    text = text.replace('"', "'")
    text = text.replace("`", "'")
    return text.strip()


train_data["text_clean"] = train_data["text"].apply(clean_text)
test_data["text_clean"] = test_data["text"].apply(clean_text)


# ============================================================
# 3. STYLOMETRIC FEATURES ENGINEERING
# ============================================================
def extract_stylometric_features(text):
    """Extract hand-crafted stylometric features from text"""
    if not isinstance(text, str) or len(text) == 0:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_len": 0,
            "avg_sent_len": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "comma_count": 0,
            "colon_count": 0,
            "semicolon_count": 0,
            "dash_count": 0,
            "quote_count": 0,
            "paren_count": 0,
            "capital_letters_pct": 0,
            "stopword_pct": 0,
            "punctuation_pct": 0,
            "unique_word_ratio": 0,
            "long_word_ratio": 0,
            "first_person_pronoun_count": 0,
            "third_person_pronoun_count": 0,
            "past_tense_count": 0,
            "present_tense_count": 0,
            "adverb_count": 0,
            "adjective_count": 0,
            "conjunction_count": 0,
            "preposition_count": 0,
            "article_count": 0,
            "num_digits": 0,
            "num_all_caps_words": 0,
            "num_ellipsis": 0,
        }

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Basic counts
    sent_count = len(re.findall(r"[.!?]+", text)) + 1  # plus one for last sentence
    avg_word_len = char_count / max(word_count, 1)
    avg_sent_len = word_count / max(sent_count, 1)

    # Punctuation counts
    excl_count = text.count("!")
    quest_count = text.count("?")
    comma_count = text.count(",")
    colon_count = text.count(":")
    semi_count = text.count(";")
    dash_count = text.count("-") + text.count("—")
    quote_count = text.count("'") // 2  # approximate pairs
    paren_count = text.count("(") + text.count(")")

    # Capital letters percentage
    capital_letters = sum(1 for c in text if c.isupper())
    capital_pct = capital_letters / max(char_count, 1)

    # Stop words
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
        "this",
        "that",
        "these",
        "those",
    }
    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    stopword_pct = stopword_count / max(word_count, 1)

    # Punctuation percentage
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_pct = punct_count / max(char_count, 1)

    # Vocabulary richness
    unique_words = len(set(w.lower() for w in words))
    unique_ratio = unique_words / max(word_count, 1)

    # Long words (>6 chars)
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / max(word_count, 1)

    # Pronoun counts (simple heuristic)
    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
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
    }
    first_person_count = sum(1 for w in words if w.lower() in first_person)
    third_person_count = sum(1 for w in words if w.lower() in third_person)

    # Tense indicators (simplified)
    past_tense_indicators = {
        "was",
        "were",
        "had",
        "did",
        "said",
        "would",
        "could",
        "should",
        "went",
        "came",
        "saw",
        "knew",
        "thought",
        "felt",
        "became",
    }
    present_tense_indicators = {
        "is",
        "are",
        "has",
        "does",
        "says",
        "will",
        "can",
        "shall",
        "go",
        "come",
        "see",
        "know",
        "think",
        "feel",
        "become",
    }
    past_count = sum(1 for w in words if w.lower() in past_tense_indicators)
    present_count = sum(1 for w in words if w.lower() in present_tense_indicators)

    # POS-like patterns (using keyword lists instead of actual POS tagger for speed)
    adverbs = {
        "very",
        "quite",
        "too",
        "so",
        "just",
        "then",
        "now",
        "here",
        "there",
        "always",
        "never",
        "often",
        "sometimes",
        "suddenly",
        "quickly",
        "slowly",
        "carefully",
        "silently",
        "strangely",
        "horribly",
        "terribly",
        "fearfully",
    }
    adjectives = {
        "great",
        "small",
        "large",
        "long",
        "short",
        "high",
        "low",
        "old",
        "new",
        "good",
        "bad",
        "beautiful",
        "ugly",
        "dark",
        "light",
        "cold",
        "hot",
        "strange",
        "terrible",
        "horrible",
        "dreadful",
        "fearful",
        "awful",
        "mysterious",
        "ancient",
        "eternal",
        "vast",
        "deep",
        "shadowy",
    }
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
        "though",
        "while",
        "since",
        "unless",
        "until",
        "after",
        "before",
    }
    prepositions = {
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "into",
        "upon",
        "through",
        "between",
        "among",
        "around",
        "beyond",
        "under",
        "over",
        "above",
        "below",
        "behind",
        "before",
        "after",
        "without",
    }
    articles = {"a", "an", "the"}

    adverb_count = sum(1 for w in words if w.lower() in adverbs)
    adjective_count = sum(1 for w in words if w.lower() in adjectives)
    conjunction_count = sum(1 for w in words if w.lower() in conjunctions)
    preposition_count = sum(1 for w in words if w.lower() in prepositions)
    article_count = sum(1 for w in words if w.lower() in articles)

    # Digit count
    digit_count = sum(1 for c in text if c.isdigit())

    # All-caps words (not first word of sentence)
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)

    # Ellipsis count
    ellipsis_count = text.count("...") + text.count("…") + text.count(". . .")

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sent_count": sent_count,
        "avg_word_len": avg_word_len,
        "avg_sent_len": avg_sent_len,
        "excl_count": excl_count,
        "quest_count": quest_count,
        "comma_count": comma_count,
        "colon_count": colon_count,
        "semi_count": semi_count,
        "dash_count": dash_count,
        "quote_count": quote_count,
        "paren_count": paren_count,
        "capital_pct": capital_pct,
        "stopword_pct": stopword_pct,
        "punct_pct": punct_pct,
        "unique_ratio": unique_ratio,
        "long_word_ratio": long_word_ratio,
        "first_person_count": first_person_count,
        "third_person_count": third_person_count,
        "past_count": past_count,
        "present_count": present_count,
        "adverb_count": adverb_count,
        "adjective_count": adjective_count,
        "conjunction_count": conjunction_count,
        "preposition_count": preposition_count,
        "article_count": article_count,
        "digit_count": digit_count,
        "all_caps_count": all_caps,
        "ellipsis_count": ellipsis_count,
    }


# Extract stylometric features for all text
stylometric_train = pd.DataFrame(
    [extract_stylometric_features(t) for t in train_data["text_clean"]]
)
stylometric_test = pd.DataFrame(
    [extract_stylometric_features(t) for t in test_data["text_clean"]]
)

print(f"Stylometric features extracted: {stylometric_train.shape[1]} features")

# ============================================================
# 4. HANDLE MISSING / INFINITE VALUES IN STYLOMETRIC FEATURES
# ============================================================
for df in [stylometric_train, stylometric_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)


# ============================================================
# 5. SCALE STYLOMETRIC FEATURES (FIT ON TRAIN ONLY!)
# ============================================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
stylometric_scaled_train = scaler.fit_transform(stylometric_train)
stylometric_scaled_test = scaler.transform(stylometric_test)

print(
    f"Stylometric features scaled: train {stylometric_scaled_train.shape}, test {stylometric_scaled_test.shape}"
)

# ============================================================
# 6. TF-IDF FEATURE EXTRACTION
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer

# Word-level TF-IDF (unigrams and bigrams)
print("Extracting word TF-IDF features...")
word_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    sublinear_tf=True,
    max_df=0.95,
    min_df=3,
    stop_words="english",
)

word_tfidf_train = word_vectorizer.fit_transform(train_data["text_clean"])
word_tfidf_test = word_vectorizer.transform(test_data["text_clean"])
print(f"Word TF-IDF: train {word_tfidf_train.shape}, test {word_tfidf_test.shape}")

# Character-level TF-IDF (3-5 grams - captures Lovecraft's unique word constructions)
print("Extracting character TF-IDF features...")
char_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(3, 5),
    lowercase=True,
    strip_accents="unicode",
    analyzer="char",
    sublinear_tf=True,
    max_df=0.90,
    min_df=3,
)

char_tfidf_train = char_vectorizer.fit_transform(train_data["text_clean"])
char_tfidf_test = char_vectorizer.transform(test_data["text_clean"])
print(f"Char TF-IDF: train {char_tfidf_train.shape}, test {char_tfidf_test.shape}")

# ============================================================
# 7. ADDITIONAL LINGUISTIC FEATURES
# ============================================================
import string
from collections import Counter

function_words = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "that",
    "it",
    "was",
    "with",
    "for",
    "but",
    "not",
    "on",
    "his",
    "had",
    "their",
    "he",
    "at",
    "by",
    "from",
    "she",
    "or",
    "which",
    "as",
    "we",
    "an",
    "all",
    "been",
    "were",
    "have",
    "are",
    "its",
    "has",
    "her",
    "they",
    "is",
    "can",
    "would",
    "so",
    "what",
    "there",
    "if",
    "no",
    "upon",
    "my",
    "one",
    "could",
    "me",
    "do",
    "your",
    "may",
    "some",
    "very",
    "more",
    "most",
    "such",
    "into",
    "shall",
    "than",
    "about",
    "then",
    "these",
    "every",
    "their",
    "were",
    "after",
    "before",
    "through",
    "between",
    "under",
    "over",
    "much",
    "many",
    "still",
    "yet",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "though",
    "until",
    "while",
    "because",
    "although",
    "unless",
    "since",
}


def count_function_words(text):
    words = text.lower().split()
    counts = {fw: 0 for fw in function_words}
    for w in words:
        if w in counts:
            counts[w] += 1
    return counts


func_word_train = pd.DataFrame(
    [count_function_words(t) for t in train_data["text_clean"]]
)
func_word_test = pd.DataFrame(
    [count_function_words(t) for t in test_data["text_clean"]]
)

print(f"Function word features extracted: {func_word_train.shape[1]}")

# ============================================================
# 8. NORMALIZE FUNCTION WORD COUNTS BY TEXT LENGTH
# ============================================================
# Add text length as denominator
train_text_lens = train_data["text_clean"].str.len().values.astype(float)
test_text_lens = test_data["text_clean"].str.len().values.astype(float)

func_word_norm_train = func_word_train.div(train_text_lens, axis=0).fillna(0).values
func_word_norm_test = func_word_test.div(test_text_lens, axis=0).fillna(0).values

print(f"Normalized function word features: train {func_word_norm_train.shape}")

# ============================================================
# 9. COMBINE ALL FEATURES
# ============================================================
from scipy.sparse import hstack, csr_matrix

# Convert stylometric to sparse
stylometric_sparse_train = csr_matrix(stylometric_scaled_train)
stylometric_sparse_test = csr_matrix(stylometric_scaled_test)

# Convert function words to sparse
func_word_sparse_train = csr_matrix(func_word_norm_train)
func_word_sparse_test = csr_matrix(func_word_norm_test)

# Combine all sparse matrices
X_train_combined = hstack(
    [
        word_tfidf_train,
        char_tfidf_train,
        stylometric_sparse_train,
        func_word_sparse_train,
    ]
).tocsr()

X_test_combined = hstack(
    [
        word_tfidf_test,
        char_tfidf_test,
        stylometric_sparse_test,
        func_word_sparse_test,
    ]
).tocsr()

print(
    f"Combined feature matrix: train {X_train_combined.shape}, test {X_test_combined.shape}"
)

# ============================================================
# 10. CREATE TRAIN/VALIDATION SPLIT (StratifiedKFold)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_data, train_data["author"]))

# Create label encoder
le = LabelEncoder()
y_train = le.fit_transform(train_data["author"])

print(f"\nClass distribution:")
print(pd.Series(le.classes_))
print(f"Train indices: {len(train_idx)} samples")
print(f"Val indices: {len(val_idx)} samples")

# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================
os.makedirs("./working", exist_ok=True)

# Save feature matrices
from scipy.sparse import save_npz

save_npz("./working/X_train.npz", X_train_combined)
save_npz("./working/X_test.npz", X_test_combined)

# Save indices and labels
np.save("./working/train_idx.npy", train_idx)
np.save("./working/val_idx.npy", val_idx)
np.save("./working/y_train.npy", y_train)
np.save("./working/y_train_full.npy", y_train)  # full training set labels
np.save("./working/train_texts.npy", np.array(train_data["text"]))
np.save("./working/test_texts.npy", np.array(test_data["text"]))
np.save("./working/test_ids.npy", np.array(test_data["id"]))

# Save label encoder classes
np.save("./working/label_classes.npy", le.classes_)

# Save feature names
feature_names = (
    list(word_vectorizer.get_feature_names_out())
    + list(char_vectorizer.get_feature_names_out())
    + [f"stylo_{col}" for col in stylometric_train.columns]
    + list(function_words)
)
np.save("./working/feature_names.npy", np.array(feature_names))

# Also save original DataFrames with cleaned text for potential neural network usage
train_data.to_pickle("./working/train_processed.pkl")
test_data.to_pickle("./working/test_processed.pkl")
stylometric_train.to_pickle("./working/stylometric_train.pkl")
stylometric_test.to_pickle("./working/stylometric_test.pkl")

print("\n=== Processing Complete ===")
print(f"Final training feature shape: {X_train_combined.shape}")
print(f"Final test feature shape: {X_test_combined.shape}")
print(f"Files saved to ./working/")
print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
print(f"Test size: {X_test_combined.shape[0]}")

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
        self.attention_pool = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_authors),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        # Learnable weighted pooling over sequence dimension
        attention_weights = self.attention_pool(hidden_states).squeeze(-1)
        attention_weights = attention_weights.masked_fill(
            attention_mask == 0, float("-inf")
        )
        attention_probs = torch.softmax(attention_weights, dim=-1)
        pooled = (hidden_states * attention_probs.unsqueeze(-1)).sum(dim=1)
        logits = self.head(pooled)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Collect backbone unfrozen params (last 8 layers)
backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-12:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

# Collect head params (now Sequential with multiple layers)
head_params = list(model.head.parameters())
# Also collect attention_pool params
attention_pool_params = list(model.attention_pool.parameters())

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
        {"params": attention_pool_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Ensure optimizer param groups are correctly ordered
# Group 0: backbone unfrozen layers (lr=2e-5)
# Group 1: head (lr=5e-5)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_unfrozen_params):,}")
print(f"Head params: {sum(p.numel() for p in head_params) + sum(p.numel() for p in attention_pool_params):,}")

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

# Use previously computed indices for train/validation split
train_texts_final = train_texts_orig[train_idx]
train_labels_final = train_labels_orig[train_idx]
val_texts_final = train_texts_orig[val_idx]
val_labels_final = train_labels_orig[val_idx]

batch_size = 16
max_length = 1024

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
os.makedirs("./submission", exist_ok=True)

import math

# Linear warmup, then cosine decay to min_lr
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

# Initialize scheduler with linear warmup and cosine decay
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    accumulation_steps = 2
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        scaler_grad.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()
            optimizer.zero_grad()

        # Apply scheduler step per optimizer step (not per batch)
        if (batch_idx + 1) % accumulation_steps == 0:
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

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")
