import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
import lightgbm as lgb
import joblib
import os
import gc
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# ============================================
# 1. Data Loading
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Clean text
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# ============================================
# Feature Engineering Functions
# ============================================

# Stopwords set
stopwords_set = set(
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
        "such",
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
        "can",
        "will",
        "just",
        "should",
        "now",
    ]
)


def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2:
        count += 1
    if count == 0:
        count = 1
    return count


# 1. Basic text statistics
def basic_stats(text):
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features = {}
    features["num_words"] = len(words)
    features["num_chars"] = len(text)
    features["num_sentences"] = max(len(sentences), 1)
    features["avg_word_len"] = np.mean([len(w) for w in words]) if words else 0
    features["avg_sentence_len"] = features["num_words"] / features["num_sentences"]
    features["num_unique_words"] = len(set(words))
    features["type_token_ratio"] = features["num_unique_words"] / max(
        features["num_words"], 1
    )
    features["num_stopwords"] = sum(1 for w in words if w in stopwords_set)
    features["stopword_ratio"] = features["num_stopwords"] / max(
        features["num_words"], 1
    )
    return features


# 2. Punctuation features
def punctuation_features(text):
    features = {}
    for punct in string.punctuation:
        features[f"count_{punct}"] = text.count(punct)
    features["punct_ratio"] = sum(text.count(p) for p in string.punctuation) / max(
        len(text), 1
    )
    features["exclamation_ratio"] = text.count("!") / max(len(text), 1)
    features["question_ratio"] = text.count("?") / max(len(text), 1)
    features["dash_ratio"] = text.count("-") / max(len(text), 1)
    features["quote_ratio"] = text.count('"') / max(len(text), 1)
    return features


# 3. Character n-grams
def char_ngram_features(texts, ngram_range=(2, 5), max_features=300):
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=False,
    )
    ngram_matrix = vectorizer.fit_transform(texts)
    return ngram_matrix, vectorizer


# 4. Word n-gram TF-IDF
def word_ngram_tfidf(texts, ngram_range=(1, 3), max_features=500):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


# 5. Readability metrics
def readability_features(text):
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_words = len(words)
    num_sentences = max(len(sentences), 1)
    num_syllables = sum(count_syllables(w) for w in words)
    features = {}
    if num_words > 0 and num_sentences > 0:
        features["flesch_kincaid"] = (
            0.39 * (num_words / num_sentences)
            + 11.8 * (num_syllables / num_words)
            - 15.59
        )
    else:
        features["flesch_kincaid"] = 0
    num_chars = len(text.replace(" ", ""))
    if num_words > 0 and num_sentences > 0:
        features["ari"] = (
            4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        )
    else:
        features["ari"] = 0
    return features


# 6. POS pattern features
def pos_pattern_features(text):
    features = {}
    words = text.lower().split()
    features["article_count"] = sum(1 for w in words if w in ["a", "an", "the"])
    features["article_ratio"] = features["article_count"] / max(len(words), 1)
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
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
    }
    features["prep_count"] = sum(1 for w in words if w in prepositions)
    features["prep_ratio"] = features["prep_count"] / max(len(words), 1)
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
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "themselves",
    }
    features["pronoun_count"] = sum(1 for w in words if w in pronouns)
    features["pronoun_ratio"] = features["pronoun_count"] / max(len(words), 1)
    conjunctions = {"and", "but", "or", "nor", "yet", "so", "for"}
    features["conj_count"] = sum(1 for w in words if w in conjunctions)
    features["conj_ratio"] = features["conj_count"] / max(len(words), 1)
    past_tense = {
        "was",
        "were",
        "had",
        "did",
        "said",
        "went",
        "came",
        "made",
        "took",
        "saw",
        "knew",
        "thought",
        "felt",
        "became",
        "began",
        "brought",
        "left",
        "told",
        "found",
        "gave",
        "held",
        "kept",
        "let",
        "put",
        "set",
        "stood",
        "turned",
        "ran",
        "sat",
        "lay",
        "rose",
        "fell",
        "struck",
        "drew",
        "broke",
        "spoke",
        "wrote",
        "drove",
        "rode",
    }
    features["past_tense_count"] = sum(1 for w in words if w in past_tense)
    features["past_tense_ratio"] = features["past_tense_count"] / max(len(words), 1)
    return features


# 7. Sentiment features
def sentiment_features(text):
    positive_words = {
        "love",
        "joy",
        "happy",
        "beautiful",
        "wonderful",
        "great",
        "good",
        "bright",
        "hope",
        "peace",
        "gentle",
        "kind",
        "sweet",
        "dear",
        "soft",
        "warm",
        "fair",
        "true",
        "noble",
        "pure",
        "pleasant",
        "bliss",
        "delight",
        "ecstasy",
        "rapture",
        "enchant",
        "charm",
        "grace",
        "mercy",
        "bless",
        "glory",
        "splendor",
        "radiant",
        "serene",
        "tranquil",
    }
    negative_words = {
        "dark",
        "fear",
        "terror",
        "horror",
        "dread",
        "gloom",
        "death",
        "pain",
        "sorrow",
        "agony",
        "anguish",
        "misery",
        "suffering",
        "grief",
        "despair",
        "madness",
        "doom",
        "curse",
        "evil",
        "shadow",
        "phantom",
        "specter",
        "ghost",
        "demon",
        "devil",
        "hell",
        "grave",
        "coffin",
        "tomb",
        "corpse",
        "blood",
        "wound",
        "scream",
        "shriek",
        "moan",
        "groan",
        "weep",
        "lament",
        "mourn",
        "desolate",
        "hideous",
        "monstrous",
        "vile",
        "foul",
        "loathsome",
        "abhorrent",
        "dreadful",
        "awful",
        "terrible",
        "horrid",
    }
    words = text.lower().split()
    features = {
        "positive_word_count": sum(1 for w in words if w in positive_words),
        "negative_word_count": sum(1 for w in words if w in negative_words),
        "sentiment_ratio": (
            sum(1 for w in words if w in positive_words)
            - sum(1 for w in words if w in negative_words)
        )
        / max(len(words), 1),
        "emotion_intensity": (
            sum(1 for w in words if w in positive_words)
            + sum(1 for w in words if w in negative_words)
        )
        / max(len(words), 1),
    }
    return features


# 8. Author vocabulary features
def author_vocab_features(text):
    words = text.lower().split()
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "squamous",
        "rugose",
        "noisome",
        "ichor",
        "blasphemous",
        "cacophony",
        "antediluvian",
        "primordial",
        "cosmic",
        "non-euclidean",
        "yog-sothoth",
        "cthulhu",
        "r'lyeh",
        "necronomicon",
        "azathoth",
        "nyarlathotep",
        "shoggoth",
        "mi-go",
        "yuggoth",
        "kadath",
    }
    poe_words = {
        "nevermore",
        "chamber",
        "dreary",
        "bleak",
        "pallid",
        "ghastly",
        "sepulchre",
        "tintinnabulation",
        "hypnagogic",
        "arabesque",
        "grotesque",
        "usher",
        "laconically",
        "perverseness",
        "imp",
    }
    shelley_words = {
        "monster",
        "creature",
        "frankenstein",
        "geneva",
        "ingolstadt",
        "wretch",
        "demon",
        "fiend",
        "sublime",
        "magnificent",
        "alpine",
        "glacier",
        "oratory",
        "eloquence",
    }
    features = {
        "lovecraft_vocab_score": sum(1 for w in words if w in lovecraft_words)
        / max(len(words), 1),
        "poe_vocab_score": sum(1 for w in words if w in poe_words) / max(len(words), 1),
        "shelley_vocab_score": sum(1 for w in words if w in shelley_words)
        / max(len(words), 1),
        "first_person_singular": sum(
            1 for w in words if w in {"i", "me", "my", "mine", "myself"}
        )
        / max(len(words), 1),
        "first_person_plural": sum(
            1 for w in words if w in {"we", "us", "our", "ours", "ourselves"}
        )
        / max(len(words), 1),
        "third_person": sum(
            1
            for w in words
            if w
            in {
                "he",
                "she",
                "it",
                "they",
                "him",
                "her",
                "them",
                "his",
                "hers",
                "its",
                "their",
            }
        )
        / max(len(words), 1),
    }
    return features


# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])

# Create stratified split first
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["author_encoded"])
):
    if fold == 0:
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        train_fold_indices = train_idx
        val_fold_indices = val_idx
        break

# ============================================
# Apply Feature Engineering on train_fold only, then transform val/test
# ============================================
# Compute engineered features on train fold only
basic_train_fold = train_df.iloc[train_fold_indices]["clean_text"].apply(basic_stats)
basic_test = test_df["clean_text"].apply(basic_stats)
basic_train_fold_df = pd.DataFrame(basic_train_fold.tolist())
basic_test_df = pd.DataFrame(basic_test.tolist())

# Transform val fold separately
basic_val = train_df.iloc[val_fold_indices]["clean_text"].apply(basic_stats)
basic_val_df = pd.DataFrame(basic_val.tolist())

punct_train_fold = train_df.iloc[train_fold_indices]["clean_text"].apply(punctuation_features)
punct_test = test_df["clean_text"].apply(punctuation_features)
punct_train_fold_df = pd.DataFrame(punct_train_fold.tolist())
punct_test_df = pd.DataFrame(punct_test.tolist())

punct_val = train_df.iloc[val_fold_indices]["clean_text"].apply(punctuation_features)
punct_val_df = pd.DataFrame(punct_val.tolist())

read_train_fold = train_df.iloc[train_fold_indices]["clean_text"].apply(readability_features)
read_test = test_df["clean_text"].apply(readability_features)
read_train_fold_df = pd.DataFrame(read_train_fold.tolist())
read_test_df = pd.DataFrame(read_test.tolist())

read_val = train_df.iloc[val_fold_indices]["clean_text"].apply(readability_features)
read_val_df = pd.DataFrame(read_val.tolist())

pos_train_fold = train_df.iloc[train_fold_indices]["clean_text"].apply(pos_pattern_features)
pos_test = test_df["clean_text"].apply(pos_pattern_features)
pos_train_fold_df = pd.DataFrame(pos_train_fold.tolist())
pos_test_df = pd.DataFrame(pos_test.tolist())

pos_val = train_df.iloc[val_fold_indices]["clean_text"].apply(pos_pattern_features)
pos_val_df = pd.DataFrame(pos_val.tolist())

sent_train_fold = train_df.iloc[train_fold_indices]["clean_text"].apply(sentiment_features)
sent_test = test_df["clean_text"].apply(sentiment_features)
sent_train_fold_df = pd.DataFrame(sent_train_fold.tolist())
sent_test_df = pd.DataFrame(sent_test.tolist())

sent_val = train_df.iloc[val_fold_indices]["clean_text"].apply(sentiment_features)
sent_val_df = pd.DataFrame(sent_val.tolist())

vocab_train_fold = train_df.iloc[train_fold_indices]["clean_text"].apply(author_vocab_features)
vocab_test = test_df["clean_text"].apply(author_vocab_features)
vocab_train_fold_df = pd.DataFrame(vocab_train_fold.tolist())
vocab_test_df = pd.DataFrame(vocab_test.tolist())

vocab_val = train_df.iloc[val_fold_indices]["clean_text"].apply(author_vocab_features)
vocab_val_df = pd.DataFrame(vocab_val.tolist())

# Fit n-gram vectorizers on train_fold only to avoid data leakage
char_ngram_vec = CountVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=300,
    lowercase=False,
)
char_ngram_vec.fit(train_df.iloc[train_fold_indices]["clean_text"].tolist())
char_ngram_train_fold = char_ngram_vec.transform(train_df.iloc[train_fold_indices]["clean_text"].tolist())
char_ngram_val = char_ngram_vec.transform(train_df.iloc[val_fold_indices]["clean_text"].tolist())
char_ngram_test = char_ngram_vec.transform(test_df["clean_text"].tolist())

char_ngram_train_fold_df = pd.DataFrame(
    char_ngram_train_fold.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_ngram_train_fold.shape[1])],
    index=train_fold_indices,
)
char_ngram_val_df = pd.DataFrame(
    char_ngram_val.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_ngram_val.shape[1])],
    index=val_fold_indices,
)
char_ngram_test_df = pd.DataFrame(
    char_ngram_test.toarray(),
    columns=[f"char_ngram_{i}" for i in range(char_ngram_test.shape[1])],
)

# Fit TF-IDF vectorizer on train_fold only to avoid data leakage
word_tfidf_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=500,
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
)
word_tfidf_vec.fit(train_df.iloc[train_fold_indices]["clean_text"].tolist())
word_tfidf_train_fold = word_tfidf_vec.transform(train_df.iloc[train_fold_indices]["clean_text"].tolist())
word_tfidf_val = word_tfidf_vec.transform(train_df.iloc[val_fold_indices]["clean_text"].tolist())
word_tfidf_test = word_tfidf_vec.transform(test_df["clean_text"].tolist())

word_tfidf_train_fold_df = pd.DataFrame(
    word_tfidf_train_fold.toarray(),
    columns=[f"word_tfidf_{i}" for i in range(word_tfidf_train_fold.shape[1])],
    index=train_fold_indices,
)
word_tfidf_val_df = pd.DataFrame(
    word_tfidf_val.toarray(),
    columns=[f"word_tfidf_{i}" for i in range(word_tfidf_val.shape[1])],
    index=val_fold_indices,
)
word_tfidf_test_df = pd.DataFrame(
    word_tfidf_test.toarray(),
    columns=[f"word_tfidf_{i}" for i in range(word_tfidf_test.shape[1])],
)

# Combine all features for train_fold
train_features = pd.concat(
    [
        basic_train_fold_df.reset_index(drop=True),
        punct_train_fold_df.reset_index(drop=True),
        read_train_fold_df.reset_index(drop=True),
        pos_train_fold_df.reset_index(drop=True),
        sent_train_fold_df.reset_index(drop=True),
        vocab_train_fold_df.reset_index(drop=True),
        char_ngram_train_fold_df.reset_index(drop=True),
        word_tfidf_train_fold_df.reset_index(drop=True),
    ],
    axis=1,
)

# Combine all features for val_fold
val_features = pd.concat(
    [
        basic_val_df.reset_index(drop=True),
        punct_val_df.reset_index(drop=True),
        read_val_df.reset_index(drop=True),
        pos_val_df.reset_index(drop=True),
        sent_val_df.reset_index(drop=True),
        vocab_val_df.reset_index(drop=True),
        char_ngram_val_df.reset_index(drop=True),
        word_tfidf_val_df.reset_index(drop=True),
    ],
    axis=1,
)

test_features = pd.concat(
    [
        basic_test_df.reset_index(drop=True),
        punct_test_df.reset_index(drop=True),
        read_test_df.reset_index(drop=True),
        pos_test_df.reset_index(drop=True),
        sent_test_df.reset_index(drop=True),
        vocab_test_df.reset_index(drop=True),
        char_ngram_test_df.reset_index(drop=True),
        word_tfidf_test_df.reset_index(drop=True),
    ],
    axis=1,
)

train_features = train_features.fillna(0).replace([np.inf, -np.inf], 0)
val_features = val_features.fillna(0).replace([np.inf, -np.inf], 0)
test_features = test_features.fillna(0).replace([np.inf, -np.inf], 0)

# Standardize using train_fold statistics only
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)
train_features_scaled_df = pd.DataFrame(
    train_features_scaled, columns=train_features.columns, index=train_fold_indices
)
val_features_scaled_df = pd.DataFrame(
    val_features_scaled, columns=val_features.columns, index=val_fold_indices
)
test_features_scaled_df = pd.DataFrame(
    test_features_scaled, columns=test_features.columns, index=test_df.index
)

# Save processed data
os.makedirs("./working", exist_ok=True)
train_features_scaled_df.to_pickle("./working/train_features.pkl")
val_features_scaled_df.to_pickle("./working/val_features.pkl")
test_features_scaled_df.to_pickle("./working/test_features.pkl")
train_df[["id", "author", "author_encoded"]].to_pickle("./working/train_labels.pkl")
test_df[["id"]].to_pickle("./working/test_ids.pkl")
train_fold.to_pickle("./working/train_fold.pkl")
val_fold.to_pickle("./working/val_fold.pkl")
joblib.dump(le, "./working/label_encoder.pkl")
joblib.dump(scaler, "./working/scaler.pkl")


# ============================================
# Model Design
# ============================================
class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
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
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


MODEL_NAME = "microsoft/deberta-v3-large"
NUM_AUTHORS = 3
MAX_LENGTH = 512
BATCH_SIZE = 8
NUM_WORKERS = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_indices = train_fold.index.tolist()
val_indices = val_fold.index.tolist()

train_texts = train_df.iloc[train_indices]["clean_text"].tolist()
train_labels_list = train_fold["author_encoded"].tolist()
val_texts = train_df.iloc[val_indices]["clean_text"].tolist()
val_labels_list = val_fold["author_encoded"].tolist()
test_texts = test_df["clean_text"].tolist()

train_dataset = AuthorshipDataset(train_texts, train_labels_list, tokenizer, MAX_LENGTH)
val_dataset = AuthorshipDataset(val_texts, val_labels_list, tokenizer, MAX_LENGTH)
test_dataset = AuthorshipDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


class HybridAuthorshipModel(nn.Module):
    def __init__(self, model_name, num_labels, num_engineered_features, dropout=0.3):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.deberta_hidden_size = self.deberta.config.hidden_size
        for i, param in enumerate(self.deberta.parameters()):
            if i < 12 * 3:
                param.requires_grad = False
        total_features = self.deberta_hidden_size + num_engineered_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Dropout(dropout),
            nn.Linear(total_features, total_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, total_features // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 4, num_labels),
        )

    def forward(self, input_ids, attention_mask, engineered_features):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_embedding, engineered_features], dim=1)
        logits = self.classifier(combined)
        return logits


num_engineered_features = train_features.shape[1]
model = HybridAuthorshipModel(
    model_name=MODEL_NAME,
    num_labels=NUM_AUTHORS,
    num_engineered_features=num_engineered_features,
    dropout=0.3,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Weighted loss
class_counts = train_fold["author"].value_counts()
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
label_encoder = joblib.load("./working/label_encoder.pkl")
weight_tensor = torch.tensor(
    [
        class_weights[label_encoder.transform([name])[0]]
        for name in label_encoder.classes_
    ],
    dtype=torch.float,
).to(device)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)

optimizer = AdamW(
    [
        {"params": model.deberta.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 2e-5},
    ],
    weight_decay=0.01,
)

total_steps = len(train_loader) * 20
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# LightGBM
lgb_params = {
    "objective": "multiclass",
    "num_class": NUM_AUTHORS,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "max_depth": -1,
    "num_threads": NUM_WORKERS,
    "seed": 42,
    "verbosity": -1,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# ============================================
# Training and Evaluation
# ============================================
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
GRADIENT_ACCUMULATION_STEPS = 2
USE_MIXED_PRECISION = True

X_train_eng = train_features_scaled_df.values.astype(np.float32)
X_val_eng = val_features_scaled_df.values.astype(np.float32)
y_train_eng = train_fold["author_encoded"].values
y_val_eng = val_fold["author_encoded"].values
X_test_eng = test_features_scaled_df.values.astype(np.float32)

scaler_gpu = GradScaler(enabled=USE_MIXED_PRECISION)
best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val LogLoss':<12} {'Best?'}")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        idx_start = step * train_loader.batch_size
        idx_end = min(idx_start + train_loader.batch_size, len(X_train_eng))
        eng_batch = torch.tensor(
            X_train_eng[idx_start:idx_end], dtype=torch.float32
        ).to(device)

        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask, eng_batch)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler_gpu.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask, eng_batch)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            idx_start = step * val_loader.batch_size
            idx_end = min(idx_start + val_loader.batch_size, len(X_val_eng))
            eng_batch = torch.tensor(
                X_val_eng[idx_start:idx_end], dtype=torch.float32
            ).to(device)
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask, eng_batch)
            else:
                logits = model(input_ids, attention_mask, eng_batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_probs = np.concatenate(all_preds, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_labels_onehot = np.zeros((len(val_labels), 3))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    val_logloss = log_loss(val_labels_onehot, val_probs_clipped)

    val_loss_tensor = criterion(
        torch.tensor(val_probs_clipped).float().log().to(device),
        torch.tensor(val_labels).long().to(device),
    ).item()

    is_best = val_logloss < best_val_loss
    if is_best:
        best_val_loss = val_logloss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"{epoch+1:<8} {train_loss:<12.6f} {val_loss_tensor:<12.6f} {val_logloss:<12.6f} {'*' if is_best else ''}"
    )

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation log loss: {best_val_loss:.6f}")

# Train LightGBM
print("Training LightGBM on engineered features...")
lgb_model.fit(
    X_train_eng,
    y_train_eng,
    eval_set=[(X_val_eng, y_val_eng)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(50, first_metric_only=False), lgb.log_evaluation(0)],
)

# Ensemble validation
model.eval()
val_preds_hybrid = []
with torch.no_grad():
    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        idx_start = step * val_loader.batch_size
        idx_end = min(idx_start + val_loader.batch_size, len(X_val_eng))
        eng_batch = torch.tensor(X_val_eng[idx_start:idx_end], dtype=torch.float32).to(
            device
        )
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask, eng_batch)
        else:
            logits = model(input_ids, attention_mask, eng_batch)
        probs = torch.softmax(logits, dim=1)
        val_preds_hybrid.append(probs.cpu().numpy())

val_preds_hybrid = np.concatenate(val_preds_hybrid, axis=0)
val_preds_hybrid_clipped = np.clip(val_preds_hybrid, 1e-15, 1 - 1e-15)
val_preds_hybrid_clipped = val_preds_hybrid_clipped / val_preds_hybrid_clipped.sum(
    axis=1, keepdims=True
)

lgb_val_probs = lgb_model.predict_proba(X_val_eng)

val_probs_ensemble = (val_preds_hybrid_clipped + lgb_val_probs) / 2.0
val_probs_ensemble = val_probs_ensemble / val_probs_ensemble.sum(axis=1, keepdims=True)

final_val_logloss = log_loss(val_labels_onehot, val_probs_ensemble)
print(f"\nFinal Validation Log Loss (ensemble): {final_val_logloss:.6f}")

# Test inference
print("Generating test predictions...")
model.eval()
test_preds_hybrid = []
with torch.no_grad():
    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        idx_start = step * test_loader.batch_size
        idx_end = min(idx_start + test_loader.batch_size, len(X_test_eng))
        eng_batch = torch.tensor(X_test_eng[idx_start:idx_end], dtype=torch.float32).to(
            device
        )
        if USE_MIXED_PRECISION:
            with autocast():
                logits = model(input_ids, attention_mask, eng_batch)
        else:
            logits = model(input_ids, attention_mask, eng_batch)
        probs = torch.softmax(logits, dim=1)
        test_preds_hybrid.append(probs.cpu().numpy())

test_preds_hybrid = np.concatenate(test_preds_hybrid, axis=0)
test_preds_hybrid = np.clip(test_preds_hybrid, 1e-15, 1 - 1e-15)
test_preds_hybrid = test_preds_hybrid / test_preds_hybrid.sum(axis=1, keepdims=True)

test_preds_lgb = lgb_model.predict_proba(X_test_eng)

test_preds_ensemble = (test_preds_hybrid + test_preds_lgb) / 2.0
test_preds_ensemble = test_preds_ensemble / test_preds_ensemble.sum(
    axis=1, keepdims=True
)

# Create submission
test_ids_loaded = pd.read_pickle("./working/test_ids.pkl")
submission = pd.DataFrame(
    {
        "id": test_ids_loaded["id"].values,
        "EAP": test_preds_ensemble[:, 0],
        "HPL": test_preds_ensemble[:, 1],
        "MWS": test_preds_ensemble[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:")
print(submission.head())

score = final_val_logloss
print(f"Final Validation Score: {score}")