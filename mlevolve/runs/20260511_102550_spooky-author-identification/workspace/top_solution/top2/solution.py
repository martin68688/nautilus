import pandas as pd
import numpy as np
import re
import string
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# ========== FEATURE ENGINEERING ==========


def extract_basic_features(text_series):
    """Extract basic text features"""
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)

    features["char_count"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = texts.str.count("[.!?]") + 1

    features["unique_words"] = texts.apply(lambda x: len(set(x.lower().split())))
    features["type_token_ratio"] = features["unique_words"] / (
        features["word_count"] + 1
    )

    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    features["period_count"] = texts.str.count(r"\.")
    features["comma_count"] = texts.str.count(",")
    features["semicolon_count"] = texts.str.count(";")
    features["colon_count"] = texts.str.count(":")
    features["dash_count"] = texts.str.count("—|-")
    features["quote_count"] = texts.str.count('"') + texts.str.count("'")
    features["paren_count"] = texts.str.count(r"\(|\)")

    total_punct = (
        features["exclamation_count"]
        + features["question_count"]
        + features["period_count"]
        + features["comma_count"]
        + features["semicolon_count"]
        + features["colon_count"]
        + features["dash_count"]
        + features["quote_count"]
        + features["paren_count"]
    )
    features["punct_density"] = total_punct / (features["char_count"] + 1)

    features["capital_ratio"] = texts.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    features["all_caps_words"] = texts.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
    )

    archaic_words = [
        "thee",
        "thou",
        "thy",
        "thine",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "oft",
        "hath",
        "doth",
        "dost",
        "shall",
        "art",
        "forsooth",
        "perchance",
        "methinks",
        "alas",
        "aye",
        "nay",
        "whence",
        "wherein",
        "whereupon",
    ]
    features["archaic_word_count"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in archaic_words
        )
    )

    features["syllable_estimate"] = texts.apply(
        lambda x: sum(_estimate_syllables(w) for w in x.split() if w)
    )
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["word_count"] / (features["sentence_count"] + 1)
        - 84.6 * features["syllable_estimate"] / (features["word_count"] + 1)
    )

    return features


def _estimate_syllables(word):
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def extract_ngram_features(text_series, ngram_range=(1, 3), max_features=5000):
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    ngram_matrix = tfidf.fit_transform(text_series.fillna("").astype(str))
    feature_names = [f"char_ngram_{i}" for i in range(ngram_matrix.shape[1])]
    return (
        pd.DataFrame(
            ngram_matrix.toarray(), columns=feature_names, index=text_series.index
        ),
        tfidf,
    )


def extract_word_ngram_features(text_series, max_features=3000):
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    ngram_matrix = tfidf.fit_transform(text_series.fillna("").astype(str))
    feature_names = [f"word_ngram_{i}" for i in range(ngram_matrix.shape[1])]
    return (
        pd.DataFrame(
            ngram_matrix.toarray(), columns=feature_names, index=text_series.index
        ),
        tfidf,
    )


def extract_pos_style_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)

    features["article_count"] = texts.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w.strip(string.punctuation) in ["a", "an", "the"]
        )
    )

    first_person = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"]
    second_person = ["you", "your", "yours", "thou", "thee", "thy"]
    third_person = [
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
    ]

    features["first_person_count"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in first_person
        )
    )
    features["second_person_count"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in second_person
        )
    )
    features["third_person_count"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in third_person
        )
    )

    conjunctions = [
        "and",
        "but",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "because",
        "although",
        "while",
        "since",
    ]
    features["conjunction_count"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in conjunctions
        )
    )

    prepositions = [
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "of",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
    ]
    features["preposition_count"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in prepositions
        )
    )

    features["ly_adverbs"] = texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.endswith("ly") and len(w) > 3)
    )
    features["ing_verbs"] = texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.endswith("ing") and len(w) > 4)
    )
    features["ed_verbs"] = texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.endswith("ed") and len(w) > 3)
    )

    negation_words = [
        "not",
        "n't",
        "never",
        "nothing",
        "no",
        "none",
        "neither",
        "nor",
        "nowhere",
    ]
    features["negation_count"] = texts.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w.strip(string.punctuation) in negation_words
        )
    )

    return features


def extract_author_specific_vocab(text_series):
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)

    lovecraft_vocab = [
        "eldritch",
        "cyclopean",
        "blasphemous",
        "indescribable",
        "unnamable",
        "antediluvian",
        "non-euclidean",
        "cosmic",
        "gibbering",
        "crawling",
        "fungus",
        "squamous",
        "ichor",
        "abyss",
        "ancient",
        "evil",
        "vault",
        "nameless",
        "hideous",
        "madness",
        "nightmare",
        "entity",
        "dimension",
        "alien",
        "slimy",
        "daemon",
        "vampire",
        "demon",
        "crypt",
        "tomb",
    ]
    poe_vocab = [
        "desolate",
        "melancholy",
        "dreary",
        "trembling",
        "terror",
        "horror",
        "phantasm",
        "spectre",
        "apparition",
        "revelry",
        "pendulum",
        "chamber",
        "sepulchre",
        "casement",
        "torch",
        "dungeon",
        "shroud",
        "coffin",
        "pallid",
        "ghastly",
        "dissolution",
        "phantasy",
        "anomalous",
        "cognizance",
    ]
    shelley_vocab = [
        "beautiful",
        "nature",
        "sublime",
        "passion",
        "affection",
        "gentle",
        "beloved",
        "friendship",
        "virtue",
        "misery",
        "despair",
        "mountain",
        "valley",
        "forest",
        "moonlight",
        "delight",
        "compassion",
        "remorse",
        "creator",
        "creature",
        "wretch",
        "innocent",
        "suffer",
        "embrace",
    ]

    features["lovecraft_vocab_score"] = texts.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w.strip(string.punctuation) in lovecraft_vocab
        )
        / (len(x.split()) + 1)
    )
    features["poe_vocab_score"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in poe_vocab
        )
        / (len(x.split()) + 1)
    )
    features["shelley_vocab_score"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in shelley_vocab
        )
        / (len(x.split()) + 1)
    )

    return features


def extract_emotion_sentiment_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)

    fear_words = [
        "fear",
        "dread",
        "terror",
        "horror",
        "panic",
        "alarm",
        "fright",
        "scared",
        "afraid",
        "terrified",
        "horrified",
        "dreadful",
        "terrible",
        "awful",
    ]
    sadness_words = [
        "sad",
        "sorrow",
        "grief",
        "misery",
        "despair",
        "mournful",
        "melancholy",
        "dreary",
        "gloomy",
        "somber",
        "weep",
        "cry",
        "tear",
        "lament",
    ]
    anger_words = [
        "anger",
        "rage",
        "fury",
        "wrath",
        "hate",
        "hatred",
        "violent",
        "furious",
        "enraged",
        "outrage",
        "vengeance",
        "revenge",
    ]
    surprise_words = [
        "surprise",
        "astonish",
        "amaze",
        "shock",
        "wonder",
        "startle",
        "sudden",
        "unexpected",
        "incredible",
        "extraordinary",
        "marvel",
    ]

    features["fear_words"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in fear_words
        )
    )
    features["sadness_words"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in sadness_words
        )
    )
    features["anger_words"] = texts.apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in anger_words
        )
    )
    features["surprise_words"] = texts.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w.strip(string.punctuation) in surprise_words
        )
    )

    total_emotion = features[
        ["fear_words", "sadness_words", "anger_words", "surprise_words"]
    ].sum(axis=1)
    features["emotional_intensity"] = total_emotion / (texts.str.split().str.len() + 1)

    return features


label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])

# First split the data, THEN do feature engineering
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    train_df["text"].values,
    train_df["author_encoded"].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

# Create DataFrames with proper indices for feature extraction
train_split_df = train_df.loc[train_df["text"].isin(X_train_texts)].copy()
val_split_df = train_df.loc[train_df["text"].isin(X_val_texts)].copy()

# Align by index using the split indices
train_indices = np.where(np.isin(train_df["text"].values, X_train_texts))[0]
val_indices = np.where(np.isin(train_df["text"].values, X_val_texts))[0]

train_split_df = train_df.iloc[train_indices].copy()
val_split_df = train_df.iloc[val_indices].copy()

print("Extracting features from training data (train split only)...")
train_basic_features = extract_basic_features(train_split_df["text"])
train_pos_features = extract_pos_style_features(train_split_df["text"])
train_author_vocab = extract_author_specific_vocab(train_split_df["text"])
train_emotion_features = extract_emotion_sentiment_features(train_split_df["text"])
train_char_ngram, char_ngram_vectorizer = extract_ngram_features(train_split_df["text"])
train_word_ngram, word_ngram_vectorizer = extract_word_ngram_features(train_split_df["text"])

train_features = pd.concat(
    [
        train_basic_features,
        train_pos_features,
        train_author_vocab,
        train_emotion_features,
        train_char_ngram,
        train_word_ngram,
    ],
    axis=1,
)

train_features = train_features.replace([np.inf, -np.inf], 0)
for col in train_features.columns:
    if train_features[col].dtype in ["float64", "float32"]:
        upper = train_features[col].quantile(0.99)
        lower = train_features[col].quantile(0.01)
        train_features[col] = train_features[col].clip(lower, upper)

print("Extracting features from validation data...")
val_basic_features = extract_basic_features(val_split_df["text"])
val_pos_features = extract_pos_style_features(val_split_df["text"])
val_author_vocab = extract_author_specific_vocab(val_split_df["text"])
val_emotion_features = extract_emotion_sentiment_features(val_split_df["text"])
val_char_ngram = pd.DataFrame(
    char_ngram_vectorizer.transform(val_split_df["text"].fillna("").astype(str)).toarray(),
    columns=train_char_ngram.columns,
    index=val_split_df.index,
)
val_word_ngram = pd.DataFrame(
    word_ngram_vectorizer.transform(val_split_df["text"].fillna("").astype(str)).toarray(),
    columns=train_word_ngram.columns,
    index=val_split_df.index,
)

val_features = pd.concat(
    [
        val_basic_features,
        val_pos_features,
        val_author_vocab,
        val_emotion_features,
        val_char_ngram,
        val_word_ngram,
    ],
    axis=1,
)

val_features = val_features.replace([np.inf, -np.inf], 0)
for col in val_features.columns:
    if col in train_features.columns and train_features[col].dtype in [
        "float64",
        "float32",
    ]:
        upper = train_features[col].quantile(0.99)
        lower = train_features[col].quantile(0.01)
        val_features[col] = val_features[col].clip(lower, upper)

print("Extracting features from test data...")
test_basic_features = extract_basic_features(test_df["text"])
test_pos_features = extract_pos_style_features(test_df["text"])
test_author_vocab = extract_author_specific_vocab(test_df["text"])
test_emotion_features = extract_emotion_sentiment_features(test_df["text"])
test_char_ngram = pd.DataFrame(
    char_ngram_vectorizer.transform(test_df["text"].fillna("").astype(str)).toarray(),
    columns=train_char_ngram.columns,
    index=test_df.index,
)
test_word_ngram = pd.DataFrame(
    word_ngram_vectorizer.transform(test_df["text"].fillna("").astype(str)).toarray(),
    columns=train_word_ngram.columns,
    index=test_df.index,
)

test_features = pd.concat(
    [
        test_basic_features,
        test_pos_features,
        test_author_vocab,
        test_emotion_features,
        test_char_ngram,
        test_word_ngram,
    ],
    axis=1,
)

test_features = test_features.replace([np.inf, -np.inf], 0)
for col in test_features.columns:
    if col in train_features.columns and train_features[col].dtype in [
        "float64",
        "float32",
    ]:
        upper = train_features[col].quantile(0.99)
        lower = train_features[col].quantile(0.01)
        test_features[col] = test_features[col].clip(lower, upper)

# ========== MODEL DESIGN ==========

NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 30
WARMUP_STEPS = 100

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)


class AuthorClassifier(nn.Module):
    def __init__(self, base_model, num_classes=NUM_AUTHORS, dropout_rate=0.2):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}


model = AuthorClassifier(
    base_model.deberta if hasattr(base_model, "deberta") else base_model,
    num_classes=NUM_AUTHORS,
    dropout_rate=0.2,
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
)

total_training_steps = len(X_train_texts) // BATCH_SIZE * NUM_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_training_steps
)

scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(f"Device: {device}")

# ========== TRAINING & EVALUATION ==========


class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if self.labels is not None:
            return input_ids, attention_mask, self.labels[idx]
        return input_ids, attention_mask


train_dataset = AuthorshipDataset(X_train_texts, y_train)
val_dataset = AuthorshipDataset(X_val_texts, y_val)
test_dataset = AuthorshipDataset(test_df["text"].values)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
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

model.train()
best_val_loss = float("inf")
best_val_score = float("inf")
patience = 5
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    train_batches = 0

    for batch in train_loader:
        input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_train_loss += loss.item()
        train_batches += 1

    avg_train_loss = total_train_loss / train_batches

    model.eval()
    val_loss = 0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            if scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=batch_labels,
                    )
                    loss = outputs["loss"]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs["loss"]

            val_loss += loss.item()
            val_batches += 1

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(batch_labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    eps = 1e-15
    val_preds = np.clip(val_preds, eps, 1 - eps)

    N = len(val_true)
    val_log_loss = 0
    for i in range(N):
        for j in range(NUM_AUTHORS):
            if val_true[i] == j:
                val_log_loss += np.log(val_preds[i, j])
    val_log_loss = -val_log_loss / N

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_log_loss:.4f}"
    )

    if val_log_loss < best_val_score:
        best_val_score = val_log_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)
model.eval()

# Final validation evaluation
all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

        if scaler is not None:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())
        all_val_labels.append(batch_labels.cpu().numpy())

val_preds = np.concatenate(all_val_preds, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)

eps = 1e-15
val_preds = np.clip(val_preds, eps, 1 - eps)

N = len(val_true)
final_val_log_loss = 0
for i in range(N):
    for j in range(NUM_AUTHORS):
        if val_true[i] == j:
            final_val_log_loss += np.log(val_preds[i, j])
final_val_log_loss = -final_val_log_loss / N

# Test inference
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]

        if scaler is not None:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

# Normalize probabilities to sum to 1
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

eps = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], eps, 1 - eps)

submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_log_loss}")