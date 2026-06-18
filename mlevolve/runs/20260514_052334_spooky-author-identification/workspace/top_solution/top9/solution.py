import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import textstat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# NOTE: Feature extraction is done separately on train and test to prevent leakage.
# The original code had a block extracting features on all_text combined, which leaked test info.

# 1. Basic text statistics
def extract_basic_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.apply(lambda x: len(sent_tokenize(str(x))))
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )
    features["unique_word_ratio"] = text_series.apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["period_count"] = text_series.str.count(r"\.")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["dash_count"] = text_series.str.count("-")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["paren_count"] = text_series.str.count(r"\(") + text_series.str.count(
        r"\)"
    )
    features["punctuation_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation) / (len(str(x)) + 1)
    )
    features["capital_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if w[0].isupper())
        / (len(str(x).split()) + 1)
    )
    features["all_caps_word_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
    )
    features["ellipsis_count"] = text_series.str.count(r"\.\.\.")
    features["ampersand_count"] = text_series.str.count("&")
    features["asterisk_count"] = text_series.str.count(r"\*")
    return features


# 2. Vocabulary richness features
def extract_vocabulary_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    stop_words = set(stopwords.words("english"))
    for idx, text in enumerate(text_series):
        if pd.isna(text):
            continue
        words = str(text).lower().split()
        content_words = [w for w in words if w not in stop_words and w.isalpha()]
        unique_words = set(words)
        unique_content = set(content_words)
        features.loc[idx, "ttr"] = len(unique_words) / (len(words) + 1)
        features.loc[idx, "content_ttr"] = len(unique_content) / (
            len(content_words) + 1
        )
        features.loc[idx, "hapax_legomena_ratio"] = sum(
            1 for w, c in Counter(words).items() if c == 1
        ) / (len(words) + 1)
        features.loc[idx, "hapax_dislegomena_ratio"] = sum(
            1 for w, c in Counter(words).items() if c == 2
        ) / (len(words) + 1)
        word_counts = Counter(words)
        S1 = sum(word_counts.values())
        S2 = sum(v * (v - 1) for v in word_counts.values())
        features.loc[idx, "yules_k"] = 10000 * (S2 / (S1 + 1))
    return features


# 3. Part-of-speech features
def extract_pos_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    for idx, text in enumerate(text_series):
        if pd.isna(text):
            continue
        try:
            tokens = word_tokenize(str(text))
            pos_tags = pos_tag(tokens)
            features.loc[idx, "noun_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("NN")
            ) / (len(tokens) + 1)
            features.loc[idx, "verb_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("VB")
            ) / (len(tokens) + 1)
            features.loc[idx, "adj_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("JJ")
            ) / (len(tokens) + 1)
            features.loc[idx, "adv_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("RB")
            ) / (len(tokens) + 1)
            features.loc[idx, "pronoun_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("PR")
            ) / (len(tokens) + 1)
            features.loc[idx, "prep_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("IN")
            ) / (len(tokens) + 1)
            features.loc[idx, "det_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("DT")
            ) / (len(tokens) + 1)
            features.loc[idx, "conj_ratio"] = sum(
                1 for _, tag in pos_tags if tag.startswith("CC")
            ) / (len(tokens) + 1)
        except:
            pass
    return features


# 4. Readability features
def extract_readability_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    for idx, text in enumerate(text_series):
        if pd.isna(text):
            continue
        try:
            features.loc[idx, "flesch_reading_ease"] = textstat.flesch_reading_ease(
                str(text)
            )
            features.loc[idx, "flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(
                str(text)
            )
            features.loc[idx, "coleman_liau_index"] = textstat.coleman_liau_index(
                str(text)
            )
            features.loc[idx, "automated_readability_index"] = (
                textstat.automated_readability_index(str(text))
            )
            features.loc[idx, "dale_chall_score"] = (
                textstat.dale_chall_readability_score(str(text))
            )
            features.loc[idx, "linsear_write_formula"] = textstat.linsear_write_formula(
                str(text)
            )
            features.loc[idx, "gunning_fog"] = textstat.gunning_fog(str(text))
            features.loc[idx, "syllable_count"] = textstat.syllable_count(str(text))
            features.loc[idx, "polysyllable_count"] = textstat.polysyllable_count(
                str(text)
            )
            features.loc[idx, "monosyllable_count"] = textstat.monosyllable_count(
                str(text)
            )
        except:
            pass
    return features


# 5. Sentiment features
def extract_sentiment_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    positive_words = set(
        [
            "good",
            "great",
            "wonderful",
            "beautiful",
            "happy",
            "love",
            "joy",
            "peace",
            "hope",
            "kind",
            "gentle",
            "sweet",
            "bright",
            "glorious",
            "magnificent",
            "splendid",
            "delightful",
            "cheerful",
            "blissful",
            "radiant",
            "heavenly",
            "divine",
            "exquisite",
            "lovely",
            "pleasant",
            "charming",
            "graceful",
            "elegant",
        ]
    )
    negative_words = set(
        [
            "bad",
            "terrible",
            "horrible",
            "awful",
            "dreadful",
            "fearful",
            "hideous",
            "ghastly",
            "frightful",
            "dark",
            "gloomy",
            "sinister",
            "ominous",
            "menacing",
            "threatening",
            "dismal",
            "dreary",
            "sorrowful",
            "grief",
            "agony",
            "torture",
            "pain",
            "suffering",
            "horror",
            "terror",
            "fear",
            "dread",
            "anguish",
            "despair",
            "hopeless",
            "macabre",
            "gruesome",
            "grotesque",
            "monstrous",
            "horrifying",
        ]
    )
    for idx, text in enumerate(text_series):
        if pd.isna(text):
            continue
        words = str(text).lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        features.loc[idx, "positive_sentiment"] = pos_count / (len(words) + 1)
        features.loc[idx, "negative_sentiment"] = neg_count / (len(words) + 1)
        features.loc[idx, "sentiment_balance"] = (pos_count - neg_count) / (
            len(words) + 1
        )
        features.loc[idx, "sentiment_abs"] = abs(pos_count - neg_count) / (
            len(words) + 1
        )
    return features


# 6. Author vocabulary features - REMOVED due to label leakage (plan requirement)
# extract_author_vocab_features function is deleted to prevent overfitting to surface patterns
def extract_author_vocab_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    # Return empty dataframe - this feature set caused label leakage
    return features


# 7. N-gram features
def extract_ngram_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    for idx, text in enumerate(text_series):
        if pd.isna(text):
            continue
        clean_text = str(text).lower()
        chars = list(clean_text)
        bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
        trigrams = ["".join(chars[i : i + 3]) for i in range(len(chars) - 2)]
        fourgrams = ["".join(chars[i : i + 4]) for i in range(len(chars) - 3)]
        features.loc[idx, "bigram_diversity"] = len(set(bigrams)) / (len(bigrams) + 1)
        features.loc[idx, "trigram_diversity"] = len(set(trigrams)) / (
            len(trigrams) + 1
        )
        features.loc[idx, "fourgram_diversity"] = len(set(fourgrams)) / (
            len(fourgrams) + 1
        )
        features.loc[idx, "th_count"] = clean_text.count("th")
        features.loc[idx, "ing_count"] = clean_text.count("ing")
        features.loc[idx, "the_count"] = clean_text.count("the ")
        features.loc[idx, "and_count"] = clean_text.count(" and ")
        features.loc[idx, "that_count"] = clean_text.count(" that ")
        features.loc[idx, "with_count"] = clean_text.count(" with ")
        features.loc[idx, "were_count"] = clean_text.count(" were ")
        features.loc[idx, "had_count"] = clean_text.count(" had ")
        features.loc[idx, "was_count"] = clean_text.count(" was ")
    return features


# 8. Structural features
def extract_structural_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    conjunctions_start = set(
        [
            "and",
            "but",
            "or",
            "yet",
            "so",
            "for",
            "nor",
            "because",
            "although",
            "though",
            "while",
            "if",
        ]
    )
    pronouns_start = set(
        [
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
        ]
    )
    question_words = set(
        ["what", "why", "when", "where", "who", "how", "which", "whose"]
    )
    for idx, text in enumerate(text_series):
        if pd.isna(text):
            continue
        sentences = sent_tokenize(str(text))
        if sentences:
            first_words = [
                sent.split()[0].lower() if sent.split() else "" for sent in sentences
            ]
            features.loc[idx, "conjunction_start_ratio"] = sum(
                1 for w in first_words if w in conjunctions_start
            ) / len(sentences)
            features.loc[idx, "pronoun_start_ratio"] = sum(
                1 for w in first_words if w in pronouns_start
            ) / len(sentences)
            features.loc[idx, "question_word_start_ratio"] = sum(
                1 for w in first_words if w in question_words
            ) / len(sentences)
            features.loc[idx, "determiner_start_ratio"] = sum(
                1
                for w in first_words
                if w in ["the", "a", "an", "this", "that", "these", "those"]
            ) / len(sentences)
        else:
            features.loc[idx, "conjunction_start_ratio"] = 0
            features.loc[idx, "pronoun_start_ratio"] = 0
            features.loc[idx, "question_word_start_ratio"] = 0
            features.loc[idx, "determiner_start_ratio"] = 0
    return features


print("Extracting train features...")
train_text_series = pd.Series(train_df["text"].values)
basic_features_train = extract_basic_features(train_text_series)
vocab_features_train = extract_vocabulary_features(train_text_series)
pos_features_train = extract_pos_features(train_text_series)
readability_features_train = extract_readability_features(train_text_series)
sentiment_features_train = extract_sentiment_features(train_text_series)
author_vocab_features_train = extract_author_vocab_features(train_text_series)
if author_vocab_features_train.shape[1] > 0:
    author_vocab_features_train = author_vocab_features_train.iloc[:, 0:0]
ngram_features_train = extract_ngram_features(train_text_series)
structural_features_train = extract_structural_features(train_text_series)

train_features = pd.concat(
    [
        basic_features_train,
        vocab_features_train,
        pos_features_train,
        readability_features_train,
        sentiment_features_train,
        author_vocab_features_train,
        ngram_features_train,
        structural_features_train,
    ],
    axis=1,
)
train_features = train_features.replace([np.inf, -np.inf], np.nan)
fill_values = train_features.median()
train_features = train_features.fillna(fill_values)

print("Extracting test features...")
test_text_series = pd.Series(test_df["text"].values)
basic_features_test = extract_basic_features(test_text_series)
vocab_features_test = extract_vocabulary_features(test_text_series)
pos_features_test = extract_pos_features(test_text_series)
readability_features_test = extract_readability_features(test_text_series)
sentiment_features_test = extract_sentiment_features(test_text_series)
author_vocab_features_test = extract_author_vocab_features(test_text_series)
if author_vocab_features_test.shape[1] > 0:
    author_vocab_features_test = author_vocab_features_test.iloc[:, 0:0]
ngram_features_test = extract_ngram_features(test_text_series)
structural_features_test = extract_structural_features(test_text_series)

test_features = pd.concat(
    [
        basic_features_test,
        vocab_features_test,
        pos_features_test,
        readability_features_test,
        sentiment_features_test,
        author_vocab_features_test,
        ngram_features_test,
        structural_features_test,
    ],
    axis=1,
)
test_features = test_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.fillna(fill_values)

# Add text column
train_features["text"] = train_df["text"].values
test_features["text"] = test_df["text"].values

label_encoder = LabelEncoder()
train_features["author"] = label_encoder.fit_transform(train_df["author"])

feature_cols = [c for c in train_features.columns if c != "text" and c != "author"]
# StandardScaler fit on train only to prevent test data leakage
scaler = StandardScaler()
train_features_numeric = train_features[feature_cols].values
test_features_numeric = test_features[feature_cols].values

# Handle any remaining NaN/inf values before scaling
train_features_numeric = np.nan_to_num(train_features_numeric, nan=0.0, posinf=0.0, neginf=0.0)
test_features_numeric = np.nan_to_num(test_features_numeric, nan=0.0, posinf=0.0, neginf=0.0)

train_features_numeric = scaler.fit_transform(train_features_numeric)
test_features_numeric = scaler.transform(test_features_numeric)

# Keep the scaled feature arrays for later use (avoid reassigning back to DataFrame to prevent dtype issues)
train_features_arr = train_features_numeric
test_features_arr = test_features_numeric

# Prepare arrays for cross-validation
train_texts_all = train_features["text"].values
train_labels_all = train_features["author"].values
train_features_all = train_features_arr
test_texts = test_features["text"].values
test_features_arr = test_features_arr

# Vocabulary will be built per-fold within cross-validation
# to prevent leakage from validation data
max_seq_len = 256
max_char_per_word = 20


def text_to_word_ids(text, max_len=max_seq_len):
    words = str(text).lower().split()[:max_len]
    ids = [word2idx.get(w, 1) for w in words]
    ids = ids + [0] * (max_len - len(ids))
    return ids


def text_to_char_ids(text, max_len=max_seq_len, max_char=max_char_per_word):
    words = str(text).lower().split()[:max_len]
    char_ids = []
    for w in words:
        chars = list(w)[:max_char]
        cids = [char2idx.get(c, 1) for c in chars]
        cids = cids + [0] * (max_char - len(cids))
        char_ids.append(cids)
    while len(char_ids) < max_len:
        char_ids.append([0] * max_char)
    return char_ids[:max_len]


from transformers import AutoModel, AutoTokenizer

# DistilBERT-based model - replacing the BiLSTM+CharCNN+feature hybrid
class SpookyTransformerClassifier(nn.Module):
    def __init__(self, num_features, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")
        # Freeze first 4 layers (plan requirement: freeze layers 0-3, fine-tune layers 4-5)
        for i, layer in enumerate(self.transformer.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False
        # Feature projection: 62 -> 64-dim with LayerNorm and GELU
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # Classifier: [CLS] (768) + projected features (64) -> 832 -> 256 -> 128 -> 3
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.feature_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, features=None):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = transformer_outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        if features is not None:
            projected_features = self.feature_proj(features)  # (batch, 64)
            combined = torch.cat([cls_output, projected_features], dim=1)  # (batch, 832)
        else:
            combined = cls_output
        logits = self.classifier(combined)
        return logits


# Initialize DistilBERT tokenizer globally
distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class SpookyDataset(Dataset):
    def __init__(self, texts, features, labels=None):
        self.texts = texts
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Use DistilBERT tokenizer instead of manual word/char tokenization
        encoding = distilbert_tokenizer(
            text,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features,
                "labels": label,
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features,
            }


# Model hyperparams (updated for DistilBERT)
DROPOUT_RATE = 0.2
NUM_FEATURES = len(feature_cols)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
all_test_probs = []

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(train_features_all, train_labels_all)
):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*50}")

    fold_train_texts = train_texts_all[train_idx]
    fold_train_features = train_features_all[train_idx]
    fold_train_labels = train_labels_all[train_idx]
    fold_val_texts = train_texts_all[valid_idx]
    fold_val_features = train_features_all[valid_idx]
    fold_val_labels = train_labels_all[valid_idx]

    # No need to build vocabulary - DistilBERT uses its own tokenizer

    train_dataset = SpookyDataset(
        fold_train_texts, fold_train_features, fold_train_labels
    )
    val_dataset = SpookyDataset(fold_val_texts, fold_val_features, fold_val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    model = SpookyTransformerClassifier(
        num_features=NUM_FEATURES,
        num_classes=3,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.06)

    # Differential learning rates:
    # - Transformer unfrozen layers (layers 4,5): 2e-5
    # - Feature projection head: 2e-4 (10x)
    # - Classifier head: 2e-4 (10x)
    transformer_params = []
    feature_proj_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "feature_proj" in name:
            feature_proj_params.append(param)
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            transformer_params.append(param)

    optimizer = AdamW(
        [
            {"params": transformer_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": feature_proj_params, "lr": 2e-4, "weight_decay": 0.01},
            {"params": classifier_params, "lr": 2e-4, "weight_decay": 0.01},
        ]
    )

    num_epochs = 25
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scaler = GradScaler()

    # Separate linear warmup scheduler for first 10% of steps, then cosine decay to 0
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    # Cosine decay to 0 after warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=0.0
    )
    # Sequential scheduler: warmup then cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    max_patience = 4

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                features = batch["features"].to(device)

                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)

                val_loss += loss.item()
                val_batches += 1
                val_probs.append(probs.cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        val_log_loss = log_loss(fold_val_labels, val_probs)
        avg_val_loss = val_loss / val_batches

        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss/train_batches:.4f} | Val Loss: {avg_val_loss:.4f} | Val Log Loss: {val_log_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    fold_scores.append(best_val_loss)

    # Test predictions
    test_dataset = SpookyDataset(test_texts, test_features_arr)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    test_probs_fold = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

            test_probs_fold.append(probs.cpu().numpy())

    test_probs_fold = np.concatenate(test_probs_fold)
    test_probs_fold = np.clip(test_probs_fold, 1e-15, 1 - 1e-15)
    test_probs_fold = test_probs_fold / test_probs_fold.sum(axis=1, keepdims=True)
    all_test_probs.append(test_probs_fold)

    print(f"Fold {fold + 1} best validation loss: {best_val_loss:.4f}")

final_test_probs = np.mean(all_test_probs, axis=0)
final_val_score = np.mean(fold_scores)

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"Final Validation Score: {final_val_score}")