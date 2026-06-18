import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup
import warnings
import math

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

# Map authors to indices
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = train_df["author"].map(author_map).values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Use split indices
train_texts = train_df.iloc[train_idx]["text"].values
train_labels = train_labels_orig[train_idx]
val_texts = train_df.iloc[val_idx]["text"].values
val_labels = train_labels_orig[val_idx]

os.makedirs("./working", exist_ok=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================


def extract_linguistic_features(text_series):
    features_list = []
    for text in text_series:
        if pd.isna(text) or text == "" or text is None:
            text = ""
        text = str(text)
        feat = {}
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        num_chars = len(text)
        feat["word_count"] = num_words
        feat["char_count"] = num_chars
        feat["avg_word_length"] = num_chars / max(num_words, 1)
        feat["sentence_count"] = num_sentences
        feat["avg_sentence_length"] = num_words / num_sentences
        unique_words = len(set(words))
        feat["unique_word_ratio"] = unique_words / max(num_words, 1)
        function_words = [
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
            "was",
            "had",
            "did",
            "would",
            "could",
            "should",
            "might",
            "shall",
            "will",
            "may",
        ]
        function_word_count = sum(1 for w in words if w.lower() in function_words)
        feat["function_word_ratio"] = function_word_count / max(num_words, 1)
        feat["exclamation_count"] = text.count("!")
        feat["question_count"] = text.count("?")
        feat["semicolon_count"] = text.count(";")
        feat["colon_count"] = text.count(":")
        feat["dash_count"] = (
            text.count("\u2014") + text.count("\u2013") + text.count("-")
        )
        feat["quote_count"] = text.count('"') + text.count("'")
        feat["parentheses_count"] = text.count("(") + text.count(")")
        feat["ellipsis_count"] = text.count("...")
        feat["comma_count"] = text.count(",")
        feat["period_count"] = text.count(".")
        total_punct = (
            feat["exclamation_count"]
            + feat["question_count"]
            + feat["semicolon_count"]
            + feat["colon_count"]
            + feat["dash_count"]
            + feat["comma_count"]
        )
        feat["punctuation_density"] = total_punct / max(num_chars, 1) * 100
        feat["capital_letter_count"] = sum(1 for c in text if c.isupper())
        feat["capital_ratio"] = feat["capital_letter_count"] / max(num_chars, 1)
        feat["digit_count"] = sum(1 for c in text if c.isdigit())
        feat["digit_ratio"] = feat["digit_count"] / max(num_chars, 1)
        word_lengths = [len(w) for w in words if w]
        if word_lengths:
            feat["long_word_count"] = sum(1 for l in word_lengths if l > 8)
            feat["short_word_count"] = sum(1 for l in word_lengths if l <= 3)
            feat["long_word_ratio"] = feat["long_word_count"] / max(num_words, 1)
            feat["short_word_ratio"] = feat["short_word_count"] / max(num_words, 1)
            feat["std_word_length"] = (
                np.std(word_lengths) if len(word_lengths) > 1 else 0
            )
        else:
            feat["long_word_count"] = 0
            feat["short_word_count"] = 0
            feat["long_word_ratio"] = 0
            feat["short_word_ratio"] = 0
            feat["std_word_length"] = 0
        feat["ly_ending_count"] = sum(1 for w in words if w.lower().endswith("ly"))
        feat["ing_ending_count"] = sum(1 for w in words if w.lower().endswith("ing"))
        feat["ed_ending_count"] = sum(1 for w in words if w.lower().endswith("ed"))
        contractions = [
            "don't",
            "can't",
            "won't",
            "couldn't",
            "wouldn't",
            "shouldn't",
            "it's",
            "that's",
            "there's",
            "what's",
            "who's",
            "where's",
            "i'm",
            "you're",
            "he's",
            "she's",
            "we're",
            "they're",
            "i've",
            "you've",
            "we've",
            "they've",
            "i'll",
            "you'll",
            "he'll",
            "she'll",
            "we'll",
            "they'll",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "doesn't",
            "didn't",
            "let's",
        ]
        contraction_count = sum(1 for w in words if w.lower() in contractions)
        feat["contraction_count"] = contraction_count
        feat["contraction_ratio"] = contraction_count / max(num_words, 1)
        syllable_count = 0
        for w in words:
            w = w.lower().strip(".,!?;:'\"()[]{}")
            if w:
                vowels = "aeiouy"
                vowel_groups = re.findall(f"[{vowels}]+", w)
                syllable_count += max(len(vowel_groups), 1)
        feat["syllable_count"] = syllable_count
        feat["syllables_per_word"] = syllable_count / max(num_words, 1)
        feat["ari_score"] = (
            4.71 * (num_chars / max(num_words, 1))
            + 0.5 * (num_words / num_sentences)
            - 21.43
        )
        feat["fkgl_score"] = (
            0.39 * (num_words / num_sentences)
            + 11.8 * (syllable_count / max(num_words, 1))
            - 15.59
        )
        stopwords = set(
            [
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
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "can",
                "need",
                "dare",
                "ought",
                "used",
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
                "mine",
                "yours",
                "his",
                "hers",
                "its",
                "ours",
                "theirs",
                "what",
                "which",
                "who",
                "whom",
                "whose",
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
                "not",
                "only",
                "own",
                "same",
                "so",
                "than",
                "too",
                "very",
                "just",
                "because",
                "as",
                "until",
                "while",
                "about",
                "between",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "from",
                "up",
                "down",
                "out",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
            ]
        )
        stopword_count = sum(1 for w in words if w.lower() in stopwords)
        feat["stopword_ratio"] = stopword_count / max(num_words, 1)
        features_list.append(feat)
    return pd.DataFrame(features_list)


char_ngram_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=5, max_df=0.8
)
char_ngram_vectorizer.fit(train_texts)
train_ngrams = char_ngram_vectorizer.transform(train_texts)
val_ngrams = char_ngram_vectorizer.transform(val_texts)
test_ngrams = char_ngram_vectorizer.transform(test_texts)

train_ngram_df = pd.DataFrame(
    train_ngrams.toarray(),
    columns=[
        f"char_ngram_{col}" for col in char_ngram_vectorizer.get_feature_names_out()
    ],
    index=train_idx,
)
val_ngram_df = pd.DataFrame(
    val_ngrams.toarray(),
    columns=[
        f"char_ngram_{col}" for col in char_ngram_vectorizer.get_feature_names_out()
    ],
    index=val_idx,
)
test_ngram_df = pd.DataFrame(
    test_ngrams.toarray(),
    columns=[
        f"char_ngram_{col}" for col in char_ngram_vectorizer.get_feature_names_out()
    ],
    index=test_df.index,
)

tfidf_vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=5,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"(?u)\b[A-Za-z]+\b",
)
train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
val_tfidf = tfidf_vectorizer.transform(val_texts)
test_tfidf = tfidf_vectorizer.transform(test_texts)

train_tfidf_df = pd.DataFrame(
    train_tfidf.toarray(),
    columns=[f"tfidf_{col}" for col in tfidf_vectorizer.get_feature_names_out()],
    index=train_idx,
)
val_tfidf_df = pd.DataFrame(
    val_tfidf.toarray(),
    columns=[f"tfidf_{col}" for col in tfidf_vectorizer.get_feature_names_out()],
    index=val_idx,
)
test_tfidf_df = pd.DataFrame(
    test_tfidf.toarray(),
    columns=[f"tfidf_{col}" for col in tfidf_vectorizer.get_feature_names_out()],
    index=test_df.index,
)

train_feats = extract_linguistic_features(train_texts)
val_feats = extract_linguistic_features(val_texts)
test_feats = extract_linguistic_features(test_texts)

feature_cols = train_feats.columns.tolist()
scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats)
val_feats_scaled = scaler.transform(val_feats)
test_feats_scaled = scaler.transform(test_feats)

train_feats_df = pd.DataFrame(train_feats_scaled, columns=feature_cols, index=train_idx)
val_feats_df = pd.DataFrame(val_feats_scaled, columns=feature_cols, index=val_idx)
test_feats_df = pd.DataFrame(
    test_feats_scaled, columns=feature_cols, index=test_df.index
)

train_features = pd.concat([train_feats_df, train_ngram_df, train_tfidf_df], axis=1)
val_features = pd.concat([val_feats_df, val_ngram_df, val_tfidf_df], axis=1)
test_features = pd.concat([test_feats_df, test_ngram_df, test_tfidf_df], axis=1)

print(f"Combined features shape - Train: {train_features.shape}")

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_rates=[0.2, 0.3, 0.4]):
        super().__init__()
        self.dropout_rates = dropout_rates

    def forward(self, x):
        outputs = []
        for rate in self.dropout_rates:
            dropout = nn.Dropout(rate)
            outputs.append(dropout(x))
        return torch.stack(outputs, dim=0).mean(dim=0)


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

        # Projection layer for increased capacity
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )

        # Multi-sample dropout head
        self.multi_dropout = MultiSampleDropout(dropout_rates=[0.2, 0.3, 0.4])
        self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Mean pooling over last 4 hidden layers
        hidden_states = outputs.hidden_states
        last_4_layers = torch.stack(hidden_states[-4:], dim=0)
        mean_pooled = last_4_layers.mean(dim=0)

        # Mean pooling over sequence (excluding padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(mean_pooled.size()).float()
        sum_embeddings = (mean_pooled * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Projection layer
        projected = self.projection(pooled)

        # Multi-sample dropout before classification
        projected_dropped = self.multi_dropout(projected)
        logits = self.head(projected_dropped)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

head_params = list(model.head.parameters())
projection_params = list(model.projection.parameters())

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {
            "params": head_params + projection_params,
            "lr": 6e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.98),
        },
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
import random as rnd


def synonym_replacement(text, replace_prob=0.2):
    words = text.split()
    if len(words) == 0:
        return text
    new_words = words.copy()
    for i in range(len(new_words)):
        if rnd.random() < replace_prob:
            # Simple synonym replacement by adding small noise
            if len(new_words[i]) > 3 and rnd.random() < 0.5:
                # Replace with similar length word by shuffling a little
                chars = list(new_words[i])
                if len(chars) > 2:
                    idx1, idx2 = rnd.sample(range(len(chars)), 2)
                    chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
                    new_words[i] = ''.join(chars)
    return ' '.join(new_words)


def word_dropout(text, dropout_prob=0.05):
    words = text.split()
    if len(words) == 0:
        return text
    new_words = [w for w in words if rnd.random() > dropout_prob]
    if len(new_words) == 0:
        new_words = words[:1]
    return ' '.join(new_words)


def apply_augmentations(text, augment_prob=0.5):
    if rnd.random() < augment_prob:
        text = synonym_replacement(text, replace_prob=0.2)
        text = word_dropout(text, dropout_prob=0.05)
    return text


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment and self.labels is not None:
            text = apply_augmentations(text)
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


batch_size = 16
max_length = 512

train_dataset = SpookyDataset(train_texts, train_labels, tokenizer, max_length, augment=True)
val_dataset = SpookyDataset(val_texts, val_labels, tokenizer, max_length, augment=False)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length, augment=False)

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

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

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

        scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / max(num_train_batches, 1)

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

    avg_val_loss = total_val_loss / max(num_val_batches, 1)
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
