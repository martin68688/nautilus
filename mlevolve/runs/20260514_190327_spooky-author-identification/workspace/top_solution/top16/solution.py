import pandas as pd
import numpy as np
import re
import os
import pickle
import math
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import warnings

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
# BASIC CLEANING
# ============================================================
def basic_clean(text):
    """Basic text cleaning - preserve structure for feature extraction"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================
def extract_basic_features(text):
    """Extract basic text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chars = len(text)
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    unique_words = len(set(w.lower() for w in words))
    return {
        "char_count": chars,
        "word_count": word_count,
        "sentence_count": sent_count,
        "avg_word_length": avg_word_len,
        "unique_word_ratio": unique_words / max(word_count, 1),
        "avg_sentence_len": word_count / sent_count,
    }


def extract_readability(text):
    """Extract readability metrics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    syllable_count = 0
    for word in words:
        word = word.lower().strip(".,!?;:\"\'()[]{}")
        if word:
            vowels = sum(1 for c in word if c in "aeiou")
            syllable_count += max(1, vowels)
    return {
        "syllable_count": syllable_count,
        "flesch_reading_ease": max(0, 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syllable_count / max(word_count, 1))),
        "avg_syllables_per_word": syllable_count / max(word_count, 1),
    }


def extract_linguistic_features(text):
    """Extract basic linguistic pattern features"""
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return {
            "stopword_ratio": 0, "punctuation_ratio": 0,
            "capitalized_ratio": 0, "numeric_ratio": 0,
            "exclamation_ratio": 0, "question_ratio": 0,
            "first_person_pronoun_count": 0,
            "past_tense_verb_ratio": 0,
        }
    stopwords = set([
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "by", "with", "from", "as", "is", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
        "so", "if", "then", "than", "that", "this", "these", "those", "i",
        "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
        "hers", "ours", "theirs", "what", "which", "who", "whom", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "only", "own", "same", "here",
        "there", "very", "just", "too", "also", "about", "above", "after",
        "again", "against", "below", "between", "through", "during", "before",
        "after", "up", "down", "out", "off", "over", "under", "into", "onto",
        "upon", "than", "because", "since", "until", "while", "although",
        "though", "even", "still", "yet", "already", "any", "anyone",
        "anything", "anywhere", "everyone", "everything", "everywhere",
        "noone", "nothing", "nowhere", "someone", "something", "somewhere",
    ])
    first_person_pronouns = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
    past_tense_verbs = {"was", "were", "had", "did", "said", "made", "took", "came",
                        "saw", "went", "gave", "knew", "thought", "felt", "found",
                        "left", "told", "became", "began", "brought", "built",
                        "bought", "caught", "chose", "drew", "drank", "drove",
                        "ate", "fell", "flew", "forgot", "grew", "held", "kept",
                        "led", "lost", "met", "paid", "put", "ran", "sent",
                        "showed", "sang", "sat", "slept", "spoke", "spent",
                        "stood", "stole", "swam", "tore", "threw", "understood",
                        "woke", "wore", "wrote", "won", "hid", "hit", "hurt",
                        "let", "set", "shut", "spread", "cut", "cost", "read"}
    lowercase_words = [w.lower().strip(".,!?;:\"\'()[]{}") for w in words]
    stopword_count = sum(1 for w in lowercase_words if w in stopwords)
    punct_count = sum(1 for c in text if c in ".,!?;:\"\'()[]{}")
    capitalized_count = sum(1 for w in words if w and w[0].isupper())
    numeric_count = sum(1 for w in words if any(c.isdigit() for c in w))
    exclamation_count = text.count("!") + text.count("?")
    question_count = text.count("?")
    first_person_count = sum(1 for w in lowercase_words if w in first_person_pronouns)
    past_tense_count = sum(1 for w in lowercase_words if w in past_tense_verbs)
    return {
        "stopword_ratio": stopword_count / total_words,
        "punctuation_ratio": punct_count / max(len(text), 1),
        "capitalized_ratio": capitalized_count / total_words,
        "numeric_ratio": numeric_count / total_words,
        "exclamation_ratio": exclamation_count / max(len(text), 1),
        "question_ratio": question_count / max(len(text), 1),
        "first_person_pronoun_count": first_person_count,
        "past_tense_verb_ratio": past_tense_count / total_words,
    }


def extract_structural_features(text):
    """Extract structural text features"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    commas = text.count(",")
    semicolons = text.count(";")
    colons = text.count(":")
    dashes = text.count("--") + text.count("—")
    quotes = text.count("\"") + text.count("\"") + text.count("'") + text.count("'")
    parentheses = text.count("(") + text.count(")")
    brackets = text.count("[") + text.count("]")
    return {
        "comma_per_word": commas / max(len(words), 1),
        "semicolon_per_word": semicolons / max(len(words), 1),
        "colon_per_word": colons / max(len(words), 1),
        "dash_per_word": dashes / max(len(words), 1),
        "quote_per_word": quotes / max(len(words), 1),
        "parentheses_per_word": parentheses / max(len(words), 1),
        "brackets_per_word": brackets / max(len(words), 1),
        "avg_paragraph_len": len(words) / max(len(sentences), 1),
    }


def extract_vocab_features(text):
    """Extract vocabulary richness features"""
    words = text.split()
    if not words:
        return {"ttr": 0, "hapax_legomena_ratio": 0, "hapax_dislegomena_ratio": 0,
                "honore_statistic": 0, "sichel_statistic": 0,
                "avg_word_frequency_rank": 0, "long_word_ratio": 0,
                "very_long_word_ratio": 0}
    word_freq = {}
    for w in words:
        w_clean = w.lower().strip(".,!?;:\"\'()[]{}")
        if w_clean:
            word_freq[w_clean] = word_freq.get(w_clean, 0) + 1
    types = len(word_freq)
    tokens = len(words)
    hapax = sum(1 for v in word_freq.values() if v == 1)
    hapax_dis = sum(1 for v in word_freq.values() if v == 2)
    ttr = types / max(tokens, 1)
    honore = 100 * np.log(tokens / max(1 - hapax / max(types, 1), 1e-10)) if types > 1 else 0
    sichel = hapax_dis / max(types, 1)
    long_words = sum(1 for w in words if len(w) > 6)
    very_long_words = sum(1 for w in words if len(w) > 10)
    return {
        "ttr": ttr,
        "hapax_legomena_ratio": hapax / max(types, 1),
        "hapax_dislegomena_ratio": hapax_dis / max(types, 1),
        "honore_statistic": honore,
        "sichel_statistic": sichel,
        "long_word_ratio": long_words / max(tokens, 1),
        "very_long_word_ratio": very_long_words / max(tokens, 1),
    }


train_df["text_clean"] = train_df["text"].apply(basic_clean)
test_df["text_clean"] = test_df["text"].apply(basic_clean)

# ============================================================
# TRAIN/VALIDATION SPLIT - FIRST before any feature engineering
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_indices = train_idx
val_indices = val_idx

# Original text data for transformer
train_texts_orig = train_df["text"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Use stratified labels
le = LabelEncoder()
train_labels_orig = le.fit_transform(train_df["author"])

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

# Split the data before any feature engineering
train_df_split = train_df.iloc[train_indices].copy()
val_df_split = train_df.iloc[val_indices].copy()

# ============================================================
# FEATURE ENGINEERING - Now done AFTER split on train only
# ============================================================
train_corpus = train_df_split["text_clean"].values
val_corpus = val_df_split["text_clean"].values
test_corpus = test_df["text_clean"].values

# 1. Basic Text Statistics (applied separately)
train_basic_features = train_df_split["text_clean"].apply(extract_basic_features)
val_basic_features = val_df_split["text_clean"].apply(extract_basic_features)
test_basic_features = test_df["text_clean"].apply(extract_basic_features)
train_basic_df = pd.DataFrame(train_basic_features.tolist())
val_basic_df = pd.DataFrame(val_basic_features.tolist())
test_basic_df = pd.DataFrame(test_basic_features.tolist())

# 2. Readability Scores (applied separately)
train_readability = train_df_split["text_clean"].apply(extract_readability)
val_readability = val_df_split["text_clean"].apply(extract_readability)
test_readability = test_df["text_clean"].apply(extract_readability)
train_readability_df = pd.DataFrame(train_readability.tolist())
val_readability_df = pd.DataFrame(val_readability.tolist())
test_readability_df = pd.DataFrame(test_readability.tolist())

# 3. POS-like Features (applied separately)
train_linguistic = train_df_split["text_clean"].apply(extract_linguistic_features)
val_linguistic = val_df_split["text_clean"].apply(extract_linguistic_features)
test_linguistic = test_df["text_clean"].apply(extract_linguistic_features)
train_linguistic_df = pd.DataFrame(train_linguistic.tolist())
val_linguistic_df = pd.DataFrame(val_linguistic.tolist())
test_linguistic_df = pd.DataFrame(test_linguistic.tolist())

# 4. TF-IDF Features - Fit ONLY on train split
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 3),
    sublinear_tf=True,
    stop_words="english",
)
count_vectorizer = CountVectorizer(
    max_features=1000, min_df=5, max_df=0.7, ngram_range=(2, 3), binary=True
)

tfidf_features = tfidf_vectorizer.fit_transform(train_corpus)
val_tfidf_features = tfidf_vectorizer.transform(val_corpus)
test_tfidf_features = tfidf_vectorizer.transform(test_corpus)

svd = TruncatedSVD(n_components=100, random_state=42)
train_tfidf_svd = svd.fit_transform(tfidf_features)
val_tfidf_svd = svd.transform(val_tfidf_features)
test_tfidf_svd = svd.transform(test_tfidf_features)

count_features = count_vectorizer.fit_transform(train_corpus)
val_count_features = count_vectorizer.transform(val_corpus)
test_count_features = count_vectorizer.transform(test_corpus)

# 5. Structural features (applied separately)
train_structural = train_df_split["text_clean"].apply(extract_structural_features)
val_structural = val_df_split["text_clean"].apply(extract_structural_features)
test_structural = test_df["text_clean"].apply(extract_structural_features)
train_structural_df = pd.DataFrame(train_structural.tolist())
val_structural_df = pd.DataFrame(val_structural.tolist())
test_structural_df = pd.DataFrame(test_structural.tolist())

# 6. Vocabulary complexity features (applied separately)
train_vocab = train_df_split["text_clean"].apply(extract_vocab_features)
val_vocab = val_df_split["text_clean"].apply(extract_vocab_features)
test_vocab = test_df["text_clean"].apply(extract_vocab_features)
train_vocab_df = pd.DataFrame(train_vocab.tolist())
val_vocab_df = pd.DataFrame(val_vocab.tolist())
test_vocab_df = pd.DataFrame(test_vocab.tolist())

# ============================================================
# COMBINE ALL FEATURES
# ============================================================
train_features = pd.concat(
    [
        train_basic_df,
        train_readability_df,
        train_linguistic_df,
        train_structural_df,
        train_vocab_df,
        pd.DataFrame(train_tfidf_svd, columns=[f"tfidf_svd_{i}" for i in range(100)]),
        pd.DataFrame(
            count_features.toarray(),
            columns=[f"count_ngram_{i}" for i in range(count_features.shape[1])],
        ),
    ],
    axis=1,
)

val_features = pd.concat(
    [
        val_basic_df,
        val_readability_df,
        val_linguistic_df,
        val_structural_df,
        val_vocab_df,
        pd.DataFrame(val_tfidf_svd, columns=[f"tfidf_svd_{i}" for i in range(100)]),
        pd.DataFrame(
            val_count_features.toarray(),
            columns=[f"count_ngram_{i}" for i in range(val_count_features.shape[1])],
        ),
    ],
    axis=1,
)

test_features = pd.concat(
    [
        test_basic_df,
        test_readability_df,
        test_linguistic_df,
        test_structural_df,
        test_vocab_df,
        pd.DataFrame(test_tfidf_svd, columns=[f"tfidf_svd_{i}" for i in range(100)]),
        pd.DataFrame(
            test_count_features.toarray(),
            columns=[f"count_ngram_{i}" for i in range(test_count_features.shape[1])],
        ),
    ],
    axis=1,
)

# Standardize numerical features - Fit ONLY on train
scaler = StandardScaler()
train_scaled = pd.DataFrame(
    scaler.fit_transform(train_features),
    columns=train_features.columns,
    index=train_features.index,
)
val_scaled = pd.DataFrame(
    scaler.transform(val_features),
    columns=val_features.columns,
    index=val_features.index,
)
test_scaled = pd.DataFrame(
    scaler.transform(test_features),
    columns=test_features.columns,
    index=test_features.index,
)

# Handle inf/nan
train_scaled = train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
val_scaled = val_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
test_scaled = test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

os.makedirs("./working", exist_ok=True)

# Save feature objects for potential later use
with open("./working/feature_objects.pkl", "wb") as f:
    pickle.dump(
        {
            "le": le,
            "scaler": scaler,
            "tfidf_vectorizer": tfidf_vectorizer,
            "count_vectorizer": count_vectorizer,
            "svd": svd,
        },
        f,
    )

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.2):
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
        # Unfreeze last 6 layers instead of 4
        for layer in self.backbone.deberta.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_authors)
        # Attention pooling over last 4 hidden layers
        self.attention_pool = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract hidden states from all layers, take last 4
        all_hidden = outputs.hidden_states  # tuple of (batch, seq_len, hidden) per layer
        last_4 = all_hidden[-4:]  # 4 tensors
        # Mean pool each layer over the sequence dimension (masked)
        pooled_list = []
        for layer_hidden in last_4:
            mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
            sum_hidden = (layer_hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            pooled = sum_hidden / (sum_mask + 1e-9)
            pooled_list.append(pooled)
        # Stack and compute attention-weighted sum
        stacked = torch.stack(pooled_list, dim=1)  # (batch, 4, hidden)
        # Compute scalar attention scores per layer
        attn_scores = self.attention_pool(stacked).squeeze(-1)  # (batch, 4)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, 4)
        weighted = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        logits = self.head(weighted)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.2)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-6:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

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
num_epochs = 12
patience = 3
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()
os.makedirs("./submission", exist_ok=True)

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)
# Use SequentialLR: LinearLR warmup then CosineAnnealingLR
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
    ],
    milestones=[warmup_steps],
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