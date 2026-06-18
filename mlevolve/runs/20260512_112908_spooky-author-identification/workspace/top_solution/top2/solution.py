import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
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
# FEATURE ENGINEERING
# ============================================================
train_df["is_train"] = 1
test_df["is_train"] = 0
all_text = pd.concat(
    [train_df[["id", "text", "is_train"]], test_df[["id", "text", "is_train"]]], axis=0
).reset_index(drop=True)


def extract_basic_features(text_series):
    df = text_series.to_frame("text").copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["avg_word_len"] = df["char_count"] / (df["word_count"] + 1)
    df["sentence_count"] = df["text"].apply(
        lambda x: len(re.findall(r"[.!?]+", str(x))) + 1
    )
    df["avg_sentence_len"] = df["word_count"] / (df["sentence_count"] + 1)
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["period_count"] = df["text"].str.count(r"\.")
    df["comma_count"] = df["text"].str.count(",")
    df["semicolon_count"] = df["text"].str.count(";")
    df["colon_count"] = df["text"].str.count(":")
    df["dash_count"] = df["text"].str.count("—")
    df["quote_count"] = df["text"].str.count('"') + df["text"].str.count("'")
    df["paren_count"] = df["text"].str.count(r"[()]")
    df["punctuation_ratio"] = (
        df["exclamation_count"]
        + df["question_count"]
        + df["period_count"]
        + df["comma_count"]
        + df["semicolon_count"]
        + df["colon_count"]
    ) / (df["word_count"] + 1)
    df["capital_words_ratio"] = df["text"].apply(
        lambda x: sum(1 for w in str(x).split() if w[0].isupper())
        / (len(str(x).split()) + 1)
    )
    df["all_caps_words"] = df["text"].apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
    )
    df["ellipsis_count"] = df["text"].str.count(r"\.\.\.")
    df["digit_count"] = df["text"].str.count(r"\d")
    return df


def extract_lexical_features(text_series):
    df = pd.DataFrame(index=text_series.index)

    def type_token_ratio(text):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)

    df["type_token_ratio"] = text_series.apply(type_token_ratio)

    def hapax_ratio(text):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0
        word_counts = Counter(words)
        hapax = sum(1 for v in word_counts.values() if v == 1)
        return hapax / len(words)

    df["hapax_ratio"] = text_series.apply(hapax_ratio)

    def rare_words_ratio(text, common_words):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0
        rare_count = sum(1 for w in words if w not in common_words)
        return rare_count / len(words)

    all_words = " ".join(text_series.values).lower().split()
    common_words = set([w for w, c in Counter(all_words).most_common(500)])
    df["rare_words_ratio"] = text_series.apply(
        lambda x: rare_words_ratio(x, common_words)
    )

    def avg_word_length_dist(text):
        words = str(text).split()
        if len(words) == 0:
            return 0, 0, 0
        lengths = [len(w) for w in words]
        return np.mean(lengths), np.std(lengths), np.max(lengths)

    df[["avg_word_len_mean", "avg_word_len_std", "max_word_len"]] = text_series.apply(
        lambda x: pd.Series(avg_word_length_dist(x))
    )
    stop_words = set(
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
            "can",
            "could",
            "shall",
            "should",
            "may",
            "might",
            "that",
            "which",
            "who",
            "whom",
            "what",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "no",
            "nor",
            "not",
            "so",
            "very",
            "too",
            "quite",
            "rather",
            "such",
            "same",
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
            "here",
            "there",
            "then",
            "than",
        ]
    )
    df["stop_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in stop_words)
        / (len(str(x).split()) + 1)
    )

    def char_ngram_features(text, n=3):
        text = str(text).lower()
        ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
        if len(ngrams) == 0:
            return 0, 0
        return len(set(ngrams)), len(ngrams) / max(1, len(text))

    df[["trigram_diversity", "trigram_density"]] = text_series.apply(
        lambda x: pd.Series(char_ngram_features(x, 3))
    )
    return df


def extract_readability_features(text_series):
    df = pd.DataFrame(index=text_series.index)

    def syllable_count(word):
        word = str(word).lower()
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
            count += 1
        return count

    def flesch_reading_ease(text):
        words = str(text).split()
        if len(words) < 2:
            return 0
        sentences = len(re.findall(r"[.!?]+", str(text))) + 1
        syllables = sum(syllable_count(w) for w in words)
        return (
            206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
        )

    df["flesch_reading_ease"] = text_series.apply(flesch_reading_ease)

    def flesch_kincaid_grade(text):
        words = str(text).split()
        if len(words) < 2:
            return 0
        sentences = len(re.findall(r"[.!?]+", str(text))) + 1
        syllables = sum(syllable_count(w) for w in words)
        if sentences == 0 or len(words) == 0:
            return 0
        return 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59

    df["flesch_kincaid_grade"] = text_series.apply(flesch_kincaid_grade)

    def honore_statistic(text):
        words = str(text).lower().split()
        if len(words) < 10:
            return 0
        word_counts = Counter(words)
        V1 = sum(1 for v in word_counts.values() if v == 1)
        N = len(words)
        V = len(word_counts)
        if V1 == 0 or V == 0:
            return 0
        return 100 * np.log(N) / (1 - V1 / V)

    df["honore_statistic"] = text_series.apply(honore_statistic)

    def sichel_measure(text):
        words = str(text).lower().split()
        if len(words) < 5:
            return 0
        word_counts = Counter(words)
        V2 = sum(1 for v in word_counts.values() if v == 2)
        V = len(word_counts)
        if V == 0:
            return 0
        return V2 / V

    df["sichel_measure"] = text_series.apply(sichel_measure)
    return df


def extract_sentiment_features(text_series):
    positive_words = set(
        [
            "love",
            "beautiful",
            "wonderful",
            "great",
            "happy",
            "joy",
            "bright",
            "glorious",
            "magnificent",
            "splendid",
            "delight",
            "pleasure",
            "peace",
            "hope",
            "grace",
            "tender",
            "gentle",
            "calm",
            "serene",
            "bliss",
        ]
    )
    negative_words = set(
        [
            "dark",
            "death",
            "fear",
            "horror",
            "dread",
            "terrible",
            "awful",
            "dismal",
            "gloomy",
            "sorrow",
            "anguish",
            "pain",
            "suffering",
            "misery",
            "hatred",
            "rage",
            "fury",
            "vicious",
            "cruel",
            "ghastly",
            "hideous",
            "monstrous",
            "shadow",
            "grief",
            "weep",
            "woe",
        ]
    )

    def sentiment_score(text):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0, 0
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        return pos_count / len(words), neg_count / len(words)

    df = pd.DataFrame(index=text_series.index)
    sentiment_scores = text_series.apply(lambda x: pd.Series(sentiment_score(x)))
    df["positive_ratio"] = sentiment_scores.iloc[:, 0].values
    df["negative_ratio"] = sentiment_scores.iloc[:, 1].values
    df["emotional_intensity"] = text_series.apply(
        lambda x: len(re.findall(r"[!]", str(x)))
        + sum(1 for w in str(x).lower().split() if w in negative_words)
    )
    return df


print("Extracting basic features...")
basic_features = extract_basic_features(all_text["text"])

print("Extracting lexical features...")
lexical_features = extract_lexical_features(all_text["text"])

print("Extracting readability features...")
readability_features = extract_readability_features(all_text["text"])

print("Extracting sentiment features...")
sentiment_features = extract_sentiment_features(all_text["text"])

handcrafted_features = pd.concat(
    [
        basic_features.drop(columns=["text"]),
        lexical_features,
        readability_features,
        sentiment_features,
    ],
    axis=1,
)
handcrafted_features = handcrafted_features.replace([np.inf, -np.inf], 0)
handcrafted_features = handcrafted_features.fillna(0)

print("Generating TF-IDF features...")
train_texts = all_text[all_text["is_train"] == 1]["text"].values
test_texts = all_text[all_text["is_train"] == 0]["text"].values

# Split train texts into train/val to prevent leakage
# Reuse the same indices from the StratifiedKFold split used later
skf_temp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_orig_idx, val_orig_idx = next(skf_temp.split(train_texts, train_df["author"].values))

train_texts_fit = train_texts[train_orig_idx]
val_texts_fit = train_texts[val_orig_idx]

tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True,
    min_df=5,
    max_df=0.9,
)
tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=3000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.85,
    stop_words="english",
)

# Fit only on training fold to prevent data leakage
tfidf_char.fit(train_texts_fit)
tfidf_word.fit(train_texts_fit)

# Transform all texts using the fitted vectorizers
char_features = tfidf_char.transform(all_text["text"].values)
word_features = tfidf_word.transform(all_text["text"].values)

char_features_df = pd.DataFrame(
    char_features.toarray()[:, :500], columns=[f"char_tfidf_{i}" for i in range(500)]
)
word_features_df = pd.DataFrame(
    word_features.toarray()[:, :200], columns=[f"word_tfidf_{i}" for i in range(200)]
)

all_features = pd.concat(
    [handcrafted_features, char_features_df, word_features_df], axis=1
)
train_mask = all_text["is_train"] == 1
feature_cols = all_features.columns

scaler = StandardScaler()
# Fit scaler ONLY on training fold (train_orig_idx)
scaler.fit(all_features.loc[train_mask, feature_cols].iloc[train_orig_idx])

all_features_scaled = pd.DataFrame(
    scaler.transform(all_features[feature_cols]),
    columns=feature_cols,
    index=all_features.index,
)
all_features_scaled["id"] = all_text["id"].values
all_features_scaled["is_train"] = all_text["is_train"].values

train_features = all_features_scaled[all_features_scaled["is_train"] == 1].copy()
test_features = all_features_scaled[all_features_scaled["is_train"] == 0].copy()
train_features["author"] = train_df["author"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_features, train_features["author"]))

train_set = train_features.iloc[train_idx].reset_index(drop=True)
val_set = train_features.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)
train_set.to_parquet("./working/train_set.parquet", index=False)
val_set.to_parquet("./working/val_set.parquet", index=False)
test_features.drop(columns=["is_train"]).to_parquet(
    "./working/test_set.parquet", index=False
)

feature_names = [c for c in feature_cols]
pd.Series(feature_names).to_csv("./working/feature_names.csv", index=False)

print(f"Train set: {train_set.shape}")
print(f"Val set: {val_set.shape}")
print(f"Test set: {test_features.shape}")
print(f"Total features: {len(feature_names)}")

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
        for layer in self.backbone.deberta.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.pooling_projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.style_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size // 4, num_authors),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        cls_pool = hidden_states[:, 0, :]
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pool = sum_embeddings / sum_mask
        hidden_states_masked = hidden_states * mask_expanded + (1 - mask_expanded) * (
            -1e9
        )
        max_pool = torch.max(hidden_states_masked, dim=1)[0]
        pooled = torch.cat([cls_pool, mean_pool, max_pool], dim=1)
        pooled = self.pooling_projection(pooled)
        attended, _ = self.style_attention(
            query=pooled.unsqueeze(1),
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=(attention_mask == 0),
        )
        attended = attended.squeeze(1)
        combined = pooled + attended
        logits = self.classifier(combined)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Collect backbone unfrozen params (last 6 layers)
backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-6:]:
    for name, param in layer.named_parameters():
        if 'bias' not in name and 'LayerNorm' not in name:
            backbone_unfrozen_params.append(param)

# Collect custom head params (pooling, attention, classifier)
head_params = []
head_params.extend(model.pooling_projection.parameters())
head_params.extend(model.style_attention.parameters())
head_params.extend(model.classifier.parameters())

optimizer = AdamW(
    [
        {"params": backbone_unfrozen_params, "lr": 2e-5, "weight_decay": 0.01, "betas": (0.9, 0.999)},
        {"params": model.backbone.classifier.parameters(), "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Ensure optimizer param groups are correctly ordered
# Group 0: backbone unfrozen layers (lr=2e-5)
# Group 1: backbone classifier (lr=5e-5) (note: this is the classification head from AutoModelForSequenceClassification)
# Group 2: custom head modules (lr=5e-5)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_unfrozen_params):,}")
print(f"Backbone classifier params: {sum(p.numel() for p in model.backbone.classifier.parameters()):,}")
print(f"Custom head params: {sum(p.numel() for p in head_params):,}")

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
train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

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
os.makedirs("./submission", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch in train_loader:
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

    # Scheduler not defined in this script; remove to avoid NameError
    pass

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