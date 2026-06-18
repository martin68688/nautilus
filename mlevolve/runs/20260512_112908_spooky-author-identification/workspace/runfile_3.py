import os
os.sched_setaffinity(0, {117})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
import re
import os
import copy
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

warnings.filterwarnings("ignore")

# Define author mapping globally
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
NUM_AUTHORS = 3
MAX_LENGTH = 512

# Model configurations
MODEL_CONFIGS = {
    "deberta-large": {
        "model_name": "microsoft/deberta-v3-large",
        "lr_backbone": 2e-5,
        "lr_head": 5e-5,
        "freeze_layers": True,  # freeze all except last 6 layers
        "unfreeze_last_n": 6,
        "weight_decay": 0.01,
    },
    "deberta-base": {
        "model_name": "microsoft/deberta-v3-base",
        "lr_backbone": 5e-5,
        "lr_head": 5e-5,
        "freeze_layers": False,  # all layers unfrozen
        "unfreeze_last_n": None,
        "weight_decay": 0.01,
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "lr_backbone": 3e-5,
        "lr_head": 3e-5,
        "freeze_layers": False,
        "unfreeze_last_n": None,
        "weight_decay": 0.01,
    },
}

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

# NOTE: TF-IDF and scaling will be handled per-fold inside the CV loop to prevent data leakage.
# Here we only store the raw handcrafted features, and TF-IDF will be fit per fold.
train_mask = all_text["is_train"] == 1
feature_cols = handcrafted_features.columns.tolist()

all_features = handcrafted_features
all_features["id"] = all_text["id"].values
all_features["is_train"] = all_text["is_train"].values

train_features = all_features[all_features["is_train"] == 1].copy()
test_features = all_features[all_features["is_train"] == 0].copy()
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
# MODEL DESIGN - Three diverse transformer models
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MeanPoolingClassifier(nn.Module):
    """
    Simple model: backbone transformer + mean pooling + dropout + linear classification head.
    This is a simpler architecture than the previous SpookyAuthorClassifier to
    standardize across models and avoid overfitting.
    """
    def __init__(self, model_name, num_authors=3, dropout_rate=0.3,
                 freeze_layers=True, unfreeze_last_n=6):
        super().__init__()
        # Use AutoModel (not AutoModelForSequenceClassification) to get full control
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        hidden_size = self.config.hidden_size

        # Freeze layers as specified
        if freeze_layers and hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_n is not None:
                for layer in self.backbone.encoder.layer[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_authors)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        last_hidden = outputs.last_hidden_state

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def create_optimizer_and_scheduler(model, config, num_training_steps):
    """Create AdamW optimizer and cosine scheduler."""
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if hasattr(model, 'backbone') and model.backbone is not None:
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append((name, param))
            else:
                head_params.append((name, param))

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
                'weight_decay': config["weight_decay"],
                'lr': config["lr_backbone"],
            },
            {
                'params': [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config["lr_backbone"],
            },
            {
                'params': [p for n, p in head_params if not any(nd in n for nd in no_decay)],
                'weight_decay': config["weight_decay"],
                'lr': config["lr_head"],
            },
            {
                'params': [p for n, p in head_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config["lr_head"],
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': config["weight_decay"],
                'lr': config["lr_backbone"],
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config["lr_backbone"],
            },
        ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config["lr_backbone"],  # default lr, overridden per group
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Cosine scheduler with warmup
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_training_steps,
        T_mult=1,
        eta_min=1e-6
    )

    return optimizer, scheduler, warmup_steps


# Initialize tokenizers for all models
tokenizers = {}
for model_key, config in MODEL_CONFIGS.items():
    tokenizers[model_key] = AutoTokenizer.from_pretrained(config["model_name"])

print("Model configurations loaded. Tokenizers initialized.")
print(f"Models: {list(MODEL_CONFIGS.keys())}")


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


# ============================================================
# PREPARE DATA FOR 5-FOLD CROSS-VALIDATION
# ============================================================
train_texts = train_df["text"].values
train_labels = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

batch_size = 16
grad_accum = 2
num_epochs = 30
patience = 5

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions and test predictions
# OOF: (n_train, 3 models x 3 classes = 9 features)
oof_predictions = np.zeros((len(train_texts), len(MODEL_CONFIGS) * NUM_AUTHORS))
oof_labels = np.zeros(len(train_texts), dtype=int)

# Test predictions: (n_test, n_folds, n_models, n_classes)
test_predictions = np.zeros((len(test_texts), 5, len(MODEL_CONFIGS), NUM_AUTHORS))

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ============================================================
# TRAINING LOOP WITH 5-FOLD CV
# ============================================================
print("=" * 60)
print("Starting 5-fold Cross Validation for 3 models")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*60}")

    fold_train_texts = train_texts[train_idx]
    fold_train_labels = train_labels[train_idx]
    fold_val_texts = train_texts[val_idx]
    fold_val_labels = train_labels[val_idx]

    fold_oof = np.zeros((len(val_idx), len(MODEL_CONFIGS) * NUM_AUTHORS))
    fold_test_probs = np.zeros((len(test_texts), len(MODEL_CONFIGS), NUM_AUTHORS))

    for model_idx, (model_key, config) in enumerate(MODEL_CONFIGS.items()):
        print(f"\nTraining {model_key}...")

        # Initialize model for this fold
        model = MeanPoolingClassifier(
            model_name=config["model_name"],
            num_authors=NUM_AUTHORS,
            dropout_rate=0.3,
            freeze_layers=config["freeze_layers"],
            unfreeze_last_n=config.get("unfreeze_last_n"),
        )
        model.to(device)

        # Create datasets
        tokenizer = tokenizers[model_key]

        train_dataset = SpookyDataset(
            fold_train_texts, fold_train_labels, tokenizer, MAX_LENGTH
        )
        val_dataset = SpookyDataset(
            fold_val_texts, fold_val_labels, tokenizer, MAX_LENGTH
        )
        test_dataset = SpookyDataset(
            test_texts, None, tokenizer, MAX_LENGTH
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        num_training_steps = len(train_loader) * num_epochs // grad_accum
        optimizer, scheduler, warmup_steps = create_optimizer_and_scheduler(
            model, config, num_training_steps
        )

        scaler_grad = GradScaler()
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            # Training
            model.train()
            total_train_loss = 0
            num_train_batches = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss = loss / grad_accum

                scaler_grad.scale(loss).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    scaler_grad.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler_grad.step(optimizer)
                    scaler_grad.update()
                    optimizer.zero_grad()

                    # Warmup and schedule
                    if epoch * len(train_loader) + batch_idx < warmup_steps:
                        # Linear warmup: scale lr
                        lr_scale = min(1.0, (epoch * len(train_loader) + batch_idx) / warmup_steps)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * lr_scale
                    else:
                        scheduler.step()

                total_train_loss += loss.item() * grad_accum
                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches

            # Validation
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            all_val_probs = []

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

            avg_val_loss = total_val_loss / num_val_batches
            val_probs = np.concatenate(all_val_probs, axis=0)

            val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
            val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
            val_score = log_loss(fold_val_labels, val_probs_clipped)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs")
                    break

        # Load best model for this fold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Get OOF predictions for this model/fold
        model.eval()
        all_val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast():
                    logits = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                all_val_probs.append(probs.cpu().numpy())

        val_probs = np.concatenate(all_val_probs, axis=0)
        start_col = model_idx * NUM_AUTHORS
        end_col = (model_idx + 1) * NUM_AUTHORS
        fold_oof[:, start_col:end_col] = val_probs

        # Get test predictions for this model/fold
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
        fold_test_probs[:, model_idx, :] = test_probs

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    # Store OOF predictions for this fold
    oof_predictions[val_idx] = fold_oof
    oof_labels[val_idx] = fold_val_labels
    test_predictions[:, fold, :, :] = fold_test_probs

    # Save fold results
    np.save(f"./working/oof_fold_{fold}.npy", fold_oof)
    np.save(f"./working/test_fold_{fold}.npy", fold_test_probs)
    print(f"Fold {fold+1} completed. OOF shape: {fold_oof.shape}")

print("\n" + "="*60)
print("All folds completed. Training meta-learner...")
print("="*60)

# ============================================================
# TRAIN META-LEARNER (Logistic Regression)
# ============================================================
# Clip and normalize OOF predictions
oof_predictions = np.clip(oof_predictions, 1e-15, 1 - 1e-15)
# Normalize each model's predictions to sum to 1
for i in range(len(MODEL_CONFIGS)):
    start = i * NUM_AUTHORS
    end = (i + 1) * NUM_AUTHORS
    row_sums = oof_predictions[:, start:end].sum(axis=1, keepdims=True)
    oof_predictions[:, start:end] = oof_predictions[:, start:end] / row_sums

meta_learner = LogisticRegression(
    C=1.0,
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42,
    tol=1e-4,
)
meta_learner.fit(oof_predictions, oof_labels)

# Evaluate meta-learner on OOF
meta_oof_probs = meta_learner.predict_proba(oof_predictions)
meta_oof_score = log_loss(oof_labels, meta_oof_probs)
print(f"Meta-learner OOF log loss: {meta_oof_score:.6f}")

# ============================================================
# TEST INFERENCE WITH ENSEMBLE AND META-LEARNER
# ============================================================
print("\n" + "="*60)
print("Generating final test predictions...")
print("="*60)

# Average test predictions across folds for each model
# test_predictions shape: (n_test, 5 folds, 3 models, 3 classes)
avg_test_probs = test_predictions.mean(axis=1)  # (n_test, 3 models, 3 classes)

# Flatten model predictions for meta-learner
test_meta_features = avg_test_probs.reshape(len(test_texts), -1)  # (n_test, 9)
test_meta_features = np.clip(test_meta_features, 1e-15, 1 - 1e-15)
for i in range(len(MODEL_CONFIGS)):
    start = i * NUM_AUTHORS
    end = (i + 1) * NUM_AUTHORS
    row_sums = test_meta_features[:, start:end].sum(axis=1, keepdims=True)
    test_meta_features[:, start:end] = test_meta_features[:, start:end] / row_sums

# Apply meta-learner
final_test_probs = meta_learner.predict_proba(test_meta_features)

# Additionally, create a simple average ensemble as a baseline comparison
simple_avg_probs = avg_test_probs.mean(axis=1)  # (n_test, 3 classes)
simple_avg_probs = simple_avg_probs / simple_avg_probs.sum(axis=1, keepdims=True)

# Use meta-learner predictions as final
test_probs = final_test_probs

print(f"Meta-learner test predictions shape: {test_probs.shape}")

# ============================================================
# VALIDATION: Compute average validation score
# ============================================================
# Compute per-model OOF scores
print("\n" + "="*60)
print("Validation Summary:")
print("="*60)
for model_idx, (model_key, config) in enumerate(MODEL_CONFIGS.items()):
    start = model_idx * NUM_AUTHORS
    end = (model_idx + 1) * NUM_AUTHORS
    model_oof = oof_predictions[:, start:end]
    # Normalize
    model_oof = model_oof / model_oof.sum(axis=1, keepdims=True)
    score = log_loss(oof_labels, model_oof)
    print(f"  {model_key}: OOF LogLoss = {score:.6f}")

print(f"  Meta-learner: OOF LogLoss = {meta_oof_score:.6f}")

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

# Final normalization
epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission_3c125689b88a46dcaaf07695194ab0d3.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Submission file: ./submission/submission_3c125689b88a46dcaaf07695194ab0d3.csv")
print(f"\nFinal Meta-learner Validation Score: {meta_oof_score}")