import os
os.sched_setaffinity(0, {11, 12, 13, 14, 15})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import re
import os
import pickle
import string
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ====================== DATA LOADING ======================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode target labels
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_label"] = train_df["author"].map(author_map)


# ====================== TEXT CLEANING ======================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ====================== FEATURE ENGINEERING ======================
def get_word_count(text):
    return len(text.split())


def get_char_count(text):
    return len(text)


def get_avg_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(w) for w in words) / len(words)


def get_punctuation_count(text):
    return sum(1 for c in text if c in string.punctuation)


def get_punctuation_density(text):
    total = len(text)
    if total == 0:
        return 0
    return get_punctuation_count(text) / total


def get_unique_word_ratio(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)


def get_sentence_count(text):
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def get_avg_sentence_length(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def get_exclamation_question_count(text):
    return sum(1 for c in text if c in "!?")


def get_comma_count(text):
    return text.count(",")


def get_quote_count(text):
    return text.count('"') + text.count("'")


def get_dash_count(text):
    return text.count("-") + text.count("—")


def get_semicolon_count(text):
    return text.count(";")


def get_colon_count(text):
    return text.count(":")


def get_contains_digit(text):
    return 1 if any(c.isdigit() for c in text) else 0


def get_uppercase_ratio(text):
    total = sum(1 for c in text if c.isalpha())
    if total == 0:
        return 0
    upper = sum(1 for c in text if c.isupper())
    return upper / total


nltk_stopwords = {
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
    "are",
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
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "not",
    "no",
    "nor",
    "so",
    "if",
    "then",
    "else",
    "when",
    "where",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "why",
    "all",
    "each",
    "every",
    "both",
    "few",
    "many",
    "some",
    "any",
    "more",
    "most",
    "other",
    "such",
    "only",
    "own",
    "same",
    "very",
    "just",
    "also",
    "too",
    "now",
    "here",
    "there",
    "up",
    "down",
    "out",
    "off",
    "over",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "once",
    "than",
    "because",
    "while",
    "though",
    "although",
    "until",
    "since",
    "yet",
}

positive_words = {
    "love",
    "beautiful",
    "wonderful",
    "happy",
    "joy",
    "delight",
    "pleasure",
    "hope",
    "kind",
    "gentle",
    "peace",
}
negative_words = {
    "death",
    "dread",
    "fear",
    "terrible",
    "horror",
    "pain",
    "dark",
    "gloom",
    "sorrow",
    "cruel",
    "fright",
}
supernatural_words = {
    "ghost",
    "spirit",
    "phantom",
    "shadow",
    "monster",
    "demon",
    "devil",
    "witch",
    "curse",
    "haunt",
    "unseen",
    "ancient",
    "beyond",
    "void",
    "abyss",
    "twilight",
    "spectral",
    "supernatural",
}


def engineer_features(df, is_train=True):
    features = pd.DataFrame()
    features["id"] = df["id"].values
    df["cleaned_text"] = df["text"].apply(clean_text)

    features["word_count"] = df["cleaned_text"].apply(get_word_count)
    features["char_count"] = df["cleaned_text"].apply(get_char_count)
    features["avg_word_length"] = df["cleaned_text"].apply(get_avg_word_length)
    features["punctuation_count"] = df["cleaned_text"].apply(get_punctuation_count)
    features["punctuation_density"] = df["cleaned_text"].apply(get_punctuation_density)
    features["unique_word_ratio"] = df["cleaned_text"].apply(get_unique_word_ratio)
    features["stopword_ratio"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w in nltk_stopwords)
        / max(len(x.split()), 1)
    )
    features["sentence_count"] = df["cleaned_text"].apply(get_sentence_count)
    features["avg_sentence_length"] = df["cleaned_text"].apply(get_avg_sentence_length)
    features["exclamation_question_count"] = df["cleaned_text"].apply(
        get_exclamation_question_count
    )
    features["comma_count"] = df["cleaned_text"].apply(get_comma_count)
    features["quote_count"] = df["cleaned_text"].apply(get_quote_count)
    features["dash_count"] = df["cleaned_text"].apply(get_dash_count)
    features["semicolon_count"] = df["cleaned_text"].apply(get_semicolon_count)
    features["colon_count"] = df["cleaned_text"].apply(get_colon_count)
    features["contains_digit"] = df["cleaned_text"].apply(get_contains_digit)
    features["word_count_squared"] = features["word_count"] ** 2
    features["char_per_word"] = np.where(
        features["word_count"] > 0, features["char_count"] / features["word_count"], 0
    )
    features["uppercase_ratio"] = df["text"].apply(get_uppercase_ratio)

    # Pronoun, article, preposition, conjunction counts
    features["pronoun_count"] = df["cleaned_text"].apply(
        lambda x: sum(
            1
            for w in x.split()
            if w.lower()
            in {
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
            }
        )
    )
    features["article_count"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w.lower() in {"a", "an", "the"})
    )
    features["preposition_count"] = df["cleaned_text"].apply(
        lambda x: sum(
            1
            for w in x.split()
            if w.lower()
            in {
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "about",
                "into",
                "through",
                "during",
                "between",
            }
        )
    )
    features["conjunction_count"] = df["cleaned_text"].apply(
        lambda x: sum(
            1
            for w in x.split()
            if w.lower()
            in {
                "and",
                "but",
                "or",
                "nor",
                "yet",
                "so",
                "for",
                "because",
                "while",
                "although",
                "since",
                "unless",
            }
        )
    )

    features["pronoun_ratio"] = features["pronoun_count"] / (features["word_count"] + 1)
    features["article_ratio"] = features["article_count"] / (features["word_count"] + 1)
    features["preposition_ratio"] = features["preposition_count"] / (
        features["word_count"] + 1
    )
    features["conjunction_ratio"] = features["conjunction_count"] / (
        features["word_count"] + 1
    )

    # First word indicators
    features["first_word_is_the"] = df["cleaned_text"].apply(
        lambda x: 1 if len(x.split()) > 0 and x.split()[0] == "the" else 0
    )
    features["first_word_is_i"] = df["cleaned_text"].apply(
        lambda x: 1 if len(x.split()) > 0 and x.split()[0] == "i" else 0
    )
    features["first_word_is_a"] = df["cleaned_text"].apply(
        lambda x: 1 if len(x.split()) > 0 and x.split()[0] == "a" else 0
    )
    features["first_word_is_my"] = df["cleaned_text"].apply(
        lambda x: 1 if len(x.split()) > 0 and x.split()[0] == "my" else 0
    )

    # End punctuation
    features["ends_with_period"] = df["text"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip().endswith(".") else 0
    )
    features["ends_with_exclamation"] = df["text"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip().endswith("!") else 0
    )
    features["ends_with_question"] = df["text"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip().endswith("?") else 0
    )

    # Capitalization
    features["capitalized_word_ratio"] = df["text"].apply(
        lambda x: sum(1 for w in str(x).split() if w[0].isupper())
        / max(len(str(x).split()), 1)
    )
    features["all_caps_word_count"] = df["text"].apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
    )

    # Sentiment and supernatural words
    features["positive_word_count"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w in positive_words)
    )
    features["negative_word_count"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w in negative_words)
    )
    features["sentiment_ratio"] = (
        features["positive_word_count"] - features["negative_word_count"]
    ) / (features["word_count"] + 1)
    features["supernatural_word_count"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w in supernatural_words)
    )
    features["contains_et"] = df["cleaned_text"].apply(
        lambda x: 1 if " et " in x else 0
    )

    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)

    feature_cols = [c for c in features.columns if c != "id"]
    X = features[feature_cols].values
    return X, feature_cols


# Engineer features
X_train_all, feature_names = engineer_features(train_df, is_train=True)
y_train_all = train_df["author_label"].values
X_test, _ = engineer_features(test_df, is_train=False)

print(f"Feature matrix shape (train): {X_train_all.shape}")
print(f"Number of features: {len(feature_names)}")

# Split into train/val (80/20)
train_texts, val_texts, train_labels, val_labels, train_idx, val_idx = train_test_split(
    train_df["text"].values,
    y_train_all,
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=y_train_all,
)

X_train = X_train_all[train_idx]
X_val = X_train_all[val_idx]
test_texts = test_df["text"].values
test_ids = test_df["id"].values

print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

# ====================== MODEL DEFINITION ======================
num_handcrafted_features = len(feature_names)


class HybridStyleClassifier(nn.Module):
    def __init__(
        self, num_labels=3, num_handcrafted=num_handcrafted_features, dropout_rate=0.2
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
        self.config.num_labels = num_labels
        self.config.output_hidden_states = True

        self.deberta = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large", config=self.config
        )
        hidden_size = self.config.hidden_size

        self.style_projection = nn.Sequential(
            nn.Linear(num_handcrafted, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        fusion_input_size = hidden_size + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.style_projection, self.fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        style_features=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        cls_embedding = sequence_output[:, 0, :]

        if style_features is not None:
            style_embedding = self.style_projection(style_features)
        else:
            style_embedding = torch.zeros(
                cls_embedding.shape[0], 128, device=cls_embedding.device
            )

        combined = torch.cat([cls_embedding, style_embedding], dim=1)
        logits = self.fusion(combined)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits


# ====================== DATASET ======================
class SpookyAuthorsDataset(Dataset):
    def __init__(self, texts, features, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.features = features
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
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "style_features": torch.FloatTensor(self.features[idx]),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ====================== METRIC ======================
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / N
    return loss


# ====================== TRAINING SETUP ======================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HybridStyleClassifier(
    num_labels=3, num_handcrafted=num_handcrafted_features, dropout_rate=0.2
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

# Datasets and loaders
train_dataset = SpookyAuthorsDataset(
    texts=train_texts,
    features=X_train,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=256,
)
val_dataset = SpookyAuthorsDataset(
    texts=val_texts,
    features=X_val,
    labels=val_labels,
    tokenizer=tokenizer,
    max_length=256,
)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

# Optimizer with layer-wise learning rates
no_decay = [
    "bias",
    "LayerNorm.weight",
    "LayerNorm.bias",
    "layernorm.weight",
    "layernorm.bias",
]
deberta_params = list(model.deberta.named_parameters())
deberta_decay = [p for n, p in deberta_params if not any(nd in n for nd in no_decay)]
deberta_no_decay = [p for n, p in deberta_params if any(nd in n for nd in no_decay)]
classifier_params = list(model.style_projection.parameters()) + list(
    model.fusion.parameters()
)

optimizer = AdamW(
    [
        {"params": deberta_decay, "weight_decay": 0.01, "lr": 2e-5},
        {"params": deberta_no_decay, "weight_decay": 0.0, "lr": 2e-5},
        {"params": classifier_params, "weight_decay": 0.01, "lr": 5e-5},
    ],
    eps=1e-8,
)

num_training_steps = len(train_loader) * 30
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)
scaler = GradScaler()

# ====================== TRAINING LOOP ======================
best_val_loss = float("inf")
patience = 5
patience_counter = 0

print("\nStarting training...")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print("-" * 60)

for epoch in range(30):
    model.train()
    train_loss = 0.0
    train_batches = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        style_features = batch["style_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                style_features=style_features,
                labels=labels,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_batches += 1

    avg_train_loss = train_loss / train_batches

    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            style_features = batch["style_features"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    style_features=style_features,
                    labels=labels,
                )

            val_loss += loss.item()
            val_batches += 1
            probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_true_labels = np.concatenate(all_val_labels, axis=0)
    val_true_onehot = np.zeros((val_true_labels.shape[0], 3))
    val_true_onehot[np.arange(val_true_labels.shape[0]), val_true_labels] = 1
    val_logloss = multiclass_log_loss(val_true_onehot, val_preds)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/30 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {current_lr:.2e}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(
            {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "./working/best_model_00a575517dd64e74a6c4d4aa38d9d570.pt",
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print(f"\nBest validation loss: {best_val_loss:.4f}")

# ====================== FINAL TRAINING ON FULL DATA ======================
print("\nTraining final model on full dataset...")
full_dataset = SpookyAuthorsDataset(
    texts=train_df["text"].values,
    features=X_train_all,
    labels=y_train_all,
    tokenizer=tokenizer,
    max_length=256,
)
full_loader = DataLoader(
    full_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)

final_model = HybridStyleClassifier(
    num_labels=3, num_handcrafted=num_handcrafted_features, dropout_rate=0.2
)
final_model.to(device)
final_model.load_state_dict(torch.load("./working/best_model_00a575517dd64e74a6c4d4aa38d9d570.pt"))

final_deberta_params = list(final_model.deberta.named_parameters())
final_deberta_decay = [p for n, p in final_deberta_params if not any(nd in n for nd in no_decay)]
final_deberta_no_decay = [p for n, p in final_deberta_params if any(nd in n for nd in no_decay)]
final_classifier_params = list(final_model.style_projection.parameters()) + list(
    final_model.fusion.parameters()
)

final_optimizer = AdamW(
    [
        {"params": final_deberta_decay, "weight_decay": 0.01, "lr": 1e-5},
        {"params": final_deberta_no_decay, "weight_decay": 0.0, "lr": 1e-5},
        {"params": final_classifier_params, "weight_decay": 0.01, "lr": 3e-5},
    ],
    eps=1e-8,
)
final_scaler = GradScaler()

for epoch in range(10):
    final_model.train()
    total_loss = 0.0
    batches = 0
    for batch in full_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        style_features = batch["style_features"].to(device)
        labels = batch["labels"].to(device)

        final_optimizer.zero_grad()
        with autocast():
            loss, logits = final_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                style_features=style_features,
                labels=labels,
            )

        final_scaler.scale(loss).backward()
        final_scaler.unscale_(final_optimizer)
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        final_scaler.step(final_optimizer)
        final_scaler.update()

        total_loss += loss.item()
        batches += 1
    avg_loss = total_loss / batches
    print(f"Full training - Epoch {epoch+1}/10 | Loss: {avg_loss:.4f}")

# ====================== TEST INFERENCE ======================
print("\nPerforming test inference...")
test_dataset = SpookyAuthorsDataset(
    texts=test_texts, features=X_test, labels=None, tokenizer=tokenizer, max_length=256
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

final_model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        style_features = batch["style_features"].to(device)

        with autocast():
            logits = final_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                style_features=style_features,
            )
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)

# ====================== FINAL VALIDATION SCORE ======================
# Recompute validation score with final model (loaded from best checkpoint)
final_model.eval()
all_val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        style_features = batch["style_features"].to(device)
        with autocast():
            logits = final_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                style_features=style_features,
            )
        probs = torch.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())

val_preds = np.concatenate(all_val_preds, axis=0)
val_true_onehot = np.zeros((val_labels.shape[0], 3))
val_true_onehot[np.arange(val_labels.shape[0]), val_labels] = 1
final_val_score = multiclass_log_loss(val_true_onehot, val_preds)

# ====================== SUBMISSION ======================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission_00a575517dd64e74a6c4d4aa38d9d570.csv", index=False)
print(f"Submission saved to ./submission/submission_00a575517dd64e74a6c4d4aa38d9d570.csv")
print(f"Submission shape: {submission.shape}")
print(f"Final Validation Score: {final_val_score}")