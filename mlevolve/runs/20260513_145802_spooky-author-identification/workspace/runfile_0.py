import os
os.sched_setaffinity(0, {5, 6, 7, 8, 9})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import re
import os
import gc
import warnings
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

warnings.filterwarnings("ignore")

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


def extract_stylometric_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    chars = list(text)
    features["word_count"] = len(words)
    features["char_count"] = len(chars)
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["avg_sentence_length"] = len(words) / features["sentence_count"]
    punctuation_counts = {
        "comma": text.count(","),
        "semicolon": text.count(";"),
        "colon": text.count(":"),
        "exclamation": text.count("!"),
        "question": text.count("?"),
        "period": text.count("."),
        "dash": text.count("-") + text.count("—") + text.count("–"),
        "quote": text.count('"') + text.count('"') + text.count("'"),
        "parentheses": text.count("(") + text.count(")"),
        "ellipsis": text.count("..."),
    }
    total_punct = sum(punctuation_counts.values()) + 1
    for punct, count in punctuation_counts.items():
        features[f"{punct}_density"] = count / total_punct
    features["punctuation_ratio"] = total_punct / max(len(chars), 1)
    caps_words = sum(1 for w in words if w[0].isupper() if w)
    features["caps_ratio"] = caps_words / max(len(words), 1)
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    features["all_caps_ratio"] = all_caps / max(len(words), 1)
    unique_words = set(w.lower() for w in words)
    features["type_token_ratio"] = len(unique_words) / max(len(words), 1)
    from collections import Counter

    word_counts = Counter(w.lower() for w in words)
    hapax = sum(1 for count in word_counts.values() if count == 1)
    features["hapax_ratio"] = hapax / max(len(words), 1)
    if len(sentences) > 0 and len(words) > 0:
        syllables = sum([max(1, len(w) // 3) for w in words])
        features["flesch_reading_ease"] = (
            206.835
            - 1.015 * (len(words) / len(sentences))
            - 84.6 * (syllables / len(words))
        )
        L = (len(chars) / len(words)) * 100
        S = (len(sentences) / len(words)) * 100
        features["coleman_liau"] = 0.0588 * L - 0.296 * S - 15.8
    else:
        features["flesch_reading_ease"] = 0
        features["coleman_liau"] = 0
    function_words = set(
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
            "up",
            "about",
            "into",
            "over",
            "after",
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
            "just",
            "also",
            "as",
            "because",
            "if",
            "then",
            "that",
            "this",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "you",
            "your",
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
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
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
            "must",
            "need",
            "dare",
        ]
    )
    function_word_count = sum(1 for w in words if w.lower() in function_words)
    features["function_word_density"] = function_word_count / max(len(words), 1)
    long_words = sum(1 for w in words if len(w) > 8)
    features["long_word_ratio"] = long_words / max(len(words), 1)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop_count = sum(1 for w in words if w.lower() in ENGLISH_STOP_WORDS)
    features["stopword_ratio"] = stop_count / max(len(words), 1)
    return features


print("Extracting stylometric features...")
train_style_features = train_df["clean_text"].apply(
    lambda x: pd.Series(extract_stylometric_features(x))
)
test_style_features = test_df["clean_text"].apply(
    lambda x: pd.Series(extract_stylometric_features(x))
)

print("Extracting n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)
char_ngrams_train = char_vectorizer.fit_transform(train_df["clean_text"])
char_ngrams_test = char_vectorizer.transform(test_df["clean_text"])

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    min_df=3,
)
word_ngrams_train = word_vectorizer.fit_transform(train_df["clean_text"])
word_ngrams_test = word_vectorizer.transform(test_df["clean_text"])

print("Combining features...")
char_ngrams_train_dense = char_ngrams_train.toarray()
char_ngrams_test_dense = char_ngrams_test.toarray()
word_ngrams_train_dense = word_ngrams_train.toarray()
word_ngrams_test_dense = word_ngrams_test.toarray()

X_train = np.hstack(
    [train_style_features.values, char_ngrams_train_dense, word_ngrams_train_dense]
)
X_test = np.hstack(
    [test_style_features.values, char_ngrams_test_dense, word_ngrams_test_dense]
)

feature_names = (
    list(train_style_features.columns)
    + [f"char_ngram_{i}" for i in range(char_ngrams_train_dense.shape[1])]
    + [f"word_ngram_{i}" for i in range(word_ngrams_train_dense.shape[1])]
)
print(f"Total features created: {len(feature_names)}")
print(f"Training feature matrix shape: {X_train.shape}")
print(f"Test feature matrix shape: {X_test.shape}")

print("Scaling features...")
scaler = StandardScaler()
num_style_cols = train_style_features.shape[1]
X_train[:, :num_style_cols] = scaler.fit_transform(X_train[:, :num_style_cols])
X_test[:, :num_style_cols] = scaler.transform(X_test[:, :num_style_cols])

print("Preparing train/validation split...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df["author"])
class_names = label_encoder.classes_
print(f"Classes: {class_names}")
print(f"Class distribution: {np.bincount(y)}")

train_val_split = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=y
)
train_indices, val_indices = train_val_split

# Refit TF-IDF vectorizers on train split only to prevent data leakage
train_texts_split = train_df.iloc[train_indices]["clean_text"]
val_texts_split = train_df.iloc[val_indices]["clean_text"]

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)
char_ngrams_train = char_vectorizer.fit_transform(train_texts_split)
char_ngrams_val = char_vectorizer.transform(val_texts_split)
char_ngrams_test = char_vectorizer.transform(test_df["clean_text"])

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    min_df=3,
)
word_ngrams_train = word_vectorizer.fit_transform(train_texts_split)
word_ngrams_val = word_vectorizer.transform(val_texts_split)
word_ngrams_test = word_vectorizer.transform(test_df["clean_text"])

print("Recombining features after split...")
char_ngrams_train_dense = char_ngrams_train.toarray()
char_ngrams_val_dense = char_ngrams_val.toarray()
char_ngrams_test_dense = char_ngrams_test.toarray()
word_ngrams_train_dense = word_ngrams_train.toarray()
word_ngrams_val_dense = word_ngrams_val.toarray()
word_ngrams_test_dense = word_ngrams_test.toarray()

train_style_features_split = train_style_features.iloc[train_indices]
val_style_features_split = train_style_features.iloc[val_indices]

X_train_final = np.hstack(
    [train_style_features_split.values, char_ngrams_train_dense, word_ngrams_train_dense]
)
X_val = np.hstack(
    [val_style_features_split.values, char_ngrams_val_dense, word_ngrams_val_dense]
)
X_test = np.hstack(
    [test_style_features.values, char_ngrams_test_dense, word_ngrams_test_dense]
)

# Re-scale style features on train split only
num_style_cols = train_style_features_split.shape[1]
scaler = StandardScaler()
X_train_final[:, :num_style_cols] = scaler.fit_transform(X_train_final[:, :num_style_cols])
X_val[:, :num_style_cols] = scaler.transform(X_val[:, :num_style_cols])
X_test[:, :num_style_cols] = scaler.transform(X_test[:, :num_style_cols])

y_train = y[train_indices]
y_val = y[val_indices]

os.makedirs("./working", exist_ok=True)
np.save("./working/X_train.npy", X_train_final.astype(np.float32))
np.save("./working/X_val.npy", X_val.astype(np.float32))
np.save("./working/X_test.npy", X_test.astype(np.float32))
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)
np.save("./working/train_indices.npy", train_indices)
np.save("./working/val_indices.npy", val_indices)
joblib.dump(feature_names, "./working/feature_names.pkl")
joblib.dump(scaler, "./working/scaler.pkl")
joblib.dump(label_encoder, "./working/label_encoder.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
train_df[["id", "text", "author"]].to_csv("./working/train_text.csv", index=False)
test_df[["id", "text"]].to_csv("./working/test_text.csv", index=False)
train_df.to_pickle("./working/train_df.pkl")
test_df.to_pickle("./working/test_df.pkl")
np.save("./working/X_full.npy", X_train.astype(np.float32))
np.save("./working/y_full.npy", y)

print("Quick validation with Logistic Regression...")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    C=1.0,
    random_state=42,
    n_jobs=-1,
)
lr.fit(X_train_final, y_train)
val_probs_lr = lr.predict_proba(X_val)
val_probs_lr = np.clip(val_probs_lr, 1e-15, 1 - 1e-15)
val_probs_lr = val_probs_lr / val_probs_lr.sum(axis=1, keepdims=True)
score_lr = log_loss(y_val, val_probs_lr)
print(f"Logistic Regression Validation Score: {score_lr:.6f}")

del X_train, X_val, X_test, char_ngrams_train, char_ngrams_test
del (
    char_ngrams_train_dense,
    char_ngrams_test_dense,
    word_ngrams_train_dense,
    word_ngrams_test_dense,
)
gc.collect()
print("Data processing and feature engineering complete!")


# =============================================================================
# MODEL DESIGN: SpookyClassifier
# =============================================================================
class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=150, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        if num_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.head = nn.Linear(hidden_size + 64, num_authors)
        else:
            self.feature_proj = None
            self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_pool = outputs.last_hidden_state[:, 0, :]
        if self.feature_proj is not None and features is not None:
            feat_embed = self.feature_proj(features)
            combined = torch.cat([cls_pool, feat_embed], dim=1)
        else:
            combined = cls_pool
        logits = self.head(combined)
        return logits


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
class SpookyDataset(Dataset):
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
            return_tensors=None,
        )
        item = {
            "input_ids": np.array(encoding["input_ids"]),
            "attention_mask": np.array(encoding["attention_mask"]),
            "features": self.features[idx].astype(np.float32),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
max_length = 256
num_epochs = 30
patience = 5
batch_size = 16
learning_rate_backbone = 2e-5
learning_rate_head = 5e-5
num_features = X_train_final.shape[1]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
fold_test_probs_list = []

test_texts = test_df["clean_text"].values
test_dataset = SpookyDataset(
    texts=test_texts,
    features=np.load("./working/X_test.npy"),
    labels=None,
    tokenizer=tokenizer,
    max_length=max_length,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

print("Starting 5-fold cross-validation training...")
for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.arange(len(train_df)), train_df["author"])
):
    print(f"FOLD {fold + 1}/5")

    train_texts = train_df.iloc[train_idx]["clean_text"].values
    val_texts = train_df.iloc[val_idx]["clean_text"].values
    X_full = np.load("./working/X_full.npy")
    y_full = np.load("./working/y_full.npy")
    train_features = X_full[train_idx]
    val_features = X_full[val_idx]
    train_labels = y_full[train_idx]
    val_labels = y_full[val_idx]

    train_dataset = SpookyDataset(
        texts=train_texts,
        features=train_features,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = SpookyDataset(
        texts=val_texts,
        features=val_features,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    model = SpookyClassifier(num_authors=3, num_features=num_features, dropout_rate=0.3)
    model.to(device)

    backbone_params = [
        p
        for layer in model.backbone.deberta.encoder.layer[-8:]
        for n, p in layer.named_parameters()
        if "bias" not in n and "LayerNorm" not in n
    ]
    head_params = list(model.head.parameters()) + (
        list(model.feature_proj.parameters()) if model.feature_proj else []
    )
    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": learning_rate_backbone,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": learning_rate_head,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
        ]
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_fold_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
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

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = initial_lrs[pg_idx] * (
                        current_step / max(1, warmup_steps)
                    )
            else:
                scheduler.step(epoch + current_step / len(train_loader))

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        model.eval()
        val_probs = []
        val_losses = []
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
                val_losses.append(loss.item())
                val_probs.append(probs.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_probs = np.concatenate(val_probs)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        val_log_loss = log_loss(val_labels, val_probs)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Log Loss: {val_log_loss:.6f}"
        )

        if avg_val_loss < best_fold_val_loss:
            best_fold_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
    val_probs = np.concatenate(val_probs)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    fold_score = log_loss(val_labels, val_probs)
    fold_scores.append(fold_score)
    print(f"Fold {fold + 1} Validation Log Loss: {fold_score:.6f}")

    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())
    fold_test_probs = np.concatenate(fold_test_probs)
    fold_test_probs = np.clip(fold_test_probs, 1e-15, 1 - 1e-15)
    fold_test_probs = fold_test_probs / fold_test_probs.sum(axis=1, keepdims=True)
    fold_test_probs_list.append(fold_test_probs)

    del model, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

final_test_probs = np.mean(fold_test_probs_list, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

mean_val_score = np.mean(fold_scores)
std_val_score = np.std(fold_scores)
print(f"Cross-Validation Results: Mean: {mean_val_score:.6f} ± {std_val_score:.6f}")
print(f"Individual fold scores: {[f'{s:.6f}' for s in fold_scores]}")

print("Creating submission file...")
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_9898e7edee7f4bb3aa4a7547e6f79417.csv", index=False)
print(f"Submission saved to ./submission/submission_9898e7edee7f4bb3aa4a7547e6f79417.csv")
print(f"Submission shape: {submission.shape}")

score = mean_val_score
print(f"Final Validation Score: {score}")