import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, DistilBertModel, DistilBertConfig
import xgboost as xgb
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Encode target
le = LabelEncoder()
train["author_encoded"] = le.fit_transform(train["author"])
num_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

# Create train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train["text"], train["author_encoded"])))

train_df = train.iloc[train_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def extract_features(text_series, is_train=True):
    features = pd.DataFrame(index=text_series.index)

    # Basic text stats
    features["char_count"] = text_series.apply(len)
    features["word_count"] = text_series.apply(lambda x: len(str(x).split()))
    features["sent_count"] = text_series.apply(
        lambda x: len(re.split(r"[.!?]+", str(x))) - 1
    )
    features["avg_word_len"] = text_series.apply(
        lambda x: (
            np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
    )

    # Punctuation features
    features["excl_count"] = text_series.apply(lambda x: str(x).count("!"))
    features["quest_count"] = text_series.apply(lambda x: str(x).count("?"))
    features["period_count"] = text_series.apply(lambda x: str(x).count("."))
    features["comma_count"] = text_series.apply(lambda x: str(x).count(","))
    features["semi_count"] = text_series.apply(lambda x: str(x).count(";"))
    features["colon_count"] = text_series.apply(lambda x: str(x).count(":"))
    features["quote_count"] = text_series.apply(
        lambda x: str(x).count('"') + str(x).count("'")
    )
    features["dash_count"] = text_series.apply(lambda x: str(x).count("-"))

    # Caps and special patterns
    features["caps_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    features["caps_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    features["digit_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isdigit())
    )

    # Readability features
    features["syllable_est"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).split()
            for c in ["a", "e", "i", "o", "u", "y"]
            if c in w.lower()
        )
    )
    features["complex_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 6)
    )

    # Quote density
    features["quote_density"] = text_series.apply(
        lambda x: str(x).count('"') / max(len(str(x)), 1)
    )

    # Stopword-based features
    stopwords = {
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
        "by",
        "with",
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
    }

    features["stopword_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in stopwords)
    )
    features["stopword_ratio"] = features["stopword_count"] / features[
        "word_count"
    ].replace(0, 1)

    # Word length distribution
    features["long_word_ratio"] = features["complex_words"] / features[
        "word_count"
    ].replace(0, 1)
    features["short_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) <= 3)
    ) / features["word_count"].replace(0, 1)

    # Vocabulary diversity
    features["unique_words"] = text_series.apply(
        lambda x: len(set(str(x).lower().split()))
    )
    features["type_token_ratio"] = features["unique_words"] / features[
        "word_count"
    ].replace(0, 1)

    # Historical/archaic word indicators
    archaic_words = {
        "thee",
        "thou",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "art",
        "hast",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "nay",
        "forsooth",
        "perchance",
        "anon",
        "prithee",
    }
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "non-euclidean",
        "ichor",
        "gibbering",
        "maddening",
        "cosmic",
        "carcosa",
        "yog-sothoth",
        "cthulhu",
    }
    poe_words = {
        "nevermore",
        "chamber",
        "tapping",
        "rapping",
        "sepulchre",
        "ghoul",
        "pallid",
        "dreary",
        "weary",
        "ebony",
    }

    features["archaic_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in archaic_words)
    )
    features["lovecraft_score"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in lovecraft_words)
    )
    features["poe_score"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in poe_words)
    )

    # First person pronoun usage
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    features["first_person_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in first_person)
    )

    # Sentence complexity
    features["avg_sent_length"] = features["word_count"] / features[
        "sent_count"
    ].replace(0, 1)

    # Character diversity
    features["unique_chars"] = text_series.apply(lambda x: len(set(str(x).lower())))
    features["char_diversity"] = features["unique_chars"] / features[
        "char_count"
    ].replace(0, 1)

    # Negation count
    negation_words = {
        "not",
        "no",
        "never",
        "nothing",
        "nowhere",
        "none",
        "neither",
        "nor",
        "cannot",
        "can't",
        "don't",
        "won't",
        "doesn't",
        "isn't",
    }
    features["negation_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in negation_words)
    )

    # Emotion/affect words
    positive_words = set(
        [
            "love",
            "beautiful",
            "wonderful",
            "happy",
            "joy",
            "delight",
            "pleasure",
            "gentle",
            "sweet",
            "tender",
            "bright",
            "hope",
        ]
    )
    negative_words = set(
        [
            "dark",
            "dread",
            "fear",
            "horror",
            "terror",
            "death",
            "shadow",
            "gloom",
            "pain",
            "sorrow",
            "misery",
            "agony",
            "anguish",
            "doom",
        ]
    )

    features["positive_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in positive_words)
    )
    features["negative_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in negative_words)
    )
    features["sentiment_ratio"] = (
        features["positive_count"] - features["negative_count"]
    ) / (features["positive_count"] + features["negative_count"] + 1)

    return features


# Extract features
train_features = extract_features(train_df["text"], is_train=True)
val_features = extract_features(val_df["text"], is_train=False)
test_features = extract_features(test["text"], is_train=False)

# Handle NaN and inf values
train_features = train_features.replace([np.inf, -np.inf], 0).fillna(0)
val_features = val_features.replace([np.inf, -np.inf], 0).fillna(0)
test_features = test_features.replace([np.inf, -np.inf], 0).fillna(0)

# Add percentile features
for df_features, df in [
    (train_features, train_df),
    (val_features, val_df),
    (test_features, test),
]:
    df_features["word_count_rank"] = df_features["word_count"].rank(pct=True)
    df_features["char_count_rank"] = df_features["char_count"].rank(pct=True)


# N-gram features
def create_ngram_count(text, n=2, ngram_type="char"):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    if ngram_type == "char":
        common_ngrams = [
            "th",
            "he",
            "in",
            "er",
            "an",
            "re",
            "nd",
            "at",
            "on",
            "nt",
            "ha",
            "ou",
            "it",
            "hi",
            "es",
            "st",
            "en",
            "ea",
            "to",
            "or",
            "ed",
            "te",
            "ar",
            "al",
            "le",
            "ve",
            "ti",
            "ra",
            "ur",
            "me",
        ]
        return sum(text.count(ng) for ng in common_ngrams) / max(len(text), 1)
    else:
        words = text.split()
        common_bigrams = [
            ("of", "the"),
            ("in", "the"),
            ("to", "the"),
            ("and", "the"),
            ("it", "was"),
            ("i", "was"),
            ("there", "was"),
            ("this", "was"),
        ]
        count = 0
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            if bigram in common_bigrams:
                count += 1
        return count / max(len(words), 1)


for prefix, df in [("train", train_df), ("val", val_df), ("test", test)]:
    text_data = df["text"]
    ngram_char = text_data.apply(lambda x: create_ngram_count(x, 2, "char"))
    ngram_word = text_data.apply(lambda x: create_ngram_count(x, 2, "word"))
    if prefix == "train":
        train_features["ngram_char_density"] = ngram_char.values
        train_features["ngram_word_density"] = ngram_word.values
    elif prefix == "val":
        val_features["ngram_char_density"] = ngram_char.values
        val_features["ngram_word_density"] = ngram_word.values
    else:
        test_features["ngram_char_density"] = ngram_char.values
        test_features["ngram_word_density"] = ngram_word.values

# Scale features
scaler = StandardScaler()
feature_columns = train_features.columns

train_features_scaled = pd.DataFrame(
    scaler.fit_transform(train_features),
    columns=feature_columns,
    index=train_features.index,
)
val_features_scaled = pd.DataFrame(
    scaler.transform(val_features), columns=feature_columns, index=val_features.index
)
test_features_scaled = pd.DataFrame(
    scaler.transform(test_features), columns=feature_columns, index=test_features.index
)

# Combine with original data
train_processed = pd.concat(
    [
        train_df[["id", "text", "author", "author_encoded"]].reset_index(drop=True),
        train_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)
val_processed = pd.concat(
    [
        val_df[["id", "text", "author", "author_encoded"]].reset_index(drop=True),
        val_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)
test_processed = pd.concat(
    [
        test[["id", "text"]].reset_index(drop=True),
        test_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed val shape: {val_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# Extract arrays
X_train_tab = train_processed[feature_columns].values
y_train = train_processed["author_encoded"].values
X_val_tab = val_processed[feature_columns].values
y_val = val_processed["author_encoded"].values
X_test_tab = test_processed[feature_columns].values

train_texts = train_processed["text"].tolist()
val_texts = val_processed["text"].tolist()
test_texts = test_processed["text"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# TRAIN XGBoost
# ============================================================
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)
xgb_val_probs = xgb_model.predict_proba(X_val_tab)
xgb_test_probs = xgb_model.predict_proba(X_test_tab)
print(f"XGBoost val log loss: {log_loss(y_val, xgb_val_probs):.6f}")

# ============================================================
# TRAIN LightGBM
# ============================================================
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_leaves=31,
    objective="multiclass",
    metric="multi_logloss",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_tab,
    y_train,
    eval_set=[(X_val_tab, y_val)],
    callbacks=[lgb.early_stopping(30)],
)
lgb_val_probs = lgb_model.predict_proba(X_val_tab)
lgb_test_probs = lgb_model.predict_proba(X_test_tab)
print(f"LightGBM val log loss: {log_loss(y_val, lgb_val_probs):.6f}")


# ============================================================
# TEXT ENCODERS (DistilBERT and RoBERTa)
# ============================================================
class TextFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        hidden_dim=512,
        num_classes=3,
        dropout=0.5,
    ):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        logits = self.classifier(pooled)
        return logits


class RoBERTaEncoder(nn.Module):
    def __init__(
        self,
        model_name="roberta-base",
        hidden_dim=384,
        num_classes=3,
        dropout=0.5,
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        logits = self.classifier(pooled)
        return logits


tokenizer_distilbert = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
text_encoder = TextFeatureEncoder(num_classes=num_classes, dropout=0.5).to(device)
roberta_encoder = RoBERTaEncoder(num_classes=num_classes, dropout=0.5).to(device)


# Dataset
# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, tokenizer, train_texts, train_labels, val_texts, val_labels, test_texts, model_name, lr=2e-5, num_epochs=10, patience=3):
    # Datasets
    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoded = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            item = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(test_texts)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer with weight_decay=0.05
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler_grad = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"Training {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            scaler_grad.scale(loss).backward()
            scaler_grad.step(optimizer)
            scaler_grad.update()
            scheduler.step(epoch + num_batches / len(train_loader))
            total_loss += loss.item()
            num_batches += 1
        avg_train_loss = total_loss / num_batches

        model.eval()
        val_loss = 0.0
        val_num_batches = 0
        all_val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_num_batches += 1
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
        avg_val_loss = val_loss / val_num_batches
        val_probs = np.concatenate(all_val_preds, axis=0)
        val_log_loss_val = log_loss(val_labels, val_probs)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Log Loss: {val_log_loss_val:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  -> New best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best {model_name} with val_loss={best_val_loss:.6f}")

    # Generate predictions
    model.eval()
    all_val_probs = []
    all_test_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu().numpy())

    val_probs = np.concatenate(all_val_probs, axis=0)
    test_probs = np.concatenate(all_test_probs, axis=0)
    return val_probs, test_probs, best_val_loss

# Train DistilBERT
distilbert_val_probs, distilbert_test_probs, distilbert_val_loss = train_model(
    text_encoder, tokenizer_distilbert, train_texts, y_train, val_texts, y_val, test_texts,
    model_name="DistilBERT", lr=2e-5
)

# Train RoBERTa
roberta_val_probs, roberta_test_probs, roberta_val_loss = train_model(
    roberta_encoder, tokenizer_roberta, train_texts, y_train, val_texts, y_val, test_texts,
    model_name="RoBERTa", lr=2e-5
)

# ============================================================
# WEIGHTED ENSEMBLE
# ============================================================
print("\nOptimizing ensemble weights...")
best_weight = 0.1
best_ensemble_val_loss = float("inf")
weights = np.arange(0.1, 1.0, 0.1)
for w in weights:
    ensemble_val_probs = w * distilbert_val_probs + (1 - w) * roberta_val_probs
    ensemble_val_loss = log_loss(y_val, ensemble_val_probs)
    if ensemble_val_loss < best_ensemble_val_loss:
        best_ensemble_val_loss = ensemble_val_loss
        best_weight = w
    print(f"  Weight DistilBERT={w:.1f}, RoBERTa={1-w:.1f} -> Val Log Loss: {ensemble_val_loss:.6f}")

print(f"Best weight: DistilBERT={best_weight:.1f}, RoBERTa={1-best_weight:.1f} with Val Log Loss: {best_ensemble_val_loss:.6f}")

# Final test predictions
test_probs_final = best_weight * distilbert_test_probs + (1 - best_weight) * roberta_test_probs

# Clip and normalize
eps = 1e-15
test_probs_final = np.clip(test_probs_final, eps, 1 - eps)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# Create submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_processed["id"].values,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Test predictions shape: {submission.shape}")
print(submission.head())
