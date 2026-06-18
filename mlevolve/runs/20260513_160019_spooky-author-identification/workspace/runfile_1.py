import os
os.sched_setaffinity(0, {163, 164})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import re
import string
from collections import Counter
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)


# ============================================================
# 3. FEATURE ENGINEERING - Stylometric & Linguistic Features
# ============================================================
def extract_stylometric_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chars = list(text)

    n_words = len(words)
    n_chars = len(chars)
    n_sentences = max(len(sentences), 1)
    n_unique_words = len(set(w.lower() for w in words))

    features["word_count"] = n_words
    features["char_count"] = n_chars
    features["sentence_count"] = n_sentences
    features["avg_word_length"] = n_chars / max(n_words, 1)
    features["avg_sentence_length_words"] = n_words / n_sentences
    features["avg_sentence_length_chars"] = n_chars / n_sentences
    features["std_word_length"] = np.std([len(w) for w in words]) if n_words > 1 else 0
    features["max_word_length"] = max([len(w) for w in words]) if words else 0
    features["min_word_length"] = min([len(w) for w in words]) if words else 0

    features["unique_word_ratio"] = n_unique_words / max(n_words, 1)
    features["hapax_legomena"] = sum(
        1 for w in Counter(w.lower() for w in words).values() if w == 1
    ) / max(n_words, 1)
    features["hapax_dislegomena"] = sum(
        1 for w in Counter(w.lower() for w in words).values() if w == 2
    ) / max(n_words, 1)

    punct_counts = {
        "period_count": text.count("."),
        "comma_count": text.count(","),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "dash_count": text.count("-") + text.count("—") + text.count("–"),
        "quote_count": text.count('"')
        + text.count('"')
        + text.count('"')
        + text.count('"')
        + text.count("'")
        + text.count('"'),
        "parenthesis_count": text.count("(") + text.count(")"),
        "apostrophe_count": text.count("'"),
        "ellipsis_count": text.count("..."),
    }
    features.update(punct_counts)

    features["comma_per_word"] = punct_counts["comma_count"] / max(n_words, 1)
    features["semicolon_per_word"] = punct_counts["semicolon_count"] / max(n_words, 1)
    features["exclamation_per_word"] = punct_counts["exclamation_count"] / max(
        n_words, 1
    )
    features["dash_per_word"] = punct_counts["dash_count"] / max(n_words, 1)
    features["quote_per_word"] = punct_counts["quote_count"] / max(n_words, 1)

    features["capitalized_word_ratio"] = sum(1 for w in words if w[0].isupper()) / max(
        n_words, 1
    )
    features["all_caps_word_ratio"] = sum(
        1 for w in words if w.isupper() and len(w) > 1
    ) / max(n_words, 1)

    syllables = sum([max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words])
    features["syllable_count"] = syllables
    features["syllables_per_word"] = syllables / max(n_words, 1)
    features["complex_word_ratio"] = sum(
        1 for w in words if len(re.findall(r"[aeiouy]+", w.lower())) >= 3
    ) / max(n_words, 1)

    char_bigrams = [text[i : i + 2].lower() for i in range(len(text) - 1)]
    common_bigrams = [
        "th",
        "he",
        "in",
        "er",
        "an",
        "re",
        "ed",
        "on",
        "es",
        "st",
        "en",
        "at",
        "to",
        "nt",
        "ha",
        "nd",
        "ou",
        "ea",
        "ng",
        "hi",
    ]
    for bg in common_bigrams:
        features[f"bigram_{bg}"] = char_bigrams.count(bg) / max(len(char_bigrams), 1)

    stop_words = {
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
        "had",
        "have",
        "has",
        "been",
        "being",
        "be",
        "are",
        "am",
        "it",
        "its",
        "that",
        "which",
        "who",
        "whom",
        "this",
        "these",
        "those",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "very",
        "just",
        "all",
        "any",
        "each",
        "every",
        "some",
        "such",
        "more",
        "most",
        "other",
        "only",
        "own",
        "same",
        "too",
        "very",
    }
    features["stop_word_ratio"] = sum(
        1 for w in words if w.lower() in stop_words
    ) / max(n_words, 1)

    if sentences:
        start_words = [
            s.split()[0].lower() if s.split() else "" for s in sentences if s
        ]
        features["avg_sentence_start_word_length"] = (
            np.mean([len(w) for w in start_words]) if start_words else 0
        )
        common_starts = [
            "the",
            "a",
            "an",
            "i",
            "he",
            "she",
            "it",
            "we",
            "they",
            "this",
            "that",
            "there",
            "here",
            "my",
            "his",
            "her",
            "and",
            "but",
            "or",
            "so",
            "for",
            "yet",
            "not",
            "if",
            "when",
            "where",
            "how",
            "what",
            "why",
            "who",
        ]
        for cs in common_starts:
            features[f"starts_with_{cs}"] = sum(
                1 for w in start_words if w == cs
            ) / max(len(start_words), 1)
    else:
        features["avg_sentence_start_word_length"] = 0

    features["has_em_dash"] = 1 if "—" in text or "--" in text else 0
    features["has_ellipsis"] = 1 if "..." in text else 0
    features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(n_chars, 1)
    features["special_char_ratio"] = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / max(n_chars, 1)

    past_tense_aux = {"was", "were", "had", "did", "would", "could", "should", "might"}
    present_tense_aux = {
        "is",
        "are",
        "am",
        "has",
        "have",
        "do",
        "does",
        "will",
        "can",
        "shall",
        "may",
    }
    features["past_aux_ratio"] = sum(
        1 for w in words if w.lower() in past_tense_aux
    ) / max(n_words, 1)
    features["present_aux_ratio"] = sum(
        1 for w in words if w.lower() in present_tense_aux
    ) / max(n_words, 1)

    first_person = {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
    }
    third_person = {
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
    }
    features["first_person_pronoun_ratio"] = sum(
        1 for w in words if w.lower() in first_person
    ) / max(n_words, 1)
    features["third_person_pronoun_ratio"] = sum(
        1 for w in words if w.lower() in third_person
    ) / max(n_words, 1)

    coord_conj = {"and", "but", "or", "nor", "for", "yet", "so"}
    features["coord_conjunction_ratio"] = sum(
        1 for w in words if w.lower() in coord_conj
    ) / max(n_words, 1)

    subord_conj = {
        "after",
        "although",
        "as",
        "because",
        "before",
        "how",
        "if",
        "once",
        "since",
        "than",
        "that",
        "though",
        "till",
        "unless",
        "until",
        "when",
        "where",
        "whether",
        "while",
        "why",
        "like",
    }
    features["subord_conjunction_ratio"] = sum(
        1 for w in words if w.lower() in subord_conj
    ) / max(n_words, 1)

    archaic_words = {
        "thy",
        "thou",
        "thee",
        "thine",
        "ye",
        "hath",
        "doth",
        "dost",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "betwixt",
        "amongst",
        "whilst",
        "perchance",
        "foremost",
        "dwell",
        "dwelt",
        "gloom",
        "eldritch",
        "unnameable",
        "unutterable",
        "cyclopean",
        "non-euclidean",
        "antediluvian",
        "primordial",
    }
    features["archaic_word_count"] = sum(1 for w in words if w.lower() in archaic_words)
    features["archaic_word_ratio"] = features["archaic_word_count"] / max(n_words, 1)

    return features


print("Extracting stylometric features...")
train_features = train_df["text"].apply(extract_stylometric_features)
test_features = test_df["text"].apply(extract_stylometric_features)

train_features_df = pd.DataFrame(train_features.tolist())
test_features_df = pd.DataFrame(test_features.tolist())

train_features_df.insert(0, "id", train_df["id"].values)
test_features_df.insert(0, "id", test_df["id"].values)
train_features_df["author"] = train_df["author"].values

train_features_df = train_features_df.replace([np.inf, -np.inf], np.nan)
test_features_df = test_features_df.replace([np.inf, -np.inf], np.nan)

numeric_cols = train_features_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = train_features_df[col].median()
    train_features_df[col] = train_features_df[col].fillna(median_val)
    test_features_df[col] = test_features_df[col].fillna(median_val)

print("Performing feature selection...")
feature_cols = [c for c in numeric_cols if c not in ["id"]]
X = train_features_df[feature_cols].values
y = train_features_df["author"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
mi_df = pd.DataFrame({"feature": feature_cols, "mi_score": mi_scores})
mi_df = mi_df.sort_values("mi_score", ascending=False)

top_features = mi_df[mi_df["mi_score"] > 0]["feature"].tolist()
if len(top_features) > 100:
    top_features = mi_df.head(100)["feature"].tolist()
print(
    f"Selected {len(top_features)} informative features out of {len(feature_cols)} total"
)

train_selected = train_features_df[["id"] + top_features + ["author"]]
test_selected = test_features_df[["id"] + top_features]

print("Scaling features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_selected[top_features])
train_selected[top_features] = scaled_features
scaled_test_features = scaler.transform(test_selected[top_features])
test_selected[top_features] = scaled_test_features

print("Creating stratified splits...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_selected, y_encoded))[0]

train_split = train_selected.iloc[train_idx].reset_index(drop=True)
val_split = train_selected.iloc[val_idx].reset_index(drop=True)
test_final = test_selected.reset_index(drop=True)

print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_final)}")

os.makedirs("./working", exist_ok=True)
train_split.to_parquet("./working/train_processed.parquet", index=False)
val_split.to_parquet("./working/val_processed.parquet", index=False)
test_final.to_parquet("./working/test_processed.parquet", index=False)

os.makedirs("./working/models", exist_ok=True)
joblib.dump(le, "./working/models/label_encoder.pkl")
joblib.dump(scaler, "./working/models/scaler.pkl")
joblib.dump(top_features, "./working/models/selected_features.pkl")


# ============================================================
# CONFIG
# ============================================================
class Config:
    model_name = "microsoft/deberta-v3-large"
    num_authors = 3
    max_length = 256
    dropout_rate = 0.3
    num_stylometric_features = len(top_features)
    batch_size = 16
    weight_decay = 0.01
    label_smoothing = 0.1


config = Config()


# ============================================================
# MODEL DEFINITION (from Step 2 template - SpookyClassifier)
# ============================================================
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


# ============================================================
# TOKENIZER AND DATASET
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(config.model_name)


class SpookyDataset(Dataset):
    def __init__(self, texts, features=None, labels=None):
        self.texts = texts
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = tokenizer(
            text,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# FEATURE EXTRACTOR FOR GBDT
# ============================================================
class FeatureExtractor:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_authors,
            output_hidden_states=True,
        ).to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def extract_embeddings(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.backbone.deberta(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


# ============================================================
# ENSEMBLE WEIGHTS OPTIMIZER
# ============================================================
def optimize_ensemble_weights(val_preds_dict, val_labels):
    from sklearn.metrics import log_loss

    best_score = float("inf")
    best_weights = {"xgb": 0.33, "lgb": 0.33, "transformer": 0.34}
    for w1 in np.arange(0.0, 1.05, 0.1):
        for w2 in np.arange(0.0, 1.05 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            for dw1 in [-0.05, 0, 0.05]:
                for dw2 in [-0.05, 0, 0.05]:
                    fw1 = w1 + dw1
                    fw2 = w2 + dw2
                    fw3 = 1.0 - fw1 - fw2
                    if fw1 < 0 or fw2 < 0 or fw3 < 0:
                        continue
                    ensemble = (
                        fw1 * val_preds_dict["xgb"]
                        + fw2 * val_preds_dict["lgb"]
                        + fw3 * val_preds_dict["transformer"]
                    )
                    ensemble = np.clip(ensemble, 1e-15, 1 - 1e-15)
                    ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
                    score = log_loss(val_labels, ensemble)
                    if score < best_score:
                        best_score = score
                        best_weights = {"xgb": fw1, "lgb": fw2, "transformer": fw3}
    return best_weights, best_score


# ============================================================
# PREPARE DATA FOR TRAINING
# ============================================================
train_processed = pd.read_parquet("./working/train_processed.parquet")
val_processed = pd.read_parquet("./working/val_processed.parquet")
test_processed = pd.read_parquet("./working/test_processed.parquet")
label_encoder = joblib.load("./working/models/label_encoder.pkl")
scaler = joblib.load("./working/models/scaler.pkl")
selected_features = joblib.load("./working/models/selected_features.pkl")

config.num_stylometric_features = len(selected_features)

# Map back to original text using ids
train_texts = train_df.loc[train_df["id"].isin(train_processed["id"]), "text"].values
val_texts = train_df.loc[train_df["id"].isin(val_processed["id"]), "text"].values

X_train_features = train_processed[selected_features].values.astype(np.float32)
X_val_features = val_processed[selected_features].values.astype(np.float32)
X_test_features = test_processed[selected_features].values.astype(np.float32)

y_train = label_encoder.transform(train_processed["author"].values)
y_val = label_encoder.transform(val_processed["author"].values)

# Full data for cross-val
X_full_texts = train_df["text"].values
X_full_features = np.vstack([X_train_features, X_val_features])
y_full = np.concatenate([y_train, y_val])

print(f"Train labels distribution: {np.bincount(y_train)}")
print(f"Val labels distribution: {np.bincount(y_val)}")


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    epoch,
    warmup_steps,
    initial_lrs,
):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features = batch.get("features", None)
        if features is not None:
            features = features.to(device)
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
            for pg in optimizer.param_groups:
                pg["lr"] = initial_lrs[0] * (current_step / max(1, warmup_steps))
        else:
            scheduler.step(epoch + current_step / len(train_loader))
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader):
    model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_labels.append(batch["labels"].cpu().numpy())
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    score = log_loss(val_labels, val_probs)
    return score, val_probs, val_labels


# ============================================================
# 5-FOLD CROSS-VALIDATION TRAINING
# ============================================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_oof_preds = np.zeros((len(y_full), config.num_authors))
all_test_probs_transformer = []
all_val_labels = []

print(f"Starting {n_splits}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_features, y_full)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{n_splits}")
    print(f"{'='*50}")

    X_fold_train_texts = X_full_texts[train_idx]
    X_fold_val_texts = X_full_texts[val_idx]
    X_fold_train_features = X_full_features[train_idx]
    X_fold_val_features = X_full_features[val_idx]
    y_fold_train = y_full[train_idx]
    y_fold_val = y_full[val_idx]

    train_dataset = SpookyDataset(
        X_fold_train_texts, X_fold_train_features, y_fold_train
    )
    val_dataset = SpookyDataset(X_fold_val_texts, X_fold_val_features, y_fold_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = SpookyClassifier(
        num_authors=config.num_authors,
        num_features=config.num_stylometric_features,
        dropout_rate=config.dropout_rate,
    )
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
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
        ]
    )

    num_epochs = 30
    patience = 5
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler_amp = GradScaler()

    best_fold_score = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler_amp,
            scheduler,
            epoch,
            warmup_steps,
            initial_lrs,
        )
        val_score, val_probs, val_labels = validate(model, val_loader)
        print(
            f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val LogLoss: {val_score:.4f}"
        )

        if val_score < best_fold_score:
            best_fold_score = val_score
            patience_counter = 0
            torch.save(
                model.state_dict(), f"./working/models/best_model_fold{fold+1}.pt"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f"./working/models/best_model_fold{fold+1}.pt"))
    _, val_probs, val_labels = validate(model, val_loader)
    all_oof_preds[val_idx] = val_probs
    all_val_labels.extend(val_labels)

    # Test predictions
    test_dataset = SpookyDataset(test_df["text"].values, X_test_features)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model.eval()
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())
    fold_test_probs = np.concatenate(fold_test_probs)
    all_test_probs_transformer.append(fold_test_probs)
    fold_log_loss = log_loss(y_fold_val, val_probs)
    print(f"Fold {fold+1} Best LogLoss: {fold_log_loss:.4f}")

# ============================================================
# TRAIN GBDT MODELS
# ============================================================
print("\n" + "=" * 50)
print("Training GBDT Models on Transformer Embeddings + Features")
print("=" * 50)

feature_extractor = FeatureExtractor(config, device)
print("Extracting embeddings for training data...")
train_embeddings = feature_extractor.extract_embeddings(X_full_texts, batch_size=32)
print("Extracting embeddings for test data...")
test_embeddings = feature_extractor.extract_embeddings(
    test_df["text"].values, batch_size=32
)

X_train_gbdt = np.hstack([train_embeddings, X_full_features])
X_test_gbdt = np.hstack([test_embeddings, X_test_features])

print("Training XGBoost...")
xgb_params = {
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
    "use_label_encoder": False,
}
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_gbdt, y_full, eval_set=[(X_train_gbdt, y_full)], verbose=False)
xgb_val_probs = xgb_model.predict_proba(X_train_gbdt)
xgb_test_probs = xgb_model.predict_proba(X_test_gbdt)

print("Training LightGBM...")
lgb_params = {
    "n_estimators": 2000,
    "num_leaves": 63,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "random_state": 42,
    "verbose": -1,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train_gbdt,
    y_full,
    eval_set=[(X_train_gbdt, y_full)],
    callbacks=[lgb.early_stopping(50)],
    verbose=False,
)
lgb_val_probs = lgb_model.predict_proba(X_train_gbdt)
lgb_test_probs = lgb_model.predict_proba(X_test_gbdt)

# ============================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================
print("\n" + "=" * 50)
print("Optimizing Ensemble Weights")
print("=" * 50)

transformer_val_probs = all_oof_preds
val_preds_dict = {
    "xgb": xgb_val_probs,
    "lgb": lgb_val_probs,
    "transformer": transformer_val_probs,
}
best_weights, best_score = optimize_ensemble_weights(val_preds_dict, y_full)
print(f"Best ensemble weights: {best_weights}")
print(f"Best ensemble validation LogLoss: {best_score:.4f}")

# ============================================================
# GENERATE FINAL TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 50)
print("Generating Final Test Predictions")
print("=" * 50)

transformer_test_probs = np.mean(all_test_probs_transformer, axis=0)
final_test_probs = (
    best_weights["xgb"] * xgb_test_probs
    + best_weights["lgb"] * lgb_test_probs
    + best_weights["transformer"] * transformer_test_probs
)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

# ============================================================
# CREATE SUBMISSION
# ============================================================
print("Creating submission file...")
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

submission.to_csv("./submission/submission_e136286532254749bdf74a569a42719a.csv", index=False)
print(f"Submission saved to ./submission/submission_e136286532254749bdf74a569a42719a.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
print(f"\n{'='*50}")
print(f"Final Validation Score: {best_score}")
print(f"{'='*50}")
