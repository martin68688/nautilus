import os
os.sched_setaffinity(0, {5, 6, 7, 8, 9, 10, 11, 20, 21, 22})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import os
import re
import json
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# -------------------------------
# Step 1: Feature Engineering
# -------------------------------


def extract_stylometric_features(text_series):
    """Extract comprehensive stylometric features from text."""
    features = pd.DataFrame()

    # Basic text statistics
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_length"] = text_series.apply(
        lambda x: (
            np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
    )
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.findall(r"[.!?]+", str(x))) + 1
    )

    # Punctuation features
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["period_count"] = text_series.str.count(r"\.")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["dash_count"] = text_series.str.count("-")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["parenthesis_count"] = text_series.str.count(
        r"\("
    ) + text_series.str.count(r"\)")

    # Punctuation density
    total_chars = features["char_count"].clip(lower=1)
    features["punctuation_density"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["period_count"]
        + features["comma_count"]
        + features["semicolon_count"]
        + features["colon_count"]
    ) / total_chars

    # Capitalization patterns
    features["capital_letters"] = text_series.str.findall(r"[A-Z]").str.len()
    features["capital_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 0 and w[0].isupper())
        / max(len(str(x).split()), 1)
    )

    # Archaic language markers
    archaic_words = [
        "thee",
        "thou",
        "thy",
        "thine",
        "hath",
        "doth",
        "art",
        "wilt",
        "shalt",
        "canst",
        "didst",
        "hast",
        "cometh",
        "maketh",
        "sayeth",
        "thence",
        "whence",
        "whither",
        "wherefore",
        "therein",
        "thereof",
        "thereto",
        "whereof",
    ]
    for word in archaic_words:
        features[f"archaic_{word}"] = text_series.str.lower().str.count(
            r"\b" + word + r"\b"
        )
    features["archaic_word_count"] = features[
        [f"archaic_{w}" for w in archaic_words]
    ].sum(axis=1)

    # Contraction usage
    contractions = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
    for c in contractions:
        features[f"contraction_{c}"] = text_series.str.lower().str.count(re.escape(c))
    features["contraction_count"] = features[
        [f"contraction_{c}" for c in contractions]
    ].sum(axis=1)

    # Horror specific words
    horror_words = [
        "fear",
        "horror",
        "terror",
        "dread",
        "dark",
        "night",
        "death",
        "dead",
        "ghost",
        "shadow",
        "strange",
        "wild",
        "terrible",
        "awful",
        "hideous",
        "frightful",
        "dreadful",
        "shudder",
        "gloom",
        "gloomy",
        "mystery",
        "mysterious",
        "ancient",
        "old",
        "curse",
        "evil",
        "demon",
        "devil",
        "hell",
        "spirit",
    ]
    for word in horror_words:
        features[f"horror_{word}"] = text_series.str.lower().str.count(
            r"\b" + word + r"\b"
        )
    features["horror_word_count"] = features[[f"horror_{w}" for w in horror_words]].sum(
        axis=1
    )

    # Word length distribution
    features["short_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) <= 3)
    )
    features["medium_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if 4 <= len(w) <= 7)
    )
    features["long_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) >= 8)
    )

    # Vocabulary richness
    features["unique_words"] = text_series.apply(
        lambda x: len(set(str(x).lower().split()))
    )
    features["type_token_ratio"] = features["unique_words"] / features[
        "word_count"
    ].clip(lower=1)

    # Starting word patterns
    features["starts_with_article"] = (
        text_series.str.lower().str.startswith("the ")
    ).astype(int)

    # Check if text starts with any conjunction - using apply since str.startswith with tuple works
    features["starts_with_conjunction"] = text_series.str.lower().str.startswith(
        ("and ", "but ", "for ", "nor ", "yet ", "so ")
    ).astype(int)

    features["starts_with_preposition"] = text_series.str.lower().str.startswith(
        ("in ", "on ", "at ", "by ", "with ", "from ", "to ")
    ).astype(int)

    return features


def get_pos_tags_indicators(text):
    """Simple POS-like indicators based on word endings."""
    text_lower = str(text).lower()
    words = text_lower.split()

    ly_words = sum(1 for w in words if w.endswith("ly"))
    adj_suffixes = sum(
        1
        for w in words
        if any(
            w.endswith(suf) for suf in ["ful", "ous", "ive", "able", "ible", "al", "ic"]
        )
    )
    noun_suffixes = sum(
        1
        for w in words
        if any(
            w.endswith(suf) for suf in ["tion", "ment", "ness", "ity", "ance", "ence"]
        )
    )
    verb_suffixes = sum(
        1 for w in words if any(w.endswith(suf) for suf in ["ed", "ing", "ize", "ify"])
    )

    return pd.Series(
        {
            "adverb_indicators": ly_words / max(len(words), 1),
            "adj_indicators": adj_suffixes / max(len(words), 1),
            "noun_indicators": noun_suffixes / max(len(words), 1),
            "verb_indicators": verb_suffixes / max(len(words), 1),
        }
    )


print("Extracting stylometric features...")
all_text = pd.concat([train_df["text"], test_df["text"]], axis=0)
stylo_features_all = extract_stylometric_features(all_text)

text_stats = pd.DataFrame()
text_stats["word_count"] = all_text.str.split().str.len()
text_stats["char_count"] = all_text.str.len()
text_stats["is_short"] = (text_stats["word_count"] <= 10).astype(int)
text_stats["is_medium"] = (
    (text_stats["word_count"] > 10) & (text_stats["word_count"] <= 30)
).astype(int)
text_stats["is_long"] = (text_stats["word_count"] > 30).astype(int)

print("Extracting POS indicators...")
pos_features = all_text.apply(get_pos_tags_indicators)

all_features = pd.concat([stylo_features_all, text_stats, pos_features], axis=1)
all_features = all_features.replace([np.inf, -np.inf], 0)
all_features = all_features.fillna(0)

train_features = all_features.iloc[: len(train_df)].reset_index(drop=True)
test_features = all_features.iloc[len(train_df) :].reset_index(drop=True)

feature_cols = train_features.columns.tolist()
scaler = StandardScaler()
train_features_scaled = pd.DataFrame(
    scaler.fit_transform(train_features), columns=feature_cols
)
test_features_scaled = pd.DataFrame(
    scaler.transform(test_features), columns=feature_cols
)

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(train_df["author"])
class_names = label_encoder.classes_

print("Extracting n-gram features...")
# Note: TF-IDF vectorizers will be fit inside each fold to prevent data leakage
# We keep the scaler for stylometric features only (already fitted correctly)
X_full_features = train_features_scaled.copy()
X_test_features = test_features_scaled.copy()

print(f"Full feature set shape: {X_full_features.shape}")
print(f"Test feature set shape: {X_test_features.shape}")
print(f"Class labels: {class_names}")

# -------------------------------
# Step 2: Model Definition
# -------------------------------


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, num_stylometric_features=150, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/deberta-v3-large")

        # Partial unfreezing: freeze first 16 layers, unfreeze last 8
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size  # 1024

        self.feature_proj = nn.Sequential(
            nn.Linear(num_stylometric_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size + 64),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size + 64, num_authors),
        )

    def forward(self, input_ids, attention_mask, stylometric_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if stylometric_features is not None:
            feat_embedding = self.feature_proj(stylometric_features)
            combined = torch.cat([cls_embedding, feat_embedding], dim=1)
        else:
            combined = cls_embedding

        logits = self.classifier(combined)
        return logits


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


# Dataset class
class SpookyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, features, labels=None, max_length=256):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -------------------------------
# Step 3: Training & Evaluation
# -------------------------------

# Hyperparameters
num_epochs = 30
batch_size = 16
patience = 5
num_folds = 5
label_smoothing = 0.1
max_grad_norm = 1.0

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_val_scores = []
fold_test_probs = []

train_texts = train_df["text"].values
test_texts = test_df["text"].values

print(f"Starting {num_folds}-fold cross-validation training...")

for fold, (train_idx_fold, val_idx_fold) in enumerate(
    skf.split(X_full_features, y_full)
):
    print(f"\n=== Fold {fold + 1}/{num_folds} ===")

    fold_train_texts = train_texts[train_idx_fold]
    fold_val_texts = train_texts[val_idx_fold]

    # Fit TF-IDF vectorizers on fold training data only (prevent data leakage)
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=300,
        sublinear_tf=True,
        lowercase=True,
    )
    char_train_fold = char_vectorizer.fit_transform(fold_train_texts)
    char_val_fold = char_vectorizer.transform(fold_val_texts)
    char_test_fold = char_vectorizer.transform(test_texts)

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=300,
        sublinear_tf=True,
        lowercase=True,
        stop_words="english",
    )
    word_train_fold = word_vectorizer.fit_transform(fold_train_texts)
    word_val_fold = word_vectorizer.transform(fold_val_texts)
    word_test_fold = word_vectorizer.transform(test_texts)

    train_char_dense = pd.DataFrame(
        char_train_fold.toarray(),
        columns=[f"char_ngram_{i}" for i in range(char_train_fold.shape[1])],
    )
    val_char_dense = pd.DataFrame(
        char_val_fold.toarray(),
        columns=[f"char_ngram_{i}" for i in range(char_val_fold.shape[1])],
    )
    test_char_dense = pd.DataFrame(
        char_test_fold.toarray(),
        columns=[f"char_ngram_{i}" for i in range(char_test_fold.shape[1])],
    )
    train_word_dense = pd.DataFrame(
        word_train_fold.toarray(),
        columns=[f"word_ngram_{i}" for i in range(word_train_fold.shape[1])],
    )
    val_word_dense = pd.DataFrame(
        word_val_fold.toarray(),
        columns=[f"word_ngram_{i}" for i in range(word_val_fold.shape[1])],
    )
    test_word_dense = pd.DataFrame(
        word_test_fold.toarray(),
        columns=[f"word_ngram_{i}" for i in range(word_test_fold.shape[1])],
    )

    train_stylo_features = X_full_features.iloc[train_idx_fold].values.astype(np.float32)
    val_stylo_features = X_full_features.iloc[val_idx_fold].values.astype(np.float32)

    fold_train_features = np.concatenate(
        [train_stylo_features, train_char_dense.values.astype(np.float32), train_word_dense.values.astype(np.float32)],
        axis=1
    )
    fold_val_features = np.concatenate(
        [val_stylo_features, val_char_dense.values.astype(np.float32), val_word_dense.values.astype(np.float32)],
        axis=1
    )

    fold_train_labels = y_full[train_idx_fold]
    fold_val_labels = y_full[val_idx_fold]

    train_dataset = SpookyDataset(
        fold_train_texts, fold_train_features, fold_train_labels
    )
    val_dataset = SpookyDataset(fold_val_texts, fold_val_features, fold_val_labels)
    fold_test_features = np.concatenate(
        [X_test_features.values.astype(np.float32), test_char_dense.values.astype(np.float32), test_word_dense.values.astype(np.float32)],
        axis=1
    )
    test_dataset = SpookyDataset(test_texts, fold_test_features)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = SpookyAuthorClassifier(
        num_authors=3,
        num_stylometric_features=fold_train_features.shape[1],
        dropout_rate=0.3,
    )
    model.to(device)

    # Differentiated learning rates
    backbone_params = []
    for layer in model.backbone.encoder.layer[-8:]:
        for n, p in layer.named_parameters():
            if "bias" not in n and "LayerNorm" not in n:
                backbone_params.append(p)

    head_params = list(model.classifier.parameters()) + list(
        model.feature_proj.parameters()
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

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler()

    best_val_score = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = initial_lrs[0] * (current_step / max(1, warmup_steps))
            else:
                scheduler.step(epoch + current_step / len(train_loader))

            total_loss += loss.item()

        # Validation
        model.eval()
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    probs = torch.softmax(logits, dim=1)

                val_probs.append(probs.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_labels_concat = np.concatenate(val_labels_list)

        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

        val_score = log_loss(val_labels_concat, val_probs)
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Val Log Loss: {val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    model.eval()

    # Final validation prediction
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
    fold_val_score = log_loss(fold_val_labels, val_probs)
    fold_val_scores.append(fold_val_score)
    print(f"Fold {fold+1} Best Validation Log Loss: {fold_val_score:.4f}")

    # Test predictions for this fold
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

            test_probs.append(probs.cpu().numpy())

    fold_test_probs.append(np.concatenate(test_probs))

    del model
    torch.cuda.empty_cache()

# Aggregate results
mean_val_score = np.mean(fold_val_scores)
std_val_score = np.std(fold_val_scores)
print(f"\n=== Cross-Validation Results ===")
print(f"Mean Val Log Loss: {mean_val_score:.6f} ± {std_val_score:.6f}")

final_test_probs = np.mean(fold_test_probs, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission_df60ef148d574da39ccbb8d55cff2374.csv", index=False)

print(f"Submission saved to ./submission/submission_df60ef148d574da39ccbb8d55cff2374.csv")
print(f"Final Validation Score: {mean_val_score}")