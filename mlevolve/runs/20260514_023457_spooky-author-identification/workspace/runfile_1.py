import os
os.sched_setaffinity(0, {12, 13})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import re
import os
from collections import Counter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.sparse import vstack
import gc

# ============================================================
# Step 1: Data Processing and Feature Engineering
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

def extract_stylometric_features(text_series):
    features = {}
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.split(r"[.!?]+", x)) - 1
    )
    features["avg_word_length"] = features["char_count"] / features["word_count"].clip(
        lower=1
    )
    features["avg_sentence_length"] = features["word_count"] / features[
        "sentence_count"
    ].clip(lower=1)
    features["exclamation_count"] = text_series.str.count(r"!")
    features["question_count"] = text_series.str.count(r"\?")
    features["semicolon_count"] = text_series.str.count(r";")
    features["colon_count"] = text_series.str.count(r":")
    features["dash_count"] = text_series.str.count(r"—|-")
    features["quote_count"] = text_series.str.count(r"\"")
    features["parenthesis_count"] = text_series.str.count(r"\(|\)")
    features["comma_count"] = text_series.str.count(r",")
    features["period_count"] = text_series.str.count(r"\.")
    features["ellipsis_count"] = text_series.str.count(r"\.\.\.")
    features["punctuation_ratio"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["semicolon_count"]
        + features["colon_count"]
        + features["dash_count"]
        + features["quote_count"]
        + features["parenthesis_count"]
        + features["comma_count"]
        + features["period_count"]
    ) / features["char_count"].clip(lower=1)
    features["all_caps_words"] = text_series.apply(
        lambda x: len(re.findall(r"\b[A-Z]{2,}\b", x))
    )
    features["capitalized_words"] = text_series.apply(
        lambda x: len(re.findall(r"\b[A-Z][a-z]+\b", x))
    )
    features["capital_ratio"] = features["capitalized_words"] / features[
        "word_count"
    ].clip(lower=1)
    features["unique_words"] = text_series.apply(lambda x: len(set(x.lower().split())))
    features["type_token_ratio"] = features["unique_words"] / features[
        "word_count"
    ].clip(lower=1)
    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "whence",
        "thence",
        "hither",
        "thither",
        "ere",
        "whilst",
        "behold",
        "perchance",
        "foreboding",
        "eldritch",
        "cyclopean",
        "ichor",
    ]
    features["archaic_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in archaic_words)
    )
    positive_words = [
        "beautiful",
        "wonderful",
        "joy",
        "love",
        "hope",
        "happy",
        "pleasure",
        "delight",
    ]
    negative_words = [
        "dread",
        "horror",
        "terror",
        "fear",
        "dark",
        "death",
        "gloom",
        "anguish",
        "despair",
        "hideous",
        "ghastly",
        "monstrous",
        "awful",
        "terrible",
        "frightful",
    ]
    features["positive_sentiment"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in positive_words)
    )
    features["negative_sentiment"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in negative_words)
    )
    features["sentiment_imbalance"] = (
        features["positive_sentiment"] - features["negative_sentiment"]
    ) / (features["positive_sentiment"] + features["negative_sentiment"] + 1)
    features["first_person_sg"] = text_series.str.count(r"\bI\b")
    features["first_person_pl"] = text_series.str.count(r"\bwe\b|\bus\b")
    features["second_person"] = text_series.str.count(r"\byou\b|\byour\b")
    features["third_person"] = text_series.str.count(
        r"\bhe\b|\bshe\b|\bit\b|\bthey\b|\bthem\b"
    )
    features["pronoun_ratio_1sg"] = features["first_person_sg"] / features[
        "word_count"
    ].clip(lower=1)
    stop_words_specific = [
        "the",
        "and",
        "of",
        "to",
        "in",
        "a",
        "that",
        "was",
        "it",
        "with",
        "for",
        "on",
        "but",
        "by",
        "not",
        "all",
        "from",
        "at",
        "as",
        "had",
        "which",
        "were",
        "been",
    ]
    for w in stop_words_specific:
        features[f"fw_{w}"] = text_series.str.count(rf"\b{w}\b")
    features["consonant_ratio"] = text_series.apply(
        lambda x: sum(1 for c in x.lower() if c in "bcdfghjklmnpqrstvwxyz")
        / max(len(x), 1)
    )
    features["vowel_ratio"] = text_series.apply(
        lambda x: sum(1 for c in x.lower() if c in "aeiou") / max(len(x), 1)
    )
    features["complex_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.split() if len(w) > 8)
    )
    features["complex_word_ratio"] = features["complex_word_count"] / features[
        "word_count"
    ].clip(lower=1)
    features["short_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.split() if len(w) <= 3)
    )
    features["long_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.split() if len(w) >= 10)
    )
    features["short_word_ratio"] = features["short_word_count"] / features[
        "word_count"
    ].clip(lower=1)
    features["long_word_ratio"] = features["long_word_count"] / features[
        "word_count"
    ].clip(lower=1)
    return pd.DataFrame(features, index=text_series.index)


print("Extracting stylometric features for train and test separately...")
train_stylo = extract_stylometric_features(train_df["text"])
test_stylo = extract_stylometric_features(test_df["text"])

print("Extracting TF-IDF features from training data only...")
tfidf_params = [
    {"name": "unigram", "ngram_range": (1, 1), "max_features": 500, "min_df": 5},
    {"name": "bigram", "ngram_range": (2, 2), "max_features": 500, "min_df": 5},
    {"name": "trigram", "ngram_range": (3, 3), "max_features": 300, "min_df": 5},
]

tfidf_features_train = []
tfidf_features_test = []
for params in tfidf_params:
    tfidf = TfidfVectorizer(
        ngram_range=params["ngram_range"],
        max_features=params["max_features"],
        min_df=params["min_df"],
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    )
    train_tfidf = tfidf.fit_transform(train_df["text"])
    test_tfidf = tfidf.transform(test_df["text"])
    feat_names = [f'{params["name"]}_{feat}' for feat in tfidf.get_feature_names_out()]
    train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=feat_names)
    test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=feat_names)
    tfidf_features_train.append(train_tfidf_df)
    tfidf_features_test.append(test_tfidf_df)

train_tfidf_combined = pd.concat(tfidf_features_train, axis=1)
test_tfidf_combined = pd.concat(tfidf_features_test, axis=1)
print(f"Train TF-IDF features shape: {train_tfidf_combined.shape}")
print(f"Test TF-IDF features shape: {test_tfidf_combined.shape}")

print("Combining all features for train and test separately...")
train_features = pd.concat([train_stylo, train_tfidf_combined], axis=1)
test_features = pd.concat([test_stylo, test_tfidf_combined], axis=1)

train_features = train_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.replace([np.inf, -np.inf], np.nan)

# Impute using only training data statistics
train_features = train_features.fillna(train_features.mean())
test_features = test_features.fillna(train_features.mean())

train_features = train_features.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

print("Scaling features using training data statistics only...")
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)
train_features_scaled = pd.DataFrame(
    train_features_scaled, columns=train_features.columns
)
test_features_scaled = pd.DataFrame(test_features_scaled, columns=test_features.columns)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["author"])
label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print(f"Label mapping: {label_mapping}")

train_features_np = train_features_scaled.values.astype(np.float32)
test_features_np = test_features_scaled.values.astype(np.float32)

print(f"Train features shape: {train_features_np.shape}")
print(f"Test features shape: {test_features_np.shape}")
print(f"Number of features: {train_features_np.shape[1]}")

# ============================================================
# Step 2 & 3: Model Design & Training/Evaluation
# ============================================================
print("\n" + "=" * 50)
print("Initializing DeBERTa-v3-large model and tokenizer...")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
MAX_LEN = 256


def tokenize_texts(texts, tokenizer, max_len=MAX_LEN):
    encoded = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


print("Tokenizing all texts...")
train_input_ids, train_attention_mask = tokenize_texts(train_df["text"], tokenizer)
test_input_ids, test_attention_mask = tokenize_texts(test_df["text"], tokenizer)


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


NUM_FOLDS = 5
NUM_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
LABEL_SMOOTHING = 0.1

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_val_scores = []
all_test_probs = []

print(f"Starting {NUM_FOLDS}-fold cross-validation...")
print(f"Training samples: {len(train_labels)}, Test samples: {len(test_df)}")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_features_np, train_labels)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{NUM_FOLDS}")
    print(f"{'='*50}")

    X_train_text_ids = train_input_ids[train_idx]
    X_train_mask = train_attention_mask[train_idx]
    X_train_feat = torch.FloatTensor(train_features_np[train_idx])
    y_train = torch.LongTensor(train_labels[train_idx])

    X_val_text_ids = train_input_ids[val_idx]
    X_val_mask = train_attention_mask[val_idx]
    X_val_feat = torch.FloatTensor(train_features_np[val_idx])
    y_val = torch.LongTensor(train_labels[val_idx])

    train_dataset = TensorDataset(X_train_text_ids, X_train_mask, X_train_feat, y_train)
    val_dataset = TensorDataset(X_val_text_ids, X_val_mask, X_val_feat, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = SpookyClassifier(
        num_authors=3, num_features=train_features_np.shape[1], dropout_rate=0.3
    )
    model.to(device)

    backbone_params = []
    for layer in model.backbone.deberta.encoder.layer[-8:]:
        for n, p in layer.named_parameters():
            if "bias" not in n and "LayerNorm" not in n:
                backbone_params.append(p)

    head_params = list(model.head.parameters())
    if model.feature_proj:
        head_params.extend(list(model.feature_proj.parameters()))

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

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state_fold = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            features = batch[2].to(device, non_blocking=True)
            labels = batch[3].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = initial_lrs[pg_idx] * (
                        current_step / max(1, warmup_steps)
                    )
            else:
                scheduler.step(epoch + batch_idx / len(train_loader))

            total_loss += loss.item()
            num_batches += 1

        model.eval()
        val_loss = 0
        val_probs = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device, non_blocking=True)
                attention_mask = batch[1].to(device, non_blocking=True)
                features = batch[2].to(device, non_blocking=True)
                labels = batch[3].to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask, features)
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)

                val_loss += loss.item()
                val_probs.append(probs.cpu().numpy())
                val_true.append(labels.cpu().numpy())

        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)

        val_probs = np.nan_to_num(val_probs, nan=1e-15, posinf=1.0, neginf=1e-15)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        score = log_loss(val_true, val_probs)

        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Log Loss: {score:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_fold = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state_fold)

    model.eval()
    val_probs = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            features = batch[2].to(device, non_blocking=True)
            labels = batch[3].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_true = np.concatenate(val_true)
    val_probs = np.nan_to_num(val_probs, nan=1e-15, posinf=1.0, neginf=1e-15)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    fold_score = log_loss(val_true, val_probs)
    fold_val_scores.append(fold_score)
    print(f"Fold {fold+1} Validation Log Loss: {fold_score:.6f}")

    model.eval()
    test_probs = []
    test_dataset = TensorDataset(
        test_input_ids, test_attention_mask, torch.FloatTensor(test_features_np)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            features = batch[2].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(test_probs)
    all_test_probs.append(fold_test_probs)

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

final_val_score = np.mean(fold_val_scores)
print(f"\n{'='*50}")
print(f"Cross-Validation Log Loss: {final_val_score:.6f}")
print(f"Individual fold scores: {[f'{s:.6f}' for s in fold_val_scores]}")

final_test_probs = np.mean(all_test_probs, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

print(f"\nCreating submission file...")
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission_b69f3cc00b594d549f600e180b9a4d1c.csv", index=False)
print(f"Submission saved to ./submission/submission_b69f3cc00b594d549f600e180b9a4d1c.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_score}")