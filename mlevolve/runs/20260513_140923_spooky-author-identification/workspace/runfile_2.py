import os
os.sched_setaffinity(0, {253, 254})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from collections import Counter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gc
import joblib

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================
# Data Loading
# ============================================
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")


# ============================================
# Data Processing & Feature Engineering
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()


train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)


# Character-level features
def extract_char_features(text):
    features = {}
    for punct in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "..."]:
        features[f"punct_{punct}"] = text.count(punct) / (len(text) + 1)
    original_text = text
    upper_count = sum(1 for c in original_text if c.isupper())
    features["uppercase_ratio"] = upper_count / (len(original_text) + 1)
    digit_count = sum(1 for c in original_text if c.isdigit())
    features["digit_ratio"] = digit_count / (len(original_text) + 1)
    words = original_text.split()
    if words:
        features["avg_word_len"] = np.mean([len(w) for w in words])
        features["max_word_len"] = max(len(w) for w in words)
        features["min_word_len"] = min(len(w) for w in words)
        features["std_word_len"] = np.std([len(w) for w in words])
    else:
        features["avg_word_len"] = 0
        features["max_word_len"] = 0
        features["min_word_len"] = 0
        features["std_word_len"] = 0
    return features


# Lexical richness features
def extract_lexical_features(text):
    features = {}
    words = text.split()
    if len(words) == 0:
        return {
            f"lex_{k}": 0
            for k in [
                "num_words",
                "num_unique_words",
                "ttr",
                "hapunax_legomena",
                "avg_syllables",
                "std_syllables",
            ]
        }
    features["lex_num_words"] = len(words)
    features["lex_num_unique_words"] = len(set(words))
    features["lex_ttr"] = features["lex_num_unique_words"] / (
        features["lex_num_words"] + 1
    )
    word_counts = Counter(words)
    hapax = sum(1 for v in word_counts.values() if v == 1)
    features["lex_hapunax_legomena"] = hapax / (len(words) + 1)

    def count_syllables(word):
        word = re.sub(r"[^a-z]", "", word)
        return max(1, len(re.findall(r"[aeiouy]+", word)))

    if words:
        syllables = [count_syllables(w) for w in words]
        features["lex_avg_syllables"] = np.mean(syllables)
        features["lex_std_syllables"] = np.std(syllables)
    else:
        features["lex_avg_syllables"] = 0
        features["lex_std_syllables"] = 0
    return features


# Sentence structure features
def extract_sentence_features(text):
    features = {}
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return {
            f"sent_{k}": 0
            for k in [
                "num_sentences",
                "avg_sent_len",
                "std_sent_len",
                "max_sent_len",
                "min_sent_len",
            ]
        }
    sent_lengths = [len(s.split()) for s in sentences]
    features["sent_num_sentences"] = len(sentences)
    features["sent_avg_sent_len"] = np.mean(sent_lengths)
    features["sent_std_sent_len"] = np.std(sent_lengths)
    features["sent_max_sent_len"] = max(sent_lengths)
    features["sent_min_sent_len"] = min(sent_lengths)
    return features


# Function word features
function_words = {
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
    "were",
    "had",
    "has",
    "have",
    "do",
    "does",
    "did",
    "shall",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "is",
    "are",
    "been",
    "being",
    "be",
    "am",
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
    "who",
    "whom",
    "which",
    "what",
    "where",
    "when",
    "why",
    "how",
    "not",
    "no",
    "never",
    "always",
    "ever",
    "often",
    "sometimes",
}


def extract_function_word_features(text):
    features = {}
    words = text.lower().split()
    if len(words) == 0:
        return {f"fw_{w}": 0 for w in function_words}
    total_words = len(words)
    for word in function_words:
        features[f"fw_{word}"] = words.count(word) / (total_words + 1)
    return features


# Extract non-ngram features (safe to compute before split)
print("Extracting features...")
char_features_train = train["text"].apply(extract_char_features)
char_features_df = pd.DataFrame(char_features_train.tolist())
lex_features_train = train["text"].apply(extract_lexical_features)
lex_features_df = pd.DataFrame(lex_features_train.tolist())
sent_features_train = train["text"].apply(extract_sentence_features)
sent_features_df = pd.DataFrame(sent_features_train.tolist())
fw_features_train = train["text"].apply(extract_function_word_features)
fw_features_df = pd.DataFrame(fw_features_train.tolist())

train_base_features = pd.concat(
    [
        char_features_df.reset_index(drop=True),
        lex_features_df.reset_index(drop=True),
        sent_features_df.reset_index(drop=True),
        fw_features_df.reset_index(drop=True),
    ],
    axis=1,
).fillna(0)

char_features_test = test["text"].apply(extract_char_features)
char_features_test_df = pd.DataFrame(char_features_test.tolist())
lex_features_test = test["text"].apply(extract_lexical_features)
lex_features_test_df = pd.DataFrame(lex_features_test.tolist())
sent_features_test = test["text"].apply(extract_sentence_features)
sent_features_test_df = pd.DataFrame(sent_features_test.tolist())
fw_features_test = test["text"].apply(extract_function_word_features)
fw_features_test_df = pd.DataFrame(fw_features_test.tolist())

test_base_features = pd.concat(
    [
        char_features_test_df.reset_index(drop=True),
        lex_features_test_df.reset_index(drop=True),
        sent_features_test_df.reset_index(drop=True),
        fw_features_test_df.reset_index(drop=True),
    ],
    axis=1,
).fillna(0)

# Align test features to match training columns (handle missing/extra columns)
missing_cols = set(train_base_features.columns) - set(test_base_features.columns)
extra_cols = set(test_base_features.columns) - set(train_base_features.columns)
if missing_cols:
    for col in missing_cols:
        test_base_features[col] = 0.0
if extra_cols:
    test_base_features = test_base_features.drop(columns=extra_cols)
test_base_features = test_base_features[train_base_features.columns]

# We'll compute n-gram features and scaling inside the CV loop to prevent leakage.
# Store base features and raw texts for use inside the loop.
train_base_features_full = train_base_features
test_base_features_full = test_base_features

NUM_FEATURES = train_base_features_full.shape[1]
print(f"Number of base features: {NUM_FEATURES}")

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(train["author"])
print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"


# ============================================
# Model Definition
# ============================================
class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=NUM_FEATURES, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # Freeze all then unfreeze last 8 layers
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


# ============================================
# Training & Evaluation
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_FOLDS = 5
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 5e-5
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, features, labels=None):
        self.texts = texts
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "features": self.features.iloc[idx].values.astype(np.float32),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
test_probabilities = []
best_overall_val_logloss = float("inf")

X_train = train_base_features_full
X_test = test_base_features_full

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*50}")

    X_train_fold = X_train.iloc[train_idx].reset_index(drop=True)
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train.iloc[val_idx].reset_index(drop=True)
    y_val_fold = y_train[val_idx]
    texts_train = train.iloc[train_idx]["text"].values
    texts_val = train.iloc[val_idx]["text"].values

    train_dataset = TextDataset(texts_train, X_train_fold, y_train_fold)
    val_dataset = TextDataset(texts_val, X_val_fold, y_val_fold)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = SpookyClassifier(
        num_authors=3, num_features=NUM_FEATURES, dropout_rate=DROPOUT_RATE
    ).to(device)

    backbone_params = []
    for layer in model.backbone.deberta.encoder.layer[-8:]:
        for name, param in layer.named_parameters():
            if "bias" not in name and "LayerNorm" not in name:
                backbone_params.append(param)
    head_params = list(model.head.parameters()) + list(model.feature_proj.parameters())

    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": LEARNING_RATE_BACKBONE,
                "weight_decay": WEIGHT_DECAY,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": LEARNING_RATE_HEAD,
                "weight_decay": WEIGHT_DECAY,
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
    scaler = torch.amp.GradScaler("cuda")

    best_val_logloss = float("inf")
    patience_counter = 0
    max_patience = 5

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long).to(device)
            attention_mask = torch.as_tensor(batch["attention_mask"], dtype=torch.long).to(device)
            features = torch.as_tensor(np.stack(batch["features"]), dtype=torch.float32).to(device)
            labels = torch.as_tensor(batch["labels"], dtype=torch.long).to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRADIENT_CLIP_NORM
            )
            scaler.step(optimizer)
            scaler.update()

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = initial_lrs[0] * (current_step / max(1, warmup_steps))
            else:
                scheduler.step(epoch + current_step / len(train_loader))

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = torch.tensor(batch["input_ids"]).to(device)
                attention_mask = torch.tensor(batch["attention_mask"]).to(device)
                features = torch.tensor(np.stack(batch["features"])).to(device)
                labels = torch.tensor(batch["labels"]).to(device)
                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)
                val_loss += loss.item()
                val_probs.append(probs.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_probs = np.concatenate(val_probs)
        val_labels_concat = np.concatenate(val_labels_list)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        val_logloss = log_loss(val_labels_concat, val_probs)

        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f}"
        )

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            torch.save(model.state_dict(), f"./working/best_model_fold_{fold}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model for test inference
    model.load_state_dict(torch.load(f"./working/best_model_fold_{fold}.pt"))
    model.eval()
    test_dataset = TextDataset(test["text"].values, X_test.iloc[:, :], labels=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long).to(device)
            attention_mask = torch.as_tensor(batch["attention_mask"], dtype=torch.long).to(device)
            features = torch.as_tensor(np.stack(batch["features"]), dtype=torch.float32).to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())
    fold_test_probs = np.concatenate(fold_test_probs)
    test_probabilities.append(fold_test_probs)
    print(f"Fold {fold+1} best validation log loss: {best_val_logloss:.6f}")
    if best_val_logloss < best_overall_val_logloss:
        best_overall_val_logloss = best_val_logloss

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

# Final test predictions
final_test_probs = np.mean(test_probabilities, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

# Create submission
class_labels = le.classes_
submission = pd.DataFrame(
    {
        "id": test["id"].values,
        class_labels[0]: final_test_probs[:, 0],
        class_labels[1]: final_test_probs[:, 1],
        class_labels[2]: final_test_probs[:, 2],
    }
)
submission.to_csv("./submission/submission_e3aa97164c2e4a56afbd8782a5e190c7.csv", index=False)
print(f"\nSubmission saved to ./submission/submission_e3aa97164c2e4a56afbd8782a5e190c7.csv")
print(f"Submission shape: {submission.shape}")
print(f"Final Validation Score: {best_overall_val_logloss:.6f}")