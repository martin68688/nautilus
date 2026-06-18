import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
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

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# BASIC TEXT CLEANING
# ============================================================
def clean_text(text):
    """Basic cleaning: lowercase, normalize whitespace, remove URLs."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)


# ============================================================
# FEATURE ENGINEERING — STYLISTIC & LINGUISTIC FEATURES
# ============================================================
def extract_features(df, text_col="text_clean"):
    """Generate handcrafted features capturing authorial style."""
    features = pd.DataFrame(index=df.index)

    # Basic length features
    features["char_count"] = df[text_col].apply(len)
    features["word_count"] = df[text_col].apply(lambda x: len(x.split()))
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = df[text_col].apply(
        lambda x: len(re.findall(r"[.!?]+", x))
    )
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )

    # Punctuation and capitalization
    features["exclamation_count"] = df[text_col].apply(lambda x: x.count("!"))
    features["question_count"] = df[text_col].apply(lambda x: x.count("?"))
    features["dash_count"] = df[text_col].apply(lambda x: x.count("-"))
    features["quote_count"] = df[text_col].apply(lambda x: x.count('"') + x.count("'"))
    features["colon_semicolon_count"] = df[text_col].apply(
        lambda x: x.count(":") + x.count(";")
    )
    features["ellipsis_count"] = df[text_col].apply(lambda x: x.count("..."))
    features["comma_count"] = df[text_col].apply(lambda x: x.count(","))
    features["capital_ratio"] = df[text_col].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )

    # Vocabulary richness
    features["unique_word_ratio"] = df[text_col].apply(
        lambda x: len(set(x.split())) / (len(x.split()) + 1)
    )
    features["stopword_ratio"] = df[text_col].apply(
        lambda x: sum(
            1
            for w in x.split()
            if w
            in {
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
            }
        )
        / (len(x.split()) + 1)
    )

    # Sentiment proxies (based on keyword counts)
    positive_words = {
        "love",
        "hope",
        "joy",
        "beautiful",
        "wonder",
        "happy",
        "gentle",
        "peace",
        "kind",
        "bright",
    }
    negative_words = {
        "fear",
        "terror",
        "death",
        "dark",
        "shadow",
        "pain",
        "dread",
        "horror",
        "gloom",
        "sorrow",
    }

    features["positive_word_count"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in positive_words)
    )
    features["negative_word_count"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in negative_words)
    )

    # Genre-specific keywords (author style markers)
    eap_keywords = {
        "raven",
        "tell-tale",
        "usher",
        "pit",
        "pendulum",
        "masque",
        "amontillado",
        "cask",
    }
    hpl_keywords = {
        "cthulhu",
        "nyarlathotep",
        "yog-sothoth",
        "necronomicon",
        "arkham",
        "innsmouth",
        "r'lyeh",
        "eldritch",
    }
    mws_keywords = {
        "frankenstein",
        "creature",
        "monster",
        "science",
        "galvanism",
        "ingolstadt",
        "geneva",
        "walton",
    }

    features["eap_keyword_count"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in eap_keywords)
    )
    features["hpl_keyword_count"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in hpl_keywords)
    )
    features["mws_keyword_count"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in mws_keywords)
    )

    # First-person pronoun usage (Poe/Shelley use more "I", Lovecraft uses more "we")
    features["first_person_singular"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in {"i", "me", "my", "myself"})
    )
    features["first_person_plural"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if w in {"we", "us", "our", "ourselves"})
    )

    # Rare words indicator (proxy for vocabulary complexity)
    features["long_word_ratio"] = df[text_col].apply(
        lambda x: sum(1 for w in x.split() if len(w) > 10) / (len(x.split()) + 1)
    )

    return features


# Extract features for train and test (computed later per split to avoid leakage)
print("Feature extraction will be performed per split")

# ============================================================
# TRAIN/VALIDATION SPLIT (WITH ENCODED TARGET)
# ============================================================
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author_encoded"]))

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING — SPLIT-SAFE
# ============================================================
train_split_features = extract_features(train_split_df, "text_clean")
val_split_features = extract_features(val_split_df, "text_clean")
test_features = extract_features(test_df, "text_clean")

# ============================================================
# TF-IDF FEATURES (N-GRAM LEVEL) — FIT ON TRAIN SPLIT ONLY
# ============================================================
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    lowercase=False,
)
train_tfidf_char = tfidf_char.fit_transform(train_split_df["text_clean"])
val_tfidf_char = tfidf_char.transform(val_split_df["text_clean"])
test_tfidf_char = tfidf_char.transform(test_df["text_clean"])

tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=1000,
    stop_words="english",
    sublinear_tf=True,
)
train_tfidf_word = tfidf_word.fit_transform(train_split_df["text_clean"])
val_tfidf_word = tfidf_word.transform(val_split_df["text_clean"])
test_tfidf_word = tfidf_word.transform(test_df["text_clean"])

# Convert TF-IDF sparse to dense numpy
train_tfidf_char_dense = train_tfidf_char.toarray() if hasattr(train_tfidf_char, "toarray") else train_tfidf_char
val_tfidf_char_dense = val_tfidf_char.toarray() if hasattr(val_tfidf_char, "toarray") else val_tfidf_char
test_tfidf_char_dense = test_tfidf_char.toarray() if hasattr(test_tfidf_char, "toarray") else test_tfidf_char

train_tfidf_word_dense = train_tfidf_word.toarray() if hasattr(train_tfidf_word, "toarray") else train_tfidf_word
val_tfidf_word_dense = val_tfidf_word.toarray() if hasattr(val_tfidf_word, "toarray") else val_tfidf_word
test_tfidf_word_dense = test_tfidf_word.toarray() if hasattr(test_tfidf_word, "toarray") else test_tfidf_word

# Create DataFrames for TF-IDF features
train_tfidf_char_df = pd.DataFrame(
    train_tfidf_char_dense,
    columns=[f"char_tfidf_{i}" for i in range(train_tfidf_char_dense.shape[1])],
)
val_tfidf_char_df = pd.DataFrame(
    val_tfidf_char_dense,
    columns=[f"char_tfidf_{i}" for i in range(val_tfidf_char_dense.shape[1])],
)
test_tfidf_char_df = pd.DataFrame(
    test_tfidf_char_dense,
    columns=[f"char_tfidf_{i}" for i in range(test_tfidf_char_dense.shape[1])],
)

train_tfidf_word_df = pd.DataFrame(
    train_tfidf_word_dense,
    columns=[f"word_tfidf_{i}" for i in range(train_tfidf_word_dense.shape[1])],
)
val_tfidf_word_df = pd.DataFrame(
    val_tfidf_word_dense,
    columns=[f"word_tfidf_{i}" for i in range(val_tfidf_word_dense.shape[1])],
)
test_tfidf_word_df = pd.DataFrame(
    test_tfidf_word_dense,
    columns=[f"word_tfidf_{i}" for i in range(test_tfidf_word_dense.shape[1])],
)

# ============================================================
# COMBINE ALL FEATURES
# ============================================================
X_train = pd.concat(
    [train_split_features.reset_index(drop=True), train_tfidf_char_df, train_tfidf_word_df],
    axis=1,
)
X_val = pd.concat(
    [val_split_features.reset_index(drop=True), val_tfidf_char_df, val_tfidf_word_df],
    axis=1,
)
X_test = pd.concat(
    [test_features.reset_index(drop=True), test_tfidf_char_df, test_tfidf_word_df],
    axis=1,
)

y_train = train_split_df["author_encoded"].reset_index(drop=True)
y_val = val_split_df["author_encoded"].reset_index(drop=True)

# ============================================================
# SCALE NUMERICAL FEATURES (FIT ON TRAIN ONLY)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

test_ids = test_df["id"].values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================
# SAVE PREPROCESSED DATA FOR NEXT STEPS
# ============================================================
os.makedirs("./working", exist_ok=True)
import joblib

joblib.dump(X_train, "./working/X_train.pkl")
joblib.dump(X_val, "./working/X_val.pkl")
joblib.dump(X_test, "./working/X_test.pkl")
joblib.dump(y_train, "./working/y_train.pkl")
joblib.dump(y_val, "./working/y_val.pkl")
joblib.dump(test_ids, "./working/test_ids.pkl")
joblib.dump(le, "./working/label_encoder.pkl")
joblib.dump(scaler, "./working/scaler.pkl")

print("Preprocessed data saved successfully")
print(f"Validation set label distribution:\n{pd.Series(y_val).value_counts()}")

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
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Extract [CLS] embedding from last hidden state (1024-dimensional)
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, 1024)
        cls_embedding = hidden_states[:, 0, :]     # (batch_size, 1024)
        return outputs.logits, cls_embedding


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

backbone_unfrozen_params = []
for name, param in model.backbone.named_parameters():
    if param.requires_grad:
        backbone_unfrozen_params.append(param)

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_unfrozen_params):,}")
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
train_texts_orig = train_df["text"].values
train_labels_orig = train_df["author_encoded"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Use previously computed indices for train/validation split
train_indices = train_idx
val_indices = val_idx

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

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
)

initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

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
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = initial_lrs[0] * warmup_factor
        else:
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
                logits, cls_emb = model(input_ids, attention_mask)
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
        if epochs_no_improve >= 4:
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
            logits, cls_emb = model(input_ids, attention_mask)
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
            logits, cls_emb = model(input_ids, attention_mask)
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