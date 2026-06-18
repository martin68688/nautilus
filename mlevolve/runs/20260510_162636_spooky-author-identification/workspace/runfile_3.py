import os
os.sched_setaffinity(0, {53, 54, 60, 62, 63})
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import re
import string
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ModernBertForSequenceClassification,
    AutoConfig,
)
import os
import joblib
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Create a combined dataframe for feature engineering
train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["author"] = np.nan
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)


# Basic text cleaning
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^\w\s\']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return ""


combined["clean_text"] = combined["text"].apply(clean_text)

# Feature 1: Character count features
combined["char_count"] = combined["text"].str.len()
combined["word_count"] = combined["text"].str.split().str.len()
combined["avg_word_len"] = combined["char_count"] / (combined["word_count"] + 1)
combined["sentence_count"] = combined["text"].str.count("[.!?]") + 1
combined["avg_sentence_len"] = combined["word_count"] / (combined["sentence_count"] + 1)


# Feature 2: Punctuation and special character frequencies
def count_punct(text):
    if not isinstance(text, str):
        return 0
    return sum(1 for c in text if c in string.punctuation)


def count_exclamation(text):
    if not isinstance(text, str):
        return 0
    return text.count("!")


def count_question(text):
    if not isinstance(text, str):
        return 0
    return text.count("?")


def count_ellipsis(text):
    if not isinstance(text, str):
        return 0
    return text.count("...")


def count_dash(text):
    if not isinstance(text, str):
        return 0
    return text.count("—") + text.count("--")


combined["punct_count"] = combined["text"].apply(count_punct)
combined["exclamation_count"] = combined["text"].apply(count_exclamation)
combined["question_count"] = combined["text"].apply(count_question)
combined["ellipsis_count"] = combined["text"].apply(count_ellipsis)
combined["dash_count"] = combined["text"].apply(count_dash)
combined["punct_ratio"] = combined["punct_count"] / (combined["char_count"] + 1)
combined["quote_count"] = combined["text"].str.count('"') + combined["text"].str.count(
    "'"
)
combined["comma_count"] = combined["text"].str.count(",")
combined["semicolon_count"] = combined["text"].str.count(";")
combined["colon_count"] = combined["text"].str.count(":")

# Feature 3: Capitalization features
combined["capital_word_ratio"] = combined["text"].apply(
    lambda x: (
        sum(1 for w in str(x).split() if w[0].isupper()) / (len(str(x).split()) + 1)
        if isinstance(x, str)
        else 0
    )
)
combined["all_caps_ratio"] = combined["text"].apply(
    lambda x: (
        sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
        / (len(str(x).split()) + 1)
        if isinstance(x, str)
        else 0
    )
)

# Feature 4: Stopword-related features
stopwords = set(
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
        "by",
        "with",
        "from",
    ]
)


def count_stopwords(text):
    if not isinstance(text, str):
        return 0
    words = text.lower().split()
    return sum(1 for w in words if w in stopwords)


combined["stopword_count"] = combined["clean_text"].apply(count_stopwords)
combined["stopword_ratio"] = combined["stopword_count"] / (combined["word_count"] + 1)

# Feature 5: Function word frequencies
function_words = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "because",
    "when",
    "while",
    "who",
    "which",
    "that",
    "this",
    "these",
    "those",
    "there",
    "here",
    "not",
    "no",
    "never",
    "nothing",
    "none",
    "nor",
    "neither",
    "very",
    "quite",
    "rather",
    "somewhat",
    "almost",
    "nearly",
    "still",
    "yet",
    "already",
    "just",
    "even",
    "only",
    "also",
    "too",
    "so",
    "such",
    "how",
    "what",
    "why",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "me",
    "you",
    "him",
    "she",
    "it",
    "us",
    "them",
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "ourselves",
    "themselves",
    "one",
    "oneself",
    "all",
    "every",
    "each",
    "both",
    "few",
    "many",
    "much",
    "some",
    "any",
    "several",
    "most",
    "other",
    "another",
    "such",
    "own",
    "same",
    "different",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "above",
    "below",
    "between",
    "through",
    "during",
    "before",
    "after",
    "since",
    "until",
    "about",
    "into",
    "within",
    "without",
}

for fw in list(function_words)[:50]:
    fw_clean = re.escape(fw)
    combined[f"fw_{fw}"] = combined["clean_text"].str.count(r"\b" + fw_clean + r"\b")

# Feature 6: Character n-grams - fit only on training data to avoid leakage
train_text_clean = combined[combined["is_train"]==1]["clean_text"].fillna("")
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=200, min_df=3, lowercase=True
)
char_vectorizer.fit(train_text_clean)
char_features_train = char_vectorizer.transform(train_text_clean)
char_features_test = char_vectorizer.transform(combined[combined["is_train"]==0]["clean_text"].fillna(""))
char_feat_names = [f"char_ngram_{i}" for i in range(char_features_train.shape[1])]
char_df_train = pd.DataFrame(
    char_features_train.toarray() if hasattr(char_features_train, "toarray") else char_features_train,
    columns=char_feat_names,
    index=combined[combined["is_train"]==1].index,
)
char_df_test = pd.DataFrame(
    char_features_test.toarray() if hasattr(char_features_test, "toarray") else char_features_test,
    columns=char_feat_names,
    index=combined[combined["is_train"]==0].index,
)
combined = pd.concat([combined, pd.concat([char_df_train, char_df_test])], axis=1)

# Feature 7: Word n-grams - fit only on training data to avoid leakage
word_vectorizer = CountVectorizer(
    ngram_range=(1, 3), max_features=300, min_df=3, stop_words="english", lowercase=True
)
word_vectorizer.fit(train_text_clean)
word_features_train = word_vectorizer.transform(train_text_clean)
word_features_test = word_vectorizer.transform(combined[combined["is_train"]==0]["clean_text"].fillna(""))
word_feat_names = [f"word_ngram_{i}" for i in range(word_features_train.shape[1])]
word_df_train = pd.DataFrame(
    word_features_train.toarray() if hasattr(word_features_train, "toarray") else word_features_train,
    columns=word_feat_names,
    index=combined[combined["is_train"]==1].index,
)
word_df_test = pd.DataFrame(
    word_features_test.toarray() if hasattr(word_features_test, "toarray") else word_features_test,
    columns=word_feat_names,
    index=combined[combined["is_train"]==0].index,
)
combined = pd.concat([combined, pd.concat([word_df_train, word_df_test])], axis=1)

# Feature 8: TF-IDF features - fit only on training data to avoid leakage
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=300,
    min_df=3,
    stop_words="english",
    sublinear_tf=True,
    lowercase=True,
)
tfidf_vectorizer.fit(train_text_clean)
tfidf_features_train = tfidf_vectorizer.transform(train_text_clean)
tfidf_features_test = tfidf_vectorizer.transform(combined[combined["is_train"]==0]["clean_text"].fillna(""))
tfidf_feat_names = [f"tfidf_{i}" for i in range(tfidf_features_train.shape[1])]
tfidf_df_train = pd.DataFrame(
    tfidf_features_train.toarray() if hasattr(tfidf_features_train, "toarray") else tfidf_features_train,
    columns=tfidf_feat_names,
    index=combined[combined["is_train"]==1].index,
)
tfidf_df_test = pd.DataFrame(
    tfidf_features_test.toarray() if hasattr(tfidf_features_test, "toarray") else tfidf_features_test,
    columns=tfidf_feat_names,
    index=combined[combined["is_train"]==0].index,
)
combined = pd.concat([combined, pd.concat([tfidf_df_train, tfidf_df_test])], axis=1)


# Feature 9: Readability and complexity features
def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2:
        count += 1
    if count == 0:
        count = 1
    return count


def flesch_reading_ease(text):
    if not isinstance(text, str) or len(text.split()) < 2:
        return 0
    words = text.split()
    sentences = len(re.findall(r"[.!?]+", text))
    if sentences == 0:
        sentences = 1
    syllables = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


combined["flesch_score"] = combined["text"].apply(flesch_reading_ease)
combined["complex_word_ratio"] = combined["text"].apply(
    lambda x: (
        sum(1 for w in str(x).split() if len(w) > 6) / (len(str(x).split()) + 1)
        if isinstance(x, str)
        else 0
    )
)
combined["unique_word_ratio"] = combined["text"].apply(
    lambda x: (
        len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
        if isinstance(x, str)
        else 0
    )
)


# Feature 10: POS-like features
def get_pos_indicators(text):
    if not isinstance(text, str):
        return [0, 0, 0, 0]
    words = text.split()
    ing_count = sum(1 for w in words if w.endswith("ing"))
    ed_count = sum(1 for w in words if w.endswith("ed"))
    ly_count = sum(1 for w in words if w.endswith("ly"))
    tion_count = sum(1 for w in words if w.endswith("tion"))
    return [ing_count, ed_count, ly_count, tion_count]


pos_features = combined["text"].apply(get_pos_indicators)
combined["ing_suffix_count"] = pos_features.apply(lambda x: x[0])
combined["ed_suffix_count"] = pos_features.apply(lambda x: x[1])
combined["ly_suffix_count"] = pos_features.apply(lambda x: x[2])
combined["tion_suffix_count"] = pos_features.apply(lambda x: x[3])

# Feature 11: Length-based features
combined["short_word_ratio"] = combined["text"].apply(
    lambda x: (
        sum(1 for w in str(x).split() if len(w) <= 3) / (len(str(x).split()) + 1)
        if isinstance(x, str)
        else 0
    )
)
combined["long_word_ratio"] = combined["text"].apply(
    lambda x: (
        sum(1 for w in str(x).split() if len(w) >= 8) / (len(str(x).split()) + 1)
        if isinstance(x, str)
        else 0
    )
)

# Feature 12: Proper noun ratio
combined["proper_noun_ratio"] = combined["text"].apply(
    lambda x: (
        sum(1 for w in str(x).split() if w[0].isupper() and not w.isupper())
        / (len(str(x).split()) + 1)
        if isinstance(x, str) and len(x.split()) > 0
        else 0
    )
)

# Feature 13: Dialogue features
combined["dialogue_ratio"] = combined["text"].apply(
    lambda x: (
        (str(x).count('"') + str(x).count("'")) / (len(str(x)) + 1)
        if isinstance(x, str)
        else 0
    )
)
combined["has_quotes"] = combined["text"].apply(
    lambda x: 1 if isinstance(x, str) and ('"' in x or "'" in x) else 0
)

# Select numeric columns for final features
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["is_train"]
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

# Handle NaN values
combined[feature_cols] = combined[feature_cols].fillna(0)

# Split back into train and test
train_processed = combined[combined["is_train"] == 1].copy()
test_processed = combined[combined["is_train"] == 0].copy()

# Encode target
le = LabelEncoder()
train_processed["author_encoded"] = le.fit_transform(train_processed["author"])

# Scale features
scaler = StandardScaler()
train_features = train_processed[feature_cols].values
scaler.fit(train_features)
train_processed[feature_cols] = scaler.transform(train_features)
test_processed[feature_cols] = scaler.transform(test_processed[feature_cols].values)

# Save processed data
train_processed.to_pickle("./working/train_processed.pkl")
test_processed.to_pickle("./working/test_processed.pkl")
np.save("./working/feature_columns.npy", feature_cols)
joblib.dump(le, "./working/label_encoder.pkl")
joblib.dump(scaler, "./working/scaler.pkl")

print(f"Training set shape: {train_processed.shape}")
print(f"Test set shape: {test_processed.shape}")
print(f"Number of features: {len(feature_cols)}")

# Model hyperparameters
NUM_LABELS = 3
MAX_LENGTH = 1024
DROPOUT_RATE = 0.2
LABEL_SMOOTHING = 0.1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Initialize ModernBERT model and tokenizer
print("Loading ModernBERT-large...")
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_config = AutoConfig.from_pretrained(model_id)
model_config.num_labels = NUM_LABELS
model_config.hidden_dropout_prob = DROPOUT_RATE


# Loss function with label smoothing
class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


criterion = LabelSmoothCrossEntropyLoss(smoothing=LABEL_SMOOTHING)

# Training hyperparameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
MAX_GRAD_NORM = 1.0
NUM_FOLDS = 5
MODEL_SAVE_PATH = "./working/best_model"


# Dataset class
class AuthorDataset(Dataset):
    def __init__(
        self, texts, labels=None, tokenizer=None, max_length=512, is_test=False
    ):
        self.texts = texts.values if isinstance(texts, pd.Series) else texts
        self.labels = (
            labels.values
            if isinstance(labels, pd.Series) and labels is not None
            else labels
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Create stratified folds
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_scores = []

# LightGBM parameters
lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


# Training function
def train_epoch(model, loader, optimizer, scaler, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * gradient_accumulation_steps
    return total_loss / len(loader)


# Validation function
@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# Two-stage training
all_valid_preds = []
all_valid_labels = []
all_test_preds_deberta = []
all_test_preds_modernbert = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["author"])):
    torch.cuda.empty_cache()
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*50}")

    # Split data
    train_texts = train_df.iloc[train_idx]["text"]
    train_labels = (
        train_df.iloc[train_idx]["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
    )
    val_texts = train_df.iloc[val_idx]["text"]
    val_labels = (
        train_df.iloc[val_idx]["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
    )

    # Create datasets and dataloaders
    train_dataset = AuthorDataset(
        train_texts, train_labels, tokenizer, MAX_LENGTH
    )
    val_dataset = AuthorDataset(
        val_texts, val_labels, tokenizer, MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model for this fold
    fold_config = AutoConfig.from_pretrained(model_id)
    fold_config.num_labels = 3
    fold_config.hidden_dropout_prob = 0.2
    fold_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, config=fold_config
    ).to(device)
    fold_model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = AdamW(fold_model.parameters(), lr=2e-5, weight_decay=0.01)

    # Training loop
    print("Training ModernBERT...")
    scaler = GradScaler()
    best_model_loss = float("inf")
    patience_counter = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            fold_model,
            train_loader,
            optimizer,
            scaler,
            GRADIENT_ACCUMULATION_STEPS,
        )
        val_preds = validate(fold_model, val_loader)
        val_loss = log_loss(val_labels, val_preds)
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_model_loss:
            best_model_loss = val_loss
            patience_counter = 0
            torch.save(
                fold_model.state_dict(),
                f"{MODEL_SAVE_PATH}_fold{fold}.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    fold_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}_fold{fold}.pt"))

    # Get validation predictions
    fold_val_preds = validate(fold_model, val_loader)
    fold_scores.append(val_loss)

    # Get test predictions
    test_dataset = AuthorDataset(
        test_df["text"], tokenizer=tokenizer, max_length=MAX_LENGTH
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_test_preds = validate(fold_model, test_loader)

    print(f"Fold {fold + 1} Validation Log Loss: {val_loss:.4f}")

    # Store for ensemble
    all_valid_preds.append(fold_val_preds)
    all_valid_labels.append(val_labels)
    all_test_preds_modernbert.append(fold_test_preds)

# Ensemble all folds for final validation score
all_valid_preds_combined = np.concatenate(all_valid_preds, axis=0)
all_valid_labels_combined = np.concatenate(all_valid_labels, axis=0)
final_val_score = log_loss(all_valid_labels_combined, all_valid_preds_combined)
print(f"\n{'='*50}")
print(f"Final Validation Log Loss: {final_val_score:.4f}")
print(f"{'='*50}")

# Generate final test predictions using simple average of fold predictions
final_test_preds = np.mean(all_test_preds_modernbert, axis=0)
final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)
final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

# Create submission file
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": final_test_preds[:, 0],
        "HPL": final_test_preds[:, 1],
        "MWS": final_test_preds[:, 2],
    }
)
submission = submission[["id", "EAP", "HPL", "MWS"]]
submission.to_csv("./submission/submission_c41e99cd4be149b5a3335bc7f69beadd.csv", index=False)
print(f"Submission saved to ./submission/submission_c41e99cd4be149b5a3335bc7f69beadd.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")