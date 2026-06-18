import os
os.sched_setaffinity(0, {19, 58, 59, 60, 61})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import re
import os
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import gc


# ============ LOAD DATA ============
train_path = "./input/train.csv"
test_path = "./input/test.csv"
submission_path = "./input/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(submission_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============ TEXT CLEANING ============
def clean_text(text):
    """Basic text cleaning while preserving stylistic elements"""
    text = str(text).lower()
    # Preserve punctuation patterns (they're stylistically informative)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# ============ LEXICAL FEATURES ============
def extract_lexical_features(text):
    features = {}
    words = text.split()
    chars = list(text)

    # Basic statistics
    features["word_count"] = len(words)
    features["char_count"] = len(chars)
    features["avg_word_len"] = np.mean([len(w) for w in words]) if words else 0
    features["sentence_len"] = len(words)  # proxy for sentence length

    # Unique word ratio (vocabulary richness)
    features["unique_word_ratio"] = len(set(words)) / max(len(words), 1)

    # Punctuation density (stylistic marker)
    punct_count = sum(1 for c in chars if c in ".,!?;:'\"-()[]{}")
    features["punct_density"] = punct_count / max(len(chars), 1)

    # Specific punctuation frequencies
    features["comma_density"] = chars.count(",") / max(len(chars), 1)
    features["excl_density"] = chars.count("!") / max(len(chars), 1)
    features["quest_density"] = chars.count("?") / max(len(chars), 1)
    features["semicolon_density"] = chars.count(";") / max(len(chars), 1)
    features["colon_density"] = chars.count(":") / max(len(chars), 1)
    features["quote_density"] = (chars.count('"') + chars.count("'")) / max(
        len(chars), 1
    )
    features["dash_density"] = chars.count("-") / max(len(chars), 1)

    # Capitalization patterns (proportion of capitalized words)
    capitalized = sum(1 for w in text.split() if w[0].isupper() if w)
    features["capital_ratio"] = capitalized / max(len(words), 1)

    # Stopword ratio (common vs unique vocabulary usage)
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
            "with",
            "by",
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
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "then",
            "than",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "them",
            "their",
            "we",
            "us",
            "our",
            "you",
            "your",
        ]
    )
    stopword_count = sum(1 for w in words if w in stopwords)
    features["stopword_ratio"] = stopword_count / max(len(words), 1)

    # Character-level features (catching morphological patterns)
    features["vowel_ratio"] = sum(1 for c in chars if c in "aeiou") / max(len(chars), 1)
    features["consonant_ratio"] = sum(
        1 for c in chars if c.isalpha() and c not in "aeiou"
    ) / max(len(chars), 1)
    features["digit_ratio"] = sum(1 for c in chars if c.isdigit()) / max(len(chars), 1)
    features["space_ratio"] = chars.count(" ") / max(len(chars), 1)

    # Special character frequencies (thematic elements)
    features["ellipsis_flag"] = 1 if "..." in text else 0
    features["excl_quest_combined"] = (chars.count("!") + chars.count("?")) / max(
        len(chars), 1
    )

    return features


# Apply feature extraction
train_lexical = train_df["clean_text"].apply(extract_lexical_features)
test_lexical = test_df["clean_text"].apply(extract_lexical_features)

train_lexical_df = pd.DataFrame(train_lexical.tolist())
test_lexical_df = pd.DataFrame(test_lexical.tolist())

print(
    f"Lexical features shape - Train: {train_lexical_df.shape}, Test: {test_lexical_df.shape}"
)

# ============ N-GRAM TF-IDF FEATURES (CHARACTER LEVEL - captures authorial style) ============
# Character n-grams from 2-6 to catch word roots, affixes, and writing tics
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 6),
    max_features=5000,
    sublinear_tf=True,
    lowercase=True,
    strip_accents="unicode",
)

# Fit ONLY on training data
char_tfidf_train = char_vectorizer.fit_transform(train_df["clean_text"])
char_tfidf_test = char_vectorizer.transform(test_df["clean_text"])

print(
    f"Char n-gram TF-IDF shape - Train: {char_tfidf_train.shape}, Test: {char_tfidf_test.shape}"
)

# ============ WORD N-GRAM TF-IDF FEATURES (captures vocabulary/style) ============
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
    lowercase=True,
    strip_accents="unicode",
    min_df=3,  # remove rare words
)

# Fit ONLY on training data
word_tfidf_train = word_vectorizer.fit_transform(train_df["clean_text"])
word_tfidf_test = word_vectorizer.transform(test_df["clean_text"])

print(
    f"Word n-gram TF-IDF shape - Train: {word_tfidf_train.shape}, Test: {word_tfidf_test.shape}"
)

# ============ COMBINE ALL FEATURES ============
# Convert sparse matrices to dense for combination with lexical features
char_tfidf_train_dense = char_tfidf_train.toarray()
char_tfidf_test_dense = char_tfidf_test.toarray()
word_tfidf_train_dense = word_tfidf_train.toarray()
word_tfidf_test_dense = word_tfidf_test.toarray()

# Create DataFrames from n-gram features
char_feat_cols = [f"char_ngram_{i}" for i in range(char_tfidf_train_dense.shape[1])]
word_feat_cols = [f"word_ngram_{i}" for i in range(word_tfidf_train_dense.shape[1])]

train_char_df = pd.DataFrame(char_tfidf_train_dense, columns=char_feat_cols)
test_char_df = pd.DataFrame(char_tfidf_test_dense, columns=char_feat_cols)
train_word_df = pd.DataFrame(word_tfidf_train_dense, columns=word_feat_cols)
test_word_df = pd.DataFrame(word_tfidf_test_dense, columns=word_feat_cols)

# Combine all features
train_features = pd.concat(
    [
        train_lexical_df.reset_index(drop=True),
        train_char_df.reset_index(drop=True),
        train_word_df.reset_index(drop=True),
    ],
    axis=1,
)

test_features = pd.concat(
    [
        test_lexical_df.reset_index(drop=True),
        test_char_df.reset_index(drop=True),
        test_word_df.reset_index(drop=True),
    ],
    axis=1,
)

print(f"Combined train features: {train_features.shape}")
print(f"Combined test features: {test_features.shape}")

# ============ ENCODE TARGET ============
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)
print(f"Classes: {label_encoder.classes_}")

# ============ CREATE STRATIFIED SPLIT ============
# Use 80/20 train/validation split with stratification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get a single validation split (hold-out)
train_idx, val_idx = next(skf.split(train_features, train_labels))

X_train = train_features.iloc[train_idx].values
y_train = train_labels[train_idx]
X_val = train_features.iloc[val_idx].values
y_val = train_labels[val_idx]

X_test = test_features.values

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")

# ============ SAVE PREPROCESSED DATA ============
os.makedirs("./working", exist_ok=True)

# Save preprocessed arrays for later steps
np.save("./working/X_train.npy", X_train)
np.save("./working/y_train.npy", y_train)
np.save("./working/X_val.npy", X_val)
np.save("./working/y_val.npy", y_val)
np.save("./working/X_test.npy", X_test)

# Save feature names for interpretability
feature_names = train_features.columns.tolist()
np.save("./working/feature_names.npy", feature_names)

# Save label encoder classes
np.save("./working/label_classes.npy", label_encoder.classes_)

# Also save original text data for transformer-based models (future steps)
train_df.to_csv("./working/train_processed.csv", index=False)
test_df.to_csv("./working/test_processed.csv", index=False)

print("Preprocessed data saved successfully.")
print(f"Feature space: {len(feature_names)} features")
print(f"Classes: {list(label_encoder.classes_)}")


# ============ MODEL CONFIGURATION ============
class ModelConfig:
    model_name = "microsoft/deberta-v3-large"
    num_labels = 3
    max_length = 512
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    dropout = 0.1  # DeBERTa already has internal dropout


config = ModelConfig()


# ============ TRAINING CONFIGURATION ============
class TrainingConfig:
    batch_size = 16
    num_epochs = 5
    max_length = 384
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    gradient_accumulation_steps = 2
    early_stopping_patience = 3
    save_path = "./working/best_model_8f81daa6afa14378bb1d5f172ab7f629.pt"


train_cfg = TrainingConfig()


# ============ DATASET CLASS ============
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=384):
        self.texts = texts
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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============ LOAD TOKENIZER AND MODEL ============
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============ LOAD DATA ============
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Load preprocessed data from previous steps
X_train = np.load("./working/X_train.npy", allow_pickle=True)
y_train = np.load("./working/y_train.npy", allow_pickle=True)
X_val = np.load("./working/X_val.npy", allow_pickle=True)
y_val = np.load("./working/y_val.npy", allow_pickle=True)
X_test = np.load("./working/X_test.npy", allow_pickle=True)
label_classes = np.load("./working/label_classes.npy", allow_pickle=True)

print(
    f"Loaded data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
)

# ============ PREPARE DATALOADERS ============
# We use the original text for DeBERTa, but we need the text from train_df/test_df
# The texts are ordered the same as the numpy arrays from stratification
# Use the cleaned text from previous step if available, otherwise use raw text
try:
    train_texts = pd.read_csv("./working/train_processed.csv")["clean_text"].tolist()
    test_texts = pd.read_csv("./working/test_processed.csv")["clean_text"].tolist()
except:
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()

# Use the same stratification ordering from earlier step
# Reconstruct from the saved y_train/y_val to find original indices
train_idx = np.load("./working/y_train.npy", allow_pickle=True)  # dummy, we need proper indices
# Instead, reload feature arrays to get index mapping
X_train_check = np.load("./working/X_train.npy", allow_pickle=True)
X_val_check = np.load("./working/X_val.npy", allow_pickle=True)

# The train_idx/val_idx are already deterministic from the saved split
# Re-derive them with the exact same random state
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_labels = train_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
# We use the original train_df order (same as when features were created)
train_idx, val_idx = next(skf.split(np.zeros(len(train_df)), all_labels))

# Verify consistency with saved data
assert len(train_idx) == X_train_check.shape[0], "Train split size mismatch!"
assert len(val_idx) == X_val_check.shape[0], "Val split size mismatch!"

train_texts_subset = [train_texts[i] for i in train_idx]
val_texts_subset = [train_texts[i] for i in val_idx]

train_dataset = SpookyDataset(
    train_texts_subset, y_train, tokenizer, train_cfg.max_length
)
val_dataset = SpookyDataset(val_texts_subset, y_val, tokenizer, train_cfg.max_length)
test_dataset = SpookyDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=train_cfg.max_length
)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=train_cfg.batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=train_cfg.batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============ OPTIMIZER AND SCHEDULER ============
optimizer = AdamW(
    model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
)
total_steps = len(train_loader) * train_cfg.num_epochs
warmup_steps = int(total_steps * train_cfg.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# ============ MIXED PRECISION ============
scaler = torch.cuda.amp.GradScaler()

# ============ TRAINING LOOP ============
best_val_loss = float("inf")
patience_counter = 0
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(train_cfg.num_epochs):
    # Training
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss = loss / train_cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * train_cfg.gradient_accumulation_steps

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    # Apply clipping to avoid log(0) - match competition evaluation
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

    val_loss = log_loss(val_true, val_preds_clipped)

    print(
        f"Epoch {epoch+1}/{train_cfg.num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Log Loss: {val_loss:.4f}"
    )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), train_cfg.save_path)
        patience_counter = 0
        print(f"  -> Saved new best model (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= train_cfg.early_stopping_patience:
            print(f"  -> Early stopping triggered")
            break

print(f"Best Validation Log Loss: {best_val_loss:.4f}")

# ============ LOAD BEST MODEL AND EVALUATE ============
model.load_state_dict(torch.load(train_cfg.save_path))
model.eval()

# Validation evaluation with best model
val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(outputs.logits, dim=1)
        val_preds.append(probs.cpu().numpy())

val_preds = np.concatenate(val_preds, axis=0)
epsilon = 1e-15
val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(y_val, val_preds_clipped)
print(f"Final Validation Score: {final_val_score}")

# ============ TEST INFERENCE ============
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(outputs.logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)

# Normalize probabilities to sum to 1 per row
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# ============ CREATE SUBMISSION ============
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission_8f81daa6afa14378bb1d5f172ab7f629.csv", index=False)
print(f"Submission saved to ./submission/submission_8f81daa6afa14378bb1d5f172ab7f629.csv")
print(f"Submission shape: {submission.shape}")