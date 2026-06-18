import pandas as pd
import numpy as np
import re
import string
import pickle
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, ModernBertForSequenceClassification
import xgboost as xgb

warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

def extract_lexical_features(text):
    """Extract lexical and readability features from text."""
    words = word_tokenize(str(text).lower())
    sentences = sent_tokenize(str(text))
    n_words = len(words)
    n_chars = len(str(text))
    n_sentences = len(sentences)
    n_unique = len(set(words))
    avg_word_len = n_chars / max(n_words, 1)
    long_words = sum(1 for w in words if len(w) > 6)
    long_words_ratio = long_words / max(n_words, 1)
    type_token_ratio = n_unique / max(n_words, 1)
    hapax = sum(1 for w in words if words.count(w) == 1) / max(n_words, 1)
    avg_sent_len = n_words / max(n_sentences, 1)

    features = {
        "n_words": n_words,
        "n_chars": n_chars,
        "n_sentences": n_sentences,
        "n_unique_words": n_unique,
        "avg_word_len": avg_word_len,
        "long_words_ratio": long_words_ratio,
        "type_token_ratio": type_token_ratio,
        "hapax_legomena_ratio": hapax,
        "char_per_word": avg_word_len,
        "avg_sent_len": avg_sent_len,
        "std_sent_len": np.std([len(s.split()) for s in sentences]) if n_sentences > 1 else 0.0,
        "max_sent_len": max([len(s.split()) for s in sentences], default=0),
        "min_sent_len": min([len(s.split()) for s in sentences], default=0),
    }
    return features

def extract_syntactic_features(text):
    """Extract POS tag based features."""
    words = word_tokenize(str(text))
    pos_tags = pos_tag(words)
    pos_counts = {"noun": 0, "verb": 0, "adj": 0, "adv": 0, "pronoun": 0}
    for word, tag in pos_tags:
        if "NN" in tag:
            pos_counts["noun"] += 1
        elif "VB" in tag:
            pos_counts["verb"] += 1
        elif "JJ" in tag:
            pos_counts["adj"] += 1
        elif "RB" in tag:
            pos_counts["adv"] += 1
        elif "PRP" in tag:
            pos_counts["pronoun"] += 1

    n_words = max(len(words), 1)
    features = {k + "_ratio": v / n_words for k, v in pos_counts.items()}
    return features

def extract_punctuation_features(text):
    """Extract punctuation density features."""
    text_str = str(text)
    n_chars = max(len(text_str), 1)
    punct_counts = {
        "punct_density": sum(1 for c in text_str if c in string.punctuation) / n_chars,
        "comma_ratio": text_str.count(",") / n_chars,
        "semicolon_ratio": text_str.count(";") / n_chars,
        "colon_ratio": text_str.count(":") / n_chars,
        "exclamation_ratio": text_str.count("!") / n_chars,
        "question_ratio": text_str.count("?") / n_chars,
        "quote_ratio": (text_str.count('"') + text_str.count("'")) / n_chars,
        "dash_ratio": text_str.count("-") / n_chars,
    }
    return punct_counts

def extract_readability_features(text):
    """Extract readability metrics."""
    words = word_tokenize(str(text))
    sentences = sent_tokenize(str(text))
    n_words = max(len(words), 1)
    n_sentences = max(len(sentences), 1)

    # Syllable count helper
    def count_syllables(word):
        word = word.lower().strip()
        if not word:
            return 1
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        for char in word:
            if char in vowels:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        if count == 0:
            count = 1
        return count

    syllables_per_word = sum(count_syllables(w) for w in words) / n_words
    poly_syllable_words = sum(1 for w in words if count_syllables(w) >= 3)
    poly_syllable_word_ratio = poly_syllable_words / n_words
    avg_words_per_sentence = n_words / n_sentences

    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * syllables_per_word
    flesch = max(0, min(100, flesch))

    # Automated Readability Index
    n_chars = len(str(text))
    ari = (4.71 * (n_chars / n_words)) + (0.5 * (n_words / n_sentences)) - 21.43

    # Coleman-Liau Index
    L = (n_chars / n_words) * 100
    S = (n_sentences / n_words) * 100
    coleman_liau = 0.0588 * L - 0.296 * S - 15.8

    # SMOG Index
    smog = (1.043 * (poly_syllable_words * (30 / n_sentences)) ** 0.5) + 3.1291

    features = {
        "flesch_reading_ease": flesch,
        "automated_readability_index": ari,
        "coleman_liau_index": coleman_liau,
        "smog_index": smog,
        "avg_words_per_sentence": avg_words_per_sentence,
        "poly_syllable_word_ratio": poly_syllable_word_ratio,
        "syllables_per_word": syllables_per_word,
    }
    return features

# ============================================================
# 3. EXTRACT STYLOMETRIC FEATURES
# ============================================================
print("Extracting lexical features...")
lex_train = train_df["text"].apply(lambda x: pd.Series(extract_lexical_features(x)))
lex_test = test_df["text"].apply(lambda x: pd.Series(extract_lexical_features(x)))

print("Extracting syntactic features...")
syn_train = train_df["text"].apply(lambda x: pd.Series(extract_syntactic_features(x)))
syn_test = test_df["text"].apply(lambda x: pd.Series(extract_syntactic_features(x)))

print("Extracting punctuation features...")
punct_train = train_df["text"].apply(lambda x: pd.Series(extract_punctuation_features(x)))
punct_test = test_df["text"].apply(lambda x: pd.Series(extract_punctuation_features(x)))

print("Extracting readability features...")
read_train = train_df["text"].apply(lambda x: pd.Series(extract_readability_features(x)))
read_test = test_df["text"].apply(lambda x: pd.Series(extract_readability_features(x)))

# Sentiment features
print("Extracting sentiment features...")
try:
    sia = SentimentIntensityAnalyzer()
    sent_train = train_df["text_clean"].apply(lambda x: pd.Series(sia.polarity_scores(str(x))))
    sent_test = test_df["text_clean"].apply(lambda x: pd.Series(sia.polarity_scores(str(x))))
except Exception:
    sent_train = pd.DataFrame({"vader_compound": [0.0] * len(train_df)}, index=train_df.index)
    sent_test = pd.DataFrame({"vader_compound": [0.0] * len(test_df)}, index=test_df.index)

# Character n-gram TF-IDF features
print("Creating character n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=15000, sublinear_tf=True
)
char_train = char_vectorizer.fit_transform(train_df["text_clean"].values)
char_test = char_vectorizer.transform(test_df["text_clean"].values)
char_train_df = pd.DataFrame(
    char_train.toarray(), columns=[f"char_{i}" for i in range(char_train.shape[1])]
)
char_test_df = pd.DataFrame(
    char_test.toarray(), columns=[f"char_{i}" for i in range(char_test.shape[1])]
)
print(f"Character n-gram features: {char_train.shape[1]}")

# Word n-gram TF-IDF features
print("Creating word n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 3), max_features=20000, sublinear_tf=True
)
word_train = word_vectorizer.fit_transform(train_df["text_clean"].values)
word_test = word_vectorizer.transform(test_df["text_clean"].values)
word_train_df = pd.DataFrame(
    word_train.toarray(), columns=[f"word_{i}" for i in range(word_train.shape[1])]
)
word_test_df = pd.DataFrame(
    word_test.toarray(), columns=[f"word_{i}" for i in range(word_test.shape[1])]
)
print(f"Word n-gram features: {word_train.shape[1]}")

# ============================================================
# 4. COMBINE ALL FEATURES & SCALING
# ============================================================
X_train_feats = pd.concat(
    [
        lex_train.reset_index(drop=True),
        syn_train.reset_index(drop=True),
        punct_train.reset_index(drop=True),
        read_train.reset_index(drop=True),
        sent_train.reset_index(drop=True),
        char_train_df.reset_index(drop=True),
        word_train_df.reset_index(drop=True),
    ],
    axis=1,
)

X_test_feats = pd.concat(
    [
        lex_test.reset_index(drop=True),
        syn_test.reset_index(drop=True),
        punct_test.reset_index(drop=True),
        read_test.reset_index(drop=True),
        sent_test.reset_index(drop=True),
        char_test_df.reset_index(drop=True),
        word_test_df.reset_index(drop=True),
    ],
    axis=1,
)

# Label encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(train_df["author"])
class_names = le.classes_
print(f"Class labels: {class_names}")
print(f"Training features shape: {X_train_feats.shape}")
print(f"Test features shape: {X_test_feats.shape}")

# Scale numeric features
numeric_cols = [
    "n_words", "n_chars", "n_sentences", "n_unique_words", "avg_word_len",
    "long_words_ratio", "type_token_ratio", "hapax_legomena_ratio", "char_per_word",
    "avg_sent_len", "std_sent_len", "max_sent_len", "min_sent_len",
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio",
    "punct_density", "comma_ratio", "semicolon_ratio", "colon_ratio",
    "exclamation_ratio", "question_ratio", "quote_ratio", "dash_ratio",
    "flesch_reading_ease", "automated_readability_index", "coleman_liau_index",
    "smog_index", "avg_words_per_sentence", "poly_syllable_word_ratio",
    "syllables_per_word", "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]
cols_to_scale = [col for col in numeric_cols if col in X_train_feats.columns]
scaler = StandardScaler()
X_train_scaled = X_train_feats.copy()
X_test_scaled = X_test_feats.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_feats[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test_feats[cols_to_scale])

# ============================================================
# 4. PREPARE MODERNBERT DATASET
# ============================================================
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
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

test_dataset = TextDataset(test_df["text"].values, None, tokenizer)

# ============================================================
# 5. 5-FOLD STRATIFIED CROSS-VALIDATION ENSEMBLE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
n_epochs = 10
learning_rate = 2e-5
batch_size = 16
effective_batch_size = 64
gradient_accumulation_steps = effective_batch_size // batch_size  # = 4
early_stopping_patience = 3

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_test_preds = []
fold_val_loglosses = []
criterion = nn.CrossEntropyLoss()

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_df)), y_train_encoded)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*60}")

    # Split data
    train_texts = train_df["text"].values[train_idx]
    train_labels = y_train_encoded[train_idx]
    val_texts = train_df["text"].values[val_idx]
    val_labels = y_train_encoded[val_idx]

    # Create dataloaders
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model for this fold
    model = ModernBertForSequenceClassification.from_pretrained(model_id, num_labels=3)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Scheduler: cosine with 20% warmup
    total_steps = (len(train_loader) // gradient_accumulation_steps) * n_epochs
    warmup_steps = int(0.2 * total_steps)

    from torch.optim.lr_scheduler import LambdaLR
    import math

    def cosine_schedule_with_warmup(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)

    best_val_log_loss = float("inf")
    best_model_state = None
    early_stopping_counter = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=-1)
                val_preds.append(probs.cpu().numpy())
                val_true.append(labels.cpu().numpy())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.concatenate(val_preds, axis=0)
        val_true = np.concatenate(val_true, axis=0)
        val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_log_loss = log_loss(val_true, val_preds_clipped)

        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
        )

        if val_log_loss < best_val_log_loss:
            best_val_log_loss = val_log_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
            print(f"  New best model! Val LogLoss: {best_val_log_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    model.load_state_dict(best_model_state)
    fold_val_loglosses.append(best_val_log_loss)
    print(f"Fold {fold+1} best Val LogLoss: {best_val_log_loss:.4f}")

    # Generate test predictions for this fold
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model.eval()
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            fold_test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(fold_test_probs, axis=0)
    fold_test_preds.append(fold_test_probs)

# Average predictions across folds
test_preds = np.mean(fold_test_preds, axis=0)
avg_fold_val_logloss = np.mean(fold_val_loglosses)
print(f"\n{'='*60}")
print(f"Average validation log loss across folds: {avg_fold_val_logloss:.4f}")
print(f"{'='*60}")

# ============================================================
# 6. GENERATE SUBMISSION
# ============================================================
print("Generating test predictions from ensemble...")
test_preds_clipped = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds_normalized = test_preds_clipped / test_preds_clipped.sum(
    axis=1, keepdims=True
)

test_ids = test_df["id"].values
submission = pd.DataFrame(
    {
        "id": test_ids,
        class_names[0]: test_preds_normalized[:, 0],
        class_names[1]: test_preds_normalized[:, 1],
        class_names[2]: test_preds_normalized[:, 2],
    }
)
submission = submission[["id", "EAP", "HPL", "MWS"]]

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(submission.head())

final_score = avg_fold_val_logloss
print(f"Final Average Validation Score: {final_score}")