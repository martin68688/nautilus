import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# Path Configuration
# ============================================================
INFERENCE_DIR = "./input"
DATA_DIR = "./input"
TRAIN_CSV = "./input/train.csv"
TEST_CSV = "./input/test.csv"
OUTPUT_CSV = "./submission/submission.csv"
WORKING_DIR = "./working"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# Configuration Parameters
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Data Loading
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# Stratified Split (CRITICAL: NO INDEX_BUG)
# ============================================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

# Use numpy indexing directly to avoid INDEX_BUG
X_train_texts = train_df["text"].values[train_idx]
y_train_labels = y_train_full[train_idx]
X_val_texts = train_df["text"].values[val_idx]
y_val_labels = y_train_full[val_idx]
X_test_texts = test_df["text"].values

print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(X_test_texts)}"
)

# Verify no leakage
assert len(set(train_idx) & set(val_idx)) == 0, "INDEX_BUG: train and val sets overlap!"


# ============================================================
# Stylometric Feature Extraction Functions
# ============================================================
def extract_stylometric_features(texts):
    """Extract 30 stylometric features from text."""
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 30)
            continue

        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        chars = list(text_str)

        text_len = len(text_str)
        word_count = len(words) if words else 1
        sent_count = len(sentences) if sentences else 1

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        sent_lens = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lens) if len(sent_lens) > 1 else 0
        sent_len_var = np.var(sent_lens) if len(sent_lens) > 1 else 0

        upper_ratio = sum(1 for c in chars if c.isupper()) / max(len(chars), 1)
        lower_ratio = sum(1 for c in chars if c.islower()) / max(len(chars), 1)
        digit_ratio = sum(1 for c in chars if c.isdigit()) / max(len(chars), 1)
        whitespace_ratio = sum(1 for c in chars if c.isspace()) / max(len(chars), 1)
        char_diversity = len(set(chars)) / max(len(chars), 1)

        punct_ratios = []
        for punct in ["!", "?", ",", ".", ";", ":", "-", '"', "'", "(", ")", "..."]:
            count = text_str.count(punct)
            punct_ratios.append(count / max(len(chars), 1))

        long_words_ratio = sum(1 for w in words if len(w) > 6) / max(word_count, 1)
        capitalized_ratio = sum(1 for w in words if w[0].isupper()) / max(word_count, 1)
        all_caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(
            word_count, 1
        )

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
        }
        function_word_ratio = sum(
            1 for w in words if w.lower() in function_words
        ) / max(word_count, 1)

        archaic_words = {
            "thee",
            "thou",
            "thy",
            "thine",
            "ye",
            "hath",
            "doth",
            "dost",
            "art",
            "wert",
            "hast",
            "canst",
            "durst",
            "whence",
            "thence",
            "whither",
            "hither",
            "thither",
            "ere",
            "whilst",
            "betwixt",
            "methinks",
            "forsooth",
            "perchance",
            "prithee",
        }
        archaic_ratio = sum(1 for w in words if w.lower() in archaic_words) / max(
            word_count, 1
        )

        emotional_words = {
            "terror",
            "horror",
            "dread",
            "fear",
            "dismal",
            "gloomy",
            "dark",
            "shadow",
            "spectral",
            "ghost",
            "phantom",
            "dreadful",
            "awful",
            "solemn",
            "melancholy",
            "dreary",
            "weary",
            "sorrow",
            "anguish",
            "agony",
            "despair",
            "mournful",
            "lament",
            "woeful",
            "hideous",
        }
        emotional_ratio = sum(1 for w in words if w.lower() in emotional_words) / max(
            word_count, 1
        )

        lovecraft_words = {
            "cyclopean",
            "eldritch",
            "squamous",
            "fungoid",
            "ichor",
            "gibbous",
            "non-euclidean",
            "cosmic",
            "aeon",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "nyarlathotep",
            "azathoth",
            "necronomicon",
            "arkham",
            "innsmouth",
            "miskatonic",
            "kadath",
            "leng",
            "hyperborean",
            "primordial",
            "unspeakable",
            "unnameable",
            "indescribable",
            "blasphemous",
        }
        lovecraft_ratio = sum(1 for w in words if w.lower() in lovecraft_words) / max(
            word_count, 1
        )

        sub_conj = {
            "although",
            "because",
            "since",
            "unless",
            "while",
            "whereas",
            "though",
            "when",
            "where",
            "after",
            "before",
            "until",
            "if",
            "even",
            "whether",
            "than",
            "that",
            "which",
            "who",
            "whom",
            "whose",
        }
        sub_conj_ratio = sum(1 for w in words if w.lower() in sub_conj) / max(
            word_count, 1
        )

        feat = [
            text_len,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            sent_len_std,
            sent_len_var,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            char_diversity,
            long_words_ratio,
            capitalized_ratio,
            all_caps_ratio,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            sub_conj_ratio,
        ]
        feat.extend(punct_ratios)
        features.append(feat)

    return np.array(features)


def create_readability_features(texts):
    """Calculate readability metrics (4 features)."""
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 4)
            continue

        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words) if words else 1
        sent_count = len(sentences) if sentences else 1
        char_count = sum(len(w) for w in words)

        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = "aeiouy"
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / max(word_count, 1)
        complex_words = sum(1 for w in words if count_syllables(w) > 2)
        complex_word_ratio = complex_words / max(word_count, 1)

        flesch = (
            206.835
            - 1.015 * (word_count / sent_count)
            - 84.6 * (total_syllables / word_count)
        )
        flesch = max(0, min(100, flesch))

        if sent_count > 0 and word_count > 0:
            ari = (
                4.71 * (char_count / word_count)
                + 0.5 * (word_count / sent_count)
                - 21.43
            )
        else:
            ari = 0
        ari = max(0, ari)

        features.append([flesch, ari, avg_syllables, complex_word_ratio])

    return np.array(features)


def create_pos_tag_approximation(texts):
    """Approximate POS tags using suffix patterns (5 features)."""
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0.0] * 5)
            continue

        words = str(text).split()
        word_count = len(words) if words else 1

        noun_suffixes = ["tion", "ment", "ness", "ity", "ance", "ence", "ship", "dom"]
        noun_count = sum(
            1 for w in words if any(w.lower().endswith(suf) for suf in noun_suffixes)
        )
        noun_ratio = noun_count / word_count

        verb_suffixes = ["ed", "ing", "ate", "ize", "ify", "en"]
        verb_count = sum(
            1 for w in words if any(w.lower().endswith(suf) for suf in verb_suffixes)
        )
        verb_ratio = verb_count / word_count

        adj_suffixes = ["ful", "ous", "ive", "able", "ible", "less", "ic", "al", "ish"]
        adj_count = sum(
            1 for w in words if any(w.lower().endswith(suf) for suf in adj_suffixes)
        )
        adj_ratio = adj_count / word_count

        adv_count = sum(1 for w in words if w.lower().endswith("ly"))
        adv_ratio = adv_count / word_count

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
            "not",
            "no",
            "nor",
            "so",
            "if",
            "all",
            "any",
            "can",
            "may",
            "will",
        }
        content_words = sum(
            1 for w in words if len(w) > 3 and w.lower() not in function_words
        )
        content_ratio = content_words / word_count

        features.append([noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_ratio])

    return np.array(features)


# ============================================================
# Extract Handcrafted Features
# ============================================================
print("\nExtracting handcrafted features...")

# Stylometric features (30)
train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(X_test_texts)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

# Remove low variance features
variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)

# Readability features (4)
train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(X_test_texts)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

# POS approximation features (5)
train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(X_test_texts)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

print(
    f"Stylo filtered: {train_stylo_filtered.shape}, Read: {train_read_scaled.shape}, POS: {train_pos_scaled.shape}"
)

# ============================================================
# N-gram Features (Character + Word + Punctuation)
# ============================================================
print("Extracting n-gram features...")

char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vectorizer_short.fit_transform(X_train_texts)
val_char_short = char_vectorizer_short.transform(X_val_texts)
test_char_short = char_vectorizer_short.transform(X_test_texts)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_train_texts)
val_char_med = char_vectorizer_med.transform(X_val_texts)
test_char_med = char_vectorizer_med.transform(X_test_texts)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_train_texts)
val_char_long = char_vectorizer_long.transform(X_val_texts)
test_char_long = char_vectorizer_long.transform(X_test_texts)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vectorizer.fit_transform(X_train_texts)
val_word = word_vectorizer.transform(X_val_texts)
test_word = word_vectorizer.transform(X_test_texts)


def extract_punctuation_sequence(text):
    return "".join([c for c in text if c in string.punctuation]) if text else ""


all_texts_for_punct = np.concatenate([X_train_texts, X_val_texts, X_test_texts])
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(X_train_texts)
n_val = len(X_val_texts)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + n_val]
test_punct = punct_features_all[n_train + n_val :]

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Sparse train shape: {train_sparse.shape}")

# ============================================================
# DeBERTa Fine-tuning
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
)
model.to(device)

train_encodings = tokenizer(
    list(X_train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    list(X_val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_encodings = tokenizer(
    list(X_test_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(y_train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(y_val_labels, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

total_steps = len(train_loader) * NUM_EPOCHS
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler() if torch.cuda.is_available() else None


def compute_log_loss(y_true, y_pred_proba):
    """Compute multi-class logarithmic loss as specified in the evaluation section."""
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(NUM_AUTHORS):
            if y_true[i] == j:
                loss -= np.log(y_pred_proba[i, j])
    return loss / n


def evaluate_deberta(model, loader):
    """Evaluate DeBERTa model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds


# Training loop with early stopping
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, _ = evaluate_deberta(model, val_loader)

    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_deberta_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest DeBERTa model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")

# Load best model
model.load_state_dict(
    torch.load(f"{WORKING_DIR}/best_deberta_model.pt", map_location=device)
)

# ============================================================
# Extract DeBERTa Embeddings for XGBoost
# ============================================================
print("\nExtracting DeBERTa embeddings...")


def extract_embeddings(model, loader):
    """Extract CLS token embeddings from DeBERTa."""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)


# Create loaders without labels for embedding extraction
train_loader_no_labels = DataLoader(
    TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
val_loader_no_labels = DataLoader(
    TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader_no_labels = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

train_embeddings = extract_embeddings(model, train_loader_no_labels)
val_embeddings = extract_embeddings(model, val_loader_no_labels)
test_embeddings = extract_embeddings(model, test_loader_no_labels)
print(
    f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}"
)

# ============================================================
# XGBoost Classifier
# ============================================================
print("\nTraining XGBoost classifier...")

# Combine handcrafted features with DeBERTa embeddings
xgb_train_features = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled, train_embeddings]
)
xgb_val_features = np.hstack(
    [val_stylo_filtered, val_read_scaled, val_pos_scaled, val_embeddings]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled, test_embeddings]
)
print(f"XGBoost train features: {xgb_train_features.shape}")

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    xgb_train_features,
    y_train_labels,
    eval_set=[(xgb_val_features, y_val_labels)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
print(
    f"XGBoost validation log loss: {compute_log_loss(y_val_labels, xgb_val_probs):.4f}"
)

# ============================================================
# Logistic Regression Classifier
# ============================================================
print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
lr_model.fit(train_sparse, y_train_labels)

lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)
print(
    f"Logistic Regression validation log loss: {compute_log_loss(y_val_labels, lr_val_probs):.4f}"
)

# ============================================================
# DeBERTa Validation and Test Probabilities
# ============================================================
print("\nGetting DeBERTa probabilities...")

val_loader_eval = DataLoader(
    TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(y_val_labels, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

_, deberta_val_loss, deberta_val_probs = evaluate_deberta(model, val_loader_eval)
print(f"DeBERTa validation log loss: {deberta_val_loss:.4f}")

# Test inference
model.eval()
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())
deberta_test_probs = np.vstack(all_test_probs)

# ============================================================
# Ensemble Weight Optimization
# ============================================================
print("\nOptimizing ensemble weights...")

val_probas = {
    "deberta": deberta_val_probs,
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
}

best_ll = float("inf")
best_weights = None

for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = (
            w1 * val_probas["deberta"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(y_val_labels, ensemble_proba)

        if ll < best_ll:
            best_ll = ll
            best_weights = {"deberta": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# Generate Test Predictions and Submission
# ============================================================
print("\nGenerating submission...")

test_probas = {
    "deberta": deberta_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}

ensemble_test_probs = (
    best_weights["deberta"] * test_probas["deberta"]
    + best_weights["xgboost"] * test_probas["xgboost"]
    + best_weights["lr"] * test_probas["lr"]
)

# Apply probability clipping and row normalization (same as evaluation)
eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

# Create submission DataFrame
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

# Save submission
submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# Final Validation Score
# ============================================================
print(f"\nFinal Validation Score: {best_ll:.6f}")

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
