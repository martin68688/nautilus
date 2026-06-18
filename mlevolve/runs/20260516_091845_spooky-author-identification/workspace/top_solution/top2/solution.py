"""
Merged Script: Spooky Author Identification
Combines feature engineering (Step 1) + model architecture (Step 2) + training/evaluation (Step 3)
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2
N_CHAR_NGRAMS = 2000
N_WORD_NGRAMS = 3000
N_FOLDS = 3
FOLD_NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED K-FOLD (3 folds)
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Store fold indices
fold_indices = []
for train_idx, val_idx in skf.split(train_df["text"].values, y_train_full):
    fold_indices.append((train_idx, val_idx))
    print(
        f"Fold - Train indices: {len(train_idx)}, Val indices: {len(val_idx)}"
    )

# For initial processing, use all data (will be split per fold during DeBERTa training)
X_all_texts = train_df["text"].values
y_all_labels = y_train_full
X_test_texts = test_df["text"].values
print(
    f"Total training samples: {len(X_all_texts)}, Test samples: {len(X_test_texts)}"
)

# ============================================================
# HANDCRAFTED FEATURES - Stylometric Features
# ============================================================
print("\n" + "=" * 60)
print("ENGINEERING HANDCRAFTED FEATURES")
print("=" * 60)

def extract_stylometric_features(texts):
    features = []
    for text in texts:
        text_str = str(text) if text else ""
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]

        char_count = len(text_str)
        word_count = len(words)
        sent_count = max(len(sentences), 1)
        avg_word_len = char_count / max(word_count, 1)
        avg_sent_len = word_count / sent_count

        upper_ratio = sum(1 for c in text_str if c.isupper()) / max(char_count, 1)
        lower_ratio = sum(1 for c in text_str if c.islower()) / max(char_count, 1)
        digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(char_count, 1)
        whitespace_ratio = sum(1 for c in text_str if c.isspace()) / max(char_count, 1)

        punct_ratios = []
        for p in [",", ".", "!", "?", ";", ":", "-", '"', "'", "(", ")", "--"]:
            punct_ratios.append(text_str.count(p) / max(char_count, 1))

        unique_chars = len(set(text_str.lower()))
        char_diversity = unique_chars / max(char_count, 1)

        long_words_ratio = sum(1 for w in words if len(w) > 6) / max(len(words), 1)
        capitalized_ratio = sum(1 for w in words if w and w[0].isupper()) / max(
            len(words), 1
        )
        all_caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(
            len(words), 1
        )

        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0

        function_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "of",
            "in",
            "to",
            "for",
            "with",
            "by",
            "at",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
        ]
        word_lower = text_str.lower().split()
        function_word_ratio = sum(1 for w in word_lower if w in function_words) / max(
            len(word_lower), 1
        )

        archaic_words = [
            "thee",
            "thou",
            "thy",
            "thine",
            "ye",
            "hath",
            "doth",
            "dost",
            "shalt",
            "wilt",
            "canst",
            "didst",
            "hast",
            "wert",
            "art",
            "wherein",
            "wherewith",
            "whereof",
            "thereunto",
            "thereto",
            "thence",
            "thither",
            "whilst",
            "amongst",
            "amidst",
            "betwixt",
            "unto",
            "ere",
            "anon",
            "hark",
            "forsooth",
            "perchance",
            "prithee",
            "alas",
            "tis",
            "twas",
            "twill",
            "twere",
            "oft",
            "ofttimes",
            "oftentimes",
            "henceforth",
            "henceforward",
            "heretofore",
            "hereafter",
            "thereafter",
            "nay",
            "aye",
            "yonder",
            "yon",
            "whence",
            "whither",
        ]
        archaic_ratio = sum(1 for w in word_lower if w in archaic_words) / max(
            len(word_lower), 1
        )

        emotional_words = [
            "horror",
            "dread",
            "terror",
            "fear",
            "dismay",
            "anguish",
            "agony",
            "torment",
            "suffering",
            "despair",
            "hopeless",
            "gloom",
            "gloomy",
            "darkness",
            "shadow",
            "phantom",
            "specter",
            "ghost",
            "apparition",
            "supernatural",
            "unearthly",
            "eerie",
            "creepy",
            "macabre",
            "grisly",
            "ghastly",
            "hideous",
            "frightful",
            "awful",
            "terrible",
            "horrible",
            "dreadful",
            "solemn",
            "melancholy",
            "mournful",
            "sorrow",
            "grief",
            "woe",
            "weary",
            "wretched",
            "misery",
            "pain",
            "torture",
            "madness",
            "insanity",
            "crazy",
            "lunatic",
            "frantic",
            "frenzy",
            "panic",
            "desolate",
            "forlorn",
            "abandoned",
            "lonely",
            "solitary",
            "bleak",
            "dreary",
            "ominous",
            "sinister",
            "menacing",
            "threatening",
            "portentous",
            "foreboding",
        ]
        emotional_ratio = sum(1 for w in word_lower if w in emotional_words) / max(
            len(word_lower), 1
        )

        lovecraft_words = [
            "cyclopean",
            "eldritch",
            "antediluvian",
            "squamous",
            "rugose",
            "ichor",
            "cryptic",
            "cacodemonical",
            "gibbous",
            "blasphemous",
            "nameless",
            "unspeakable",
            "indescribable",
            "ineffable",
            "unmentionable",
            "faceless",
            "formless",
            "shapeless",
            "void",
            "abyss",
            "chasm",
            "gulf",
            "pit",
            "fhtagn",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "azathoth",
            "nyarlathotep",
            "shoggoth",
            "mi-go",
            "necronomicon",
            "arkham",
            "innsmouth",
            "dunwich",
            "providence",
            "miskatonic",
            "kadath",
            "leng",
            "yuggoth",
        ]
        lovecraft_ratio = sum(1 for w in word_lower if w in lovecraft_words) / max(
            len(word_lower), 1
        )

        sub_conj = [
            "although",
            "though",
            "while",
            "whereas",
            "because",
            "since",
            "unless",
            "until",
            "after",
            "before",
            "once",
            "if",
            "when",
            "where",
            "whether",
            "that",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "whatever",
            "whichever",
            "whoever",
        ]
        sub_conj_ratio = sum(1 for w in word_lower if w in sub_conj) / max(
            len(word_lower), 1
        )

        features.append(
            [
                char_count,
                word_count,
                sent_count,
                avg_word_len,
                avg_sent_len,
                upper_ratio,
                lower_ratio,
                digit_ratio,
                whitespace_ratio,
            ]
            + punct_ratios
            + [
                char_diversity,
                long_words_ratio,
                capitalized_ratio,
                all_caps_ratio,
                sent_len_std,
                sent_len_var,
                function_word_ratio,
                archaic_ratio,
                emotional_ratio,
                lovecraft_ratio,
                sub_conj_ratio,
            ]
        )
    return np.array(features)

def create_readability_features(texts):
    features = []
    for text in texts:
        text_str = str(text) if text else ""
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        word_count = max(len(words), 1)
        sent_count = max(len(sentences), 1)

        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = "aeiou"
            count = 0
            for i, char in enumerate(word):
                if char in vowels and (i == 0 or word[i - 1] not in vowels):
                    count += 1
            if word.endswith("e"):
                count = max(count - 1, 1)
            if word.endswith("le") and len(word) > 2:
                count += 1
            return max(count, 1)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_word_ratio = complex_words / word_count

        flesch = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * avg_syllables
        ari = (
            4.71 * (len(text_str) / word_count)
            + 0.5 * (word_count / sent_count)
            - 21.43
        )

        features.append([flesch, ari, avg_syllables, complex_word_ratio])
    return np.array(features)

def create_pos_tag_approximation(texts):
    features = []
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ism",
        "ist",
        "or",
        "er",
        "ing",
    ]
    verb_suffixes = ["ate", "ify", "ize", "en", "ing", "ed"]
    adj_suffixes = [
        "ous",
        "ious",
        "eous",
        "ful",
        "less",
        "ive",
        "able",
        "ible",
        "al",
        "ial",
        "ic",
        "ical",
        "ish",
        "like",
        "ly",
        "y",
    ]
    adv_suffixes = ["ly", "wise", "ward", "wards", "ways", "style"]

    for text in texts:
        text_str = str(text) if text else ""
        words = text_str.split()
        word_count = max(len(words), 1)

        noun_count = verb_count = adj_count = adv_count = 0
        for word in words:
            w = word.lower()
            if len(w) >= 3:
                if any(w.endswith(s) for s in noun_suffixes):
                    noun_count += 1
                if any(w.endswith(s) for s in verb_suffixes):
                    verb_count += 1
                if any(w.endswith(s) for s in adj_suffixes):
                    adj_count += 1
                if any(w.endswith(s) for s in adv_suffixes):
                    adv_count += 1

        content_words = noun_count + verb_count + adj_count + adv_count
        features.append(
            [
                noun_count / word_count,
                verb_count / word_count,
                adj_count / word_count,
                adv_count / word_count,
                content_words / word_count,
            ]
        )
    return np.array(features)

# Extract stylometric features on full data
train_stylo = extract_stylometric_features(X_all_texts)
test_stylo = extract_stylometric_features(X_test_texts)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)

# Extract readability features on full data
train_read = create_readability_features(X_all_texts)
test_read = create_readability_features(X_test_texts)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
test_read_scaled = read_scaler.transform(test_read)

# Extract POS features on full data
train_pos = create_pos_tag_approximation(X_all_texts)
test_pos = create_pos_tag_approximation(X_test_texts)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# N-GRAM FEATURES
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
train_char_short = char_vectorizer_short.fit_transform(X_all_texts)
test_char_short = char_vectorizer_short.transform(X_test_texts)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_all_texts)
test_char_med = char_vectorizer_med.transform(X_test_texts)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_all_texts)
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
train_word = word_vectorizer.fit_transform(X_all_texts)
test_word = word_vectorizer.transform(X_test_texts)

char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vectorizer_short.fit_transform(X_all_texts)
test_char_short = char_vectorizer_short.transform(X_test_texts)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_all_texts)
test_char_med = char_vectorizer_med.transform(X_test_texts)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_all_texts)
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
train_word = word_vectorizer.fit_transform(X_all_texts)
test_word = word_vectorizer.transform(X_test_texts)

def extract_punctuation_sequence(text):
    return "".join([c for c in text if c in string.punctuation]) if text else ""

all_texts_for_punct = np.concatenate(
    [X_all_texts, X_test_texts]
)
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(X_all_texts)
train_punct = punct_features_all[:n_train]
test_punct = punct_features_all[n_train:]

# Chi-squared feature selection on n-grams
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Sparse train shape before chi2 selection: {train_sparse.shape}")

# Apply MaxAbsScaler to ensure non-negative values for chi2
scaler_sparse = MaxAbsScaler()
train_sparse_scaled = scaler_sparse.fit_transform(train_sparse)
test_sparse_scaled = scaler_sparse.transform(test_sparse)

# Chi-squared feature selection (k=10000)
chi2_selector = SelectKBest(chi2, k=10000)
train_sparse = chi2_selector.fit_transform(train_sparse_scaled, y_all_labels)
test_sparse = chi2_selector.transform(test_sparse_scaled)
print(f"Sparse train shape after chi2 selection: {train_sparse.shape}")

# ============================================================
# DEBERTA FINE-TUNING WITH 3-FOLD CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE WITH 3-FOLD CV")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize all data at once
all_train_encodings = tokenizer(
    list(X_all_texts),
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

def compute_log_loss(y_true, y_pred_proba):
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

def extract_embeddings_from_loader(model, loader):
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

# Initialize arrays for OOF predictions and test predictions
oof_train_embeddings = np.zeros((len(X_all_texts), 1024), dtype=np.float32)
oof_train_probs = np.zeros((len(X_all_texts), NUM_AUTHORS), dtype=np.float32)
test_embeddings_list = []
test_probs_list = []

for fold, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"\n{'='*40}")
    print(f"FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*40}")

    # Create model for this fold with differential learning rates
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )
    model.to(device)

    # Use standard AdamW optimizer with weight decay groups only
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

    # Cosine annealing with warm restarts (longer period for fewer epochs)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=3,
        T_mult=2,
        eta_min=1e-6
    )

    # Create fold datasets
    train_texts_fold = X_all_texts[train_idx]
    val_texts_fold = X_all_texts[val_idx]
    train_labels_fold = y_all_labels[train_idx]
    val_labels_fold = y_all_labels[val_idx]

    train_encodings_fold = {
        'input_ids': all_train_encodings['input_ids'][train_idx],
        'attention_mask': all_train_encodings['attention_mask'][train_idx],
    }
    val_encodings_fold = {
        'input_ids': all_train_encodings['input_ids'][val_idx],
        'attention_mask': all_train_encodings['attention_mask'][val_idx],
    }

    train_dataset = TensorDataset(
        train_encodings_fold['input_ids'],
        train_encodings_fold['attention_mask'],
        torch.tensor(train_labels_fold, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        val_encodings_fold['input_ids'],
        val_encodings_fold['attention_mask'],
        torch.tensor(val_labels_fold, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        test_encodings['input_ids'], test_encodings['attention_mask']
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

    # Linear warmup scheduler (separate from cosine)
    total_steps = len(train_loader) * FOLD_NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    linear_warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler() if torch.cuda.is_available() else None

    # Training loop for this fold (exactly 3 epochs)
    for epoch in range(FOLD_NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                linear_warmup_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Evaluate on validation set
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                val_preds.append(probs.cpu().numpy())
        val_preds = np.vstack(val_preds)
        val_loss = compute_log_loss(val_labels_fold, val_preds)
        val_acc = np.mean(np.argmax(val_preds, axis=1) == val_labels_fold)

        print(f"  Epoch {epoch+1}/{FOLD_NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Extract embeddings for OOF validation set
    val_loader_no_labels = DataLoader(
        TensorDataset(val_encodings_fold['input_ids'], val_encodings_fold['attention_mask']),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_val_embeddings = extract_embeddings_from_loader(model, val_loader_no_labels)
    oof_train_embeddings[val_idx] = fold_val_embeddings
    oof_train_probs[val_idx] = val_preds

    # Extract test embeddings and probs
    test_loader_no_labels = DataLoader(
        TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask']),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    fold_test_embeddings = extract_embeddings_from_loader(model, test_loader_no_labels)
    test_embeddings_list.append(fold_test_embeddings)

    # Also get test probs
    model.eval()
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())
    fold_test_probs = np.vstack(fold_test_probs)
    test_probs_list.append(fold_test_probs)

    # Clean up to save memory
    del model, optimizer, scheduler, linear_warmup_scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Average test embeddings across folds
test_embeddings = np.mean(test_embeddings_list, axis=0)
deberta_test_probs = np.mean(test_probs_list, axis=0)
print(f"\nOOF train embeddings: {oof_train_embeddings.shape}")
print(f"Test embeddings: {test_embeddings.shape}")

# Use only first fold's train/val split for XGBoost training (no leakage)
train_idx_first = fold_indices[0][0]
val_idx_first = fold_indices[0][1]

# Build training features only from first fold's training indices
xgb_train_features = np.hstack(
    [train_stylo_filtered[train_idx_first],
     train_read_scaled[train_idx_first],
     train_pos_scaled[train_idx_first],
     oof_train_embeddings[train_idx_first]]
)
xgb_val_features = np.hstack(
    [train_stylo_filtered[val_idx_first],
     train_read_scaled[val_idx_first],
     train_pos_scaled[val_idx_first],
     oof_train_embeddings[val_idx_first]]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled, test_embeddings]
)
print(f"XGBoost train features: {xgb_train_features.shape}")
print(f"XGBoost val features: {xgb_val_features.shape}")

xgb_train_labels = y_all_labels[train_idx_first]
xgb_val_labels = y_all_labels[val_idx_first]

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
    xgb_train_labels,
    eval_set=[(xgb_val_features, xgb_val_labels)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
# OOF for all training data: use the model to predict on all OOF embeddings
# (but this is only for ensemble weight optimization, not for evaluation)
xgb_oof_all = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled, oof_train_embeddings]
)
xgb_oof_probs = xgb_model.predict_proba(xgb_oof_all)
print(
    f"XGBoost validation log loss (fold1): {compute_log_loss(xgb_val_labels, xgb_val_probs):.4f}"
)

# ============================================================
# LOGISTIC REGRESSION (on chi2-selected n-grams)
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
lr_model.fit(train_sparse, y_all_labels)

lr_oof_probs = lr_model.predict_proba(train_sparse)
lr_val_probs = lr_model.predict_proba(train_sparse[val_idx_first])
lr_test_probs = lr_model.predict_proba(test_sparse)
print(
    f"Logistic Regression validation log loss (fold1): {compute_log_loss(y_all_labels[val_idx_first], lr_val_probs):.4f}"
)

# ============================================================
# DEBERTA VALIDATION PROBS (OOF)
# ============================================================
print("\nDeBERTa OOF validation log loss: ", end="")
deberta_val_loss = compute_log_loss(y_all_labels[val_idx_first], oof_train_probs[val_idx_first])
print(f"{deberta_val_loss:.4f}")
deberta_val_probs = oof_train_probs[val_idx_first]

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION (on OOF validation set)
# ============================================================
print("\nOptimizing ensemble weights on OOF validation...")
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
        ll = compute_log_loss(y_all_labels[val_idx_first], ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"deberta": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss on OOF: {best_ll:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
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

eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())
print(f"Final Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
