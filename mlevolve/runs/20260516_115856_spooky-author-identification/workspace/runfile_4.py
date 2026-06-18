import os
os.sched_setaffinity(0, {36, 37})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
# Paths
# ============================================================
DATA_DIR = "./input"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
WORKING_DIR = "./working"
OUTPUT_CSV = "./submission/submission_15d1ec7c12664762a2dc380cfbe79da0.csv"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# Configuration
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
# 1. LOAD DATA
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
# 2. STRATIFIED TRAIN/VAL SPLIT (using numpy indexing to avoid INDEX_BUG)
# ============================================================
print("\n" + "=" * 60)
print("TRAIN/VAL SPLIT")
print("=" * 60)

X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print(f"Training samples: {len(X_train_texts)}")
print(f"Validation samples: {len(X_val_texts)}")
print(f"Test samples: {len(test_df)}")

# Verifiy no overlap
assert (
    len(
        set(np.arange(len(X_train_texts)))
        & set(np.arange(len(X_train_texts)) + len(X_train_texts))
    )
    == 0
)

# ============================================================
# 3. STYLOMETRIC FEATURES (30 features)
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING STYLOMETRIC FEATURES (30)")
print("=" * 60)

ARCHAIC_WORDS = {
    "thou",
    "thee",
    "thy",
    "thine",
    "hath",
    "doth",
    "dost",
    "whence",
    "thence",
    "hither",
    "whither",
    "forsooth",
    "perchance",
    "methinks",
    "prithee",
    "anon",
    "ere",
    "nay",
    "yea",
    "therefor",
    "wherefore",
    "wouldst",
    "couldst",
    "shouldst",
    "didst",
    "hast",
}
EMOTIONAL_WORDS = {
    "love",
    "hate",
    "fear",
    "dread",
    "terror",
    "horror",
    "joy",
    "sorrow",
    "grief",
    "anguish",
    "despair",
    "hope",
    "desire",
    "longing",
    "passion",
    "rage",
    "wrath",
    "fury",
    "delight",
    "pleasure",
    "pain",
    "agony",
    "misery",
    "woe",
    "bliss",
    "ecstasy",
    "rapture",
    "lament",
}
LOVECRAFT_WORDS = {
    "elder",
    "outer",
    "cosmic",
    "nameless",
    "unspeakable",
    "forbidden",
    "ancient",
    "cyclopean",
    "non-euclidean",
    "blasphemous",
    "eldritch",
    "maddening",
    "indescribable",
    "cryptic",
    "antediluvian",
    "primordial",
    "prehistoric",
    "otherworldly",
    "ineffable",
    "unknowable",
    "abyss",
    "chasm",
    "void",
    "nightmare",
    "carcass",
    "slime",
    "fungus",
    "cylindrical",
    "ichor",
    "squamous",
    "rugose",
    "miasmal",
    "noisome",
    "gibbous",
    "necronomicon",
    "cthulhu",
    "yog-sothoth",
    "r'lyeh",
    "kadath",
    "arkham",
    "innsmouth",
    "dunwich",
    "miskatonic",
}
FUNCTION_WORDS = {
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
    "because",
    "although",
    "while",
    "if",
    "whether",
    "that",
    "which",
    "who",
    "whom",
    "this",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "need",
    "dare",
    "ought",
    "used",
    "about",
    "across",
    "along",
    "among",
    "around",
    "atop",
    "beside",
    "beyond",
    "inside",
    "outside",
    "within",
    "without",
    "upon",
    "toward",
    "towards",
    "throughout",
    "via",
    "notwithstanding",
    "despite",
}
SUB_CONJUNCTIONS = {
    "after",
    "although",
    "as",
    "because",
    "before",
    "even",
    "if",
    "once",
    "since",
    "though",
    "unless",
    "until",
    "when",
    "whenever",
    "where",
    "wherever",
    "while",
    "whereas",
    "whilst",
    "supposing",
    "provided",
    "providing",
    "considering",
    "given",
    "granted",
    "assuming",
}

PUNCTUATION_MARKS = list(string.punctuation)

def extract_stylometric_features(texts):
    """Extract 30 stylometric features from text array."""
    features = []
    for text in texts:
        if isinstance(text, str):
            text_len = len(text)
            words = text.split()
            word_count = len(words) if words else 1
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sent_count = len(sentences) if sentences else 1

            avg_word_len = (
                sum(len(w) for w in words) / word_count if word_count > 0 else 0
            )
            avg_sent_len = word_count / sent_count if sent_count > 0 else 0

            upper_ratio = (
                sum(1 for c in text if c.isupper()) / text_len if text_len > 0 else 0
            )
            lower_ratio = (
                sum(1 for c in text if c.islower()) / text_len if text_len > 0 else 0
            )
            digit_ratio = (
                sum(1 for c in text if c.isdigit()) / text_len if text_len > 0 else 0
            )
            whitespace_ratio = (
                sum(1 for c in text if c.isspace()) / text_len if text_len > 0 else 0
            )

            punct_ratios = []
            for punct in PUNCTUATION_MARKS:
                count = text.count(punct)
                punct_ratios.append(count / text_len if text_len > 0 else 0)

            char_diversity = len(set(text.lower())) / text_len if text_len > 0 else 0
            long_words = (
                sum(1 for w in words if len(w) >= 8) / word_count
                if word_count > 0
                else 0
            )
            capitalized = (
                sum(1 for w in words if w[0].isupper()) / word_count
                if word_count > 0
                else 0
            )
            all_caps = (
                sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
                if word_count > 0
                else 0
            )

            sent_lengths = [len(s.split()) for s in sentences]
            sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
            sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0

            words_lower = [
                w.lower().strip(string.punctuation)
                for w in words
                if w.strip(string.punctuation)
            ]
            word_count_clean = len(words_lower) if words_lower else 1

            function_word_ratio = (
                sum(1 for w in words_lower if w in FUNCTION_WORDS) / word_count_clean
            )
            archaic_ratio = (
                sum(1 for w in words_lower if w in ARCHAIC_WORDS) / word_count_clean
            )
            emotional_ratio = (
                sum(1 for w in words_lower if w in EMOTIONAL_WORDS) / word_count_clean
            )
            lovecraft_ratio = (
                sum(1 for w in words_lower if w in LOVECRAFT_WORDS) / word_count_clean
            )
            sub_conj_ratio = (
                sum(1 for w in words_lower if w in SUB_CONJUNCTIONS) / word_count_clean
            )

            feature_vector = [
                text_len,
                word_count,
                sent_count,
                avg_word_len,
                avg_sent_len,
                upper_ratio,
                lower_ratio,
                digit_ratio,
                whitespace_ratio,
                *punct_ratios,
                char_diversity,
                long_words,
                capitalized,
                all_caps,
                sent_len_std,
                sent_len_var,
                function_word_ratio,
                archaic_ratio,
                emotional_ratio,
                lovecraft_ratio,
                sub_conj_ratio,
            ]
            features.append(feature_vector)
        else:
            features.append([0] * 30)
    return np.array(features)

# Extract stylometric features
train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(test_df["text"].values)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)
print(f"Stylometric features after variance filtering: {train_stylo_filtered.shape[1]}")

# ============================================================
# 4. READABILITY FEATURES (4 features)
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING READABILITY FEATURES (4)")
print("=" * 60)

def create_readability_features(texts):
    """Extract 4 readability features."""
    features = []
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            word_count = len(words) if words else 1
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sent_count = len(sentences) if sentences else 1
            syllables = 0
            complex_words = 0
            for word in words:
                clean_word = word.strip(string.punctuation).lower()
                if clean_word:
                    vowels = "aeiouy"
                    syl_count = 0
                    prev_vowel = False
                    for char in clean_word:
                        if char in vowels:
                            if not prev_vowel:
                                syl_count += 1
                            prev_vowel = True
                        else:
                            prev_vowel = False
                    if syl_count == 0:
                        syl_count = 1
                    syllables += syl_count
                    if syl_count >= 3:
                        complex_words += 1

            avg_syllables = syllables / word_count if word_count > 0 else 0
            if word_count > 0 and sent_count > 0:
                flesch = (
                    206.835 - 1.015 * (word_count / sent_count) - 84.6 * avg_syllables
                )
                flesch = max(0, min(100, flesch))
            else:
                flesch = 0

            char_count = len([c for c in text if c.isalpha()])
            if word_count > 0 and sent_count > 0:
                ari = (
                    4.71 * (char_count / word_count)
                    + 0.5 * (word_count / sent_count)
                    - 21.43
                )
                ari = max(0, ari)
            else:
                ari = 0

            complex_word_ratio = complex_words / word_count if word_count > 0 else 0
            feature_vector = [flesch, ari, avg_syllables, complex_word_ratio]
            features.append(feature_vector)
        else:
            features.append([0, 0, 0, 0])
    return np.array(features)

train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

# ============================================================
# 5. POS TAG APPROXIMATION FEATURES (5 features)
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING POS APPROXIMATION FEATURES (5)")
print("=" * 60)

NOUN_SUFFIXES = [
    "tion",
    "sion",
    "ment",
    "ness",
    "ity",
    "ence",
    "ance",
    "ship",
    "dom",
    "hood",
    "ist",
    "ism",
    "age",
    "ure",
    "al",
    "ing",
    "er",
    "or",
    "ant",
    "ent",
]
VERB_SUFFIXES = ["ate", "ify", "ize", "ise", "en", "ing", "ed", "s", "es", "ed"]
ADJ_SUFFIXES = [
    "able",
    "ible",
    "al",
    "ial",
    "ary",
    "ory",
    "ful",
    "ic",
    "ical",
    "ive",
    "less",
    "ous",
    "eous",
    "ious",
    "y",
    "ish",
    "an",
    "ian",
    "ar",
    "ile",
    "ine",
    "like",
    "ly",
    "ose",
    "some",
]
ADV_SUFFIXES = ["ly", "ward", "wards", "wise", "ways"]

def create_pos_tag_approximation(texts):
    """Extract 5 POS approximation features."""
    features = []
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            word_count = len(words) if words else 1
            noun_count = 0
            verb_count = 0
            adj_count = 0
            adv_count = 0
            content_word_count = 0

            for word in words:
                clean_word = word.strip(string.punctuation).lower()
                if not clean_word or not clean_word[0].isalpha():
                    continue
                content_word_count += 1
                if any(clean_word.endswith(suffix) for suffix in NOUN_SUFFIXES):
                    noun_count += 1
                if any(clean_word.endswith(suffix) for suffix in VERB_SUFFIXES):
                    verb_count += 1
                if any(clean_word.endswith(suffix) for suffix in ADJ_SUFFIXES):
                    adj_count += 1
                if any(clean_word.endswith(suffix) for suffix in ADV_SUFFIXES):
                    adv_count += 1

            noun_ratio = noun_count / word_count if word_count > 0 else 0
            verb_ratio = verb_count / word_count if word_count > 0 else 0
            adj_ratio = adj_count / word_count if word_count > 0 else 0
            adv_ratio = adv_count / word_count if word_count > 0 else 0
            content_word_ratio = (
                content_word_count / word_count if word_count > 0 else 0
            )

            feature_vector = [
                noun_ratio,
                verb_ratio,
                adj_ratio,
                adv_ratio,
                content_word_ratio,
            ]
            features.append(feature_vector)
        else:
            features.append([0, 0, 0, 0, 0])
    return np.array(features)

train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# 6. N-GRAM FEATURES (Character & Word)
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING N-GRAM FEATURES")
print("=" * 60)

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
test_char_short = char_vectorizer_short.transform(test_df["text"].values)

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
test_char_med = char_vectorizer_med.transform(test_df["text"].values)

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
test_char_long = char_vectorizer_long.transform(test_df["text"].values)

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
test_word = word_vectorizer.transform(test_df["text"].values)

# Punctuation sequence features
def extract_punctuation_sequence(text):
    return (
        "".join([c for c in text if c in string.punctuation])
        if isinstance(text, str)
        else ""
    )

train_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_train_texts]
val_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_val_texts]
test_punct_sequences = [extract_punctuation_sequence(str(t)) for t in test_df["text"].values]

punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
train_punct = punct_vectorizer.fit_transform(train_punct_sequences)
val_punct = punct_vectorizer.transform(val_punct_sequences)
test_punct = punct_vectorizer.transform(test_punct_sequences)

# Combine sparse features
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
# 7. DEBERTA FINE-TUNING
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
    list(test_df["text"].values),
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
    test_encodings["input_ids"], test_encodings["attention_mask"]
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
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        loss -= np.log(y_pred_proba[i, y_true[i]])
    return loss / n

def evaluate_deberta(model, loader):
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
model.load_state_dict(
    torch.load(f"{WORKING_DIR}/best_deberta_model.pt", map_location=device)
)

# ============================================================
# 8. EXTRACT DEBERTA EMBEDDINGS
# ============================================================
print("\nExtracting DeBERTa embeddings...")

def extract_embeddings(model, loader):
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
# 9. XGBOOST ON DEBERTA EMBEDDINGS + HANDCRAFTED FEATURES
# ============================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

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
xgb_val_ll = compute_log_loss(y_val_labels, xgb_val_probs)
print(f"XGBoost validation log loss: {xgb_val_ll:.4f}")

# ============================================================
# 10. LOGISTIC REGRESSION ON SPARSE N-GRAMS
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

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
lr_val_ll = compute_log_loss(y_val_labels, lr_val_probs)
print(f"Logistic Regression validation log loss: {lr_val_ll:.4f}")

# ============================================================
# 11. DEBERTA VALIDATION & TEST PROBABILITIES
# ============================================================
print("\n" + "=" * 60)
print("GETTING DEBERTA PROBABILITIES")
print("=" * 60)

_, deberta_val_loss, deberta_val_probs = evaluate_deberta(model, val_loader)
print(f"DeBERTa validation log loss: {deberta_val_loss:.4f}")

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
# 12. ENSEMBLE WEIGHT OPTIMIZATION (GRID SEARCH)
# ============================================================
print("\n" + "=" * 60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("=" * 60)

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
print(f"Ensemble validation log loss: {best_ll:.6f}")

# ============================================================
# 13. GENERATE TEST ENSEMBLE PREDICTIONS & SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

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

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
print(f"\nFinal Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()