import os
os.sched_setaffinity(0, {62, 63})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import re
import string
import os
import warnings
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import joblib

warnings.filterwarnings("ignore")

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
    if not isinstance(text, str):
        return ""
    text_clean = text.strip()
    text_clean = re.sub(r"\s+", " ", text_clean)
    return text_clean


train_df["cleaned_text"] = train_df["text"].apply(clean_text)
test_df["cleaned_text"] = test_df["text"].apply(clean_text)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    prev_char_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_char_vowel:
            count += 1
        prev_char_vowel = is_vowel
    if word.endswith("e"):
        count = max(count - 1, 1)
    if count == 0:
        count = 1
    return count


def extract_basic_features(text):
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]

    features = {}
    features["word_count"] = len(words)
    features["char_count"] = len(text)
    features["avg_word_len"] = np.mean([len(w) for w in words]) if words else 0
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_sentence_len_words"] = len(words) / features["sentence_count"]
    features["avg_sentence_len_chars"] = len(text) / features["sentence_count"]

    punct_counts = Counter(text)
    features["period_count"] = punct_counts.get(".", 0)
    features["comma_count"] = punct_counts.get(",", 0)
    features["exclamation_count"] = punct_counts.get("!", 0)
    features["question_count"] = punct_counts.get("?", 0)
    features["semicolon_count"] = punct_counts.get(";", 0)
    features["colon_count"] = punct_counts.get(":", 0)
    features["dash_count"] = text.count("--") + text.count("—")
    features["quote_count"] = (
        text.count('"') + text.count('"') + text.count("'") + text.count("'")
    )
    features["paren_count"] = text.count("(") + text.count(")")

    total_punct = sum(
        features[k]
        for k in [
            "period_count",
            "comma_count",
            "exclamation_count",
            "question_count",
            "semicolon_count",
            "colon_count",
        ]
    )
    features["punct_ratio"] = total_punct / max(len(words), 1)

    features["capital_ratio"] = sum(1 for w in words if w[0].isupper()) / max(
        len(words), 1
    )
    features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)

    stopwords = {
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
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "are",
        "it",
        "its",
        "that",
        "this",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
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
        "our",
        "their",
        "not",
    }
    words_lower = text.lower().split()
    features["stopword_ratio"] = sum(1 for w in words_lower if w in stopwords) / max(
        len(words_lower), 1
    )

    features["unique_word_ratio"] = len(set(words_lower)) / max(len(words_lower), 1)

    total_syllables = sum(count_syllables(w) for w in words)
    features["syllable_count"] = total_syllables
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["avg_sentence_len_words"]
        - 84.6 * (total_syllables / max(len(words), 1))
    )

    return features


def extract_char_pattern_features(text):
    features = {}
    vowels = sum(1 for c in text if c in "aeiouAEIOU")
    consonants = sum(1 for c in text if c.isalpha() and c not in "aeiouAEIOU")
    features["vowel_ratio"] = vowels / max(consonants + vowels, 1)
    features["whitespace_ratio"] = text.count(" ") / max(len(text), 1)
    features["digit_count"] = sum(1 for c in text if c.isdigit())
    rare_chars = ["æ", "œ", "à", "è", "é", "ê", "ë", "ï", "ô", "ö", "ü", "û", "ÿ", "ñ"]
    features["rare_char_count"] = sum(1 for c in text.lower() if c in rare_chars)
    contraction_pattern = r"\b(can't|don't|won't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|didn't|it's|i'm|i'll|i've|i'd|you're|you'll|you've|you'd|he's|he'll|she's|she'll|we're|we'll|we've|we'd|they're|they'll|they've|they'd|that's|there's|here's)\b"
    features["contraction_count"] = len(re.findall(contraction_pattern, text.lower()))
    return features


basic_features_train = train_df["cleaned_text"].apply(
    lambda x: extract_basic_features(x)
)
basic_features_test = test_df["cleaned_text"].apply(lambda x: extract_basic_features(x))
char_pattern_train = train_df["cleaned_text"].apply(
    lambda x: extract_char_pattern_features(x)
)
char_pattern_test = test_df["cleaned_text"].apply(
    lambda x: extract_char_pattern_features(x)
)

basic_train_df = pd.DataFrame(basic_features_train.tolist())
basic_test_df = pd.DataFrame(basic_features_test.tolist())
char_train_df = pd.DataFrame(char_pattern_train.tolist())
char_test_df = pd.DataFrame(char_pattern_test.tolist())

# N-gram features
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)
char_ngrams_train = char_vectorizer.fit_transform(train_df["cleaned_text"])
char_ngrams_test = char_vectorizer.transform(test_df["cleaned_text"])

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=10000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    min_df=3,
    max_df=0.8,
)
word_ngrams_train = word_vectorizer.fit_transform(train_df["cleaned_text"])
word_ngrams_test = word_vectorizer.transform(test_df["cleaned_text"])

# Author vocabulary features
eap_words = [
    "nevermore",
    "raven",
    "chamber",
    "sepulchre",
    "pallid",
    "ghastly",
    "drear",
    "dreary",
    "ebony",
    "visage",
    "ominous",
    "uncanny",
    "preternatural",
    "terrified",
    "agony",
    "despair",
    "chilling",
    "horror",
    "phantasm",
    "spectre",
    "ghost",
    "dying",
    "death",
    "pale",
    "grew",
    "feeling",
    "knew",
    "eyes",
    "door",
    "never",
    "more",
]
hpl_words = [
    "eldritch",
    "cyclopean",
    "gibbering",
    "blasphemous",
    "squamous",
    "subliminal",
    "indescribable",
    "non-euclidean",
    "cosmic",
    "crawling",
    "antediluvian",
    "cryptic",
    "nameless",
    "unspeakable",
    "primordial",
    "abnormal",
    "blasphemy",
    "inconceivable",
    "amorphous",
    "night-gaunt",
    "necronomicon",
    "yogsothoth",
    "cthulhu",
    "rlyeh",
    "dunwich",
    "innsmouth",
    "arkham",
    "miskatonic",
    "tentacle",
    "abyss",
]
mws_words = [
    "created",
    "monster",
    "dearest",
    "beloved",
    "friend",
    "nature",
    "mountain",
    "glacier",
    "sunset",
    "sublime",
    "sympathy",
    "affection",
    "cottage",
    "wretch",
    "demon",
    "fate",
    "destiny",
    "science",
    "discovery",
    "switzerland",
    "geneva",
    "ingolstadt",
    "clerval",
    "elizabeth",
    "justine",
    "pursuit",
    "vengeance",
    "miserable",
    "unhappy",
    "forlorn",
]


def count_author_vocab(text, word_list):
    words = text.lower().split()
    return sum(1 for w in words if w in word_list)


train_df["eap_vocab_count"] = train_df["cleaned_text"].apply(
    lambda x: count_author_vocab(x, eap_words)
)
train_df["hpl_vocab_count"] = train_df["cleaned_text"].apply(
    lambda x: count_author_vocab(x, hpl_words)
)
train_df["mws_vocab_count"] = train_df["cleaned_text"].apply(
    lambda x: count_author_vocab(x, mws_words)
)
test_df["eap_vocab_count"] = test_df["cleaned_text"].apply(
    lambda x: count_author_vocab(x, eap_words)
)
test_df["hpl_vocab_count"] = test_df["cleaned_text"].apply(
    lambda x: count_author_vocab(x, hpl_words)
)
test_df["mws_vocab_count"] = test_df["cleaned_text"].apply(
    lambda x: count_author_vocab(x, mws_words)
)

# Combine numeric features
numeric_features_train = pd.concat(
    [
        basic_train_df,
        char_train_df,
        train_df[["eap_vocab_count", "hpl_vocab_count", "mws_vocab_count"]],
    ],
    axis=1,
)
numeric_features_test = pd.concat(
    [
        basic_test_df,
        char_test_df,
        test_df[["eap_vocab_count", "hpl_vocab_count", "mws_vocab_count"]],
    ],
    axis=1,
)

numeric_features_train = numeric_features_train.replace([np.inf, -np.inf], 0).fillna(0)
numeric_features_test = numeric_features_test.replace([np.inf, -np.inf], 0).fillna(0)

scaler = StandardScaler()
numeric_scaled_train = scaler.fit_transform(numeric_features_train)
numeric_scaled_test = scaler.transform(numeric_features_test)

# Create final feature matrices
numeric_sparse_train = csr_matrix(numeric_scaled_train)
numeric_sparse_test = csr_matrix(numeric_scaled_test)

X_train = hstack([char_ngrams_train, word_ngrams_train, numeric_sparse_train])
X_test = hstack([char_ngrams_test, word_ngrams_test, numeric_sparse_test])

print(f"Feature matrix shape (train): {X_train.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train_raw = label_encoder.fit_transform(train_df["author"])

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_raw, test_size=0.1, random_state=42, stratify=y_train_raw
)
print(f"Train samples: {X_train_final.shape[0]}, Val samples: {X_val.shape[0]}")

# Save processed data and get text for transformer
train_texts_df = train_df[["id", "cleaned_text", "author"]].copy()
test_texts_df = test_df[["id", "cleaned_text"]].copy()

train_idx, val_idx = train_test_split(
    np.arange(len(train_texts_df)),
    test_size=0.1,
    random_state=42,
    stratify=train_texts_df["author"].values,
)

train_texts = train_texts_df.iloc[train_idx]["cleaned_text"].values
val_texts = train_texts_df.iloc[val_idx]["cleaned_text"].values
test_texts = test_texts_df["cleaned_text"].values

train_labels = y_train_final
val_labels = y_val
test_ids = test_df["id"].values


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class ContrastiveDeBERTa(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_labels=3,
        projection_dim=256,
        dropout=0.15,
        temperature=0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, projection_dim),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_labels),
        )

        self.temperature = temperature
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        logits = self.classifier(cls_embedding)
        if labels is not None:
            return logits, projected
        return logits


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, contrastive_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, projected, labels):
        ce_loss = self.ce_loss(logits, labels)

        projected = F.normalize(projected, dim=1, p=2)
        similarity_matrix = torch.matmul(projected, projected.T) / self.temperature

        labels_expanded = labels.unsqueeze(0)
        labels_expanded_t = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded_t).float()

        batch_size = labels.size(0)
        self_mask = torch.eye(batch_size, device=labels.device)
        pos_mask = pos_mask - self_mask

        similarity_matrix_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix_stable = similarity_matrix - similarity_matrix_max.detach()

        exp_sim = torch.exp(similarity_matrix_stable)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = (exp_sim * (1 - self_mask)).sum(dim=1)
        all_sum = torch.clamp(all_sum, min=1e-10)

        contrastive_loss = -torch.log(torch.clamp(pos_sum / all_sum, min=1e-10))

        valid_samples = (pos_mask.sum(dim=1) > 0).float()
        if valid_samples.sum() > 0:
            contrastive_loss = (
                contrastive_loss * valid_samples
            ).sum() / valid_samples.sum()
        else:
            contrastive_loss = torch.tensor(0.0, device=labels.device)

        total_loss = (
            1 - self.contrastive_weight
        ) * ce_loss + self.contrastive_weight * contrastive_loss
        return total_loss, ce_loss, contrastive_loss


# Initialize model components
model_name = "microsoft/deberta-v3-large"
num_authors = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ContrastiveDeBERTa(
    model_name=model_name,
    num_labels=num_authors,
    projection_dim=256,
    dropout=0.15,
    temperature=0.1,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

optimizer = AdamW(
    [
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.projection.parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 5e-5},
    ],
    weight_decay=0.01,
)

criterion = SupervisedContrastiveLoss(temperature=0.1, contrastive_weight=0.3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)
scaler = torch.cuda.amp.GradScaler()

# Tokenize
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels, dtype=torch.long),
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long),
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
num_epochs = 25
warmup_epochs = 5
early_stop_patience = 5
best_val_score = float("inf")
patience_counter = 0
best_model_state = None
ema_decay = 0.999
ema_model_state = None

print("Starting training with progressive unfreezing...")
print(f"Epochs: {num_epochs}, Warmup: {warmup_epochs}, Batch size: {batch_size}")

for epoch in range(1, num_epochs + 1):
    if epoch <= warmup_epochs:
        model.encoder.eval()
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif epoch == warmup_epochs + 1:
        model.encoder.train()
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("Unfreezing encoder at epoch 6")
    else:
        model.encoder.train()
        for param in model.encoder.parameters():
            param.requires_grad = True

    model.train()
    train_loss = 0.0
    train_ce_loss = 0.0
    train_cont_loss = 0.0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits, projected = model(input_ids, attention_mask, labels)
            loss, ce, cont = criterion(logits, projected, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_ce_loss += ce.item()
        train_cont_loss += cont.item()
        num_train_batches += 1

    avg_train_loss = train_loss / num_train_batches
    avg_train_ce = train_ce_loss / num_train_batches
    avg_train_cont = train_cont_loss / num_train_batches

    if ema_model_state is None:
        ema_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    else:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema_model_state:
                    ema_model_state[k] = (
                        ema_decay * ema_model_state[k]
                        + (1 - ema_decay) * v.clone().detach()
                    )

    model.eval()
    val_loss = 0.0
    all_val_probs = []
    all_val_labels = []
    num_val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            all_val_labels.append(labels.cpu().numpy())

            with torch.cuda.amp.autocast():
                logits, projected = model(input_ids, attention_mask, labels)
                loss, _, _ = criterion(logits, projected, labels)

            val_loss += loss.item()
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_val_probs.append(probs)
            num_val_batches += 1

    avg_val_loss = val_loss / num_val_batches
    val_probs = np.vstack(all_val_probs)
    val_labels_concat = np.concatenate(all_val_labels)
    val_log_loss = log_loss(val_labels_concat, val_probs)

    scheduler.step()
    lr_current = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:2d}/{num_epochs} | LR: {lr_current:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f} | CE: {avg_train_ce:.4f} | Cont: {avg_train_cont:.4f}"
    )

    if val_log_loss < best_val_score:
        best_val_score = val_log_loss
        best_model_state = {
            k: v.clone().detach().cpu() for k, v in model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after epoch {epoch}")
            break

# Load best model and evaluate
print("\nLoading best model for final evaluation...")
model.load_state_dict(best_model_state)

model.eval()
all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        all_val_labels.append(labels.cpu().numpy())
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)

val_probs = np.vstack(all_val_probs)
val_labels_concat = np.concatenate(all_val_labels)
val_log_loss = log_loss(val_labels_concat, val_probs)

# Test inference
print("Performing test inference...")
all_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

test_probs = np.vstack(all_test_probs)

epsilon = 1e-15
test_probs = np.clip(test_probs, epsilon, 1 - epsilon)

submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_0c32c477dd324d77910e1155635a4173.csv", index=False)

print(f"\nSubmission saved to ./submission/submission_0c32c477dd324d77910e1155635a4173.csv")
print(f"Test predictions shape: {test_probs.shape}")
print(submission.head())

# Final validation score
score = val_log_loss
print(f"Final Validation Score: {score}")
