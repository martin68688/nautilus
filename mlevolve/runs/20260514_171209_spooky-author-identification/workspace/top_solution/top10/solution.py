import pandas as pd
import numpy as np
import re
import os
import warnings
import math

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import log_loss

from textblob import TextBlob

import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs("./working", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# BASIC CLEANING (NO FEATURE ENGINEERING BEFORE SPLIT)
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s\'\.,!?;:\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# ============================================================
# SPLIT DATA FIRST - THEN ENGINEER FEATURES SEPARATELY
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_indices, val_indices = next(skf.split(train_df, train_df["author"]))

train_split_df = train_df.iloc[train_indices].copy()
val_split_df = train_df.iloc[val_indices].copy()

# ============================================================
# TF-IDF N-GRAM FEATURES (FIT ONLY ON TRAIN SPLIT)
# ============================================================
vectorizer_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=False,
    use_idf=True,
    smooth_idf=True,
)

vectorizer_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    use_idf=True,
    smooth_idf=True,
    max_df=0.95,
    min_df=5,
)

train_texts = train_split_df["text"].values
val_texts = val_split_df["text"].values
test_texts = test_df["text"].values

char_features_train = vectorizer_char.fit_transform(train_texts)
char_features_val = vectorizer_char.transform(val_texts)
char_features_test = vectorizer_char.transform(test_texts)

word_features_train = vectorizer_word.fit_transform(train_texts)
word_features_val = vectorizer_word.transform(val_texts)
word_features_test = vectorizer_word.transform(test_texts)

char_features_train = char_features_train.toarray()
char_features_val = char_features_val.toarray()
char_features_test = char_features_test.toarray()
word_features_train = word_features_train.toarray()
word_features_val = word_features_val.toarray()
word_features_test = word_features_test.toarray()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def syllable_count(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_char_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_char_is_vowel:
            count += 1
        prev_char_is_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count


def split_sentences(text):
    """Simple sentence splitting using regex to avoid NLTK punkt_tab dependency."""
    if not text or not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    return sentences


def flesch_kincaid(text):
    sentences = split_sentences(text)
    words = word_tokenize(text)
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    num_words = len(words)
    num_sentences = len(sentences)
    num_syllables = sum(syllable_count(w) for w in words)
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return score


def get_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


def pos_ratios(text):
    try:
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return 0.0, 0.0, 0.0, 0.0
        pos_tags = pos_tag(tokens)
        noun_tags = {"NN", "NNS", "NNP", "NNPS"}
        verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
        adj_tags = {"JJ", "JJR", "JJS"}
        adv_tags = {"RB", "RBR", "RBS"}
        noun_count = sum(1 for _, tag in pos_tags if tag in noun_tags)
        verb_count = sum(1 for _, tag in pos_tags if tag in verb_tags)
        adj_count = sum(1 for _, tag in pos_tags if tag in adj_tags)
        adv_count = sum(1 for _, tag in pos_tags if tag in adv_tags)
        total = len(tokens)
        return noun_count / total, verb_count / total, adj_count / total, adv_count / total
    except:
        return 0.0, 0.0, 0.0, 0.0


def engineer_features(df):
    df["char_count"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["avg_word_len"] = df["char_count"] / (df["word_count"] + 1)
    df["sentence_count"] = df["text_clean"].apply(lambda x: len(sent_tokenize(x)))
    df["avg_sent_len"] = df["word_count"] / (df["sentence_count"] + 1)
    df["punct_count"] = df["text_clean"].apply(
        lambda x: sum(1 for c in x if c in ".,!?;:'\"-")
    )
    df["punct_ratio"] = df["punct_count"] / (df["char_count"] + 1)
    df["cap_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    df["unique_words_ratio"] = df["text_clean"].apply(
        lambda x: len(set(x.split())) / (len(x.split()) + 1)
    )
    df["stopword_ratio"] = df["text_clean"].apply(
        lambda x: sum(1 for w in x.split() if w in set(stopwords.words("english")))
        / (len(x.split()) + 1)
    )
    df["avg_syllables_per_word"] = df["text_clean"].apply(
        lambda x: sum(syllable_count(w) for w in word_tokenize(x)) / (len(word_tokenize(x)) + 1)
    )
    df["flesch_kincaid"] = df["text_clean"].apply(flesch_kincaid)
    df["complex_word_ratio"] = df["text_clean"].apply(
        lambda x: sum(1 for w in word_tokenize(x) if syllable_count(w) >= 3)
        / (len(word_tokenize(x)) + 1)
    )
    df["polarity"] = df["text_clean"].apply(lambda x: get_sentiment(x)[0])
    df["subjectivity"] = df["text_clean"].apply(lambda x: get_sentiment(x)[1])
    pos_feats = df["text_clean"].apply(pos_ratios)
    df["noun_ratio"] = pos_feats.apply(lambda x: x[0])
    df["verb_ratio"] = pos_feats.apply(lambda x: x[1])
    df["adj_ratio"] = pos_feats.apply(lambda x: x[2])
    df["adv_ratio"] = pos_feats.apply(lambda x: x[3])
    return df


train_split_df = engineer_features(train_split_df)
val_split_df = engineer_features(val_split_df)
test_df = engineer_features(test_df)

feature_cols = [
    "char_count",
    "word_count",
    "avg_word_len",
    "sentence_count",
    "avg_sent_len",
    "punct_count",
    "punct_ratio",
    "cap_ratio",
    "unique_words_ratio",
    "stopword_ratio",
    "avg_syllables_per_word",
    "flesch_kincaid",
    "complex_word_ratio",
    "polarity",
    "subjectivity",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
]

# Scale and combine manual features
scaler = StandardScaler()
X_manual_train = scaler.fit_transform(train_split_df[feature_cols].values)
X_manual_val = scaler.transform(val_split_df[feature_cols].values)
X_manual_test = scaler.transform(test_df[feature_cols].values)

X_train_final = np.concatenate(
    [char_features_train, word_features_train, X_manual_train], axis=1
)
X_val_final = np.concatenate(
    [char_features_val, word_features_val, X_manual_val], axis=1
)
X_test_final = np.concatenate(
    [char_features_test, word_features_test, X_manual_test], axis=1
)

le = LabelEncoder()
y_train_final = le.fit_transform(train_df["author"].values[train_indices])
y_val_final = le.transform(train_df["author"].values[val_indices])

print(
    f"Train size: {X_train_final.shape[0]}, Val size: {X_val_final.shape[0]}, Test size: {X_test_final.shape[0]}"
)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        # Initially freeze all backbone layers
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_authors)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        cls_pool = hidden_states[:, 0, :]
        cls_pool = self.dropout(cls_pool)
        logits = self.head(cls_pool)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"Total model params: {sum(p.numel() for p in model.parameters()):,}")


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
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_ids = test_df["id"].values
test_texts = test_df["text"].values

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
# SINGLE-PHASE TRAINING WITH PROPER SCHEDULING
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
            total_loss += loss.item()
            num_batches += 1
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / num_batches
    probs = np.concatenate(all_probs, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    logloss = log_loss(true_labels, probs_clipped)
    return avg_loss, logloss


scaler_grad = GradScaler()
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

best_val_score = float("inf")

# --------------------------------------------------
# UNFREEZE ALL BACKBONE LAYERS IMMEDIATELY
# --------------------------------------------------
for param in model.backbone.deberta.parameters():
    param.requires_grad = True

# Separate parameters: no weight decay for bias and LayerNorm
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.backbone.deberta.named_parameters() if not any(nd in n for nd in no_decay)],
        "lr": 2e-5,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
    },
    {
        "params": [p for n, p in model.backbone.deberta.named_parameters() if any(nd in n for nd in no_decay)],
        "lr": 2e-5,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
    },
    {
        "params": model.head.parameters(),
        "lr": 5e-5,
        "weight_decay": 0.01,
        "betas": (0.9, 0.98),
    },
]

optimizer = AdamW(optimizer_grouped_parameters)

total_steps = len(train_loader) * 10  # 10 epochs
warmup_steps = int(0.1 * total_steps)

# Linear warmup then cosine decay scheduler
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
print("\n" + "="*60)
print("TRAINING: All layers unfrozen, linear warmup + cosine decay")
print("="*60)

num_epochs = 10
patience = 3
epochs_no_improve = 0
global_step = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler_grad, device, scheduler=scheduler)
    val_loss, val_logloss = evaluate(model, val_loader, criterion, device)
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f} | LR: {current_lr:.2e}"
    )

    if val_logloss < best_val_score:
        best_val_score = val_logloss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# --------------------------------------------------
# Load best model and final evaluation
# --------------------------------------------------
if os.path.exists("./working/best_model.pt"):
    model.load_state_dict(torch.load("./working/best_model.pt"))
    print("Loaded best model")
else:
    print("No best model found, using current model")

model.eval()

# Final validation evaluation
_, final_val_score = evaluate(model, val_loader, criterion, device)
print(f"Final Validation LogLoss: {final_val_score:.4f}")

# ============================================================
# TEST INFERENCE
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
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

print(f"Final Validation LogLoss: {final_val_score:.6f}")