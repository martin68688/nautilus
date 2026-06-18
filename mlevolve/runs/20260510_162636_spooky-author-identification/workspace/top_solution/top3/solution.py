import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import os
import gc
import random
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# SEED FOR REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


set_seed(42)

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s\.\,\!\?\-\'\";:\(\)\[\]\{\}]", " ", text)
        return text
    return ""


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# ============================================================
# LABEL ENCODING
# ============================================================
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_.tolist()
print(f"Classes: {class_names}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def extract_features(text_series):
    features = pd.DataFrame()
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_length"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.split("[.!?]+").str.len()
    features["avg_sentence_length"] = features["word_count"] / (
        features["sentence_count"] + 1
    )
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["dash_count"] = text_series.str.count("-")
    features["parenthesis_count"] = text_series.str.count(r"\(") + text_series.str.count(
        r"\)"
    )
    features["capital_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    features["all_caps_words"] = text_series.apply(
        lambda x: len(re.findall(r"\b[A-Z]{2,}\b", str(x)))
    )
    archaic_words = [
        "thee",
        "thou",
        "thy",
        "thine",
        "hath",
        "doth",
        "wert",
        "dost",
        "canst",
        "shalt",
        "wilt",
        "art",
        "ye",
        "thence",
        "whither",
        "thence",
        "whence",
        "hither",
        "therein",
        "thereof",
        "whereof",
        "wherein",
        "unto",
    ]
    features["archaic_word_count"] = text_series.apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in archaic_words)
    )
    horror_words = [
        "death",
        "dark",
        "shadow",
        "terror",
        "fear",
        "horror",
        "ghost",
        "spirit",
        "night",
        "dread",
        "awful",
        "fright",
        "gloom",
        "mystery",
        "strange",
        "shadow",
        "corpse",
        "grave",
        "tomb",
        "spectre",
        "phantom",
        "demon",
        "devil",
        "hell",
    ]
    features["horror_word_count"] = text_series.apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in horror_words)
    )
    features["first_person_singular"] = text_series.apply(
        lambda x: sum(
            1
            for word in str(x).lower().split()
            if word in ["i", "me", "my", "mine", "myself"]
        )
    )
    features["first_person_plural"] = text_series.apply(
        lambda x: sum(
            1
            for word in str(x).lower().split()
            if word in ["we", "us", "our", "ours", "ourselves"]
        )
    )
    features["third_person"] = text_series.apply(
        lambda x: sum(
            1
            for word in str(x).lower().split()
            if word in ["he", "she", "it", "they", "him", "her", "them"]
        )
    )
    features["past_tense_verbs"] = text_series.apply(
        lambda x: len(re.findall(r"\b\w+ed\b", str(x)))
    )
    features["ing_verbs"] = text_series.apply(
        lambda x: len(re.findall(r"\b\w+ing\b", str(x)))
    )
    features["ly_adverbs"] = text_series.apply(
        lambda x: len(re.findall(r"\b\w+ly\b", str(x)))
    )
    return features


# Extract and clean features
train_features_raw = extract_features(train_df["text_clean"])
test_features_raw = extract_features(test_df["text_clean"])

train_features_raw = train_features_raw.replace([np.inf, -np.inf], 0).fillna(0)
test_features_raw = test_features_raw.replace([np.inf, -np.inf], 0).fillna(0)

# TF-IDF character n-grams
tfidf_char = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=5000, sublinear_tf=True, min_df=5
)
tfidf_char_train = tfidf_char.fit_transform(train_df["text_clean"])
tfidf_char_test = tfidf_char.transform(test_df["text_clean"])

tfidf_char_train_df = pd.DataFrame(
    tfidf_char_train.toarray(),
    columns=[f"char_ngram_{i}" for i in range(tfidf_char_train.shape[1])],
)
tfidf_char_test_df = pd.DataFrame(
    tfidf_char_test.toarray(),
    columns=[f"char_ngram_{i}" for i in range(tfidf_char_test.shape[1])],
)

# TF-IDF word n-grams
tfidf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=8000,
    sublinear_tf=True,
    min_df=5,
    stop_words="english",
)
tfidf_word_train = tfidf_word.fit_transform(train_df["text_clean"])
tfidf_word_test = tfidf_word.transform(test_df["text_clean"])

tfidf_word_train_df = pd.DataFrame(
    tfidf_word_train.toarray(),
    columns=[f"word_ngram_{i}" for i in range(tfidf_word_train.shape[1])],
)
tfidf_word_test_df = pd.DataFrame(
    tfidf_word_test.toarray(),
    columns=[f"word_ngram_{i}" for i in range(tfidf_word_test.shape[1])],
)

# Combine features
numerical_cols = [
    "char_count",
    "word_count",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length",
    "exclamation_count",
    "question_count",
    "comma_count",
    "semicolon_count",
    "colon_count",
    "quote_count",
    "dash_count",
    "parenthesis_count",
    "capital_ratio",
    "all_caps_words",
    "archaic_word_count",
    "horror_word_count",
    "first_person_singular",
    "first_person_plural",
    "third_person",
    "past_tense_verbs",
    "ing_verbs",
    "ly_adverbs",
]

scaler = StandardScaler()
train_features_raw[numerical_cols] = scaler.fit_transform(
    train_features_raw[numerical_cols]
)
test_features_raw[numerical_cols] = scaler.transform(test_features_raw[numerical_cols])

# Final feature sets (for potential non-DL use, but we use ModernBERT for main model)
X_train_ml = pd.concat(
    [
        train_features_raw.reset_index(drop=True),
        tfidf_char_train_df.reset_index(drop=True),
        tfidf_word_train_df.reset_index(drop=True),
    ],
    axis=1,
)
X_test_ml = pd.concat(
    [
        test_features_raw.reset_index(drop=True),
        tfidf_char_test_df.reset_index(drop=True),
        tfidf_word_test_df.reset_index(drop=True),
    ],
    axis=1,
)

y_full = train_df["author_encoded"].values

# ============================================================
# MODERNBERT SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# ============================================================
# DATASET CLASS
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=512):
        self.texts = list(texts)
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
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


# ============================================================
# EVALUATION FUNCTION
# ============================================================
def compute_log_loss(y_true, y_pred_proba):
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return log_loss(y_true, y_pred_proba)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_val_loss = val_loss / len(val_loader)
    score = compute_log_loss(all_labels, all_preds)
    return avg_val_loss, score, all_preds


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_fold(train_texts, val_texts, train_labels, val_labels, fold, device):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*50}")

    train_dataset = TextDataset(train_texts, train_labels, max_length=512)
    val_dataset = TextDataset(val_texts, val_labels, max_length=512)

    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ModernBertForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes
    )
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.config.hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler()

    best_val_score = float("inf")
    patience = 3
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        val_loss, val_score, _ = evaluate(model, val_loader, criterion, device)
        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val LogLoss: {val_score:.4f}"
        )
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    model.to(device)
    return model, best_val_score


# ============================================================
# CROSS-VALIDATION TRAINING
# ============================================================
full_train_texts = train_df["text_clean"].tolist()
full_train_labels = train_df["author_encoded"].values
test_texts = test_df["text_clean"].tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
fold_test_preds = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(full_train_texts, full_train_labels)
):
    fold_train_texts = [full_train_texts[i] for i in train_idx]
    fold_val_texts = [full_train_texts[i] for i in val_idx]
    fold_train_labels = full_train_labels[train_idx]
    fold_val_labels = full_train_labels[val_idx]

    # Fit preprocessing on fold train only to avoid data leakage
    fold_train_series = pd.Series(fold_train_texts)
    fold_val_series = pd.Series(fold_val_texts)
    test_series = pd.Series(test_texts)

    # Features
    fold_train_features_raw = extract_features(fold_train_series)
    fold_val_features_raw = extract_features(fold_val_series)
    test_features_raw_fold = extract_features(test_series)

    fold_train_features_raw = fold_train_features_raw.replace([np.inf, -np.inf], 0).fillna(0)
    fold_val_features_raw = fold_val_features_raw.replace([np.inf, -np.inf], 0).fillna(0)
    test_features_raw_fold = test_features_raw_fold.replace([np.inf, -np.inf], 0).fillna(0)

    # Scale numerical features on fold train only
    scaler_fold = StandardScaler()
    fold_train_features_raw[numerical_cols] = scaler_fold.fit_transform(
        fold_train_features_raw[numerical_cols]
    )
    fold_val_features_raw[numerical_cols] = scaler_fold.transform(
        fold_val_features_raw[numerical_cols]
    )
    test_features_raw_fold[numerical_cols] = scaler_fold.transform(
        test_features_raw_fold[numerical_cols]
    )

    # TF-IDF char n-grams on fold train only
    tfidf_char_fold = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), max_features=5000, sublinear_tf=True, min_df=5
    )
    tfidf_char_train_fold = tfidf_char_fold.fit_transform(fold_train_series)
    tfidf_char_val_fold = tfidf_char_fold.transform(fold_val_series)
    tfidf_char_test_fold = tfidf_char_fold.transform(test_series)

    tfidf_char_train_fold_df = pd.DataFrame(
        tfidf_char_train_fold.toarray(),
        columns=[f"char_ngram_{i}" for i in range(tfidf_char_train_fold.shape[1])],
    )
    tfidf_char_val_fold_df = pd.DataFrame(
        tfidf_char_val_fold.toarray(),
        columns=[f"char_ngram_{i}" for i in range(tfidf_char_val_fold.shape[1])],
    )
    tfidf_char_test_fold_df = pd.DataFrame(
        tfidf_char_test_fold.toarray(),
        columns=[f"char_ngram_{i}" for i in range(tfidf_char_test_fold.shape[1])],
    )

    # TF-IDF word n-grams on fold train only
    tfidf_word_fold = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        min_df=5,
        stop_words="english",
    )
    tfidf_word_train_fold = tfidf_word_fold.fit_transform(fold_train_series)
    tfidf_word_val_fold = tfidf_word_fold.transform(fold_val_series)
    tfidf_word_test_fold = tfidf_word_fold.transform(test_series)

    tfidf_word_train_fold_df = pd.DataFrame(
        tfidf_word_train_fold.toarray(),
        columns=[f"word_ngram_{i}" for i in range(tfidf_word_train_fold.shape[1])],
    )
    tfidf_word_val_fold_df = pd.DataFrame(
        tfidf_word_val_fold.toarray(),
        columns=[f"word_ngram_{i}" for i in range(tfidf_word_val_fold.shape[1])],
    )
    tfidf_word_test_fold_df = pd.DataFrame(
        tfidf_word_test_fold.toarray(),
        columns=[f"word_ngram_{i}" for i in range(tfidf_word_test_fold.shape[1])],
    )

    model, fold_score = train_fold(
        fold_train_texts,
        fold_val_texts,
        fold_train_labels,
        fold_val_labels,
        fold,
        device,
    )
    fold_scores.append(fold_score)

    # Test predictions
    test_dataset = TextDataset(test_texts, max_length=512)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )
    model.eval()
    fold_test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            fold_test_pred.append(probs)
    fold_test_pred = np.concatenate(fold_test_pred, axis=0)
    fold_test_preds.append(fold_test_pred)

    del model
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*50}")
print("Cross-validation Results:")
print(f"{'='*50}")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1} Validation LogLoss: {score:.6f}")

mean_cv_score = np.mean(fold_scores)
std_cv_score = np.std(fold_scores)
print(f"Mean CV Validation LogLoss: {mean_cv_score:.6f} ± {std_cv_score:.6f}")

final_val_score = mean_cv_score

# ============================================================
# ENSEMBLE TEST PREDICTIONS AND SUBMISSION
# ============================================================
test_preds_ensemble = np.mean(fold_test_preds, axis=0)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
    }
)

# Use label_encoder.classes_ to get correct column order
for i, class_name in enumerate(label_encoder.classes_):
    submission[class_name] = test_preds_ensemble[:, i]

# Normalize probabilities to sum to 1 per row
class_cols = label_encoder.classes_.tolist()
row_sums = submission[class_cols].sum(axis=1)
for col in class_cols:
    submission[col] = submission[col] / row_sums

epsilon = 1e-15
for col in class_cols:
    submission[col] = submission[col].clip(epsilon, 1 - epsilon)

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"First 5 rows:\n{submission.head()}")

print(f"Final Validation Score: {final_val_score}")