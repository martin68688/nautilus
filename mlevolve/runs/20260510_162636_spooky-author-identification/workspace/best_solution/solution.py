import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========== DATA LOADING ==========
def load_data():
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")
    return train, test


# ========== FEATURE ENGINEERING ==========
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_lexical_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    features["word_count"] = len(words)
    features["char_count"] = len(text)
    features["sentence_count"] = max(len(sentences), 1)
    features["avg_word_len"] = np.mean([len(w) for w in words]) if words else 0
    features["avg_sentence_len"] = features["word_count"] / features["sentence_count"]
    punct_count = sum(1 for c in text if c in string.punctuation)
    features["punct_density"] = punct_count / max(len(text), 1)
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["semicolon_count"] = text.count(";")
    features["colon_count"] = text.count(":")
    features["dash_count"] = text.count("—") + text.count("-")
    features["quote_count"] = text.count('"') + text.count("'")
    capital_words = sum(1 for w in words if w and w[0].isupper())
    features["capital_word_ratio"] = capital_words / max(len(words), 1)
    features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)
    features["long_words_ratio"] = sum(1 for w in words if len(w) > 6) / max(
        len(words), 1
    )
    features["very_long_words"] = sum(1 for w in words if len(w) > 10)
    stop_words = {
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
    }
    article_words = {"the", "a", "an"}
    features["stop_word_rate"] = sum(1 for w in words if w.lower() in stop_words) / max(
        len(words), 1
    )
    features["article_rate"] = sum(
        1 for w in words if w.lower() in article_words
    ) / max(len(words), 1)
    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "dost",
        "art",
        "whence",
        "thence",
        "hither",
        "thither",
        "ere",
        "whilst",
        "ye",
        "alas",
        "forsooth",
        "betwixt",
        "perchance",
    ]
    features["archaic_word_count"] = sum(1 for w in words if w.lower() in archaic_words)
    return features


def extract_structural_features(text):
    features = {}
    words = text.split()
    features["ing_verbs"] = len([w for w in words if w.lower().endswith("ing")])
    features["ed_verbs"] = len([w for w in words if w.lower().endswith("ed")])
    features["ly_adverbs"] = len([w for w in words if w.lower().endswith("ly")])
    features["tion_nouns"] = len([w for w in words if w.lower().endswith("tion")])
    features["ness_nouns"] = len([w for w in words if w.lower().endswith("ness")])
    features["ment_nouns"] = len([w for w in words if w.lower().endswith("ment")])
    features["contains_numbers"] = int(bool(re.search(r"\d", text)))
    features["number_count"] = len(re.findall(r"\d+", text))
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    second_person = {"you", "your", "yours", "yourself", "ye"}
    third_person = {
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
    }
    features["first_person_count"] = sum(1 for w in words if w.lower() in first_person)
    features["second_person_count"] = sum(
        1 for w in words if w.lower() in second_person
    )
    features["third_person_count"] = sum(1 for w in words if w.lower() in third_person)
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "squamous",
        "rugose",
        "ichor",
        "gibbous",
        "noisome",
        "foetid",
        "fetid",
        "cacodaemon",
        "antediluvian",
        "non-euclidean",
        "aeonian",
        "eon",
        "tentacle",
        "blasphemous",
        "immemorial",
        "unfathomable",
        "nameless",
        "unspeakable",
        "indescribable",
        "formless",
        "abyss",
        "void",
        "maddening",
        "loathsome",
        "hideous",
        "grotesque",
        "monstrous",
    }
    poe_words = {
        "nevermore",
        "chamber",
        "tapping",
        "rapping",
        "dreary",
        "bleak",
        "charnel",
        "sepulchre",
        "ghastly",
        "spectral",
        "pallid",
        "hue",
        "trembling",
        "quivering",
        "shudder",
        "desolate",
        "forlorn",
        "melancholy",
        "ominous",
        "portentous",
        "supernatural",
        "unearthly",
        "weird",
    }
    shelley_words = {
        "creature",
        "monster",
        "fiend",
        "demon",
        "spirit",
        "soul",
        "eternal",
        "immortal",
        "mortal",
        "tempest",
        "forest",
        "mountain",
        "passion",
        "emotion",
        "despair",
        "anguish",
        "wretch",
        "being",
        "nature",
        "human",
        "science",
        "philosophy",
        "knowledge",
    }
    features["lovecraft_vocab"] = sum(1 for w in words if w.lower() in lovecraft_words)
    features["poe_vocab"] = sum(1 for w in words if w.lower() in poe_words)
    features["shelley_vocab"] = sum(1 for w in words if w.lower() in shelley_words)
    return features


def engineer_features(train_df, test_df):
    train_df["cleaned_text"] = train_df["text"].apply(clean_text)
    test_df["cleaned_text"] = test_df["text"].apply(clean_text)
    all_texts = pd.concat([train_df["cleaned_text"], test_df["cleaned_text"]], axis=0)

    print("Extracting character n-gram features...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_features = char_vectorizer.fit_transform(all_texts)

    print("Extracting word n-gram features...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        stop_words="english",
        min_df=3,
    )
    word_features = word_vectorizer.fit_transform(all_texts)
    ngram_features = hstack([char_features, word_features])

    print("Extracting handcrafted features...")
    train_lexical = (
        train_df["cleaned_text"].apply(extract_lexical_features).apply(pd.Series)
    )
    test_lexical = (
        test_df["cleaned_text"].apply(extract_lexical_features).apply(pd.Series)
    )
    train_structural = (
        train_df["cleaned_text"].apply(extract_structural_features).apply(pd.Series)
    )
    test_structural = (
        test_df["cleaned_text"].apply(extract_structural_features).apply(pd.Series)
    )
    train_handcrafted = pd.concat([train_lexical, train_structural], axis=1)
    test_handcrafted = pd.concat([test_lexical, test_structural], axis=1)

    scaler = StandardScaler()
    train_handcrafted_scaled = scaler.fit_transform(train_handcrafted.fillna(0))
    test_handcrafted_scaled = scaler.transform(test_handcrafted.fillna(0))

    train_features = hstack(
        [ngram_features[: len(train_df)], csr_matrix(train_handcrafted_scaled)]
    )
    test_features = hstack(
        [ngram_features[len(train_df) :], csr_matrix(test_handcrafted_scaled)]
    )

    return train_features, test_features


# ========== DATASET CLASS ==========
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ========== MAIN PIPELINE ==========
print("Loading data...")
train, test = load_data()
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Feature engineering - note: TF-IDF vectorizers and scaler fitting happen inside cross-validation to avoid leakage
print("Feature engineering will be done per fold to avoid data leakage...")

# Encode labels
author_to_idx = {"EAP": 0, "HPL": 1, "MWS": 2}
idx_to_author = {0: "EAP", 1: "HPL", 2: "MWS"}
y_train = train["author"].map(author_to_idx).values

# Cross-validation setup
MODEL_ID = "answerdotai/ModernBERT-large"
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 4
NUM_FOLDS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.3
GRAD_ACCUM_STEPS = 2
WARMUP_RATIO = 0.1
PATIENCE = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

val_predictions = np.zeros((len(train), 3))
test_predictions_list = []

print(f"Starting {NUM_FOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train["text"], y_train)):
    print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")

    train_texts = train.iloc[train_idx]["text"].values
    val_texts = train.iloc[val_idx]["text"].values
    train_labels = y_train[train_idx]
    val_labels = y_train[val_idx]

    # Fit TF-IDF vectorizers and scaler on fold training data only (avoid leakage)
    fold_train_df = train.iloc[train_idx].copy()
    fold_val_df = train.iloc[val_idx].copy()
    fold_train_df["cleaned_text"] = fold_train_df["text"].apply(clean_text)
    fold_val_df["cleaned_text"] = fold_val_df["text"].apply(clean_text)
    fold_test_df = test.copy()
    fold_test_df["cleaned_text"] = fold_test_df["text"].apply(clean_text)

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_features_train = char_vectorizer.fit_transform(fold_train_df["cleaned_text"])
    char_features_val = char_vectorizer.transform(fold_val_df["cleaned_text"])
    char_features_test = char_vectorizer.transform(fold_test_df["cleaned_text"])

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        stop_words="english",
        min_df=3,
    )
    word_features_train = word_vectorizer.fit_transform(fold_train_df["cleaned_text"])
    word_features_val = word_vectorizer.transform(fold_val_df["cleaned_text"])
    word_features_test = word_vectorizer.transform(fold_test_df["cleaned_text"])

    from scipy.sparse import hstack, csr_matrix
    ngram_features_train = hstack([char_features_train, word_features_train])
    ngram_features_val = hstack([char_features_val, word_features_val])
    ngram_features_test = hstack([char_features_test, word_features_test])

    train_lexical = fold_train_df["cleaned_text"].apply(extract_lexical_features).apply(pd.Series)
    val_lexical = fold_val_df["cleaned_text"].apply(extract_lexical_features).apply(pd.Series)
    test_lexical = fold_test_df["cleaned_text"].apply(extract_lexical_features).apply(pd.Series)
    train_structural = fold_train_df["cleaned_text"].apply(extract_structural_features).apply(pd.Series)
    val_structural = fold_val_df["cleaned_text"].apply(extract_structural_features).apply(pd.Series)
    test_structural = fold_test_df["cleaned_text"].apply(extract_structural_features).apply(pd.Series)

    train_handcrafted = pd.concat([train_lexical, train_structural], axis=1)
    val_handcrafted = pd.concat([val_lexical, val_structural], axis=1)
    test_handcrafted = pd.concat([test_lexical, test_structural], axis=1)

    scaler = StandardScaler()
    train_handcrafted_scaled = scaler.fit_transform(train_handcrafted.fillna(0))
    val_handcrafted_scaled = scaler.transform(val_handcrafted.fillna(0))
    test_handcrafted_scaled = scaler.transform(test_handcrafted.fillna(0))

    X_train_feat_fold = hstack([ngram_features_train, csr_matrix(train_handcrafted_scaled)])
    X_val_feat_fold = hstack([ngram_features_val, csr_matrix(val_handcrafted_scaled)])
    X_test_feat_fold = hstack([ngram_features_test, csr_matrix(test_handcrafted_scaled)])

    train_dataset = AuthorDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = AuthorDataset(val_texts, val_labels, tokenizer, MAX_LEN)

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

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_labels = 3
    config.hidden_dropout_prob = DROPOUT_RATE
    config.attention_probs_dropout_prob = DROPOUT_RATE
    model = ModernBertForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
    )
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    best_fold_score = float("inf")
    patience_counter = 0
    fold_best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_true.append(batch["labels"].numpy())

        val_probs = np.concatenate(val_probs, axis=0)
        val_true = np.concatenate(val_true, axis=0)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(
            axis=1, keepdims=True
        )
        fold_score = log_loss(val_true, val_probs_clipped)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val LogLoss: {fold_score:.4f}"
        )

        if fold_score < best_fold_score:
            best_fold_score = fold_score
            fold_best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_predictions[val_idx] = val_probs

    model.load_state_dict(fold_best_state)
    model.eval()

    test_dataset = AuthorDataset(test["text"].values, None, tokenizer, MAX_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(fold_test_probs, axis=0)
    fold_test_probs = np.clip(fold_test_probs, 1e-15, 1 - 1e-15)
    fold_test_probs = fold_test_probs / fold_test_probs.sum(axis=1, keepdims=True)
    test_predictions_list.append(fold_test_probs)

    print(f"Fold {fold+1} Best Val LogLoss: {best_fold_score:.4f}")

val_probs_clipped = np.clip(val_predictions, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
overall_val_score = log_loss(y_train, val_probs_clipped)

print(f"\nOverall Validation LogLoss: {overall_val_score:.4f}")

test_probs_avg = np.mean(test_predictions_list, axis=0)
test_probs_avg = np.clip(test_probs_avg, 1e-15, 1 - 1e-15)
test_probs_avg = test_probs_avg / test_probs_avg.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test["id"].values,
        "EAP": test_probs_avg[:, 0],
        "HPL": test_probs_avg[:, 1],
        "MWS": test_probs_avg[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")

print(f"Final Validation Score: {overall_val_score}")