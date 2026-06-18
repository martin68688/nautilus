import os
os.sched_setaffinity(0, {23, 24, 25, 26, 27})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import AdamW
import torch
import re
import string
import os
import warnings
import joblib

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
submission_df = pd.read_csv("./input/sample_submission.csv")

train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["author"] = "EAP"
all_data = pd.concat([train_df, test_df], ignore_index=True)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


all_data["cleaned_text"] = all_data["text"].apply(clean_text)


def extract_stylometric_features(text):
    if not isinstance(text, str) or len(text) == 0:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_len": 0,
            "avg_sentence_len_words": 0,
            "avg_sentence_len_chars": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "dash_count": 0,
            "semicolon_count": 0,
            "colon_count": 0,
            "quotes_count": 0,
            "parentheses_count": 0,
            "comma_count": 0,
            "period_count": 0,
            "capital_letters_pct": 0,
            "punctuation_pct": 0,
            "unique_words_pct": 0,
            "stopword_pct": 0,
            "contraction_count": 0,
            "archaic_words_count": 0,
            "emotional_markers": 0,
            "descriptive_adj_count": 0,
            "first_person_singular": 0,
            "first_person_plural": 0,
            "passive_voice_markers": 0,
        }
    chars = len(text)
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
    avg_sent_len_words = word_count / sentence_count
    avg_sent_len_chars = chars / sentence_count
    return {
        "char_count": chars,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_len": avg_word_len,
        "avg_sentence_len_words": avg_sent_len_words,
        "avg_sentence_len_chars": avg_sent_len_chars,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "dash_count": text.count("—") + text.count("-"),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "quotes_count": text.count('"') + text.count("'") // 2,
        "parentheses_count": text.count("(") + text.count(")"),
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "capital_letters_pct": sum(1 for c in text if c.isupper()) / max(chars, 1),
        "punctuation_pct": sum(1 for c in text if c in string.punctuation)
        / max(chars, 1),
        "unique_words_pct": len(set(w.lower() for w in words)) / max(word_count, 1),
        "stopword_pct": sum(
            1
            for w in words
            if w.lower()
            in {
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
                "was",
                "were",
                "had",
                "have",
                "has",
                "been",
                "being",
                "is",
                "are",
                "be",
                "it",
                "its",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "they",
                "we",
                "my",
                "your",
                "his",
                "her",
                "its",
                "our",
                "their",
                "me",
                "him",
                "us",
                "them",
            }
        )
        / max(word_count, 1),
        "contraction_count": len(re.findall(r"\w+'\w+", text)),
        "archaic_words_count": sum(
            1
            for w in words
            if w.lower()
            in [
                "thou",
                "thee",
                "thy",
                "thine",
                "ye",
                "hath",
                "doth",
                "art",
                "wast",
                "dost",
                "canst",
                "wouldst",
                "couldst",
                "shouldst",
                "shalt",
                "wilt",
                "whence",
                "thence",
                "hither",
                "thither",
                "whither",
                "ere",
                "betwixt",
                "unto",
                "abaft",
                "nigh",
                "perchance",
                "methinks",
                "forsooth",
                "anon",
            ]
        ),
        "emotional_markers": sum(
            1
            for w in words
            if w.lower()
            in [
                "never",
                "always",
                "horror",
                "terror",
                "fear",
                "dread",
                "awful",
                "terrible",
                "strange",
                "mysterious",
                "dismal",
                "ghastly",
                "hideous",
                "monstrous",
                "frightful",
                "appalling",
                "shocking",
                "dreadful",
                "fearful",
            ]
        ),
        "descriptive_adj_count": sum(
            1
            for w in words
            if w.lower()
            in [
                "ancient",
                "vast",
                "gigantic",
                "immense",
                "unfathomable",
                "incomprehensible",
                "eldritch",
                "cyclopean",
                "unspeakable",
                "nameless",
                "indescribable",
                "infinite",
                "eternal",
                "cosmic",
            ]
        ),
        "first_person_singular": len(re.findall(r"\b(I|me|my|mine|myself)\b", text)),
        "first_person_plural": len(re.findall(r"\b(we|us|our|ours|ourselves)\b", text)),
        "passive_voice_markers": len(re.findall(r"\bwas \w+ed\b", text)),
    }


stylo_features = all_data["cleaned_text"].apply(extract_stylometric_features)
stylo_df = pd.DataFrame(stylo_features.tolist())

train_mask = all_data["is_train"] == 1
label_encoder = LabelEncoder().fit(["EAP", "HPL", "MWS"])

train_indices = all_data[all_data["is_train"] == 1].index
test_indices = all_data[all_data["is_train"] == 0].index
y_train = label_encoder.transform(all_data.loc[train_indices, "author"].values)
test_ids = test_df["id"].values
train_texts = train_df["text"].values
test_texts = test_df["text"].values

# Build TF-IDF features from cleaned text
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
)
tfidf_train = tfidf_vectorizer.fit_transform(all_data.loc[train_indices, "cleaned_text"])
tfidf_test = tfidf_vectorizer.transform(all_data.loc[test_indices, "cleaned_text"])

# Scale stylometric features
stylo_cols = stylo_df.columns.tolist()
stylo_scaler = StandardScaler()
stylo_train_scaled = stylo_scaler.fit_transform(stylo_df.loc[train_indices].values)
stylo_test_scaled = stylo_scaler.transform(stylo_df.loc[test_indices].values)

# Combine features
from scipy.sparse import hstack, csr_matrix
X_train_sparse = hstack([
    tfidf_train,
    csr_matrix(stylo_train_scaled)
])
X_test_sparse = hstack([
    tfidf_test,
    csr_matrix(stylo_test_scaled)
])

# For simplicity, keep as dense arrays since we need to feed to PyTorch
X_train = X_train_sparse.toarray().astype(np.float32)
X_test = X_test_sparse.toarray().astype(np.float32)

num_features = X_train.shape[1]
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Number of features: {num_features}")


# ============================================================
# MODEL DEFINITION
# ============================================================
class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=150, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        if num_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.head = nn.Linear(hidden_size + 64, num_authors)
        else:
            self.feature_proj = None
            self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_pool = outputs.last_hidden_state[:, 0, :]
        cls_pool = torch.nan_to_num(cls_pool, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.feature_proj is not None and features is not None:
            features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
            feat_embed = self.feature_proj(features)
            feat_embed = torch.nan_to_num(feat_embed, nan=0.0, posinf=1e4, neginf=-1e4)
            combined = torch.cat([cls_pool, feat_embed], dim=1)
        else:
            combined = cls_pool
        logits = self.head(combined)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits


# ============================================================
# TOKENIZER & DATASET
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


def prepare_dataset(texts, features, labels=None, max_length=256):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    if hasattr(features, "toarray"):
        features_dense = features.toarray()
    else:
        features_dense = features
    features_tensor = torch.FloatTensor(features_dense)
    if labels is not None:
        labels_tensor = torch.LongTensor(labels)
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            features_tensor,
            labels_tensor,
        )
    else:
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], features_tensor
        )
    return dataset


# ============================================================
# 5-FOLD CROSS-VALIDATION TRAINING
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_test_probs = []
fold_scores = []
num_epochs = 30
patience = 5
batch_size = 16

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold + 1}/{n_splits}")
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]
    train_texts_fold = train_texts[train_idx]
    val_texts_fold = train_texts[val_idx]

    train_dataset = prepare_dataset(train_texts_fold, X_fold_train, y_fold_train)
    val_dataset = prepare_dataset(val_texts_fold, X_fold_val, y_fold_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = SpookyClassifier(num_authors=3, num_features=num_features, dropout_rate=0.3)
    model.to(device)

    backbone_params = [
        p
        for layer in model.backbone.deberta.encoder.layer[-8:]
        for n, p in layer.named_parameters()
        if "bias" not in n and "LayerNorm" not in n
    ]
    head_params = list(model.head.parameters()) + (
        list(model.feature_proj.parameters()) if model.feature_proj else []
    )
    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
        ]
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, features, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = initial_lrs[i] * (current_step / max(1, warmup_steps))
            else:
                scheduler.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, features, labels = [
                    b.to(device) for b in batch
                ]
                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                    probs = torch.softmax(logits, dim=1)
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                val_probs.append(probs.cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_probs = np.nan_to_num(val_probs, nan=0.0, posinf=1.0, neginf=0.0)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        val_probs = np.nan_to_num(val_probs, nan=1.0/3.0, posinf=1.0, neginf=0.0)
        val_loss = log_loss(y_fold_val, val_probs)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    model.to(device)

    test_dataset = prepare_dataset(test_texts, X_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model.eval()
    fold_test_probs_fold = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, features = [b.to(device) for b in batch]
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs_fold.append(probs.cpu().numpy())
    fold_test_probs_fold = np.concatenate(fold_test_probs_fold)
    fold_test_probs_fold = np.nan_to_num(fold_test_probs_fold, nan=0.0, posinf=1.0, neginf=0.0)
    fold_test_probs_fold = np.clip(fold_test_probs_fold, 1e-15, 1 - 1e-15)
    fold_test_probs_fold = fold_test_probs_fold / fold_test_probs_fold.sum(
        axis=1, keepdims=True
    )
    fold_test_probs_fold = np.nan_to_num(fold_test_probs_fold, nan=1.0/3.0, posinf=1.0, neginf=0.0)

    fold_test_probs.append(fold_test_probs_fold)
    fold_scores.append(best_val_loss)
    print(f"Fold {fold + 1} best validation log-loss: {best_val_loss:.4f}")

# ============================================================
# FINAL SUBMISSION
# ============================================================
final_test_probs = np.mean(fold_test_probs, axis=0)
final_test_probs = np.nan_to_num(final_test_probs, nan=0.0, posinf=1.0, neginf=0.0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
final_test_probs = np.nan_to_num(final_test_probs, nan=1.0/3.0, posinf=1.0, neginf=0.0)

submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_b8ba3baacca749b896a38b9696890487.csv", index=False)
print(f"Submission saved to ./submission/submission_b8ba3baacca749b896a38b9696890487.csv")

val_score = np.mean(fold_scores)
print(f"Final Validation Score: {val_score}")