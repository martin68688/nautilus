import os
os.sched_setaffinity(0, {248, 250, 251, 252, 254, 120, 253, 122, 123, 124, 125, 126})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import re
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, lr_scheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from collections import Counter
import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from scipy.sparse import hstack, csr_matrix

# Download NLTK resources silently
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# ============================================================
# 3. FEATURE ENGINEERING FUNCTIONS (defined, not applied yet)
# ============================================================
def extract_punctuation_features(text):
    total_chars = len(text) + 1
    features = {
        "comma_count": text.count(",") / total_chars,
        "period_count": text.count(".") / total_chars,
        "exclam_count": text.count("!") / total_chars,
        "question_count": text.count("?") / total_chars,
        "semicolon_count": text.count(";") / total_chars,
        "colon_count": text.count(":") / total_chars,
        "dash_count": len(re.findall(r"—|-{2,}", text)) / total_chars,
        "quote_count": (text.count('"') + text.count('"')) / total_chars,
        "apostrophe_count": text.count("'") / total_chars,
        "ellipsis_count": len(re.findall(r"\.{3,}", text)) / total_chars,
        "paren_count": (text.count("(") + text.count(")")) / total_chars,
        "capital_ratio": sum(1 for c in text if c.isupper()) / max(1, total_chars),
        "punct_density": sum(1 for c in text if c in '.,!?;:—"-()[]{}') / total_chars,
    }
    return features


def extract_syntactic_features(text):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = [text]
    num_sentences = max(1, len(sentences))
    words = text.split()
    num_words = len(words)
    features = {
        "num_sentences": num_sentences,
        "avg_sentence_len": num_words / num_sentences,
        "num_words": num_words,
        "avg_word_len": sum(len(w) for w in words) / max(1, num_words),
        "num_chars": len(text),
        "vocab_richness": len(set(w.lower() for w in words)) / max(1, num_words),
        "stopword_ratio": sum(
            1 for w in words if w.lower() in nltk.corpus.stopwords.words("english")
        )
        / max(1, num_words),
        "long_word_ratio": sum(1 for w in words if len(w) > 8) / max(1, num_words),
        "unique_word_ratio": len(set(w.lower() for w in words)) / max(1, num_words),
    }
    try:
        pos_tags = pos_tag(words)
        pos_counts = Counter(tag for _, tag in pos_tags)
        total_pos = sum(pos_counts.values()) + 1
        for tag in [
            "NN",
            "NNS",
            "NNP",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "JJ",
            "JJR",
            "JJS",
            "RB",
            "RBR",
            "RBS",
            "IN",
            "DT",
            "CC",
            "PRP",
        ]:
            features[f"pos_{tag}"] = pos_counts.get(tag, 0) / total_pos
    except:
        pass
    return features


def extract_all_features(text):
    features = {}
    features.update(extract_punctuation_features(text))
    features.update(extract_syntactic_features(text))
    return features


# ============================================================
# 4. ENCODE TARGET
# ============================================================
le = LabelEncoder()
y_train = le.fit_transform(train_df["author"])
class_names = le.classes_
print(f"Classes: {class_names}")
print(f"Class distribution: {np.bincount(y_train)}")

# ============================================================
# 5. PREPARE TEXT DATA (NO feature engineering before split)
# ============================================================
print(f"Text data prepared. All feature engineering will happen inside CV loop.")


# ============================================================
# 10. DATASET AND MODEL DEFINITIONS
# ============================================================
class SpookyDataset(Dataset):
    def __init__(
        self, texts, labels=None, features=None, tokenizer=None, max_length=256
    ):
        self.texts = texts
        self.labels = labels
        self.features = features
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
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


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
        if self.feature_proj is not None and features is not None:
            feat_embed = self.feature_proj(features)
            combined = torch.cat([cls_pool, feat_embed], dim=1)
        else:
            combined = cls_pool
        logits = self.head(combined)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        if self.label_smoothing > 0:
            num_classes = logits.size(1)
            smoothed_targets = torch.zeros_like(log_probs)
            smoothed_targets.fill_(self.label_smoothing / (num_classes - 1))
            smoothed_targets.scatter_(
                1, targets.unsqueeze(1), 1.0 - self.label_smoothing
            )
            p_t = (probs * smoothed_targets).sum(dim=1)
            focal_weight = (1 - p_t) ** self.gamma
            loss = -(smoothed_targets * log_probs).sum(dim=1)
            loss = focal_weight * loss
            return loss.mean()
        else:
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze()
            focal_weight = (1 - p_t) ** self.gamma
            loss = -focal_weight * torch.log(p_t + 1e-15)
            return loss.mean()


# ============================================================
# 11. TOKENIZER SETUP
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
max_length = 256

# ============================================================
# 12. STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
all_test_probs = []
best_val_score = float("inf")

train_text = train_df["clean_text"].values
test_text = test_df["clean_text"].values
test_ids = test_df["id"].values

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
    print(f"\n========== Fold {fold+1}/5 ==========")

    train_texts = train_text[train_idx]
    val_texts = train_text[val_idx]
    train_labels = y_train[train_idx]
    val_labels = y_train[val_idx]

    # Fit TF-IDF vectorizers on training fold only to prevent data leakage
    char_vectorizer_fold = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
    )
    word_vectorizer_fold = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    train_char_fold = char_vectorizer_fold.fit_transform(train_texts)
    test_char_fold = char_vectorizer_fold.transform(test_text)
    train_word_fold = word_vectorizer_fold.fit_transform(train_texts)
    test_word_fold = word_vectorizer_fold.transform(test_text)

    # Build style features per fold (only on training fold data)
    train_feat_list_fold = [extract_all_features(text) for text in train_texts]
    test_feat_list_fold = [extract_all_features(text) for text in test_text]
    train_feat_df_fold = pd.DataFrame(train_feat_list_fold).fillna(0)
    test_feat_df_fold = pd.DataFrame(test_feat_list_fold).fillna(0)

    scaler_fold = StandardScaler()
    train_feat_scaled_fold = scaler_fold.fit_transform(train_feat_df_fold)
    test_feat_scaled_fold = scaler_fold.transform(test_feat_df_fold)

    train_feat_sparse_fold = csr_matrix(train_feat_scaled_fold)
    test_feat_sparse_fold = csr_matrix(test_feat_scaled_fold)
    X_test_fold = hstack([test_char_fold, test_word_fold, test_feat_sparse_fold])

    # Build train/val splits properly using the val indices from the fold
    val_indices_in_fold = np.arange(len(train_texts), len(train_texts) + len(val_texts))

    # Get val set for char features by indexing separately
    val_char_fold = char_vectorizer_fold.transform(val_texts)
    val_word_fold = word_vectorizer_fold.transform(val_texts)
    val_feat_list_fold = [extract_all_features(text) for text in val_texts]
    val_feat_df_fold = pd.DataFrame(val_feat_list_fold).fillna(0)
    val_feat_scaled_fold = scaler_fold.transform(val_feat_df_fold)
    val_feat_sparse_fold = csr_matrix(val_feat_scaled_fold)

    X_val_fold = hstack([val_char_fold, val_word_fold, val_feat_sparse_fold])
    X_train_fold = hstack([train_char_fold, train_word_fold, train_feat_sparse_fold])

    # PCA on fold data - fit only on training part
    n_components = min(150, X_train_fold.shape[1] - 1)
    pca_fold = PCA(n_components=n_components, random_state=42)
    train_features = pca_fold.fit_transform(
        X_train_fold.toarray() if hasattr(X_train_fold, "toarray") else X_train_fold
    )
    val_features = pca_fold.transform(
        X_val_fold.toarray() if hasattr(X_val_fold, "toarray") else X_val_fold
    )
    test_features_fold = pca_fold.transform(
        X_test_fold.toarray() if hasattr(X_test_fold, "toarray") else X_test_fold
    )

    # Use the correct texts and labels for the split datasets
    train_dataset = SpookyDataset(
        train_texts, train_labels, train_features, tokenizer, max_length
    )
    val_dataset = SpookyDataset(
        val_texts, val_labels, val_features, tokenizer, max_length
    )
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
    )

    model = SpookyClassifier(
        num_authors=3, num_features=train_features.shape[1], dropout_rate=0.3
    )
    model.to(device)

    backbone_params = []
    for name, param in model.backbone.deberta.named_parameters():
        if param.requires_grad and "bias" not in name and "LayerNorm" not in name:
            if any(f"layer.{i}" in name for i in range(16, 24)):
                backbone_params.append(param)

    head_params = list(model.head.parameters())
    if model.feature_proj is not None:
        head_params.extend(list(model.feature_proj.parameters()))

    bias_norm_params = []
    for name, param in model.backbone.deberta.named_parameters():
        if param.requires_grad and ("bias" in name or "LayerNorm" in name):
            if any(f"layer.{i}" in name for i in range(16, 24)):
                bias_norm_params.append(param)

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
            {
                "params": bias_norm_params,
                "lr": 2e-5,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
            },
        ]
    )

    num_epochs = 30
    patience = 5
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    class_counts = np.bincount(train_labels)
    total = class_counts.sum()
    alpha = total / (3 * class_counts.astype(float))
    alpha = alpha / alpha.sum() * 3
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha_tensor, label_smoothing=0.1)
    scaler = GradScaler()

    best_fold_score = float("inf")
    best_fold_model = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)

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
                warmup_factor = current_step / max(1, warmup_steps)
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = initial_lrs[pg_idx] * warmup_factor
            else:
                scheduler.step(epoch + current_step / len(train_loader))

            epoch_loss += loss.item()

        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                features = batch.get("features", None)
                if features is not None:
                    features = features.to(device)
                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_true.append(labels.cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        score = log_loss(val_true, val_probs)

        print(
            f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f} - Val Log Loss: {score:.4f}"
        )

        if score < best_fold_score:
            best_fold_score = score
            best_fold_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_fold_model is not None:
        model.load_state_dict(best_fold_model)

    model.eval()
    test_dataset = SpookyDataset(
        test_text,
        labels=None,
        features=test_features_fold,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
    )
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch.get("features", None)
            if features is not None:
                features = features.to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())
    fold_test_probs = np.concatenate(fold_test_probs)
    all_test_probs.append(fold_test_probs)
    fold_scores.append(best_fold_score)

    if best_fold_score < best_val_score:
        best_val_score = best_fold_score

    print(f"Fold {fold+1} Best Val Log Loss: {best_fold_score:.4f}")

# ============================================================
# 13. ENSEMBLE FOLD PREDICTIONS
# ============================================================
print(f"\n========== ENSEMBLING FOLDS ==========")
print(f"Fold scores: {fold_scores}")
print(f"Mean fold score: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

final_test_probs = np.mean(all_test_probs, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

# ============================================================
# 14. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)
submission.to_csv("./submission/submission_b62c59bfb11e48428bac213d311287d9.csv", index=False)

print(f"\nSubmission saved to ./submission/submission_b62c59bfb11e48428bac213d311287d9.csv")
print(f"Submission shape: {submission.shape}")
final_score = np.mean(fold_scores)
print(f"Final Validation Score: {final_score}")