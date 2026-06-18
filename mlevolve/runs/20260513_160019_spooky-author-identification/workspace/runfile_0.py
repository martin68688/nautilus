import os
os.sched_setaffinity(0, {32, 159, 160, 35, 36, 157, 158, 29, 30, 31})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import gc
import math
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss as sklearn_log_loss
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

test_ids = test_df["id"].values.copy()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Authors: {train_df['author'].value_counts().to_dict()}")


# ============================================================
# 2. TEXT CLEANING FUNCTION
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.strip()
    return text


train_df["cleaned_text"] = train_df["text"].apply(clean_text)
test_df["cleaned_text"] = test_df["text"].apply(clean_text)

# ============================================================
# 3. LABEL ENCODING
# ============================================================
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
class_names = label_encoder.classes_
print(f"Classes: {class_names}")
print(f"Class mapping: {dict(zip(class_names, range(len(class_names))))}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
ARCHAIC_WORDS = {
    "thou",
    "thee",
    "thy",
    "thine",
    "hath",
    "doth",
    "whence",
    "thence",
    "hither",
    "thither",
    "ye",
    "ere",
    "anon",
    "perchance",
    "methinks",
    "forsooth",
    "wherefore",
    "thenceforth",
}
GOTHIC_WORDS = {
    "darkness",
    "shadow",
    "gloom",
    "horror",
    "dread",
    "spectral",
    "phantom",
    "apparition",
    "supernatural",
    "doom",
    "curse",
    "haunt",
    "coffin",
    "tomb",
    "grave",
    "terror",
    "ghost",
}
SCIENTIFIC_WORDS = {
    "science",
    "experiment",
    "mechanism",
    "element",
    "substance",
    "chemical",
    "physical",
    "phenomenon",
    "theory",
    "calculate",
    "formula",
    "biology",
    "analysis",
    "specimen",
}
STOP_WORDS = {
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
    "are",
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
    "not",
    "no",
    "nor",
    "so",
    "if",
    "than",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "their",
    "my",
    "our",
    "your",
    "his",
    "her",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
}


def extract_stylometric_features(text_series):
    features = {}
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_length"] = features["char_count"] / (features["word_count"] + 1)

    for punct in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")"]:
        features[f"count_{punct}"] = text_series.str.count(re.escape(punct))

    features["count_dash"] = text_series.str.count("--")
    features["count_ellipsis"] = text_series.str.count(r"\.{3,}")
    features["count_exclamation_period"] = text_series.str.count(re.escape("!")) / (
        features["word_count"] + 1
    )
    features["count_question_ratio"] = text_series.str.count(re.escape("?")) / (
        features["word_count"] + 1
    )

    features["capital_letters"] = text_series.str.findall(r"[A-Z]").str.len()
    features["capital_ratio"] = features["capital_letters"] / (
        features["char_count"] + 1
    )
    features["capitalized_words"] = text_series.str.findall(
        r"\b[A-Z][a-z]*\b"
    ).str.len()
    features["all_caps_words"] = text_series.str.findall(r"\b[A-Z]{2,}\b").str.len()

    features["stopword_count"] = (
        text_series.str.lower()
        .str.split()
        .apply(
            lambda x: sum(1 for w in x if w in STOP_WORDS) if isinstance(x, list) else 0
        )
    )
    features["stopword_ratio"] = features["stopword_count"] / (
        features["word_count"] + 1
    )
    features["unique_word_ratio"] = text_series.str.split().apply(
        lambda x: len(set(x)) / (len(x) + 1) if isinstance(x, list) else 0
    )

    features["syllable_count"] = text_series.str.findall(
        r"[aeiouy]+", flags=re.IGNORECASE
    ).str.len()
    features["flesch_ease"] = (
        206.835
        - 1.015 * features["word_count"]
        - 84.6 * (features["syllable_count"] / (features["word_count"] + 1))
    )

    return pd.DataFrame(features)


def compute_vocabulary_scores(text_series):
    df = pd.DataFrame(index=text_series.index)
    df["archaic_word_count"] = (
        text_series.str.lower()
        .str.split()
        .apply(
            lambda x: (
                sum(1 for w in x if w in ARCHAIC_WORDS) if isinstance(x, list) else 0
            )
        )
    )
    df["gothic_word_count"] = (
        text_series.str.lower()
        .str.split()
        .apply(
            lambda x: (
                sum(1 for w in x if w in GOTHIC_WORDS) if isinstance(x, list) else 0
            )
        )
    )
    df["scientific_word_count"] = (
        text_series.str.lower()
        .str.split()
        .apply(
            lambda x: (
                sum(1 for w in x if w in SCIENTIFIC_WORDS) if isinstance(x, list) else 0
            )
        )
    )
    word_counts = text_series.str.split().str.len()
    df["archaic_word_ratio"] = df["archaic_word_count"] / (word_counts + 1)
    df["gothic_word_ratio"] = df["gothic_word_count"] / (word_counts + 1)
    df["scientific_word_ratio"] = df["scientific_word_count"] / (word_counts + 1)
    return df


print("Extracting stylometric features...")
train_features = extract_stylometric_features(train_df["cleaned_text"])
test_features = extract_stylometric_features(test_df["cleaned_text"])

train_vocab = compute_vocabulary_scores(train_df["cleaned_text"])
test_vocab = compute_vocabulary_scores(test_df["cleaned_text"])

feature_columns = [
    col
    for col in train_features.columns
    if col not in ["archaic_word_score", "gothic_word_score", "scientific_word_score"]
]
train_features = pd.concat([train_features[feature_columns], train_vocab], axis=1)
test_features = pd.concat([test_features[feature_columns], test_vocab], axis=1)

train_features = train_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.replace([np.inf, -np.inf], np.nan)

ratio_columns = [
    col for col in train_features.columns if "ratio" in col or "ease" in col
]
count_columns = [col for col in train_features.columns if col not in ratio_columns]

for col in count_columns:
    train_features[col] = train_features[col].fillna(0)
    test_features[col] = test_features[col].fillna(0)
for col in ratio_columns:
    col_mean = train_features[col].mean()
    train_features[col] = train_features[col].fillna(col_mean)
    test_features[col] = test_features[col].fillna(col_mean)

feature_names = train_features.columns.tolist()
train_features_scaled = pd.DataFrame(train_features, columns=feature_names)
test_features_scaled = pd.DataFrame(test_features, columns=feature_names)

train_features_scaled["text"] = train_df["cleaned_text"].values
test_features_scaled["text"] = test_df["cleaned_text"].values


# ============================================================
# 5. MODEL ARCHITECTURE
# ============================================================
class ExpertModule(nn.Module):
    def __init__(self, hidden_size, expert_size=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expert_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_size, expert_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_experts=3, gating_hidden=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gating_hidden),
            nn.GELU(),
            nn.Linear(gating_hidden, num_experts),
        )

    def forward(self, x):
        logits = self.gate(x)
        return F.softmax(logits / 0.5, dim=-1)


class MixtureOfExpertsClassifier(nn.Module):
    def __init__(
        self,
        num_authors=3,
        num_experts=3,
        hidden_size=1024,
        expert_size=256,
        dropout=0.2,
        use_features=False,
        num_features=150,
        feature_proj_size=64,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = True

        self.hidden_size = hidden_size
        self.use_features = use_features

        if use_features and num_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, feature_proj_size),
                nn.LayerNorm(feature_proj_size),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            combined_size = hidden_size + feature_proj_size
        else:
            self.feature_proj = None
            combined_size = hidden_size

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertModule(combined_size, expert_size, dropout)
                for _ in range(num_experts)
            ]
        )
        self.gating = GatingNetwork(combined_size, num_experts, gating_hidden=64)

        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_size // 2, expert_size // 4),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(expert_size // 4, num_authors),
                )
                for _ in range(num_experts)
            ]
        )

        self.direct_head = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, combined_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_size // 2, num_authors),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [
            self.experts,
            self.gating,
            self.expert_heads,
            self.direct_head,
            self.feature_proj,
        ]:
            if module is None:
                continue
            if isinstance(module, nn.ModuleList):
                for m in module:
                    self._apply_xavier(m)
            else:
                self._apply_xavier(module)

    def _apply_xavier(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

        if self.use_features and self.feature_proj is not None and features is not None:
            feat_embed = self.feature_proj(features)
            combined = torch.cat([sentence_embedding, feat_embed], dim=1)
        else:
            combined = sentence_embedding

        gating_weights = self.gating(combined)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(combined))
        expert_outputs = torch.stack(expert_outputs, dim=1)

        expert_logits = []
        for i, head in enumerate(self.expert_heads):
            expert_logits.append(head(expert_outputs[:, i, :]))
        expert_logits = torch.stack(expert_logits, dim=1)

        gating_expanded = gating_weights.unsqueeze(-1)
        moe_logits = torch.sum(expert_logits * gating_expanded, dim=1)
        direct_logits = self.direct_head(combined)

        residual_weight = 0.2
        final_logits = (
            1 - residual_weight
        ) * moe_logits + residual_weight * direct_logits

        return final_logits, gating_weights


# ============================================================
# 6. DATASET CLASS
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, features=None, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx], dtype=torch.float)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 7. PREPARE DATA FOR TRAINING
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
max_len = 256

train_texts = train_features_scaled["text"].values
test_texts = test_features_scaled["text"].values

feature_cols = [c for c in train_features_scaled.columns if c != "text"]
train_features_arr = train_features_scaled[feature_cols].values.astype(np.float32)
test_features_arr = test_features_scaled[feature_cols].values.astype(np.float32)

label_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = train_df["author"].map(label_mapping).values

print(f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")
print(f"Features shape: {train_features_arr.shape}")
print(f"Class distribution: {np.bincount(train_labels)}")

# ============================================================
# 8. TRAINING SETTINGS
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 30
patience = 5
batch_size = 16
num_folds = 5
lr_backbone = 2e-5
lr_head = 5e-5
lr_moe = 4e-5
weight_decay = 0.01
grad_clip = 1.0
label_smoothing = 0.1

# ============================================================
# 9. CROSS-VALIDATION & TRAINING LOOP
# ============================================================
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_scores = []
all_test_preds = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_features_arr, train_labels)
):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}/{num_folds}")
    print(f"{'='*50}")

    X_train_fold = train_features_arr[train_idx]
    y_train_fold = train_labels[train_idx]
    X_val_fold = train_features_arr[val_idx]
    y_val_fold = train_labels[val_idx]
    texts_train_fold = train_texts[train_idx]
    texts_val_fold = train_texts[val_idx]

    train_dataset = SpookyDataset(
        texts=texts_train_fold,
        features=X_train_fold,
        labels=y_train_fold,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    val_dataset = SpookyDataset(
        texts=texts_val_fold,
        features=X_val_fold,
        labels=y_val_fold,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    test_dataset = SpookyDataset(
        texts=test_texts,
        features=test_features_arr,
        labels=None,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = MixtureOfExpertsClassifier(
        num_authors=3,
        num_experts=3,
        hidden_size=1024,
        expert_size=256,
        dropout=0.2,
        use_features=True,
        num_features=train_features_arr.shape[1],
        feature_proj_size=64,
    )
    model.to(device)

    backbone_params = []
    head_params = []
    moe_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "expert" in name or "gating" in name:
            moe_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.999),
            },
            {
                "params": moe_params,
                "lr": lr_moe,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.98),
            },
        ]
    )

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_bal = nn.KLDivLoss(reduction="batchmean")
    uniform_target = torch.full((1, 3), 1.0 / 3, device=device)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    def get_lr_multiplier(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device) if "features" in batch else None

            current_step = epoch * len(train_loader) + batch_idx
            lr_scale = get_lr_multiplier(current_step)
            for pg_idx, pg in enumerate(optimizer.param_groups):
                base_lr = [lr_backbone, lr_head, lr_moe][pg_idx]
                pg["lr"] = base_lr * lr_scale

            optimizer.zero_grad()
            with autocast():
                logits, gating_weights = model(input_ids, attention_mask, features)
                cls_loss = criterion_cls(logits, labels)
                avg_gating = gating_weights.mean(dim=0)
                bal_loss = criterion_bal(avg_gating.unsqueeze(0).log(), uniform_target)
                loss = cls_loss + 0.1 * bal_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                features = batch["features"].to(device) if "features" in batch else None

                with autocast():
                    logits, _ = model(input_ids, attention_mask, features)
                    probs = torch.softmax(logits, dim=1)

                val_preds.append(probs.cpu().numpy())
                val_labels_list.append(labels.numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels_list)

        val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)

        val_loss = sklearn_log_loss(val_labels, val_preds)

        print(
            f"Epoch {epoch+1:2d}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Log Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"./working/best_model_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f"./working/best_model_fold{fold}.pt"))

    model.eval()
    val_preds_final = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device) if "features" in batch else None

            with autocast():
                logits, _ = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

            val_preds_final.append(probs.cpu().numpy())

    val_preds_final = np.concatenate(val_preds_final)
    val_preds_final = np.clip(val_preds_final, 1e-15, 1 - 1e-15)
    val_preds_final = val_preds_final / val_preds_final.sum(axis=1, keepdims=True)
    final_val_loss = sklearn_log_loss(val_labels, val_preds_final)

    fold_scores.append(final_val_loss)
    print(f"Fold {fold+1} Best Val Log Loss: {final_val_loss:.4f}")

    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device) if "features" in batch else None

            with autocast():
                logits, _ = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    all_test_preds.append(test_preds)

    del model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================
# 10. AGGREGATE RESULTS & SUBMISSION
# ============================================================
mean_val_score = np.mean(fold_scores)
std_val_score = np.std(fold_scores)
print(f"\n{'='*50}")
print(f"CROSS-VALIDATION RESULTS")
print(f"{'='*50}")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"Mean: {mean_val_score:.4f} (+/- {std_val_score:.4f})")

final_test_preds = np.mean(all_test_preds, axis=0)
final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)
final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_test_preds[:, 0],
        "HPL": final_test_preds[:, 1],
        "MWS": final_test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission_f02869227e0f431a922652fdc32cc15f.csv", index=False)
print(f"\nSubmission saved to ./submission/submission_f02869227e0f431a922652fdc32cc15f.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Sample predictions:")
print(submission_df.head())

final_score = mean_val_score
print(f"Final Validation Score: {final_score}")