import pandas as pd
import numpy as np
import re
import os
import warnings
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# ============================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================
def create_text_features(df, is_train=True):
    features = pd.DataFrame()
    features["id"] = df["id"].values
    text = df["text"].values

    features["char_count"] = [len(t) for t in text]
    features["word_count"] = [len(t.split()) for t in text]
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = [len(re.findall(r"[.!?]+", t)) for t in text]
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )

    features["exclamation_count"] = [t.count("!") for t in text]
    features["question_count"] = [t.count("?") for t in text]
    features["period_count"] = [t.count(".") for t in text]
    features["comma_count"] = [t.count(",") for t in text]
    features["semicolon_count"] = [t.count(";") for t in text]
    features["colon_count"] = [t.count(":") for t in text]
    features["dash_count"] = [t.count("-") for t in text]
    features["quote_count"] = [t.count('"') + t.count("'") for t in text]
    features["paren_count"] = [t.count("(") + t.count(")") for t in text]
    features["punctuation_ratio"] = features[
        [
            "exclamation_count",
            "question_count",
            "period_count",
            "comma_count",
            "semicolon_count",
            "colon_count",
            "dash_count",
            "quote_count",
            "paren_count",
        ]
    ].sum(axis=1) / (features["char_count"] + 1)

    features["capital_ratio"] = [
        sum(1 for c in t if c.isupper()) / (len(t) + 1) for t in text
    ]
    features["first_word_cap"] = [1 if t and t[0].isupper() else 0 for t in text]
    features["unique_word_ratio"] = [
        len(set(t.lower().split())) / (len(t.split()) + 1) for t in text
    ]
    features["stopword_ratio"] = [
        sum(
            1
            for w in t.lower().split()
            if w
            in set(
                [
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
                    "is",
                    "was",
                    "are",
                    "were",
                    "have",
                    "has",
                    "had",
                    "be",
                    "been",
                    "being",
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
                    "i",
                    "you",
                    "he",
                    "she",
                    "it",
                    "we",
                    "they",
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
                ]
            )
        )
        / (len(t.split()) + 1)
        for t in text
    ]

    char_vectorizer = CountVectorizer(
        analyzer="char", ngram_range=(1, 3), max_features=500
    )
    char_features = char_vectorizer.fit_transform(text)
    char_feat_names = [f"char_ngram_{i}" for i in range(char_features.shape[1])]
    char_df = pd.DataFrame(char_features.toarray(), columns=char_feat_names)

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=1000, stop_words="english", sublinear_tf=True
    )
    tfidf_features = tfidf_vectorizer.fit_transform(text)
    tfidf_feat_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_feat_names)

    features = pd.concat([features, char_df, tfidf_df], axis=1)

    if is_train:
        return features, char_vectorizer, tfidf_vectorizer
    return features


X = train_df["text"].values
y = train_df["author"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(X, y))[0]

train_texts = X[train_idx]
val_texts = X[val_idx]
train_labels = y[train_idx]
val_labels = y[val_idx]

train_sub = train_df.iloc[train_idx].reset_index(drop=True)
val_sub = train_df.iloc[val_idx].reset_index(drop=True)

train_features, char_vectorizer, tfidf_vectorizer = create_text_features(
    train_sub, is_train=True
)


def extract_basic_features(text, features_df):
    features_df["char_count"] = [len(t) for t in text]
    features_df["word_count"] = [len(t.split()) for t in text]
    features_df["avg_word_len"] = features_df["char_count"] / (
        features_df["word_count"] + 1
    )
    features_df["sentence_count"] = [len(re.findall(r"[.!?]+", t)) for t in text]
    features_df["avg_sentence_len"] = features_df["word_count"] / (
        features_df["sentence_count"] + 1
    )
    features_df["exclamation_count"] = [t.count("!") for t in text]
    features_df["question_count"] = [t.count("?") for t in text]
    features_df["period_count"] = [t.count(".") for t in text]
    features_df["comma_count"] = [t.count(",") for t in text]
    features_df["semicolon_count"] = [t.count(";") for t in text]
    features_df["colon_count"] = [t.count(":") for t in text]
    features_df["dash_count"] = [t.count("-") for t in text]
    features_df["quote_count"] = [t.count('"') + t.count("'") for t in text]
    features_df["paren_count"] = [t.count("(") + t.count(")") for t in text]
    features_df["punctuation_ratio"] = features_df[
        [
            "exclamation_count",
            "question_count",
            "period_count",
            "comma_count",
            "semicolon_count",
            "colon_count",
            "dash_count",
            "quote_count",
            "paren_count",
        ]
    ].sum(axis=1) / (features_df["char_count"] + 1)
    features_df["capital_ratio"] = [
        sum(1 for c in t if c.isupper()) / (len(t) + 1) for t in text
    ]
    features_df["first_word_cap"] = [1 if t and t[0].isupper() else 0 for t in text]
    features_df["unique_word_ratio"] = [
        len(set(t.lower().split())) / (len(t.split()) + 1) for t in text
    ]
    features_df["stopword_ratio"] = [
        sum(
            1
            for w in t.lower().split()
            if w
            in set(
                [
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
                    "is",
                    "was",
                    "are",
                    "were",
                    "have",
                    "has",
                    "had",
                    "be",
                    "been",
                    "being",
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
                    "i",
                    "you",
                    "he",
                    "she",
                    "it",
                    "we",
                    "they",
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
                ]
            )
        )
        / (len(t.split()) + 1)
        for t in text
    ]
    return features_df


val_features = pd.DataFrame()
val_features["id"] = val_sub["id"].values
val_features = extract_basic_features(val_texts, val_features)
val_char_features = char_vectorizer.transform(val_texts)
val_char_df = pd.DataFrame(
    val_char_features.toarray(),
    columns=[f"char_ngram_{i}" for i in range(val_char_features.shape[1])],
)
val_tfidf_features = tfidf_vectorizer.transform(val_texts)
val_tfidf_df = pd.DataFrame(
    val_tfidf_features.toarray(),
    columns=[f"tfidf_{i}" for i in range(val_tfidf_features.shape[1])],
)
val_features = pd.concat([val_features, val_char_df, val_tfidf_df], axis=1)

test_features = pd.DataFrame()
test_features["id"] = test_df["id"].values
test_texts = test_df["text"].values
test_features = extract_basic_features(test_texts, test_features)
test_char_features = char_vectorizer.transform(test_texts)
test_char_df = pd.DataFrame(
    test_char_features.toarray(),
    columns=[f"char_ngram_{i}" for i in range(test_char_features.shape[1])],
)
test_tfidf_features = tfidf_vectorizer.transform(test_texts)
test_tfidf_df = pd.DataFrame(
    test_tfidf_features.toarray(),
    columns=[f"tfidf_{i}" for i in range(test_tfidf_features.shape[1])],
)
test_features = pd.concat([test_features, test_char_df, test_tfidf_df], axis=1)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

train_features_for_model = train_features.drop("id", axis=1)
val_features_for_model = val_features.drop("id", axis=1)
test_features_for_model = test_features.drop("id", axis=1)

basic_feat_cols = list(range(18))
scaler = StandardScaler()
train_basic_scaled = scaler.fit_transform(train_features_for_model.iloc[:, :18])
val_basic_scaled = scaler.transform(val_features_for_model.iloc[:, :18])
test_basic_scaled = scaler.transform(test_features_for_model.iloc[:, :18])

train_features_scaled = np.concatenate(
    [train_basic_scaled, train_features_for_model.iloc[:, 18:].values], axis=1
)
val_features_scaled = np.concatenate(
    [val_basic_scaled, val_features_for_model.iloc[:, 18:].values], axis=1
)
test_features_scaled = np.concatenate(
    [test_basic_scaled, test_features_for_model.iloc[:, 18:].values], axis=1
)

os.makedirs("./working", exist_ok=True)
np.save("./working/train_features.npy", train_features_scaled)
np.save("./working/val_features.npy", val_features_scaled)
np.save("./working/test_features.npy", test_features_scaled)
np.save("./working/train_labels.npy", train_labels_encoded)
np.save("./working/val_labels.npy", val_labels_encoded)
np.save("./working/test_ids.npy", test_df["id"].values)
np.save("./working/label_classes.npy", label_encoder.classes_)
feature_names = list(train_features_for_model.columns)
np.save("./working/feature_names.npy", feature_names)

print(f"Train features shape: {train_features_scaled.shape}")
print(f"Validation features shape: {val_features_scaled.shape}")
print(f"Test features shape: {test_features_scaled.shape}")

# ============================================================
# MODEL DESIGN - SpookyAuthorClassifier
# ============================================================
NUM_CLASSES = 3
PRETRAINED_MODEL = "microsoft/deberta-v3-small"
MAX_SEQ_LENGTH = 256
HIDDEN_SIZE = 768


class MultiScaleConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2
        )
        self.conv4 = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=7, padding=3
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        c1 = self.relu(self.conv1(x_perm))
        c2 = self.relu(self.conv2(x_perm))
        c3 = self.relu(self.conv3(x_perm))
        c4 = self.relu(self.conv4(x_perm))
        pooled1 = F.adaptive_max_pool1d(c1, 1).squeeze(-1)
        pooled2 = F.adaptive_max_pool1d(c2, 1).squeeze(-1)
        pooled3 = F.adaptive_max_pool1d(c3, 1).squeeze(-1)
        pooled4 = F.adaptive_max_pool1d(c4, 1).squeeze(-1)
        combined = torch.cat([pooled1, pooled2, pooled3, pooled4], dim=-1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        return combined


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        scores = self.attention_weights(hidden_states).squeeze(-1)
        if attention_mask is not None:
            # Use -10000.0 instead of -1e9 to avoid float16 overflow in mixed precision
            scores = scores.masked_fill(attention_mask == 0, -10000.0)
        attention_weights = F.softmax(scores, dim=-1)
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(
            1
        )
        return weighted_sum


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_bert=True, dropout=0.3):
        super().__init__()
        self.deberta = None
        self.bert_dim = HIDDEN_SIZE
        self.multi_scale_conv = MultiScaleConvBlock(
            input_dim=self.bert_dim, hidden_dim=256, dropout=dropout
        )
        self.attention_pool = AttentionPooling(self.bert_dim)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 + self.bert_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_backbone(self, model_name=PRETRAINED_MODEL):
        self.deberta = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        conv_features = self.multi_scale_conv(sequence_output)
        attended_features = self.attention_pool(sequence_output, attention_mask)
        cls_features = sequence_output[:, 0, :]
        combined_features = torch.cat([conv_features, attended_features], dim=-1)
        logits = self.classifier(combined_features)
        if return_embeddings:
            return logits, combined_features
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0)
            true_dist = true_dist * (1.0 - self.smoothing) + self.smoothing / n_classes
            if self.weight is not None:
                weights = self.weight[target].unsqueeze(1)
                true_dist = true_dist * weights
        log_probs = F.log_softmax(pred, dim=-1)
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        label_smoothing=0.1,
        focal_gamma=2.0,
        smoothing_weight=0.7,
        focal_weight=0.3,
        class_weights=None,
    ):
        super().__init__()
        self.smoothing_loss = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=class_weights
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.smoothing_weight = smoothing_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        loss1 = self.smoothing_loss(pred, target)
        loss2 = self.focal_loss(pred, target)
        return self.smoothing_weight * loss1 + self.focal_weight * loss2


def compute_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        weights[cls] = total / (num_classes * count)
    weights = weights / weights.mean()
    return weights.float()


# ============================================================
# TRAINING AND EVALUATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
max_length = MAX_SEQ_LENGTH


def tokenize_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


print("Tokenizing training data...")
train_input_ids, train_attention_mask = tokenize_texts(
    train_texts, tokenizer, max_length
)
print("Tokenizing validation data...")
val_input_ids, val_attention_mask = tokenize_texts(val_texts, tokenizer, max_length)
print("Tokenizing test data...")
test_input_ids, test_attention_mask = tokenize_texts(test_texts, tokenizer, max_length)

train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)
val_labels_tensor = torch.tensor(val_labels_encoded, dtype=torch.long)

batch_size = 8
train_dataset = TensorDataset(
    train_input_ids, train_attention_mask, train_labels_tensor
)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels_tensor)
test_dataset = TensorDataset(test_input_ids, test_attention_mask)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

print(
    f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
)

# Initialize model
model = SpookyAuthorClassifier(num_classes=NUM_CLASSES, freeze_bert=True)
model.initialize_backbone()
for param in model.deberta.parameters():
    param.requires_grad = False

model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,}, Total params: {total_params:,}")

class_weights = compute_class_weights(train_labels_encoded).to(device)
criterion = CombinedLoss(
    label_smoothing=0.1,
    focal_gamma=2.0,
    smoothing_weight=0.7,
    focal_weight=0.3,
    class_weights=class_weights,
)

optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "deberta" not in n
        ],
        "lr": 2e-5,
        "weight_decay": 0.01,
    }
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

scaler = GradScaler()
num_epochs = 20
best_val_loss = float("inf")
best_model_state = None
patience = 4
patience_counter = 0

# Enable gradient checkpointing for memory efficiency
if hasattr(model.deberta, "gradient_checkpointing_enable"):
    model.deberta.gradient_checkpointing_enable()

print("\n===== Starting Training =====")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            total_val_loss += loss.item()
            num_val_batches += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_val_preds.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_preds_concat = np.concatenate(all_val_preds, axis=0)
    val_labels_concat = np.concatenate(all_val_labels, axis=0)

    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds_concat, epsilon, 1 - epsilon)
    row_sums = val_preds_clipped.sum(axis=1, keepdims=True)
    val_preds_normalized = val_preds_clipped / row_sums
    val_log_loss = log_loss(val_labels_concat, val_preds_normalized)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f} | LR: {current_lr:.2e}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

print(f"\n===== Training Complete =====")
print(f"Best validation log-loss: {best_val_loss:.6f}")

# Load best model
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Final validation
print("\nComputing final validation metrics...")
all_val_probs = []
all_val_labels_final = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)
        all_val_labels_final.append(labels.cpu().numpy())

val_probs_concat = np.concatenate(all_val_probs, axis=0)
val_labels_concat = np.concatenate(all_val_labels_final, axis=0)

epsilon = 1e-15
val_probs_clipped = np.clip(val_probs_concat, epsilon, 1 - epsilon)
row_sums = val_probs_clipped.sum(axis=1, keepdims=True)
val_probs_normalized = val_probs_clipped / row_sums
final_val_log_loss = log_loss(val_labels_concat, val_probs_normalized)
print(f"Final Validation Log-Loss: {final_val_log_loss:.6f}")

# Test inference
print("\nGenerating test predictions...")
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

test_probs_concat = np.concatenate(all_test_probs, axis=0)
test_probs_clipped = np.clip(test_probs_concat, epsilon, 1 - epsilon)
row_sums_test = test_probs_clipped.sum(axis=1, keepdims=True)
test_probs_normalized = test_probs_clipped / row_sums_test

test_ids = np.load("./working/test_ids.npy", allow_pickle=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_normalized[:, 0],
        "HPL": test_probs_normalized[:, 1],
        "MWS": test_probs_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_log_loss}")