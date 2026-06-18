import os
os.sched_setaffinity(0, {62, 63})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import re
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

warnings.filterwarnings("ignore")

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
    """Clean text while preserving stylistic elements"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# ============================================================
# TARGET ENCODING
# ============================================================
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
num_classes = len(le.classes_)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def extract_features(texts):
    """Extract comprehensive stylistic features"""
    features = []
    for text in texts:
        feat = {}
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        feat["word_count"] = len(words)
        feat["char_count"] = len(text)
        feat["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        feat["max_word_length"] = max([len(w) for w in words]) if words else 0
        feat["sentence_count"] = len(sentences)
        feat["avg_sentence_length"] = (
            np.mean([len(s.split()) for s in sentences]) if sentences else 0
        )
        feat["sentence_length_std"] = (
            np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
        )
        feat["exclamation_count"] = text.count("!")
        feat["question_count"] = text.count("?")
        feat["dash_count"] = text.count("--") + text.count("-")
        feat["semicolon_count"] = text.count(";")
        feat["colon_count"] = text.count(":")
        feat["quote_count"] = text.count('"') + text.count("'")
        feat["comma_count"] = text.count(",")
        feat["ellipsis_count"] = text.count("...") + text.count(". . .")
        feat["bracket_count"] = text.count("(") + text.count(")")
        total_punct = (
            feat["exclamation_count"]
            + feat["question_count"]
            + feat["dash_count"]
            + feat["semicolon_count"]
            + feat["colon_count"]
            + feat["quote_count"]
            + feat["comma_count"]
        )
        feat["punct_density"] = total_punct / max(len(text), 1)

        if words:
            unique_words = set(words)
            feat["unique_words"] = len(unique_words)
            feat["type_token_ratio"] = len(unique_words) / len(words)
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            hapax_count = sum(1 for v in word_freq.values() if v == 1)
            feat["hapax_ratio"] = hapax_count / max(len(words), 1)
        else:
            feat["unique_words"] = 0
            feat["type_token_ratio"] = 0
            feat["hapax_ratio"] = 0

        char_counts = {}
        for c in text:
            if c.strip():
                char_counts[c] = char_counts.get(c, 0) + 1
        total_chars = sum(char_counts.values())
        if total_chars > 0:
            entropy = -sum(
                (count / total_chars) * np.log2(count / total_chars)
                for count in char_counts.values()
            )
            feat["char_entropy"] = entropy
        else:
            feat["char_entropy"] = 0

        feat["capital_word_ratio"] = 0
        if words:
            capital_count = sum(1 for w in words if w and w[0].isupper())
            feat["capital_word_ratio"] = capital_count / len(words)

        common_stops = set(
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
                "by",
                "with",
            ]
        )
        if words:
            stop_count = sum(1 for w in words if w in common_stops)
            feat["stop_word_ratio"] = stop_count / len(words)
        else:
            feat["stop_word_ratio"] = 0

        if words:
            long_word_count = sum(1 for w in words if len(w) > 8)
            feat["long_word_ratio"] = long_word_count / len(words)
        else:
            feat["long_word_ratio"] = 0

        features.append(feat)
    return pd.DataFrame(features)


print("Extracting features...")
train_features = extract_features(train_df["text_clean"].tolist())
test_features = extract_features(test_df["text_clean"].tolist())

train_df["text_len"] = train_df["text"].str.len()
test_df["text_len"] = test_df["text"].str.len()

# First split the data
train_texts_full = train_df["text"].tolist()
y_full = train_df["author_encoded"].values

X_train_texts, X_val_texts, y_train_final, y_val = train_test_split(
    train_texts_full, y_full, test_size=0.15, random_state=42, stratify=y_full
)

# Create temporary dfs for feature engineering on train only
train_split_df = train_df.iloc[train_df.index[train_df["text"].isin(X_train_texts)]]
val_split_df = train_df.iloc[train_df.index[train_df["text"].isin(X_val_texts)]]

# TF-IDF features - fit ONLY on training split
print("Creating TF-IDF features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    strip_accents="unicode",
    lowercase=True,
)
char_train = char_vectorizer.fit_transform(train_split_df["text"])
char_val = char_vectorizer.transform(val_split_df["text"])
char_test = char_vectorizer.transform(test_df["text"])

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    strip_accents="unicode",
    lowercase=True,
    max_df=0.85,
    min_df=3,
    sublinear_tf=True,
)
word_train = word_vectorizer.fit_transform(train_split_df["text"])
word_val = word_vectorizer.transform(val_split_df["text"])
word_test = word_vectorizer.transform(test_df["text"])

# Combine features for train
train_feat_df = pd.concat(
    [
        train_split_df[["id", "author_encoded"]],
        train_features.add_prefix("styl_").iloc[train_split_df.index],
        train_split_df[["text_len"]],
    ],
    axis=1,
)
val_feat_df = pd.concat(
    [
        val_split_df[["id", "author_encoded"]],
        train_features.add_prefix("styl_").iloc[val_split_df.index],
        val_split_df[["text_len"]],
    ],
    axis=1,
)
test_feat_df = pd.concat(
    [test_df[["id"]], test_features.add_prefix("styl_"), test_df[["text_len"]]], axis=1
)

train_tfidf_dense = np.hstack([char_train.toarray(), word_train.toarray()])
val_tfidf_dense = np.hstack([char_val.toarray(), word_val.toarray()])
test_tfidf_dense = np.hstack([char_test.toarray(), word_test.toarray()])

X_train_combined = np.hstack(
    [train_feat_df.drop(["id", "author_encoded"], axis=1).values, train_tfidf_dense]
)
X_val_combined = np.hstack(
    [val_feat_df.drop(["id", "author_encoded"], axis=1).values, val_tfidf_dense]
)
X_test_combined = np.hstack(
    [test_feat_df.drop(["id"], axis=1).values, test_tfidf_dense]
)

y_train = train_feat_df["author_encoded"].values
y_val_labels = val_feat_df["author_encoded"].values

# Scale features - fit ONLY on training data
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_combined)
X_val = scaler.transform(X_val_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Save processed data
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)
np.save("./working/X_train.npy", X_train_final)
np.save("./working/X_val.npy", X_val)
np.save("./working/y_train.npy", y_train_final)
np.save("./working/y_val.npy", y_val)
np.save("./working/X_test.npy", X_test_scaled)
np.save("./working/test_ids.npy", test_df["id"].values)
np.save("./working/num_classes.npy", np.array([num_classes]))
np.save("./working/label_encoder_classes.npy", le.classes_)

STYLISTIC_FEATURE_DIM = X_train_final.shape[1]

# ============================================================
# MODEL DEFINITION
# ============================================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
NUM_CLASSES = 3
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DebertaWithStylisticFeatures(nn.Module):
    def __init__(self, model_name, num_labels, stylistic_dim, dropout_rate=0.2):
        super(DebertaWithStylisticFeatures, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config
        )
        hidden_size = self.config.hidden_size
        self.stylistic_projection = nn.Sequential(
            nn.Linear(stylistic_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(256, num_labels)
        self._init_weights()

    def _init_weights(self):
        for module in [self.stylistic_projection, self.fusion_layer, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, input_ids, attention_mask, stylistic_features=None):
        outputs = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        if stylistic_features is not None:
            stylistic_emb = self.stylistic_projection(stylistic_features)
            combined = torch.cat([cls_output, stylistic_emb], dim=1)
        else:
            padding = torch.zeros(cls_output.shape[0], 128, device=cls_output.device)
            combined = torch.cat([cls_output, padding], dim=1)
        fused = self.fusion_layer(combined)
        logits = self.classifier(fused)
        return logits


# ============================================================
# DATASET CLASS
# ============================================================
class SpookyTextDataset(Dataset):
    def __init__(
        self, texts, stylistic_features, labels=None, tokenizer=None, max_length=512
    ):
        self.texts = texts
        self.stylistic_features = (
            torch.tensor(stylistic_features, dtype=torch.float32)
            if stylistic_features is not None
            else None
        )
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )
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
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        if self.stylistic_features is not None:
            stylistic = self.stylistic_features[idx]
        else:
            stylistic = torch.zeros(STYLISTIC_FEATURE_DIM, dtype=torch.float32)
        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, attention_mask, stylistic, label
        else:
            return input_ids, attention_mask, stylistic


# ============================================================
# PREPARE DATA SPLITS
# ============================================================
train_texts_full = train_df["text"].tolist()
y_full = train_df["author_encoded"].values

X_train_texts, X_val_texts, _, _ = train_test_split(
    train_texts_full, y_full, test_size=0.15, random_state=42, stratify=y_full
)

# Load stylistic features from saved arrays
X_train_styl = np.load("./working/X_train.npy")
X_val_styl = np.load("./working/X_val.npy")
y_train_labels = np.load("./working/y_train.npy")
y_val_labels = np.load("./working/y_val.npy")
X_test_styl = np.load("./working/X_test.npy")
test_ids = np.load("./working/test_ids.npy", allow_pickle=True)

# ============================================================
# TOKENIZER AND MODEL INITIALIZATION
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaWithStylisticFeatures(
    model_name=MODEL_NAME,
    num_labels=NUM_CLASSES,
    stylistic_dim=STYLISTIC_FEATURE_DIM,
    dropout_rate=0.2,
)
model.to(device)

# ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================
class_weights = torch.tensor([0.4, 0.35, 0.25], device=device)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

deberta_params = []
new_params = []
for name, param in model.named_parameters():
    if "deberta" in name:
        deberta_params.append(param)
    else:
        new_params.append(param)

optimizer = AdamW(
    [
        {"params": deberta_params, "lr": LEARNING_RATE},
        {"params": new_params, "lr": LEARNING_RATE * 10},
    ],
    weight_decay=WEIGHT_DECAY,
)

# Create datasets and dataloaders
train_dataset = SpookyTextDataset(
    texts=X_train_texts,
    stylistic_features=X_train_styl,
    labels=y_train_labels,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)
val_dataset = SpookyTextDataset(
    texts=X_val_texts,
    stylistic_features=X_val_styl,
    labels=y_val_labels,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)
scaler = GradScaler()

# ============================================================
# TRAINING LOOP
# ============================================================
best_val_loss = float("inf")
best_model_state = None
patience = 5
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, stylistic, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylistic_features=stylistic,
            )
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, stylistic, labels = [b.to(device) for b in batch]
            with autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stylistic_features=stylistic,
                )
                loss = criterion(logits, labels)
            total_val_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds_clipped = np.clip(all_preds, 1e-15, 1 - 1e-15)
    row_sums = all_preds_clipped.sum(axis=1, keepdims=True)
    all_preds_normalized = all_preds_clipped / row_sums
    N = len(all_labels)
    log_loss = 0.0
    for i in range(N):
        for j in range(NUM_CLASSES):
            y_ij = 1.0 if all_labels[i] == j else 0.0
            log_loss += y_ij * np.log(all_preds_normalized[i, j])
    log_loss = -log_loss / N

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Log Loss: {log_loss:.4f}"
    )

    if log_loss < best_val_loss:
        best_val_loss = log_loss
        torch.save(model.state_dict(), "./working/best_model_0b429f229bd74cbfa4e763daa597b964.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# FINAL VALIDATION EVALUATION
# ============================================================
model.load_state_dict(torch.load("./working/best_model_0b429f229bd74cbfa4e763daa597b964.pt"))
model.to(device)
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, stylistic, labels = [b.to(device) for b in batch]
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylistic_features=stylistic,
            )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_preds_clipped = np.clip(all_preds, 1e-15, 1 - 1e-15)
row_sums = all_preds_clipped.sum(axis=1, keepdims=True)
all_preds_normalized = all_preds_clipped / row_sums
N = len(all_labels)
log_loss = 0.0
for i in range(N):
    for j in range(NUM_CLASSES):
        y_ij = 1.0 if all_labels[i] == j else 0.0
        log_loss += y_ij * np.log(all_preds_normalized[i, j])
final_val_score = -log_loss / N

print(f"Final Validation Score: {final_val_score}")

# ============================================================
# TEST INFERENCE AND SUBMISSION
# ============================================================
test_texts = test_df["text"].tolist()
test_dataset = SpookyTextDataset(
    texts=test_texts,
    stylistic_features=X_test_styl,
    labels=None,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

all_test_probs = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, stylistic = [b.to(device) for b in batch]
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylistic_features=stylistic,
            )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

all_test_probs = np.concatenate(all_test_probs, axis=0)
all_test_probs_clipped = np.clip(all_test_probs, 1e-15, 1 - 1e-15)
row_sums_test = all_test_probs_clipped.sum(axis=1, keepdims=True)
all_test_probs_normalized = all_test_probs_clipped / row_sums_test

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": all_test_probs_normalized[:, 0],
        "HPL": all_test_probs_normalized[:, 1],
        "MWS": all_test_probs_normalized[:, 2],
    }
)

submission_df.to_csv(
    "./submission/submission_0b429f229bd74cbfa4e763daa597b964.csv", index=False, columns=["id", "EAP", "HPL", "MWS"]
)
print("Submission saved to ./submission/submission_0b429f229bd74cbfa4e763daa597b964.csv")