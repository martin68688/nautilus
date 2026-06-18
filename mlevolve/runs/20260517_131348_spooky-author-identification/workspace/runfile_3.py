import os
os.sched_setaffinity(0, {110, 109, 38})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import os
import re
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f'Classes: {train_df["author"].unique()}')

# Encode target labels
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
num_classes = 3
author_to_idx = {author: idx for idx, author in enumerate(le.classes_)}
print(f"Author mapping: {author_to_idx}")


# Feature engineering functions
def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count


def extract_text_features(text):
    features = {}
    features["char_count"] = len(text)
    features["word_count"] = len(text.split())
    features["sentence_count"] = len(re.split(r"[.!?]+", text)) - 1
    if features["sentence_count"] == 0:
        features["sentence_count"] = 1

    words = text.split()
    word_lengths = [len(w) for w in words]
    features["avg_word_length"] = np.mean(word_lengths) if word_lengths else 0
    features["max_word_length"] = max(word_lengths) if word_lengths else 0
    features["std_word_length"] = np.std(word_lengths) if word_lengths else 0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        sent_lengths = [len(s.split()) for s in sentences]
        features["avg_sentence_length"] = np.mean(sent_lengths)
        features["std_sentence_length"] = np.std(sent_lengths)
        features["max_sentence_length"] = max(sent_lengths)
    else:
        features["avg_sentence_length"] = 0
        features["std_sentence_length"] = 0
        features["max_sentence_length"] = 0

    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["comma_count"] = text.count(",")
    features["semicolon_count"] = text.count(";")
    features["colon_count"] = text.count(":")
    features["dash_count"] = text.count("--") + text.count("—")
    features["quote_count"] = text.count('"') + text.count("'")
    features["ellipsis_count"] = text.count("...") + text.count("…")
    features["paren_count"] = text.count("(") + text.count(")")

    total_punct = sum(
        [
            features[k]
            for k in [
                "exclamation_count",
                "question_count",
                "comma_count",
                "semicolon_count",
                "colon_count",
                "dash_count",
                "paren_count",
            ]
        ]
    )
    features["punct_density"] = total_punct / max(features["word_count"], 1)

    features["capital_word_ratio"] = sum(1 for w in words if w[0].isupper()) / max(
        len(words), 1
    )
    features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)

    if words and features["sentence_count"] > 0:
        total_syllables = sum(count_syllables(w) for w in words)
        features["fk_grade"] = (
            0.39 * (features["word_count"] / features["sentence_count"])
            + 11.8 * (total_syllables / max(features["word_count"], 1))
            - 15.59
        )
        features["syllables_per_word"] = total_syllables / max(len(words), 1)
    else:
        features["fk_grade"] = 0
        features["syllables_per_word"] = 0

    unique_words = set(w.lower() for w in words)
    features["type_token_ratio"] = len(unique_words) / max(len(words), 1)

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
        "are",
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
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
    }
    words_lower = [w.lower() for w in words]
    features["stopword_ratio"] = sum(1 for w in words_lower if w in stopwords) / max(
        len(words), 1
    )

    first_person = sum(
        1 for w in words_lower if w in ["i", "me", "my", "mine", "myself"]
    )
    features["first_person_count"] = first_person
    features["first_person_ratio"] = first_person / max(len(words), 1)

    archaic_words = {
        "thou",
        "thee",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "dost",
        "art",
        "wast",
        "wert",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "betwixt",
        "amidst",
        "amongst",
        "whilst",
        "ere",
        "anon",
        "perchance",
        "whence",
    }
    features["archaic_word_ratio"] = sum(
        1 for w in words_lower if w in archaic_words
    ) / max(len(words), 1)

    features["dialogue_indicator"] = (
        1
        if (
            text.count('"') >= 2
            or text.count("'") >= 2
            or any(
                w.lower()
                in [
                    "said",
                    "asked",
                    "replied",
                    "cried",
                    "exclaimed",
                    "whispered",
                    "shouted",
                ]
                for w in words_lower
            )
        )
        else 0
    )

    features["char_per_word"] = features["char_count"] / max(features["word_count"], 1)
    features["word_per_sentence"] = features["word_count"] / max(
        features["sentence_count"], 1
    )

    emotion_words = {
        "fear",
        "horror",
        "terror",
        "dread",
        "anguish",
        "agony",
        "despair",
        "grief",
        "sorrow",
        "weep",
        "cry",
        "wail",
        "sob",
        "lament",
        "mourn",
        "gloom",
        "dark",
        "shadow",
        "phantom",
        "spectre",
        "ghost",
        "demon",
        "devil",
        "hell",
        "death",
        "dead",
        "dying",
        "corpse",
        "coffin",
        "grave",
        "tomb",
        "skeleton",
    }
    features["emotion_word_ratio"] = sum(
        1 for w in words_lower if w in emotion_words
    ) / max(len(words), 1)

    negative_words = {
        "no",
        "not",
        "never",
        "nothing",
        "none",
        "nor",
        "neither",
        "nowhere",
        "nobody",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    }
    features["negative_word_ratio"] = sum(
        1 for w in words_lower if w in negative_words
    ) / max(len(words), 1)

    return features


print("Extracting features from training data...")
train_features = [extract_text_features(text) for text in train_df["text"].values]
train_feat_df = pd.DataFrame(train_features)

print("Extracting features from test data...")
test_features = [extract_text_features(text) for text in test_df["text"].values]
test_feat_df = pd.DataFrame(test_features)

train_feat_df = train_feat_df.fillna(0)
test_feat_df = test_feat_df.fillna(0)

train_combined = pd.concat(
    [train_df[["id", "text", "author", "author_encoded"]], train_feat_df], axis=1
)
test_combined = pd.concat([test_df[["id", "text"]], test_feat_df], axis=1)

train_combined.columns = [
    str(col).replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
    for col in train_combined.columns
]
test_combined.columns = [
    str(col).replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
    for col in test_combined.columns
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_combined, train_combined["author_encoded"]))

# CORRECT split handling - use indices directly (INDEX_BUG prevention)
train_texts = train_combined["text"].values[train_idx]
train_labels = train_combined["author_encoded"].values[train_idx]
val_texts = train_combined["text"].values[val_idx]
val_labels = train_combined["author_encoded"].values[val_idx]
test_texts = test_combined["text"].values

feature_cols = [
    col
    for col in train_combined.columns
    if col not in ["id", "text", "author", "author_encoded"]
]
X_train_feat = train_combined[feature_cols].values[train_idx]
X_val_feat = train_combined[feature_cols].values[val_idx]
X_test_feat = test_combined[feature_cols].values

scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)
X_test_feat = scaler.transform(X_test_feat)

print("Loading ModernBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
max_length = 256

print("Tokenizing training data...")
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt",
)

print("Tokenizing validation data...")
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt",
)

print("Tokenizing test data...")
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt",
)

print(f"\nDataset splits:")
print(f"Train samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")
print(f"Feature dimensions: {X_train_feat.shape[1]} engineered features")

assert (
    len(set(train_idx) & set(val_idx)) == 0
), "Data leakage detected: train and validation indices overlap!"
print("✓ No data leakage detected between train and validation sets")


# Model Definition
class TwoStreamAuthorClassifier(nn.Module):
    def __init__(self, num_classes=3, feature_dim=31, hidden_dim=256, dropout_rate=0.3):
        super().__init__()

        self.bert = ModernBertForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-large",
            num_labels=hidden_dim,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.bert.classifier = nn.Identity()

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.num_samples = 4
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(self.num_samples)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.feature_encoder, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, features, labels=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        bert_cls = bert_outputs.hidden_states[-1][:, 0, :]

        encoded_features = self.feature_encoder(features)

        cls_expanded = bert_cls.unsqueeze(1)
        feat_expanded = encoded_features.unsqueeze(1)

        attended_features, _ = self.cross_attention(
            query=cls_expanded, key=feat_expanded, value=feat_expanded
        )
        attended_features = attended_features.squeeze(1)

        logits_list = []
        for dropout in self.dropout_layers:
            bert_dropped = dropout(bert_cls)
            feat_dropped = dropout(attended_features)
            combined = torch.cat([bert_dropped, feat_dropped], dim=1)
            logits = self.classifier(combined)
            logits_list.append(logits)

        logits = torch.stack(logits_list).mean(dim=0)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels)
            return loss, logits

        return logits


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

feature_dim = X_train_feat.shape[1]
model = TwoStreamAuthorClassifier(num_classes=3, feature_dim=feature_dim).to(device)

bert_params = list(model.bert.parameters())
other_params = list(set(model.parameters()) - set(bert_params))

optimizer = AdamW(
    [
        {"params": bert_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": other_params, "lr": 2e-5 * 5, "weight_decay": 0.01},
    ]
)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Create DataLoaders
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(X_train_feat, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(X_val_feat, dtype=torch.float32),
    torch.tensor(val_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

# Training loop
num_epochs = 10
scaler_grad = torch.cuda.amp.GradScaler()

best_val_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, features, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
                labels=labels,
            )

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds_list = []
    val_labels_list = []
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, features, labels = [x.to(device) for x in batch]
            with torch.cuda.amp.autocast():
                val_loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features,
                    labels=labels,
                )
            total_val_loss += val_loss.item()
            val_preds_list.append(F.softmax(logits, dim=1).cpu().numpy())
            val_labels_list.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_preds_all = np.clip(np.vstack(val_preds_list), 1e-15, 1 - 1e-15)
    val_labels_all = np.concatenate(val_labels_list)
    val_log_loss = log_loss(val_labels_all, val_preds_all)

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_log_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "./working/best_model_a0d11613c5284573bb065220454ab7ed.pt")

    scheduler.step()

# Load best model for test predictions
model.load_state_dict(torch.load("./working/best_model_a0d11613c5284573bb065220454ab7ed.pt"))
model.eval()

# Test prediction
test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
    torch.tensor(X_test_feat, dtype=torch.float32),
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

test_preds_list = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, features = [x.to(device) for x in batch]
        with torch.cuda.amp.autocast():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, features=features
            )
        probs = F.softmax(logits, dim=1).cpu().numpy()
        test_preds_list.append(probs)

test_preds = np.vstack(test_preds_list)
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission
os.makedirs("./submission", exist_ok=True)
test_ids = test_df["id"].values
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission_a0d11613c5284573bb065220454ab7ed.csv", index=False)

print(f"Submission saved to ./submission/submission_a0d11613c5284573bb065220454ab7ed.csv")
print(f"Submission shape: {submission.shape}")
print(f"Final Validation Score: {val_log_loss:.6f}")
