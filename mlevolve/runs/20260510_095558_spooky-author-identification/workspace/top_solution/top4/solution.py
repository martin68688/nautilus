import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import re
import os
import json
import math

# ============================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_id"] = train_df["author"].map(author_mapping)
y = train_df["author_id"].values


def extract_stylometric_features(text_series):
    features = []
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        punct_counts = {
            "num_commas": text.count(","),
            "num_periods": text.count("."),
            "num_exclamation": text.count("!"),
            "num_question": text.count("?"),
            "num_colons": text.count(":"),
            "num_semicolons": text.count(";"),
            "num_dashes": text.count("—") + text.count("-"),
            "num_quotes": text.count('"') + text.count("'"),
            "num_parentheses": text.count("(") + text.count(")"),
        }
        word_lengths = [len(w) for w in words]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        std_word_length = np.std(word_lengths) if len(word_lengths) > 1 else 0
        max_word_length = max(word_lengths) if word_lengths else 0
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
        std_sent_length = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        ellipsis_count = text.count("...") + text.count("…")
        mdash_count = text.count("—")
        lovecraftian_words = [
            "eldritch",
            "cyclopean",
            "non-euclidean",
            "carcosa",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "nyarlathotep",
            "azathoth",
            "necronomicon",
            "unmentionable",
            "unnamable",
            "indescribable",
            "inconceivable",
            "antediluvian",
            "primordial",
            "forbidden",
            "accursed",
            "blasphemous",
        ]
        lovecraft_score = sum(
            1 for w in lovecraftian_words if w.lower() in text.lower()
        )
        poe_words = [
            "nevermore",
            "raven",
            "chamber",
            "ghastly",
            "grim",
            "phantasm",
            "sepulchre",
            "tomb",
            "corpse",
            "pallid",
            "countenance",
            "visage",
            "melancholy",
            "dreary",
            "weary",
            "chilling",
            "dismal",
        ]
        poe_score = sum(1 for w in poe_words if w.lower() in text.lower())
        shelley_words = [
            "nature",
            "spirit",
            "soul",
            "eternal",
            "mortal",
            "immortal",
            "creator",
            "creation",
            "monster",
            "fiend",
            "wretch",
            "daemon",
            "alpine",
            "glacier",
            "mountain",
            "cottage",
            "geneva",
        ]
        shelley_score = sum(1 for w in shelley_words if w.lower() in text.lower())
        repeated_letter_count = sum(1 for w in words if len(set(w)) < len(w))
        feat = [
            avg_word_length,
            std_word_length,
            max_word_length,
            avg_sent_length,
            std_sent_length,
            capital_ratio,
            punct_counts["num_commas"] / max(len(sentences), 1),
            punct_counts["num_periods"] / max(len(sentences), 1),
            punct_counts["num_exclamation"] / max(len(sentences), 1),
            punct_counts["num_question"] / max(len(sentences), 1),
            punct_counts["num_colons"] / max(len(sentences), 1),
            punct_counts["num_semicolons"] / max(len(sentences), 1),
            punct_counts["num_dashes"] / max(len(sentences), 1),
            punct_counts["num_quotes"] / max(len(sentences), 1),
            punct_counts["num_parentheses"] / max(len(sentences), 1),
            ellipsis_count,
            mdash_count,
            lovecraft_score,
            poe_score,
            shelley_score,
            repeated_letter_count / max(len(words), 1),
        ]
        features.append(feat)
    return np.array(features)


train_stylo = extract_stylometric_features(train_df["text"])
test_stylo = extract_stylometric_features(test_df["text"])

tfidf = TfidfVectorizer(
    max_features=5000,
    analyzer="char_wb",
    ngram_range=(2, 5),
    sublinear_tf=True,
    min_df=3,
    max_df=0.9,
)
train_tfidf = tfidf.fit_transform(train_df["text"]).toarray()
test_tfidf = tfidf.transform(test_df["text"]).toarray()

train_features = np.concatenate([train_tfidf, train_stylo], axis=1)
test_features = np.concatenate([test_tfidf, test_stylo], axis=1)

train_texts = train_df["text"].values
test_texts = test_df["text"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(train_df, y):
    y_train, y_val = y[train_idx], y[val_idx]
    train_texts_fold, val_texts = train_texts[train_idx], train_texts[val_idx]
    break

# Fit TF-IDF and stylometric features only on training fold
train_texts_fold_series = train_df["text"].iloc[train_idx]
val_texts_series = train_df["text"].iloc[val_idx]

tfidf = TfidfVectorizer(
    max_features=5000,
    analyzer="char_wb",
    ngram_range=(2, 5),
    sublinear_tf=True,
    min_df=3,
    max_df=0.9,
)
train_tfidf_fold = tfidf.fit_transform(train_texts_fold_series).toarray()
val_tfidf = tfidf.transform(val_texts_series).toarray()
test_tfidf = tfidf.transform(test_df["text"]).toarray()

train_stylo_fold = extract_stylometric_features(train_texts_fold_series)
val_stylo = extract_stylometric_features(val_texts_series)
test_stylo = extract_stylometric_features(test_df["text"])

X_train = np.concatenate([train_tfidf_fold, train_stylo_fold], axis=1)
X_val = np.concatenate([val_tfidf, val_stylo], axis=1)
test_features = np.concatenate([test_tfidf, test_stylo], axis=1)

# ============================================================
# MODEL DESIGN
# ============================================================


class MultiRepAuthorshipModel(nn.Module):
    def __init__(
        self,
        num_authors=3,
        bert_model_name="bert-base-cased",
        stylo_feature_dim=512,
        hidden_dim=256,
        dropout_rate=0.3,
    ):
        super(MultiRepAuthorshipModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_dim = self.bert.config.hidden_size
        self.bert_projection = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.stylo_projection = nn.Sequential(
            nn.Linear(stylo_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, num_authors),
        )

    def forward(self, input_ids, attention_mask, stylo_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]
        bert_rep = self.bert_projection(bert_cls)
        stylo_rep = self.stylo_projection(stylo_features)
        combined = torch.cat([bert_rep, stylo_rep], dim=1)
        gates = self.gate_network(combined)
        gated_bert = gates[:, 0:1] * bert_rep
        gated_stylo = gates[:, 1:2] * stylo_rep
        fused_representation = torch.cat([gated_bert, gated_stylo], dim=1)
        logits = self.classifier(fused_representation)
        return logits


NUM_AUTHORS = 3
stylo_feature_dim = X_train.shape[1]

model = MultiRepAuthorshipModel(
    num_authors=NUM_AUTHORS,
    bert_model_name="bert-base-cased",
    stylo_feature_dim=stylo_feature_dim,
    hidden_dim=256,
    dropout_rate=0.3,
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
bert_params = [p for n, p in model.named_parameters() if "bert" in n]
task_params = [p for n, p in model.named_parameters() if "bert" not in n]
optimizer = AdamW(
    [{"params": bert_params, "lr": 2e-5}, {"params": task_params, "lr": 5e-4}],
    weight_decay=0.01,
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model device: {device}")

# ============================================================
# TRAINING AND EVALUATION
# ============================================================

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)


def tokenize_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


print("Tokenizing texts...")
train_input_ids, train_attention_masks = tokenize_texts(train_texts_fold, tokenizer)
val_input_ids, val_attention_masks = tokenize_texts(val_texts, tokenizer)
test_input_ids, test_attention_masks = tokenize_texts(test_texts, tokenizer)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(test_features, dtype=torch.float32)

batch_size = 16
train_dataset = torch.utils.data.TensorDataset(
    train_input_ids, train_attention_masks, X_train_tensor, y_train_tensor
)
val_dataset = torch.utils.data.TensorDataset(
    val_input_ids, val_attention_masks, X_val_tensor, y_val_tensor
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

num_epochs = 20
best_val_loss = float("inf")
best_model_state = None
patience = 4
patience_counter = 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.cuda.amp.GradScaler()

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_masks, stylo_features, labels = [
            b.to(device) for b in batch
        ]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_masks, stylo_features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_masks, stylo_features, labels = [
                b.to(device) for b in batch
            ]
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_masks, stylo_features)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)
    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
    log_loss_value = 0.0
    for i in range(len(val_labels)):
        for j in range(3):
            y_ij = 1.0 if val_labels[i] == j else 0.0
            log_loss_value -= y_ij * math.log(val_preds_clipped[i, j])
    log_loss_value /= len(val_labels)
    scheduler.step()
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {log_loss_value:.4f}"
    )
    if log_loss_value < best_val_loss:
        best_val_loss = log_loss_value
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  New best model! Val LogLoss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

model.load_state_dict(best_model_state)

# Final validation prediction
model.eval()
all_val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_masks, stylo_features, _ = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_masks, stylo_features)
        probs = torch.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())
val_preds = np.concatenate(all_val_preds, axis=0)
val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
final_val_loss = 0.0
for i in range(len(val_labels)):
    for j in range(3):
        y_ij = 1.0 if val_labels[i] == j else 0.0
        final_val_loss -= y_ij * math.log(val_preds_clipped[i, j])
final_val_loss /= len(val_labels)

print(f"Final Validation Score: {final_val_loss}")

# Test inference
print("Running test inference...")
test_dataset = torch.utils.data.TensorDataset(
    test_input_ids, test_attention_masks, X_test_tensor
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
all_test_preds = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_masks, stylo_features = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_masks, stylo_features)
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())
test_preds = np.concatenate(all_test_preds, axis=0)
test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")