import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from collections import Counter
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class FocalLoss(nn.Module):
    """Focal Loss for focusing on hard examples"""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        probs = torch.exp(log_probs)
        target_one_hot = F.one_hot(target, num_classes=n_classes).float()
        pt = torch.sum(probs * target_one_hot, dim=1)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weight = self.alpha[target] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal_loss = -alpha_weight * focal_weight * torch.log(pt + 1e-12)
        else:
            focal_loss = -focal_weight * torch.log(pt + 1e-12)
        return focal_loss.mean()


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


# Create text-level features
def extract_text_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.count("[.!?]") + 1
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["dash_count"] = text_series.str.count("-")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["period_count"] = text_series.str.count(r"\.")
    features["punct_count"] = text_series.str.count(r"[^\w\s]")
    features["punct_density"] = features["punct_count"] / (features["char_count"] + 1)
    features["capital_words"] = text_series.str.findall(r"\b[A-Z][a-z]*\b").str.len()
    features["all_caps_words"] = text_series.str.findall(r"\b[A-Z]{2,}\b").str.len()
    features["capital_ratio"] = features["capital_words"] / (features["word_count"] + 1)
    features["ellipsis_count"] = text_series.str.count(r"\.\.\.")
    features["has_ellipsis"] = (features["ellipsis_count"] > 0).astype(int)
    features["short_words"] = text_series.str.findall(r"\b\w{1,3}\b").str.len()
    features["medium_words"] = text_series.str.findall(r"\b\w{4,6}\b").str.len()
    features["long_words"] = text_series.str.findall(r"\b\w{7,}\b").str.len()
    features["short_word_ratio"] = features["short_words"] / (
        features["word_count"] + 1
    )
    features["long_word_ratio"] = features["long_words"] / (features["word_count"] + 1)
    features["unique_words"] = text_series.apply(lambda x: len(set(x.lower().split())))
    features["lexical_diversity"] = features["unique_words"] / (
        features["word_count"] + 1
    )
    features["dialog_markers"] = text_series.str.count(r'["\'\u201c\u201d]')
    features["has_dialog"] = (features["dialog_markers"] > 0).astype(int)
    features["first_person_singular"] = text_series.str.contains(
        r"\bI\b|\bme\b|\bmy\b|\bmine\b|\bmyself\b", case=False
    ).astype(int)
    features["first_person_plural"] = text_series.str.contains(
        r"\bwe\b|\bus\b|\bour\b|\bows\b|\bourselves\b", case=False
    ).astype(int)
    features["third_person"] = text_series.str.contains(
        r"\bhe\b|\bshe\b|\bit\b|\bthey\b|\bhim\b|\bher\b|\bthem\b", case=False
    ).astype(int)
    features["past_tense"] = text_series.str.contains(
        r"\bwas\b|\bwere\b|\bhad\b|\bdid\b|\bsaid\b", case=False
    ).astype(int)
    features["present_tense"] = text_series.str.contains(
        r"\bis\b|\bare\b|\bhas\b|\bdo\b|\bdoes\b", case=False
    ).astype(int)
    features["adverb_ly"] = text_series.str.findall(r"\b\w+ly\b").str.len()
    features["adjective_markers"] = text_series.str.findall(
        r"\b\w+ful\b|\b\w+ous\b|\b\w+ive\b|\b\w+able\b"
    ).str.len()
    features["syllables_estimate"] = text_series.apply(
        lambda x: sum(
            1 for word in str(x).split() for vowel in "aeiou" if vowel in word.lower()
        )
    )
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["word_count"] / (features["sentence_count"] + 1)
        - 84.6 * features["syllables_estimate"] / (features["word_count"] + 1)
    )
    # Advanced stylometric features: archaic/supernatural word counts
    archaic_words_list = r"\bthee\b|\bthou\b|\bthy\b|\bthine\b|\bhath\b|\bdoth\b|\bdost\b|\bwilt\b|\bspake\b|\bbetook\b|\bbetwixt\b|\bperchance\b|\bforsooth\b|\bhark\b|\bwhence\b|\bwhither\b|\bmethinks\b|\bverily\b|\begad\b|\btwas\b|\btis\b"
    features["archaic_word_count"] = text_series.str.findall(archaic_words_list, flags=re.IGNORECASE).str.len()
    supernatural_words_list = r"\bghost\b|\bphantom\b|\bspirit\b|\bdemon\b|\bdevil\b|\bhell\b|\bcurse\b|\bhaunt\b|\bapparition\b|\bspecter\b|\bfiend\b|\bwitch\b|\bwizard\b|\bsupernatural\b|\bdeath\b|\bdarkness\b|\bshadow\b|\bhorror\b|\bterror\b|\bfear\b"
    features["supernatural_word_count"] = text_series.str.findall(supernatural_words_list, flags=re.IGNORECASE).str.len()
    # Sentence length statistics
    features["avg_sentence_len"] = features["word_count"] / (features["sentence_count"] + 1)
    features["std_sentence_len"] = text_series.apply(
        lambda x: np.std([len(s.split()) for s in str(x).split('.') if len(s.split()) > 0]) if len(str(x).split('.')) > 1 else 0
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    return features


# Word dropout augmentation for training texts
def word_dropout_augmentation(text, dropout_rate=0.1, mask_token="[MASK]"):
    """Randomly replace dropout_rate of words with [MASK] token."""
    words = str(text).split()
    if len(words) == 0:
        return text
    num_to_mask = max(1, int(len(words) * dropout_rate))
    indices = np.random.choice(len(words), num_to_mask, replace=False)
    for idx in indices:
        words[idx] = mask_token
    return " ".join(words)


print("Creating train/validation split first to prevent data leakage...")
y_train = train_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
train_df_final = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Extracting text features (on train split only for fitting)...")
train_text_features = extract_text_features(train_df_final["text"])
val_text_features = extract_text_features(val_df["text"])
test_text_features = extract_text_features(test_df["text"])

print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    analyzer="word",
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.7,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
)
train_tfidf = tfidf_vectorizer.fit_transform(train_df_final["text"]).toarray()
val_tfidf = tfidf_vectorizer.transform(val_df["text"]).toarray()
test_tfidf = tfidf_vectorizer.transform(test_df["text"]).toarray()

char_vectorizer = TfidfVectorizer(
    max_features=2000,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
)
train_char = char_vectorizer.fit_transform(train_df_final["text"]).toarray()
val_char = char_vectorizer.transform(val_df["text"]).toarray()
test_char = char_vectorizer.transform(test_df["text"]).toarray()

# Part of Speech features
try:
    import nltk

    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt", quiet=True)

    def get_pos_features(texts):
        pos_features = []
        for text in texts:
            tokens = nltk.word_tokenize(str(text))
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = Counter(tag for word, tag in pos_tags)
            pos_features.append(pos_counts)
        return pos_features

    train_pos = get_pos_features(train_df_final["text"].tolist())
    val_pos = get_pos_features(val_df["text"].tolist())
    test_pos = get_pos_features(test_df["text"].tolist())
    all_pos_tags = set()
    for pos_dict in train_pos:
        all_pos_tags.update(pos_dict.keys())
    pos_cols = [f"pos_{tag}" for tag in all_pos_tags]
    train_pos_df = pd.DataFrame(
        [
            {f"pos_{tag}": pos_dict.get(tag, 0) for tag in all_pos_tags}
            for pos_dict in train_pos
        ]
    )
    val_pos_df = pd.DataFrame(
        [
            {f"pos_{tag}": pos_dict.get(tag, 0) for tag in all_pos_tags}
            for pos_dict in val_pos
        ]
    )
    test_pos_df = pd.DataFrame(
        [
            {f"pos_{tag}": pos_dict.get(tag, 0) for tag in all_pos_tags}
            for pos_dict in test_pos
        ]
    )
    train_pos_sum = train_pos_df.sum(axis=1).replace(0, 1)
    val_pos_sum = val_pos_df.sum(axis=1).replace(0, 1)
    test_pos_sum = test_pos_df.sum(axis=1).replace(0, 1)
    train_pos_df = train_pos_df.div(train_pos_sum, axis=0).fillna(0)
    val_pos_df = val_pos_df.div(val_pos_sum, axis=0).fillna(0)
    test_pos_df = test_pos_df.div(test_pos_sum, axis=0).fillna(0)
    print(f"Extracted {len(pos_cols)} POS tag features")
except Exception as e:
    print(f"POS tagging failed (using fallback): {e}")
    train_pos_df = pd.DataFrame()
    val_pos_df = pd.DataFrame()
    test_pos_df = pd.DataFrame()

print("Combining features...")
X_train_final = np.hstack([train_text_features.values, train_tfidf, train_char])
X_val = np.hstack([val_text_features.values, val_tfidf, val_char])
X_test = np.hstack([test_text_features.values, test_tfidf, test_char])
if not train_pos_df.empty and not val_pos_df.empty and not test_pos_df.empty:
    X_train_final = np.hstack([X_train_final, train_pos_df.values])
    X_val = np.hstack([X_val, val_pos_df.values])
    X_test = np.hstack([X_test, test_pos_df.values])

y_train_final = train_df_final["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
y_val = val_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values

print(f"Training samples: {len(train_df_final)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Feature dimension: {X_train_final.shape[1]}")

# Save feature matrices for potential reuse
os.makedirs("./working", exist_ok=True)
np.save("./working/X_train.npy", X_train_final)
np.save("./working/X_val.npy", X_val)
np.save("./working/X_test.npy", X_test)
np.save("./working/y_train.npy", y_train_final)
np.save("./working/y_val.npy", y_val)
train_df_final.to_pickle("./working/train_df.pkl")
val_df.to_pickle("./working/val_df.pkl")
test_df.to_pickle("./working/test_df.pkl")
import joblib

joblib.dump(tfidf_vectorizer, "./working/tfidf_vectorizer.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")

print("Data processing and feature engineering complete!")

# ===== MODEL DESIGN PHASE =====
NUM_LABELS = 3
MAX_LEN = 256
BATCH_SIZE = 16

class DistilRoBERTaBiLSTM(nn.Module):
    """DistilRoBERTa with a 2-layer BiLSTM head for sequential style patterns."""

    def __init__(self, model_name="distilroberta-base", num_labels=3, lstm_hidden=256, lstm_layers=2):
        super().__init__()
        from transformers import AutoModel
        self.roberta = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden)
        # Mean pooling over the sequence length dimension
        mean_pooled = torch.mean(lstm_out, dim=1)
        out = self.dropout(mean_pooled)
        logits = self.classifier(out)
        return logits


set_seed(42)
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DistilRoBERTaBiLSTM(
    model_name=model_name,
    num_labels=NUM_LABELS,
    lstm_hidden=256,
    lstm_layers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = FocalLoss(gamma=2.0, alpha=None)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Gradient accumulation: effective batch size = BATCH_SIZE * ACCUMULATION_STEPS = 16 * 4 = 64
ACCUMULATION_STEPS = 4
effective_batch_size = BATCH_SIZE * ACCUMULATION_STEPS
total_steps = len(train_df_final) // effective_batch_size * 10
warmup_steps = total_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"Models designed: DistilRoBERTa + BiLSTM, Focal Loss (gamma=2.0), gradient accumulation ({ACCUMULATION_STEPS} steps)")
print(
    f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}"
)

# ===== TRAINING AND EVALUATION PHASE =====

# Prepare datasets with word dropout augmentation for training
train_texts = train_df_final["text"].tolist()
val_texts = val_df["text"].tolist()
test_texts = test_df["text"].tolist()

# Augment training texts with word dropout
train_texts_augmented = [word_dropout_augmentation(t, dropout_rate=0.1) if np.random.random() > 0.5 else t for t in train_texts]

train_dataset = TextDataset(train_texts_augmented, y_train_final, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_texts, y_val, tokenizer, MAX_LEN)
test_dataset = TextDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    model_idx,
    epochs=10,
    patience=3,
    accumulation_steps=ACCUMULATION_STEPS,
):
    scaler = GradScaler()
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            total_train_loss += loss.item()
            train_batches += 1
        avg_train_loss = total_train_loss / train_batches

        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast():
                    logits = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    loss = criterion(logits, labels)
                total_val_loss += loss.item()
                val_batches += 1
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / val_batches
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)
        val_preds_clamped = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds_clamped = val_preds_clamped / val_preds_clamped.sum(
            axis=1, keepdims=True
        )
        val_log_loss = -np.mean(
            np.log(val_preds_clamped[np.arange(len(val_labels)), val_labels])
        )

        print(
            f"Model {model_idx} Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
        )

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for Model {model_idx}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_loss


print("Training Model (DistilRoBERTa + BiLSTM)...")
model, val_loss = train_model(
    model, optimizer, scheduler, train_loader, val_loader, 1
)

# Validation predictions
model.eval()
all_val_probs = []
all_val_labels_final = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        with autocast():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)
        all_val_labels_final.append(labels.numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_probs_clamped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clamped = val_probs_clamped / val_probs_clamped.sum(axis=1, keepdims=True)
val_labels_final = np.concatenate(all_val_labels_final, axis=0)
final_val_log_loss = -np.mean(
    np.log(val_probs_clamped[np.arange(len(val_labels_final)), val_labels_final])
)

print(f"Final Validation Score: {final_val_log_loss}")

# Test inference
model.eval()
all_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)

test_probs = np.concatenate(all_test_probs, axis=0)
test_probs_final = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# Create submission file
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")