import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import warnings
import gc
import scipy.sparse as sp
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train authors: {train_df['author'].value_counts().to_dict()}")

# Encode target labels
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_encoded"] = train_df["author"].map(author_map)

# Split into train and validation sets (stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_df["text"], train_df["author_encoded"]))[0]

train_texts = train_df["text"].iloc[train_idx].values
val_texts = train_df["text"].iloc[val_idx].values
test_texts = test_df["text"].values

train_labels = train_df["author_encoded"].iloc[train_idx].values
val_labels = train_df["author_encoded"].iloc[val_idx].values

print(
    f"Train size: {len(train_texts)}, Val size: {len(val_texts)}, Test size: {len(test_texts)}"
)


# ========== FEATURE ENGINEERING ==========
def extract_features(texts, fit_vectorizers=False):
    """Extract linguistic and stylistic features from text corpus."""
    features_list = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_words = len(words)
        num_chars = len(text)
        num_sentences = max(len(sentences), 1)
        num_unique_words = len(set(words))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = num_words / num_sentences
        capital_ratio = sum(1 for c in text if c.isupper()) / max(num_chars, 1)
        punct_ratio = sum(1 for c in text if c in ".,;:!?'\"-()[]{}") / max(
            num_chars, 1
        )
        comma_count = text.count(",")
        period_count = text.count(".")
        exclamation_count = text.count("!")
        question_count = text.count("?")
        semicolon_count = text.count(";")
        colon_count = text.count(":")
        dash_count = text.count("-")
        quote_count = text.count('"') + text.count("'")
        type_token_ratio = num_unique_words / max(num_words, 1)
        hapax_ratio = sum(1 for w, c in Counter(words).items() if c == 1) / max(
            num_words, 1
        )
        function_words = [
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
            "is",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "not",
            "no",
            "so",
            "if",
            "then",
            "than",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "us",
            "them",
            "who",
            "which",
            "what",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "some",
            "any",
            "many",
            "much",
            "more",
            "most",
            "few",
            "little",
            "very",
            "too",
            "quite",
            "rather",
            "just",
            "only",
            "also",
            "even",
            "still",
            "yet",
        ]
        func_word_counts = [words.count(fw) for fw in function_words]
        func_word_ratios = [c / max(num_words, 1) for c in func_word_counts]
        ellipses_count = text.count("...")
        em_dash_count = len(re.findall(r"--", text))
        exclamation_ratio = exclamation_count / max(num_sentences, 1)
        question_ratio = question_count / max(num_sentences, 1)
        first_person_singular = sum(
            1 for w in words if w.lower() in ["i", "me", "my", "mine", "myself"]
        )
        first_person_plural = sum(
            1 for w in words if w.lower() in ["we", "us", "our", "ours", "ourselves"]
        )
        feature_row = {
            "num_words": num_words,
            "num_chars": num_chars,
            "num_sentences": num_sentences,
            "num_unique_words": num_unique_words,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "capital_ratio": capital_ratio,
            "punct_ratio": punct_ratio,
            "comma_count": comma_count,
            "period_count": period_count,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "semicolon_count": semicolon_count,
            "colon_count": colon_count,
            "dash_count": dash_count,
            "quote_count": quote_count,
            "type_token_ratio": type_token_ratio,
            "hapax_ratio": hapax_ratio,
            "ellipses_count": ellipses_count,
            "em_dash_count": em_dash_count,
            "exclamation_ratio": exclamation_ratio,
            "question_ratio": question_ratio,
            "first_person_singular": first_person_singular,
            "first_person_plural": first_person_plural,
        }
        for i, fw in enumerate(function_words):
            feature_row[f"fw_{fw}_ratio"] = func_word_ratios[i]
        char_ngrams = []
        for n in [2, 3]:
            text_clean = text.lower()
            for i in range(len(text_clean) - n + 1):
                char_ngrams.append(text_clean[i : i + n])
        char_ngram_counts = Counter(char_ngrams)
        common_char_ngrams = [
            "th",
            "he",
            "in",
            "er",
            "an",
            "re",
            "nd",
            "on",
            "at",
            "en",
            "the",
            "and",
            "ing",
            "her",
            "hat",
            "ith",
            "tha",
            "ere",
            "ent",
            "was",
        ]
        for ng in common_char_ngrams:
            feature_row[f"charn2_{ng}"] = char_ngram_counts.get(ng, 0) / max(
                num_chars, 1
            )
        features_list.append(feature_row)
    features_df = pd.DataFrame(features_list)
    if fit_vectorizers:
        char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=500, lowercase=True
        )
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=500,
            lowercase=True,
            min_df=3,
            max_df=0.9,
        )
        char_tfidf = char_vectorizer.fit_transform(texts)
        word_tfidf = word_vectorizer.fit_transform(texts)
        vectorizers = {"char": char_vectorizer, "word": word_vectorizer}
    else:
        char_tfidf = char_vectorizer_fitted.transform(texts)
        word_tfidf = word_vectorizer_fitted.transform(texts)
        vectorizers = None
    return features_df, char_tfidf, word_tfidf, vectorizers


print("Extracting features from training data...")
train_features, train_char_tfidf, train_word_tfidf, vectorizers = extract_features(
    train_texts, fit_vectorizers=True
)
char_vectorizer_fitted = vectorizers["char"]
word_vectorizer_fitted = vectorizers["word"]

print("Extracting features from validation data...")
val_features, val_char_tfidf, val_word_tfidf, _ = extract_features(
    val_texts, fit_vectorizers=False
)

print("Extracting features from test data...")
test_features, test_char_tfidf, test_word_tfidf, _ = extract_features(
    test_texts, fit_vectorizers=False
)


def combine_features(base_features, char_tfidf, word_tfidf):
    return sp.hstack([sp.csr_matrix(base_features.values), char_tfidf, word_tfidf])


X_train = combine_features(train_features, train_char_tfidf, train_word_tfidf)
X_val = combine_features(val_features, val_char_tfidf, val_word_tfidf)
X_test = combine_features(test_features, test_char_tfidf, test_word_tfidf)

y_train = train_labels
y_val = val_labels

print(
    f"Feature matrix shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
)
print(f"Number of engineered features: {len(train_features.columns)}")

# Save processed data
os.makedirs("./working", exist_ok=True)
np.save("./working/X_train.npy", X_train.toarray())
np.save("./working/X_val.npy", X_val.toarray())
np.save("./working/X_test.npy", X_test.toarray())
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)
pd.DataFrame(train_features).to_pickle("./working/train_features.pkl")
pd.DataFrame(val_features).to_pickle("./working/val_features.pkl")
pd.DataFrame(test_features).to_pickle("./working/test_features.pkl")
np.save("./working/train_texts.npy", train_texts)
np.save("./working/val_texts.npy", val_texts)
np.save("./working/test_texts.npy", test_texts)

print("Data preprocessing and feature engineering complete!")

# ========== MODEL DESIGN ==========
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class DebertaAuthorClassifier(nn.Module):
    def __init__(self, num_labels=3, dropout=0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_authors = 3
model = DebertaAuthorClassifier(num_labels=num_authors).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999))
total_steps = 20 * (len(train_texts) // 16 + 1)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(
    f"Model: DeBERTa-v3-small with {sum(p.numel() for p in model.parameters()):,} parameters"
)
print(f"Device: {device}")


# ========== TRAINING AND EVALUATION ==========
def create_dataloader(texts, labels=None, batch_size=16, shuffle=False):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )
    if labels is not None:
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
    else:
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"], encodings["attention_mask"]
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )


train_loader = create_dataloader(train_texts, train_labels, batch_size=8, shuffle=True)
val_loader = create_dataloader(val_texts, val_labels, batch_size=8, shuffle=False)
test_loader = create_dataloader(test_texts, batch_size=8, shuffle=False)

num_epochs = 20
gradient_accumulation_steps = 2
best_val_loss = float("inf")
patience = 5
patience_counter = 0
best_model_path = "./working/best_model.pt"


def label_smoothing_loss(logits, labels, smoothing=0.1):
    n_classes = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = label_smoothing_loss(logits, labels, smoothing=0.1)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    model.eval()
    val_loss = 0
    val_num_batches = 0
    all_val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            val_num_batches += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_val_probs.append(probs)
    avg_val_loss = val_loss / val_num_batches
    val_probs = np.concatenate(all_val_probs, axis=0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(
        axis=1, keepdims=True
    )
    val_log_loss = log_loss(val_labels, val_probs_normalized)
    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {val_log_loss:.4f}"
    )
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        patience_counter = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_log_loss": val_log_loss,
                "epoch": epoch,
            },
            best_model_path,
        )
        print(f"  -> New best model saved! Val Log Loss: {val_log_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_val_probs = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_val_probs.append(probs)
val_probs = np.concatenate(all_val_probs, axis=0)
epsilon = 1e-15
val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
val_log_loss = log_loss(val_labels, val_probs_normalized)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        if len(batch) == 2:
            input_ids, attention_mask = [b.to(device) for b in batch]
        else:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_probs.append(probs)
test_probs = np.concatenate(all_test_probs, axis=0)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission.iloc[:, 1:] = submission.iloc[:, 1:].div(
    submission.iloc[:, 1:].sum(axis=1), axis=0
)
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(epsilon, 1 - epsilon)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)

del train_loader, val_loader, test_loader
gc.collect()
torch.cuda.empty_cache()

print(f"Final Validation Score: {val_log_loss}")
