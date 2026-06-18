import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
import scipy.sparse as sparse
import gc

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_texts = train_df["text"].values
train_authors = train_df["author"].values
train_ids = train_df["id"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Encode author labels
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = np.array([author_mapping[a] for a in train_authors])
num_authors = len(author_mapping)

print(f"Training samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")
print(f"Class distribution: {Counter(train_authors)}")

# ============================================================
# 2. STRATIFIED SPLIT (80-20 train-val)
# ============================================================
X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
    train_texts,
    train_labels,
    np.arange(len(train_texts)),
    test_size=0.2,
    stratify=train_labels,
    random_state=42,
)

print(f"Train set: {len(X_train)}, Val set: {len(X_val)}")
print(f"Train distribution: {np.bincount(y_train)}")
print(f"Val distribution: {np.bincount(y_val)}")


# ============================================================
# 3. FEATURE ENGINEERING - Stylometric Features
# ============================================================
def extract_stylometric_features(texts):
    features = []
    for text in texts:
        char_count = len(text)
        word_count = len(text.split())
        sent_count = len(re.split(r"[.!?]+", text))
        avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0
        comma_count = text.count(",") / max(char_count, 1)
        period_count = text.count(".") / max(char_count, 1)
        excl_count = text.count("!") / max(char_count, 1)
        quest_count = text.count("?") / max(char_count, 1)
        semi_count = text.count(";") / max(char_count, 1)
        colon_count = text.count(":") / max(char_count, 1)
        quote_count = (text.count('"') + text.count("'")) / max(char_count, 1)
        dash_count = text.count("-") / max(char_count, 1)
        paren_count = (text.count("(") + text.count(")")) / max(char_count, 1)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        lowercase_ratio = sum(1 for c in text if c.islower()) / max(char_count, 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
        space_ratio = sum(1 for c in text if c.isspace()) / max(char_count, 1)
        punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(
            char_count, 1
        )
        unique_words = len(set(text.lower().split()))
        ttr = unique_words / max(word_count, 1)
        word_counter = Counter(text.lower().split())
        hapax_count = sum(1 for v in word_counter.values() if v == 1)
        hapax_ratio = hapax_count / max(word_count, 1)
        function_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "of",
            "to",
            "for",
            "with",
            "by",
            "at",
            "from",
            "that",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "it",
            "he",
            "she",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "not",
            "no",
            "never",
            "none",
            "nothing",
            "all",
            "each",
            "every",
            "some",
            "any",
            "both",
            "many",
            "much",
            "few",
            "more",
            "most",
            "such",
            "only",
            "very",
            "too",
            "so",
            "as",
            "than",
            "then",
            "now",
            "then",
            "here",
            "there",
            "like",
            "just",
            "also",
            "always",
            "never",
            "often",
            "sometimes",
            "still",
            "yet",
            "already",
            "about",
            "above",
            "after",
            "again",
            "against",
            "among",
            "before",
            "behind",
            "below",
            "between",
            "beyond",
            "during",
            "except",
            "inside",
            "into",
            "near",
            "outside",
            "over",
            "through",
            "under",
            "upon",
            "within",
            "without",
        ]
        words_lower = text.lower().split()
        fw_counts = {}
        for fw in function_words:
            fw_counts[f"fw_{fw}"] = words_lower.count(fw) / max(word_count, 1)
        feat = [
            char_count,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            comma_count,
            period_count,
            excl_count,
            quest_count,
            semi_count,
            colon_count,
            quote_count,
            dash_count,
            paren_count,
            uppercase_ratio,
            lowercase_ratio,
            digit_ratio,
            space_ratio,
            punct_ratio,
            ttr,
            hapax_ratio,
        ]
        feat.extend(fw_counts.values())
        features.append(feat)
    return np.array(features)


train_stylo = extract_stylometric_features(X_train)
val_stylo = extract_stylometric_features(X_val)
test_stylo = extract_stylometric_features(test_texts)

print(f"Stylometric features: {train_stylo.shape[1]} per sample")

# ============================================================
# 4. CHARACTER N-GRAM FEATURES
# ============================================================
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True,
)
char_features_train = char_vectorizer.fit_transform(X_train)
char_features_val = char_vectorizer.transform(X_val)
char_features_test = char_vectorizer.transform(test_texts)
print(f"Character n-gram features: {char_features_train.shape[1]}")

# ============================================================
# 5. WORD N-GRAM FEATURES
# ============================================================
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=10000,
    sublinear_tf=True,
    min_df=2,
)
word_features_train = word_vectorizer.fit_transform(X_train)
word_features_val = word_vectorizer.transform(X_val)
word_features_test = word_vectorizer.transform(test_texts)
print(f"Word n-gram features: {word_features_train.shape[1]}")

# ============================================================
# 6. SENTENCE-BERT EMBEDDINGS
# ============================================================
sbert_model = SentenceTransformer("all-mpnet-base-v2")
sbert_model.eval()

train_sbert = sbert_model.encode(
    X_train.tolist(), convert_to_tensor=False, show_progress_bar=False
)
val_sbert = sbert_model.encode(
    X_val.tolist(), convert_to_tensor=False, show_progress_bar=False
)
test_sbert = sbert_model.encode(
    test_texts.tolist(), convert_to_tensor=False, show_progress_bar=False
)
print(f"Sentence-BERT embeddings: {train_sbert.shape[1]} dims")

# ============================================================
# 7. COMBINE FEATURES (stored but we'll use raw text for DeBERTa)
# ============================================================
stylo_train_sparse = csr_matrix(train_stylo)
stylo_val_sparse = csr_matrix(val_stylo)
stylo_test_sparse = csr_matrix(test_stylo)
sbert_train_sparse = csr_matrix(train_sbert)
sbert_val_sparse = csr_matrix(val_sbert)
sbert_test_sparse = csr_matrix(test_sbert)

X_train_combined = hstack(
    [stylo_train_sparse, char_features_train, word_features_train, sbert_train_sparse]
)
X_val_combined = hstack(
    [stylo_val_sparse, char_features_val, word_features_val, sbert_val_sparse]
)
X_test_combined = hstack(
    [stylo_test_sparse, char_features_test, word_features_test, sbert_test_sparse]
)

print(f"Combined train features: {X_train_combined.shape}")
print(f"Combined val features: {X_val_combined.shape}")
print(f"Combined test features: {X_test_combined.shape}")

# ============================================================
# 8. SAVE PROCESSED DATA
# ============================================================
save_dir = "./working"
os.makedirs(save_dir, exist_ok=True)

# Only save minimal essential data (no heavy sparse matrices or large CSV files)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "y_val.npy"), y_val)

print(f"\nData processing complete!")

# ============================================================
# 9. MODEL DESIGN - DeBERTa-v3-small (memory-efficient)
# ============================================================
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_authors,
    ignore_mismatched_sizes=True,
)
# Gradient checkpointing disabled to avoid compatibility issues with autocast/GradScaler
# We'll manage memory with batch size instead
# Increase dropout for better regularization
model.config.hidden_dropout_prob = 0.3
model.config.attention_probs_dropout_prob = 0.3


# ============================================================
# 10. FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class_counts = np.bincount(y_train)
class_weights = torch.tensor(
    [1.0 / count for count in class_counts], dtype=torch.float32
)
class_weights = class_weights / class_weights.sum() * len(class_counts)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class_weights = class_weights.to(device)
criterion = FocalLoss(gamma=2.0, alpha=class_weights)

# ============================================================
# 11. TOKENIZE ALL DATA
# ============================================================
print("Tokenizing data...")
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
val_encodings = tokenizer(
    X_val.tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)

train_labels_tensor = torch.tensor(y_train, dtype=torch.long)
val_labels_tensor = torch.tensor(y_val, dtype=torch.long)

# ============================================================
# 12. CREATE DATALOADERS
# ============================================================
batch_size = 4

train_dataset = TensorDataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], train_labels_tensor
)
val_dataset = TensorDataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], val_labels_tensor
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

# Use num_workers=0 to avoid shared memory crashing due to disk space constraints
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ============================================================
# 13. OPTIMIZER, SCHEDULER, MIXED PRECISION
# ============================================================
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
)

epochs = 15
gradient_accumulation_steps = 1
total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

# ============================================================
# 14. TRAINING LOOP WITH EARLY STOPPING & MEMORY MANAGEMENT
# ============================================================
best_val_logloss = float("inf")
best_val_accuracy = 0.0
patience = 5
patience_counter = 0
best_model_state = None

print(f"\nStarting training for {epochs} epochs...")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_train_loss += loss.item()

        # Only step optimizer and scheduler at the end of gradient accumulation
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Free memory periodically
        if step % 50 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / len(train_loader)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    all_val_logits = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_val_logits.append(outputs.logits.cpu())
            all_val_labels.append(labels.cpu())

    val_logits = torch.cat(all_val_logits, dim=0)
    val_labels = torch.cat(all_val_labels, dim=0)
    val_probs = F.softmax(val_logits, dim=1).numpy()
    val_labels_np = val_labels.numpy()

    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    log_loss = 0.0
    n = len(val_labels_np)
    for i in range(n):
        for j in range(num_authors):
            y_ij = 1.0 if val_labels_np[i] == j else 0.0
            log_loss += y_ij * np.log(val_probs_clipped[i, j])
    log_loss = -log_loss / n

    val_preds = np.argmax(val_probs_clipped, axis=1)
    accuracy = (val_preds == val_labels_np).mean()

    print(
        f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {log_loss:.6f} | Val Acc: {accuracy:.4f}"
    )

    if log_loss < best_val_logloss:
        best_val_logloss = log_loss
        best_val_accuracy = accuracy
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        # Free memory from old best state if it exists
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  -> New best model! Val LogLoss: {best_val_logloss:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# ============================================================
# 15. FINAL VALIDATION SCORE
# ============================================================
print("\nLoading best model for final validation...")
model.load_state_dict(best_model_state)
if device.type == "cuda":
    torch.cuda.empty_cache()
model.to(device)
model.eval()

all_val_logits = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_val_logits.append(outputs.logits.cpu())

val_logits = torch.cat(all_val_logits, dim=0)
val_probs = F.softmax(val_logits, dim=1).numpy()
eps = 1e-15
val_probs = np.clip(val_probs, eps, 1 - eps)
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

log_loss = 0.0
n = len(y_val)
for i in range(n):
    for j in range(num_authors):
        y_ij = 1.0 if y_val[i] == j else 0.0
        log_loss += y_ij * np.log(val_probs[i, j])
final_val_logloss = -log_loss / n

print(f"\nBest validation log loss: {best_val_logloss:.6f}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")

# Free memory before test inference
if device.type == "cuda":
    torch.cuda.empty_cache()

# ============================================================
# 16. TEST INFERENCE
# ============================================================
print("\nGenerating test predictions...")
all_test_logits = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_test_logits.append(outputs.logits.cpu())

test_logits = torch.cat(all_test_logits, dim=0)
test_probs = F.softmax(test_logits, dim=1).numpy()
test_probs = np.clip(test_probs, eps, 1 - eps)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# ============================================================
# 17. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# ============================================================
# 18. FINAL OUTPUT
# ============================================================
print(f"Final Validation Score: {best_val_logloss}")
