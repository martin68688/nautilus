"""
Merged Script: Spooky Author Identification
Combines data processing, feature engineering, bi-encoder model design, and training/evaluation.
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & CONFIGURATION
# ============================================================
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "./working"
SUBMISSION_PATH = "./submission/submission.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
HIDDEN_SIZE = 1024
EMBEDDING_DIM = 256
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2
NUM_EXPERTS = 6
TOP_K_EXPERTS = 2
TEMPERATURE = 0.05

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# FEATURE EXTRACTION FUNCTIONS (Simplified: TF-IDF + Punctuation only)
# ============================================================
def extract_punctuation_sequence(text):
    """Extract punctuation characters as a string"""
    return ''.join([c for c in str(text) if c in string.punctuation])

# No hand-crafted features - using only transformer embeddings and simple TF-IDF

# ============================================================
# CLASS-BALANCED LOSS
# ============================================================
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.999, num_classes=3):
        super().__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, labels):
        """Compute class-balanced cross-entropy loss.
        Uses effective number of samples per class for re-weighting.
        """
        with torch.no_grad():
            # Count samples per class in the current batch
            class_counts = torch.zeros(self.num_classes, device=logits.device)
            for c in range(self.num_classes):
                class_counts[c] = (labels == c).sum().float()
            # Effective number of samples per class
            effective_num = 1.0 - self.beta ** class_counts
            # Avoid division by zero: if class count is 0, effective_num is 0
            weights = (1.0 - self.beta) / (effective_num + 1e-8)
            # Normalize weights so they sum to num_classes
            weights = weights / weights.sum() * self.num_classes
        # Standard cross-entropy
        loss = F.cross_entropy(logits, labels, weight=weights)
        return loss

# ============================================================
# MODEL DEFINITION: Standard AutoModelForSequenceClassification with multi-sample dropout
# ============================================================
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

# No custom wrapper - use standard AutoModelForSequenceClassification directly

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT
# ============================================================
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)

# ============================================================
# SIMPLIFIED N-GRAM FEATURES (Pure TF-IDF + Punctuation, no hand-crafted)
# ============================================================
print("Extracting n-gram features (simplified)...")

# Character n-grams: (2-4), (3-5), (4-6) with max_features=2000 each
char_vectorizer_1 = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_1 = char_vectorizer_1.fit_transform(X_train_texts)
val_char_1 = char_vectorizer_1.transform(X_val_texts)
test_char_1 = char_vectorizer_1.transform(test_df["text"].values)

char_vectorizer_2 = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_2 = char_vectorizer_2.fit_transform(X_train_texts)
val_char_2 = char_vectorizer_2.transform(X_val_texts)
test_char_2 = char_vectorizer_2.transform(test_df["text"].values)

char_vectorizer_3 = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_3 = char_vectorizer_3.fit_transform(X_train_texts)
val_char_3 = char_vectorizer_3.transform(X_val_texts)
test_char_3 = char_vectorizer_3.transform(test_df["text"].values)

# Word n-grams: (1-3) with max_features=4000
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=4000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vectorizer.fit_transform(X_train_texts)
val_word = word_vectorizer.transform(X_val_texts)
test_word = word_vectorizer.transform(test_df["text"].values)

# Punctuation n-grams: (2-4) with CountVectorizer - fit ONLY on training data
train_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_train_texts]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
train_punct = punct_vectorizer.fit_transform(train_punct_sequences)

val_punct_sequences = [extract_punctuation_sequence(str(t)) for t in X_val_texts]
val_punct = punct_vectorizer.transform(val_punct_sequences)

test_punct_sequences = [extract_punctuation_sequence(str(t)) for t in test_df["text"].values]
test_punct = punct_vectorizer.transform(test_punct_sequences)

train_sparse = hstack(
    [train_char_1, train_char_2, train_char_3, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_1, val_char_2, val_char_3, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_1, test_char_2, test_char_3, test_word, test_punct]
).tocsr()
print(f"Sparse train shape: {train_sparse.shape}")

# ============================================================
# TOKENIZATION
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    list(X_train_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    list(X_val_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_encodings = tokenizer(
    list(test_df["text"].values),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(y_train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(y_val_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
# ============================================================
# STANDARD TRANSFORMER MODEL WITH MUTLI-SAMPLE DROPOUT AND ONECYCLELR
# ============================================================
print("\nSetting up standard AutoModelForSequenceClassification with multi-sample dropout...")

# Re-load model to set up with standard configuration
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
model.to(device)

# Standard AdamW optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Linear schedule with warmup
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# Standard cross-entropy loss for classification
ce_loss_fn = nn.CrossEntropyLoss()

scaler = GradScaler() if torch.cuda.is_available() else None

# ============================================================
# TRAINING LOOP WITH ONECYCLELR (NO SWA)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING TRANSFORMER WITH ONECYCLELR")
print("=" * 60)

def compute_log_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    y_true_onehot = np.eye(NUM_AUTHORS)[y_true]
    loss = -np.sum(y_true_onehot * np.log(y_pred_proba)) / n
    return loss

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches

    # Evaluate with current weights
    val_loss, val_acc, val_probs = evaluate(model, val_loader)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
    current_val_loss = val_loss

    # Early stopping on validation loss
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        # Save the model
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_transformer_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest Transformer model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")

# Load best model
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_transformer_model.pt", map_location=device))

# ============================================================
# GET TRANSFORMER PREDICTIONS
# ============================================================
print("\nGetting Transformer probabilities...")
_, _, val_probs = evaluate(model, val_loader)
print(f"Transformer validation log loss: {compute_log_loss(y_val_labels, val_probs):.4f}")

model.eval()
all_test_probs = []
test_loader = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())
transformer_test_probs = np.vstack(all_test_probs)

# ============================================================
# ENSEMBLE: TRANSFORMER + XGBOOST + LOGISTIC REGRESSION
# ============================================================
print("\nExtracting embeddings for ensemble...")
def extract_embeddings(model, texts_encodings):
    model.eval()
    all_embeddings = []
    loader = DataLoader(
        TensorDataset(texts_encodings["input_ids"], texts_encodings["attention_mask"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

train_embeddings = extract_embeddings(model, train_encodings)
val_embeddings = extract_embeddings(model, val_encodings)
test_embeddings = extract_embeddings(model, test_encodings)
print(f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")

# XGBoost ensemble (using only embeddings, no hand-crafted features)
print("\nTraining XGBoost classifier...")
xgb_train_features = train_embeddings
xgb_val_features = val_embeddings
xgb_test_features = test_embeddings
print(f"XGBoost train features: {xgb_train_features.shape}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    xgb_train_features,
    y_train_labels,
    eval_set=[(xgb_val_features, y_val_labels)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
print(f"XGBoost validation log loss: {compute_log_loss(y_val_labels, xgb_val_probs):.4f}")

# Logistic Regression ensemble
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
lr_model.fit(train_sparse, y_train_labels)

lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)
print(f"Logistic Regression validation log loss: {compute_log_loss(y_val_labels, lr_val_probs):.4f}")

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "transformer": val_probs,
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
}

best_ll = float("inf")
best_weights = None
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = (
            w1 * val_probas["transformer"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(y_val_labels, ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"transformer": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
test_probas = {
    "transformer": transformer_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
ensemble_test_probs = (
    best_weights["transformer"] * test_probas["transformer"]
    + best_weights["xgboost"] * test_probas["xgboost"]
    + best_weights["lr"] * test_probas["lr"]
)

eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nFinal Validation Score: {best_ll:.6f}")