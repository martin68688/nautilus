import os
os.sched_setaffinity(0, {32, 33, 34, 35, 36})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
SUBMISSION_PATH = "./submission/submission_903dd849b3884cceb11769f09e430a77.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-base"
HIDDEN_SIZE = 768
EMBEDDING_DIM = 256
MAX_LENGTH = 320
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
WARMUP_RATIO = 0.1
PATIENCE = 3
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

# ============================================================
# SELF-TRAINING DATA PIPELINE
# ============================================================
def generate_pseudo_labels(model, test_encodings, test_texts, batch_size=32, confidence_threshold=0.9):
    """Generate pseudo-labels on test set with confidence threshold."""
    model.eval()
    test_loader = DataLoader(
        TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.vstack(all_probs)
    max_probs = np.max(all_probs, axis=1)
    high_conf_mask = max_probs >= confidence_threshold
    pseudo_texts = test_texts[high_conf_mask]
    pseudo_probs = all_probs[high_conf_mask]
    pseudo_labels = np.argmax(pseudo_probs, axis=1)
    print(f"Pseudo-labels: {len(pseudo_labels)} samples above {confidence_threshold} confidence")
    return pseudo_texts, pseudo_labels, pseudo_probs

def augment_training_data(original_texts, original_labels, pseudo_texts, pseudo_labels, pseudo_probs):
    """Merge pseudo-labeled data into training set with soft labels."""
    augmented_texts = np.concatenate([original_texts, pseudo_texts])
    augmented_labels = np.concatenate([original_labels, pseudo_labels])
    # For soft labels, we store probabilities alongside
    soft_labels = np.zeros((len(augmented_labels), NUM_AUTHORS))
    # Original labels are one-hot
    original_onehot = np.eye(NUM_AUTHORS)[original_labels]
    soft_labels[:len(original_labels)] = original_onehot
    # Pseudo-labels are soft (probabilities)
    soft_labels[len(original_labels):] = pseudo_probs
    return augmented_texts, augmented_labels, soft_labels

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

# ============================================================
# CONFIDENCE-WEIGHTED LOSS WITH LABEL SMOOTHING
# ============================================================
class ConfidenceWeightedLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels, sample_weights=None):
        """Compute label-smoothed cross-entropy loss with optional sample weights.
        Args:
            logits: (batch, num_classes) raw logits
            labels: (batch,) hard labels OR (batch, num_classes) soft labels
            sample_weights: (batch,) weights for each sample, or None
        """
        num_classes = logits.size(1)
        # For hard labels (1D), convert to one-hot
        if labels.dim() == 1:
            with torch.no_grad():
                true_labels = labels.unsqueeze(1)
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (num_classes - 1))
                true_dist.scatter_(1, true_labels, 1.0 - self.smoothing)
        else:
            # Soft labels already provided
            true_dist = labels

        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(true_dist * log_probs, dim=1)

        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()

ce_loss_fn = nn.CrossEntropyLoss()
confidence_loss_fn = ConfidenceWeightedLoss(smoothing=0.2)

scaler = GradScaler() if torch.cuda.is_available() else None

# ============================================================
# HELPER FUNCTION FOR EMBEDDING EXTRACTION (works with AutoModelForSequenceClassification)
# ============================================================
def extract_embeddings_from_model(model, texts_encodings, batch_size=32):
    """Extract mean-pooled embeddings from the transformer encoder.
    Works with AutoModelForSequenceClassification by accessing model.deberta (the encoder).
    """
    model.eval()
    all_embeddings = []
    loader = DataLoader(
        TensorDataset(texts_encodings["input_ids"], texts_encodings["attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                # Access the base encoder (deberta) through the sequence classification model
                outputs = model.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden = outputs.last_hidden_state
                # Mean pooling over non-padded tokens
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

# ============================================================
# TRAINING LOOP (NO ONECYCLELR, use linear schedule)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING TRANSFORMER")
print("=" * 60)

# ============================================================
# HELPER: MERGE AND RE-TOKENIZE FOR SELF-TRAINING
# ============================================================
def re_tokenize_texts(texts, tokenizer, max_length=320):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings

def compute_log_loss(y_true, y_pred_proba):
    """Compute multi-class logarithmic loss."""
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
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

def multi_sample_dropout_forward(model, input_ids, attention_mask, num_samples=4):
    """Forward pass with multi-sample dropout: average logits over K dropout masks."""
    logits_list = []
    for _ in range(num_samples):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits_list.append(outputs.logits)
    return torch.stack(logits_list).mean(dim=0)

def compute_weighted_loss(model, batch, use_soft_labels=False, sample_weights=None):
    """Compute loss with optional multi-sample dropout and sample weighting."""
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)

    if use_soft_labels:
        # Multi-sample dropout for robustness
        logits = multi_sample_dropout_forward(model, input_ids, attention_mask)
        loss = confidence_loss_fn(logits, labels, sample_weights)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    return loss

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

# ============================================================
# ITERATIVE SELF-TRAINING LOOP (3 ROUNDS)
# ============================================================
print("\n" + "=" * 60)
print("SELF-TRAINING LOOP (PSEUDO-LABELING)")
print("=" * 60)

NUM_SELF_TRAINING_ROUNDS = 3
confidence_threshold = 0.9

# Keep track of the best ensemble found
best_ensemble_ll = float("inf")
best_ensemble_weights = None
best_round_models = {}

# Current training data starts as original
current_train_texts = X_train_texts.copy()
current_train_labels = y_train_labels.copy()
current_train_soft_labels = np.eye(NUM_AUTHORS)[y_train_labels]  # one-hot for original
current_train_sparse = train_sparse.copy()

for round_idx in range(NUM_SELF_TRAINING_ROUNDS):
    print(f"\n--- Self-training Round {round_idx + 1}/{NUM_SELF_TRAINING_ROUNDS} ---")
    print(f"Current training samples: {len(current_train_texts)}")

    # Re-tokenize the augmented training data
    current_train_encodings = tokenizer(
        list(current_train_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Use the ORIGINAL vectorizers fitted on initial training data only
    # to avoid leaking information from pseudo-labeled (test) data
    train_char_1_new = char_vectorizer_1.transform(current_train_texts)
    train_char_2_new = char_vectorizer_2.transform(current_train_texts)
    train_char_3_new = char_vectorizer_3.transform(current_train_texts)
    train_word_new = word_vectorizer.transform(current_train_texts)
    train_punct_new = punct_vectorizer.transform([extract_punctuation_sequence(str(t)) for t in current_train_texts])
    current_train_sparse_new = hstack([train_char_1_new, train_char_2_new, train_char_3_new, train_word_new, train_punct_new]).tocsr()

    # Re-train transformer from scratch on augmented data
    print("Re-training transformer from scratch on augmented data...")
    model_new = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model_new.to(device)

    optimizer_new = AdamW(model_new.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # OneCycleLR scheduler
    total_steps_new = len(train_loader) * 15  # max 15 epochs per round
    scheduler_new = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_new,
        max_lr=2e-5,
        total_steps=total_steps_new,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4,
    )

    scaler_new = GradScaler() if torch.cuda.is_available() else None

    # Create dataloaders for augmented data
    aug_train_dataset = TensorDataset(
        current_train_encodings["input_ids"],
        current_train_encodings["attention_mask"],
        torch.tensor(current_train_labels, dtype=torch.long),
    )
    # We also create a dataset with soft labels for weighted loss
    aug_train_loader = DataLoader(
        aug_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Training loop for this round
    best_val_loss_round = float("inf")
    patience_counter_round = 0

    for epoch in range(15):
        model_new.train()
        total_loss_round = 0.0
        num_batches_round = 0

        for batch in aug_train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer_new.zero_grad()

            with autocast():
                # Compute sample weights based on whether it's original or pseudo-labeled
                batch_size_curr = input_ids.size(0)
                # We need to know which samples are pseudo-labeled (those beyond original size)
                # Approximate: use confidence weights from soft labels
                batch_indices = list(range(num_batches_round * BATCH_SIZE,
                                          min((num_batches_round + 1) * BATCH_SIZE, len(current_train_labels))))
                batch_sample_weights = None
                if round_idx > 0:
                    # For pseudo-labeled samples, weight = confidence^2; for original, weight = 1.0
                    batch_sample_weights = torch.ones(batch_size_curr, device=device)
                    for idx_in_batch, global_idx in enumerate(batch_indices):
                        if global_idx >= len(y_train_labels):  # pseudo-labeled
                            max_conf = current_train_soft_labels[global_idx].max()
                            batch_sample_weights[idx_in_batch] = max_conf ** 2

                # Use multi-sample dropout for pseudo-labeled rounds
                if round_idx > 0:
                    logits = multi_sample_dropout_forward(model_new, input_ids, attention_mask)
                    # Use soft labels for pseudo-labeled data
                    soft_labels_batch = torch.tensor(
                        current_train_soft_labels[batch_indices],
                        device=device,
                        dtype=torch.float
                    )
                    loss = confidence_loss_fn(logits, soft_labels_batch, batch_sample_weights)
                else:
                    outputs = model_new(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

            if scaler_new is not None:
                scaler_new.scale(loss).backward()
                scaler_new.unscale_(optimizer_new)
                torch.nn.utils.clip_grad_norm_(model_new.parameters(), max_norm=1.0)
                scaler_new.step(optimizer_new)
                scaler_new.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_new.parameters(), max_norm=1.0)
                optimizer_new.step()

            scheduler_new.step()
            total_loss_round += loss.item()
            num_batches_round += 1

        avg_train_loss_round = total_loss_round / max(num_batches_round, 1)

        # Evaluate
        val_loss_round, val_acc_round, _ = evaluate(model_new, val_loader)
        print(f"  Epoch {epoch+1:2d}/15 | Train Loss: {avg_train_loss_round:.4f} | Val Loss: {val_loss_round:.4f} | Val Acc: {val_acc_round:.4f}")

        if val_loss_round < best_val_loss_round:
            best_val_loss_round = val_loss_round
            patience_counter_round = 0
            torch.save(model_new.state_dict(), f"{OUTPUT_DIR}/best_transformer_round{round_idx}.pt")
        else:
            patience_counter_round += 1
            if patience_counter_round >= 5:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model from this round
    model_new.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_transformer_round{round_idx}.pt", map_location=device))

    # Update ensemble with current transformer
    print("\nExtracting embeddings for ensemble...")
    def extract_embeddings_local(model, texts_encodings):
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

    # Get embeddings for current augmented training data
    current_train_embeddings = extract_embeddings_local(model_new, current_train_encodings)
    val_embeddings_new = extract_embeddings_local(model_new, val_encodings)
    test_embeddings_new = extract_embeddings_local(model_new, test_encodings)

    # Train XGBoost on current embeddings
    print("Training XGBoost...")
    xgb_model_new = xgb.XGBClassifier(
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
    xgb_model_new.fit(
        current_train_embeddings,
        current_train_labels,
        eval_set=[(val_embeddings_new, y_val_labels)],
        verbose=False,
    )

    # Train Logistic Regression on current TF-IDF
    print("Training Logistic Regression...")
    lr_model_new = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="saga",
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    # Transform validation and test with the (re-fitted) vectorizers
    val_sparse_new = hstack([
        char_vectorizer_1.transform(X_val_texts),
        char_vectorizer_2.transform(X_val_texts),
        char_vectorizer_3.transform(X_val_texts),
        word_vectorizer.transform(X_val_texts),
        punct_vectorizer.transform([extract_punctuation_sequence(str(t)) for t in X_val_texts])
    ]).tocsr()

    test_sparse_new = hstack([
        char_vectorizer_1.transform(test_df["text"].values),
        char_vectorizer_2.transform(test_df["text"].values),
        char_vectorizer_3.transform(test_df["text"].values),
        word_vectorizer.transform(test_df["text"].values),
        punct_vectorizer.transform([extract_punctuation_sequence(str(t)) for t in test_df["text"].values])
    ]).tocsr()

    lr_model_new.fit(current_train_sparse_new, current_train_labels)

    # Get predictions for ensemble
    _, _, val_probs_new = evaluate(model_new, val_loader)
    xgb_val_probs_new = xgb_model_new.predict_proba(val_embeddings_new)
    lr_val_probs_new = lr_model_new.predict_proba(val_sparse_new)

    # Optimize ensemble weights
    val_probas_new = {
        "transformer": val_probs_new,
        "xgboost": xgb_val_probs_new,
        "lr": lr_val_probs_new,
    }

    best_ll_new = float("inf")
    best_weights_new = None
    for w1 in np.arange(0.1, 0.9, 0.05):
        for w2 in np.arange(0.1, 0.9, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05 or w3 > 0.9:
                continue
            ensemble_proba_new = (
                w1 * val_probas_new["transformer"]
                + w2 * val_probas_new["xgboost"]
                + w3 * val_probas_new["lr"]
            )
            ll_new = compute_log_loss(y_val_labels, ensemble_proba_new)
            if ll_new < best_ll_new:
                best_ll_new = ll_new
                best_weights_new = {"transformer": w1, "xgboost": w2, "lr": w3}

    print(f"Round {round_idx + 1} ensemble validation log loss: {best_ll_new:.4f}")
    print(f"Ensemble weights: {best_weights_new}")

    if best_ll_new < best_ensemble_ll:
        best_ensemble_ll = best_ll_new
        best_ensemble_weights = best_weights_new
        best_round_models = {
            "transformer": model_new,
            "xgboost": xgb_model_new,
            "lr": lr_model_new,
            "char_vectorizer_1": char_vectorizer_1_new,
            "char_vectorizer_2": char_vectorizer_2_new,
            "char_vectorizer_3": char_vectorizer_3_new,
            "word_vectorizer": word_vectorizer_new,
            "punct_vectorizer": punct_vectorizer_new,
        }
        # Save best round models
        torch.save(model_new.state_dict(), f"{OUTPUT_DIR}/best_transformer_final.pt")

    # If not the last round, generate pseudo-labels for next round
    if round_idx < NUM_SELF_TRAINING_ROUNDS - 1:
        print("\nGenerating pseudo-labels for next round...")

        # Get test predictions from current ensemble
        model_new.eval()
        test_loader_new = DataLoader(
            TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        all_test_probs_new = []
        with torch.no_grad():
            for batch in test_loader_new:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                with autocast():
                    outputs = model_new(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                all_test_probs_new.append(probs.cpu().numpy())
        transformer_test_probs_new = np.vstack(all_test_probs_new)

        xgb_test_probs_new = xgb_model_new.predict_proba(test_embeddings_new)
        lr_test_probs_new = lr_model_new.predict_proba(test_sparse_new)

        ensemble_test_probs_new = (
            best_weights_new["transformer"] * transformer_test_probs_new
            + best_weights_new["xgboost"] * xgb_test_probs_new
            + best_weights_new["lr"] * lr_test_probs_new
        )

        # Adaptive threshold: decrease if too few samples
        max_probs_ensemble = np.max(ensemble_test_probs_new, axis=1)
        n_above = np.sum(max_probs_ensemble >= confidence_threshold)
        print(f"Samples above confidence threshold {confidence_threshold}: {n_above}")

        if n_above < 50 and confidence_threshold > 0.85:
            confidence_threshold -= 0.05
            print(f"Too few samples, lowering threshold to {confidence_threshold}")
            n_above = np.sum(max_probs_ensemble >= confidence_threshold)

        high_conf_mask_ensemble = max_probs_ensemble >= confidence_threshold
        pseudo_texts_new = test_df["text"].values[high_conf_mask_ensemble]
        pseudo_probs_new = ensemble_test_probs_new[high_conf_mask_ensemble]
        pseudo_labels_new = np.argmax(pseudo_probs_new, axis=1)

        print(f"Adding {len(pseudo_texts_new)} pseudo-labeled samples to training set")

        # Augment training data
        current_train_texts = np.concatenate([current_train_texts, pseudo_texts_new])
        current_train_labels = np.concatenate([current_train_labels, pseudo_labels_new])
        current_train_soft_labels = np.concatenate([current_train_soft_labels, pseudo_probs_new])

        # Rebuild TF-IDF for next round
        train_char_1_next = char_vectorizer_1_new.fit_transform(current_train_texts)
        train_char_2_next = char_vectorizer_2_new.fit_transform(current_train_texts)
        train_char_3_next = char_vectorizer_3_new.fit_transform(current_train_texts)
        train_word_next = word_vectorizer_new.fit_transform(current_train_texts)
        train_punct_next = punct_vectorizer_new.fit_transform(
            [extract_punctuation_sequence(str(t)) for t in current_train_texts]
        )
        current_train_sparse = hstack([train_char_1_next, train_char_2_next, train_char_3_next, train_word_next, train_punct_next]).tocsr()

        # Clean up to avoid memory build-up
        del model_new, xgb_model_new, lr_model_new
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================
# FINAL ENSEMBLE WITH BEST MODELS
# ============================================================
print("\n" + "=" * 60)
print("FINAL ENSEMBLE PREDICTION")
print("=" * 60)

# Load best transformer
print("\nLoading best transformer for final predictions...")
best_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
best_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_transformer_final.pt", map_location=device))
best_model.to(device)
best_model.eval()

# Get final validation predictions
_, _, val_probs_final = evaluate(best_model, val_loader)
print(f"Final Transformer validation log loss: {compute_log_loss(y_val_labels, val_probs_final):.4f}")

# Get test predictions from best transformer
final_test_loader = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
final_test_probs_list = []
with torch.no_grad():
    for batch in final_test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        final_test_probs_list.append(probs.cpu().numpy())
transformer_test_probs_final = np.vstack(final_test_probs_list)

# Get embeddings for final ensemble
train_encodings_final = re_tokenize_texts(current_train_texts, tokenizer, MAX_LENGTH)
train_embeddings_final = extract_embeddings_from_model(best_model, train_encodings_final, BATCH_SIZE)
val_embeddings_final = extract_embeddings_from_model(best_model, val_encodings, BATCH_SIZE)
test_embeddings_final = extract_embeddings_from_model(best_model, test_encodings, BATCH_SIZE)

# Train final XGBoost
print("Training final XGBoost...")
xgb_model_final = xgb.XGBClassifier(
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
xgb_model_final.fit(
    train_embeddings_final,
    current_train_labels,
    eval_set=[(val_embeddings_final, y_val_labels)],
    verbose=False,
)

xgb_val_probs_final = xgb_model_final.predict_proba(val_embeddings_final)
xgb_test_probs_final = xgb_model_final.predict_proba(test_embeddings_final)

# Train final Logistic Regression
print("Training final Logistic Regression...")
lr_model_final = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
lr_model_final.fit(current_train_sparse, current_train_labels)

# Transform validation and test with final vectorizers
final_val_char_1 = best_round_models["char_vectorizer_1"].transform(X_val_texts)
final_val_char_2 = best_round_models["char_vectorizer_2"].transform(X_val_texts)
final_val_char_3 = best_round_models["char_vectorizer_3"].transform(X_val_texts)
final_val_word = best_round_models["word_vectorizer"].transform(X_val_texts)
final_val_punct = best_round_models["punct_vectorizer"].transform(
    [extract_punctuation_sequence(str(t)) for t in X_val_texts]
)
final_val_sparse = hstack([final_val_char_1, final_val_char_2, final_val_char_3, final_val_word, final_val_punct]).tocsr()

final_test_char_1 = best_round_models["char_vectorizer_1"].transform(test_df["text"].values)
final_test_char_2 = best_round_models["char_vectorizer_2"].transform(test_df["text"].values)
final_test_char_3 = best_round_models["char_vectorizer_3"].transform(test_df["text"].values)
final_test_word = best_round_models["word_vectorizer"].transform(test_df["text"].values)
final_test_punct = best_round_models["punct_vectorizer"].transform(
    [extract_punctuation_sequence(str(t)) for t in test_df["text"].values]
)
final_test_sparse = hstack([final_test_char_1, final_test_char_2, final_test_char_3, final_test_word, final_test_punct]).tocsr()

lr_val_probs_final = lr_model_final.predict_proba(final_val_sparse)
lr_test_probs_final = lr_model_final.predict_proba(final_test_sparse)

# Final ensemble
print("\nComputing final ensemble...")
val_probas_final = {
    "transformer": val_probs_final,
    "xgboost": xgb_val_probs_final,
    "lr": lr_val_probs_final,
}

best_ll_final = float("inf")
best_weights_final = None
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba_final = (
            w1 * val_probas_final["transformer"]
            + w2 * val_probas_final["xgboost"]
            + w3 * val_probas_final["lr"]
        )
        ll_final = compute_log_loss(y_val_labels, ensemble_proba_final)
        if ll_final < best_ll_final:
            best_ll_final = ll_final
            best_weights_final = {"transformer": w1, "xgboost": w2, "lr": w3}

print(f"Final optimized ensemble weights: {best_weights_final}")
print(f"Final ensemble validation log loss: {best_ll_final:.4f}")

# Generate final submission
test_probas_final = {
    "transformer": transformer_test_probs_final,
    "xgboost": xgb_test_probs_final,
    "lr": lr_test_probs_final,
}
ensemble_test_probs_final = (
    best_weights_final["transformer"] * test_probas_final["transformer"]
    + best_weights_final["xgboost"] * test_probas_final["xgboost"]
    + best_weights_final["lr"] * test_probas_final["lr"]
)

eps = 1e-15
ensemble_test_probs_final = np.clip(ensemble_test_probs_final, eps, 1 - eps)
row_sums = ensemble_test_probs_final.sum(axis=1, keepdims=True)
ensemble_test_probs_final = ensemble_test_probs_final / row_sums
ensemble_test_probs_final = np.clip(ensemble_test_probs_final, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs_final[:, 0],
        "HPL": ensemble_test_probs_final[:, 1],
        "MWS": ensemble_test_probs_final[:, 2],
    }
)

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nFinal Validation Score: {best_ll_final:.6f}")