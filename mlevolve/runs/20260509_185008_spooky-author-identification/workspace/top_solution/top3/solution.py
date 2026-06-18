"""
Merged script: Data Processing + Model Design + Training & Evaluation
For Halloween Spooky Author Identification competition.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
import torch.optim.swa_utils as swa_utils
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# SEED AND CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "microsoft/deberta-v3-base"
NUM_AUTHORS = 3
MAX_LENGTH = 256
DROPOUT = 0.1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 12
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
TEMPERATURE = 0.1
CONTRASTIVE_WEIGHT = 0.3

# SWA configuration
SWA_START_EPOCH = 5  # start averaging from this epoch
SWA_LR_FACTOR = 0.5  # SWA learning rate factor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# DATA LOADING
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Ensure text column is string
train_df["text"] = train_df["text"].astype(str)
test_df["text"] = test_df["text"].astype(str)
test_ids = test_df["id"].values

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(train_df["author"].values)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# ============================================================
# STRATIFIED SPLIT
# ============================================================
X_temp_texts = train_df["text"].values
y_temp = labels

train_indices, temp_indices = train_test_split(
    np.arange(len(X_temp_texts)),
    test_size=0.2,
    stratify=y_temp,
    random_state=SEED,
)

temp_labels = y_temp[temp_indices]
val_indices, test_indices = train_test_split(
    np.arange(len(temp_indices)),
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED,
)

val_indices = temp_indices[val_indices]
test_indices = temp_indices[test_indices]

train_texts = train_df["text"].values[train_indices]
val_texts = train_df["text"].values[val_indices]
test_texts = train_df["text"].values[test_indices]

train_labels = labels[train_indices]
val_labels = labels[val_indices]
test_labels = labels[test_indices]

print(
    f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
)
print("Train distribution:", np.bincount(train_labels))
print("Val distribution:", np.bincount(val_labels))
print("Test distribution:", np.bincount(test_labels))

# ============================================================
# TOKENIZER SETUP
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_LENGTH
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"


# ============================================================
# MODEL DEFINITION (DeBERTa with multi-task learning)
# ============================================================
class DebertaForAuthorshipAttribution(nn.Module):
    """
    DeBERTa-v3-base with standard classification head (no contrastive projection)
    """

    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_AUTHORS, dropout=DROPOUT):
        super().__init__()

        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        self.classifier_dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        logits = outputs.logits
        logits = self.classifier_dropout(logits)

        result = {
            "logits": logits,
            "loss": outputs.loss,
        }

        return result


class MultiTaskLoss(nn.Module):
    """CrossEntropy loss with label smoothing (0.1)"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, labels):
        ce_loss_val = self.ce_loss(logits, labels)
        return ce_loss_val


# ============================================================
# TOKENIZE DATA
# ============================================================
print("Tokenizing data...")
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
full_test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

# ============================================================
# WORD-LEVEL PERTURBATION FUNCTION
# ============================================================
def word_level_perturbation(input_ids, tokenizer, swap_prob=0.1, delete_prob=0.05):
    """
    Apply mild word-level perturbations to input token sequences:
    - Random swap of two adjacent words with swap_prob probability
    - Random deletion of a word with delete_prob probability
    Operates on token ids without decoding to avoid slowdown.
    """
    batch_size, seq_len = input_ids.shape
    perturbed_ids = input_ids.clone()

    for i in range(batch_size):
        # Get the actual sequence length (excluding padding)
        seq = perturbed_ids[i].tolist()
        # Find actual tokens (ignore padding and special tokens)
        special_tokens = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id}
        # We'll work on a per-token basis
        # Determine which positions are real words (not padding/special)
        # For simplicity, apply to all non-pad tokens between cls and sep

        # Find positions of CLS and SEP
        try:
            cls_pos = seq.index(tokenizer.cls_token_id)
        except ValueError:
            cls_pos = 0
        try:
            # Find last occurrence of sep token
            sep_pos = len(seq) - 1 - seq[::-1].index(tokenizer.sep_token_id)
        except ValueError:
            sep_pos = len(seq) - 1

        # Only perturb content tokens between CLS and SEP
        content_positions = list(range(cls_pos + 1, sep_pos))

        if len(content_positions) < 2:
            continue

        # Random swap: swap two adjacent content tokens
        if np.random.random() < swap_prob and len(content_positions) >= 2:
            swap_idx = np.random.randint(0, len(content_positions) - 1)
            pos1 = content_positions[swap_idx]
            pos2 = content_positions[swap_idx + 1]
            # Only swap if neither is a special token
            if (seq[pos1] not in special_tokens and
                seq[pos2] not in special_tokens):
                perturbed_ids[i, pos1], perturbed_ids[i, pos2] = (
                    perturbed_ids[i, pos2],
                    perturbed_ids[i, pos1],
                )

        # Random deletion: set a content token to a mask token or just shift
        if np.random.random() < delete_prob and len(content_positions) >= 1:
            del_idx = np.random.randint(0, len(content_positions))
            pos = content_positions[del_idx]
            if seq[pos] not in special_tokens:
                # Replace with pad token (effectively delete)
                perturbed_ids[i, pos] = tokenizer.pad_token_id

    return perturbed_ids


def get_augmented_collate_fn(tokenizer, swap_prob=0.1, delete_prob=0.05):
    """Returns a collate function that applies word-level perturbations."""
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])

        # Apply perturbations during training
        perturbed_input_ids = word_level_perturbation(
            input_ids, tokenizer, swap_prob=swap_prob, delete_prob=delete_prob
        )

        return perturbed_input_ids, attention_mask, labels
    return collate_fn


# ============================================================
# DATASETS AND DATALOADERS
# ============================================================
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long),
)
full_test_dataset = TensorDataset(
    full_test_encodings["input_ids"],
    full_test_encodings["attention_mask"],
)

# Use augmented collate for training
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
    collate_fn=get_augmented_collate_fn(tokenizer, swap_prob=0.1, delete_prob=0.05),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
full_test_loader = DataLoader(
    full_test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("Initializing model...")
model = DebertaForAuthorshipAttribution(
    model_name=MODEL_NAME, num_labels=num_classes, dropout=DROPOUT
)
model.to(device)

criterion = MultiTaskLoss()

# ============================================================
# OPTIMIZER AND SCHEDULER
# ============================================================
no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.deberta.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            p
            for n, p in model.deberta.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        "lr": LEARNING_RATE,
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * MAX_EPOCHS // GRADIENT_ACCUMULATION_STEPS
num_warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)

scaler = GradScaler()

# ============================================================
# TRAINING LOOP (with SWA support)
# ============================================================
print("\nStarting training...")
os.makedirs("./working", exist_ok=True)

best_val_loss = float("inf")
patience_counter = 0

# SWA setup
swa_model = None
swa_scheduler = None

for epoch in range(MAX_EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits, labels
            )
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_train_loss / len(train_loader)

    # Update SWA model after SWA_START_EPOCH
    if epoch + 1 >= SWA_START_EPOCH:
        if swa_model is None:
            swa_model = swa_utils.AveragedModel(model)
            swa_scheduler = swa_utils.SWALR(
                optimizer,
                anneal_strategy="cos",
                anneal_epochs=1,
                swa_lr=LEARNING_RATE * SWA_LR_FACTOR
            )
        else:
            swa_model.update_parameters(model)

    # Validation phase
    model.eval()
    val_loss = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                logits = outputs["logits"]

                val_batch_loss = criterion(logits, labels)
                val_loss += val_batch_loss.item()

                probs = torch.softmax(logits, dim=1)
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    # Compute log loss
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(
        axis=1, keepdims=True
    )
    val_log_loss = -np.mean(
        np.sum(np.eye(num_classes)[val_true] * np.log(val_probs_normalized), axis=1)
    )

    print(
        f"Epoch {epoch + 1}/{MAX_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
    )

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Step SWA scheduler if active
    if swa_scheduler is not None and epoch + 1 >= SWA_START_EPOCH:
        swa_scheduler.step()

# After training, optionally apply SWA averaging
if swa_model is not None:
    print("Applying SWA averaging...")
    swa_utils.update_bn(train_loader, swa_model, device=device)
    # Save SWA model separately
    torch.save(swa_model.state_dict(), "./working/swa_model.pt")
    print("SWA model saved.")

# ============================================================
# FINAL VALIDATION SCORE (use SWA model if available, else best model)
# ============================================================
swa_model_path = "./working/swa_model.pt"
best_model_path = "./working/best_model.pt"

# Try loading SWA model first (should be better due to averaging)
if os.path.exists(swa_model_path):
    print("Loading SWA model for inference...")
    # Recreate a fresh model and load SWA weights
    model_for_inference = DebertaForAuthorshipAttribution(
        model_name=MODEL_NAME, num_labels=num_classes, dropout=DROPOUT
    )
    # SWA model needs to be wrapped in AveragedModel to load correctly
    swa_model_loaded = swa_utils.AveragedModel(model_for_inference)
    swa_model_loaded.load_state_dict(torch.load(swa_model_path))
    model_for_inference = swa_model_loaded.module
    model_for_inference.to(device)
    model_for_inference.eval()
elif os.path.exists(best_model_path):
    print("Loading best standard model for inference...")
    model_for_inference = DebertaForAuthorshipAttribution(
        model_name=MODEL_NAME, num_labels=num_classes, dropout=DROPOUT
    )
    model_for_inference.load_state_dict(torch.load(best_model_path))
    model_for_inference.to(device)
    model_for_inference.eval()
else:
    print("Warning: No checkpoint found, using current model state")
    model_for_inference = model
    model_for_inference.eval()

# Compute final validation score using the validation set
all_final_probs = []
all_final_labels = []

with torch.no_grad():
    for input_ids, attention_mask, labels in val_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model_for_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            all_final_probs.append(probs.cpu().numpy())
            all_final_labels.append(labels.cpu().numpy())

val_probs_final = np.concatenate(all_final_probs, axis=0)
val_labels_final = np.concatenate(all_final_labels, axis=0)

val_probs_clipped = np.clip(val_probs_final, 1e-15, 1 - 1e-15)
val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_score = -np.mean(
    np.sum(np.eye(num_classes)[val_labels_final] * np.log(val_probs_normalized), axis=1)
)

# ============================================================
# INFERENCE ON FULL TEST SET
# ============================================================
print("\nPerforming inference on test set...")
all_test_probs = []

with torch.no_grad():
    for input_ids, attention_mask in full_test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with autocast():
            outputs = model_for_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

# Clip and normalize probabilities
test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs_normalized = test_probs_clipped / test_probs_clipped.sum(
    axis=1, keepdims=True
)

# ============================================================
# GENERATE SUBMISSION FILE
# ============================================================
print("Generating submission file...")
# Ensure correct column order: the sample_submission.csv has columns: id,EAP,HPL,MWS
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_normalized[:, 0],
        "HPL": test_probs_normalized[:, 1],
        "MWS": test_probs_normalized[:, 2],
    }
)

submission_df = submission_df[["id", "EAP", "HPL", "MWS"]]

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# ============================================================
# FINAL OUTPUT
# ============================================================
print(f"Final Validation Score: {final_val_score}")