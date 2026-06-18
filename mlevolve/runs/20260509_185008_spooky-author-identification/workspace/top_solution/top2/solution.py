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
GRADIENT_ACCUMULATION_STEPS = 8
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
TEMPERATURE = 0.1
CONTRASTIVE_WEIGHT = 0.3

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
    DeBERTa-v3-large with multi-task learning:
    1. Standard author classification (CrossEntropy)
    2. Supervised contrastive learning in embedding space
    """

    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_AUTHORS, dropout=DROPOUT):
        super().__init__()

        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        self.hidden_size = self.deberta.config.hidden_size

        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
        )

        self.classifier_dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_embeddings=False
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits
        logits = self.classifier_dropout(logits)

        result = {
            "logits": logits,
            "loss": outputs.loss,
            "cls_embedding": None,
            "projected_embedding": None,
        }

        return result


class MultiTaskLoss(nn.Module):
    """Simple CrossEntropy loss - no contrastive component to avoid NaN issues"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, labels, projected_embeddings=None):
        ce_loss_val = self.ce_loss(logits, labels)
        return ce_loss_val, ce_loss_val, torch.tensor(0.0, device=logits.device)


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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
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
# TRAINING LOOP
# ============================================================
print("\nStarting training...")
os.makedirs("./working", exist_ok=True)

best_val_loss = float("inf")
patience_counter = 0

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
                return_embeddings=False,
            )

            logits = outputs["logits"]

            loss, ce_loss_val, contrastive_loss_val = criterion(
                logits, labels, None
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
                    return_embeddings=False,
                )

                logits = outputs["logits"]

                val_batch_loss, _, _ = criterion(logits, labels, None)
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

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
best_model_path = "./working/best_model.pt"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
else:
    print("Warning: No best model checkpoint found, using current model state")
model.eval()

# Compute final validation score using the validation set
all_final_probs = []
all_final_labels = []

with torch.no_grad():
    for input_ids, attention_mask, labels in val_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeddings=False,
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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeddings=False,
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