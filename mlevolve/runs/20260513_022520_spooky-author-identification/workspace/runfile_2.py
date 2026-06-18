import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9, 11, 12, 13, 14, 15})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import os
import gc
import warnings

warnings.filterwarnings("ignore")

# ============ CONFIGURATION ============
MAX_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS_FROZEN = 3
EPOCHS_FULL = 6
LEARNING_RATE = 2e-5
FROZEN_LR = 5e-5
NUM_LABELS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============ LOAD DATA ============
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Stratified train/val split
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df, train_df["author"]))

train_texts = train_df.iloc[train_idx]["text"].values
train_labels = train_df.iloc[train_idx]["author"].values
val_texts = train_df.iloc[val_idx]["text"].values
val_labels = train_df.iloc[val_idx]["author"].values
test_texts = test_df["text"].values

# Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
val_labels_enc = le.transform(val_labels)
num_authors = len(le.classes_)

print(
    f"Train: {len(train_labels_enc)}, Val: {len(val_labels_enc)}, Test: {len(test_texts)}"
)
print(f"Authors: {le.classes_}")

# ============ TOKENIZE ============
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)

# ============ DATALOADERS ============
class_counts = np.bincount(train_labels_enc)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels_enc]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels_enc),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels_enc),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============ MODEL ============
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_authors,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ============ LOSS & OPTIMIZER ============
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.deberta.named_parameters()],
        "lr": LEARNING_RATE,
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.classifier.named_parameters()],
        "lr": LEARNING_RATE * 5,
        "weight_decay": 0.01,
    },
]


# ============ TRAINING ============
def compute_log_loss(y_true, y_pred_probs):
    eps = 1e-15
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=True)
    n = len(y_true)
    loss = 0.0
    for i in range(n):
        for j in range(num_authors):
            if y_true[i] == j:
                loss += -np.log(y_pred_probs[i, j])
    return loss / n


best_val_loss = float("inf")
patience = 4
no_improve_count = 0

# Phase 1: Freeze backbone, train only classifier
print("Phase 1: Training classifier with frozen backbone")
for param in model.deberta.parameters():
    param.requires_grad = False

optimizer_phase1 = torch.optim.AdamW(
    model.classifier.parameters(), lr=FROZEN_LR, weight_decay=0.1
)
scaler_phase1 = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS_FROZEN):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(model.device) for b in batch]

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        scaler_phase1.scale(loss).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler_phase1.step(optimizer_phase1)
            scaler_phase1.update()
            optimizer_phase1.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches

    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(model.device) for b in batch]
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs.loss.item()

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_logloss = compute_log_loss(val_targets, val_preds)

    print(
        f"Epoch {epoch+1}/{EPOCHS_FROZEN} - Train Loss: {avg_train_loss:.4f} - Val Log Loss: {val_logloss:.4f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        torch.save(model.state_dict(), "./working/best_model_phase1.pt")
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Phase 2: Unfreeze and fine-tune entire model
print("\nPhase 2: Fine-tuning entire model with discriminative learning rates")
for param in model.deberta.parameters():
    param.requires_grad = True

model.load_state_dict(torch.load("./working/best_model_phase1.pt"))

optimizer = torch.optim.AdamW(
    [
        {
            "params": [p for n, p in model.deberta.named_parameters()],
            "lr": LEARNING_RATE,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": LEARNING_RATE * 3,
            "weight_decay": 0.01,
        },
    ],
    lr=LEARNING_RATE,
)

total_steps = len(train_loader) * EPOCHS_FULL // GRAD_ACCUM_STEPS
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler()
no_improve_count = 0

for epoch in range(EPOCHS_FULL):
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(model.device) for b in batch]

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1

    avg_train_loss = total_loss / num_batches

    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(model.device) for b in batch]
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs.loss.item()

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_logloss = compute_log_loss(val_targets, val_preds)

    print(
        f"Epoch {epoch+1}/{EPOCHS_FULL} - Train Loss: {avg_train_loss:.4f} - Val Log Loss: {val_logloss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        torch.save(model.state_dict(), "./working/best_model_02052befc56a4c5d9c04dc32e0c9ca95.pt")
        no_improve_count = 0
        print(f"  New best model saved! Val Log Loss: {val_logloss:.4f}")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ============ LOAD BEST MODEL AND EVALUATE ============
print("\n" + "=" * 50)
print("Loading best model for final evaluation and inference")
model.load_state_dict(torch.load("./working/best_model_02052befc56a4c5d9c04dc32e0c9ca95.pt"))
model.eval()

val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask = [b.to(model.device) for b in batch[:2]]
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        val_preds.append(probs.cpu().numpy())

val_preds = np.concatenate(val_preds)
val_logloss = compute_log_loss(val_labels_enc, val_preds)
print(f"Final Validation Log Loss: {val_logloss:.6f}")

# ============ TEST INFERENCE ============
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(model.device) for b in batch]
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds)
print(f"Test predictions shape: {test_preds.shape}")

# ============ CREATE SUBMISSION ============
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        le.classes_[0]: test_preds[:, 0],
        le.classes_[1]: test_preds[:, 1],
        le.classes_[2]: test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission_02052befc56a4c5d9c04dc32e0c9ca95.csv", index=False)
print(f"Submission saved to ./submission/submission_02052befc56a4c5d9c04dc32e0c9ca95.csv")
print(f"Submission shape: {submission.shape}")
print(f"First 5 rows:\n{submission.head()}")

# Final validation score line
print(f"Final Validation Score: {val_logloss}")
