import os
os.sched_setaffinity(0, {0, 1, 2, 3, 53})
import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========== DATA LOADING ==========
def load_data():
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")
    return train, test


# ========== FEATURE ENGINEERING ==========
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ========== DATASET CLASS ==========
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ========== MAIN PIPELINE ==========
print("Loading data...")
train, test = load_data()
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Feature engineering - note: TF-IDF vectorizers and scaler fitting happen inside cross-validation to avoid leakage
print("Feature engineering will be done per fold to avoid data leakage...")

# Encode labels
author_to_idx = {"EAP": 0, "HPL": 1, "MWS": 2}
idx_to_author = {0: "EAP", 1: "HPL", 2: "MWS"}
y_train = train["author"].map(author_to_idx).values

# Cross-validation setup
MODEL_ID = "answerdotai/ModernBERT-large"
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 6
NUM_FOLDS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.3
GRAD_ACCUM_STEPS = 2
WARMUP_RATIO = 0.1
PATIENCE = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

val_predictions = np.zeros((len(train), 3))
test_predictions_list = []

print(f"Starting {NUM_FOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train["text"], y_train)):
    print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")

    train_texts = train.iloc[train_idx]["text"].values
    val_texts = train.iloc[val_idx]["text"].values
    train_labels = y_train[train_idx]
    val_labels = y_train[val_idx]

    train_dataset = AuthorDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = AuthorDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_labels = 3
    config.hidden_dropout_prob = DROPOUT_RATE
    config.attention_probs_dropout_prob = DROPOUT_RATE
    model = ModernBertForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
    )
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS

    # Cosine annealing with linear warmup
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
    import math

    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    best_fold_score = float("inf")
    patience_counter = 0
    fold_best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Original forward pass
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            # -- Adversarial Weight Perturbation (AWP) --
            # Save original parameters and gradients
            orig_params = []
            orig_grads = []
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    orig_params.append(param.data.clone().detach())
                    orig_grads.append(param.grad.clone().detach())

            # Compute perturbation magnitude: epsilon * (param_norm / grad_norm)
            with torch.no_grad():
                param_norms = []
                grad_norms = []
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param_norms.append(param.data.norm().item())
                        grad_norms.append(param.grad.norm().item())
                avg_param_norm = sum(param_norms) / max(len(param_norms), 1)
                avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)

            epsilon = 0.001
            if avg_grad_norm > 1e-8:
                scale_factor = epsilon * (avg_param_norm / avg_grad_norm)
            else:
                scale_factor = epsilon

            # Perturb weights adversarially: w -> w + epsilon * sign(grad)
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.data.add_(param.grad.sign(), alpha=scale_factor)

            # Forward pass on perturbed weights
            with autocast():
                outputs_adv = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_adv = outputs_adv.logits
                loss_adv = criterion(logits_adv, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss_adv).backward()

            # Restore original weights
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    if param.requires_grad and param.grad is not None:
                        if i < len(orig_params):
                            param.data.copy_(orig_params[i])

            # -- End AWP --

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += (loss.item() + loss_adv.item()) * GRAD_ACCUM_STEPS

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_true.append(batch["labels"].numpy())

        val_probs = np.concatenate(val_probs, axis=0)
        val_true = np.concatenate(val_true, axis=0)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(
            axis=1, keepdims=True
        )
        fold_score = log_loss(val_true, val_probs_clipped)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val LogLoss: {fold_score:.4f}"
        )

        if fold_score < best_fold_score:
            best_fold_score = fold_score
            fold_best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Use best model state for all predictions
    if fold_best_state is not None:
        model.load_state_dict(fold_best_state)
    model.eval()

    # Recompute validation predictions with best model
    val_probs = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_true.append(batch["labels"].numpy())
    val_probs = np.concatenate(val_probs, axis=0)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_predictions[val_idx] = val_probs

    test_dataset = AuthorDataset(test["text"].values, None, tokenizer, MAX_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(fold_test_probs, axis=0)
    fold_test_probs = np.clip(fold_test_probs, 1e-15, 1 - 1e-15)
    fold_test_probs = fold_test_probs / fold_test_probs.sum(axis=1, keepdims=True)
    test_predictions_list.append(fold_test_probs)

    print(f"Fold {fold+1} Best Val LogLoss: {best_fold_score:.4f}")

val_probs_clipped = np.clip(val_predictions, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
overall_val_score = log_loss(y_train, val_probs_clipped)

print(f"\nOverall Validation LogLoss: {overall_val_score:.4f}")

test_probs_avg = np.mean(test_predictions_list, axis=0)
test_probs_avg = np.clip(test_probs_avg, 1e-15, 1 - 1e-15)
test_probs_avg = test_probs_avg / test_probs_avg.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test["id"].values,
        "EAP": test_probs_avg[:, 0],
        "HPL": test_probs_avg[:, 1],
        "MWS": test_probs_avg[:, 2],
    }
)
submission.to_csv("./submission/submission_cab3038c1f534545aaf10c5a4b7e8ff4.csv", index=False)
print("Submission saved to ./submission/submission_cab3038c1f534545aaf10c5a4b7e8ff4.csv")

print(f"Final Validation Score: {overall_val_score}")