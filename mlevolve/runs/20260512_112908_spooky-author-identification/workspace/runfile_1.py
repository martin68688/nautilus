import os
os.sched_setaffinity(0, {110, 111, 51, 52, 53})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

# ==================== DATA LOADING ====================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode labels
# We'll encode labels per fold to avoid data leakage
num_authors = train_df["author"].nunique()
author_names = sorted(train_df["author"].unique())
print(f"Authors: {author_names}")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Map author names to indices for consistent ordering across folds
author_to_idx = {author: i for i, author in enumerate(author_names)}
train_labels_original = train_df["author"].map(author_to_idx).values

train_texts = train_df["text"].tolist()
train_labels = train_labels_original
test_texts = test_df["text"].tolist()
test_ids = test_df["id"].values

# ==================== MODEL CONFIGURATION ====================
print("Loading DeBERTa-v3-large model...")
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 256
batch_size = 8  # Smaller batch for large model
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== ENSEMBLE MODEL CLASS ====================
class EnsembleModel(nn.Module):
    def __init__(self, num_labels, model_name, dropout, weight_decay, lr):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


# ==================== DATASET CLASS ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, attention_mask, label
        return input_ids, attention_mask


# ==================== TRAINING FUNCTION ====================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ==================== EVALUATION FUNCTION ====================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ==================== MAIN TRAINING LOOP ====================
print("Setting up stratified k-fold with ensemble of 5 models...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
best_model_paths = []

# Define hyperparameter sampling ranges (seed will be set per fold for reproducibility)
sample_dropout_range = [0.05, 0.15]
sample_weight_decay_range = [0.005, 0.02]
sample_lr_range = [1e-5, 3e-5]

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    # Set per-fold seed to avoid any cross-fold leakage
    np.random.seed(42 + fold)
    print(f"\n===== Fold {fold+1}/5 =====")

    # Sample diverse hyperparameters for this fold's model
    fold_dropout = np.random.uniform(sample_dropout_range[0], sample_dropout_range[1])
    fold_weight_decay = np.random.uniform(sample_weight_decay_range[0], sample_weight_decay_range[1])
    fold_lr = np.random.uniform(sample_lr_range[0], sample_lr_range[1])
    print(f"  Dropout: {fold_dropout:.4f}, Weight Decay: {fold_weight_decay:.6f}, LR: {fold_lr:.6f}")

    fold_train_texts = [train_texts[i] for i in train_idx]
    fold_train_labels = train_labels[train_idx]
    fold_val_texts = [train_texts[i] for i in val_idx]
    fold_val_labels = train_labels[val_idx]

    # Ensure we have 3 unique classes present (should always be true with stratified split)
    assert len(np.unique(fold_train_labels)) == 3, "All 3 classes must be in training fold"

    # Create datasets and dataloaders
    train_dataset = TextDataset(fold_train_texts, fold_train_labels)
    val_dataset = TextDataset(fold_val_texts, fold_val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model with sampled hyperparameters
    model = EnsembleModel(num_authors, model_name, fold_dropout, fold_weight_decay, fold_lr)
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=fold_lr, weight_decay=fold_weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Focal Loss to combat overfitting by down-weighting easy examples
    # gamma=2 focuses on hard examples, alpha=1 is balanced for 3 classes
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        train_loss = train_epoch(
            model.model, train_loader, optimizer, scheduler, criterion, scaler, device
        )

        val_preds, val_labels = evaluate(model.model, val_loader, device)

        # Clip probabilities as per competition requirements
        epsilon = 1e-15
        val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
        val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(
            axis=1, keepdims=True
        )

        val_loss = log_loss(val_labels, val_preds_clipped)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Log Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best model
            os.makedirs("./working", exist_ok=True)
            torch.save(model.state_dict(), f"./working/best_model_fold_{fold+1}.pt")

    best_model_paths.append(f"./working/best_model_fold_{fold+1}.pt")
    val_scores.append(best_val_loss)
    print(f"Fold {fold+1} best validation log loss: {best_val_loss:.4f} (epoch {best_epoch+1})")

print(f"\nAverage validation log loss: {np.mean(val_scores):.4f}")
print(f"Best model paths: {best_model_paths}")

# ==================== LOAD ENSEMBLE CHECKPOINTS FOR TEST ====================
print("\n===== Loading all 5 checkpoints for ensemble predictions =====")
ensemble_models = []
for fold_idx, path in enumerate(best_model_paths):
    # We need the hyperparameters used for this fold to instantiate the model correctly
    # Since we sampled randomly, we re-instantiate with a placeholder and load state_dict
    # (the state_dict has the same structure regardless of dropout config since it's just weights)
    model_for_loading = EnsembleModel(num_authors, model_name, 0.1, 0.01, 2e-5)
    model_for_loading.load_state_dict(torch.load(path, map_location=device))
    model_for_loading.to(device)
    model_for_loading.eval()
    ensemble_models.append(model_for_loading)
    print(f"Loaded model from {path}")

# ==================== TEST PREDICTIONS ====================
print("\nGenerating ensemble test predictions...")
test_dataset = TextDataset(test_texts)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

all_test_preds = []
for model_idx, model in enumerate(ensemble_models):
    print(f"Generating predictions from model {model_idx+1}...")
    fold_test_preds = []
    with torch.no_grad():
        for input_ids, attention_mask in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)

            fold_test_preds.append(probs.cpu().numpy())
    all_test_preds.append(np.concatenate(fold_test_preds, axis=0))

# Average predictions from all 5 models
test_preds = np.mean(all_test_preds, axis=0)

# Clip and normalize probabilities
epsilon = 1e-15
test_preds_clipped = np.clip(test_preds, epsilon, 1 - epsilon)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

# ==================== SUBMISSION ====================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_a2583d29b8c84e25b5d689163c073008.csv", index=False)
print(f"Submission saved to ./submission/submission_a2583d29b8c84e25b5d689163c073008.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

final_val_score = np.mean(val_scores)
print(f"Final Validation Score: {final_val_score}")