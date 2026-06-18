import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

# ==================== SYNTACTIC FEATURE EXTRACTION ====================
def extract_syntactic_features(texts):
    """
    Extract hierarchical syntactic features from each text:
    1. Number of clauses (using conjunction heuristics)
    2. Average phrase length (based on punctuation splits)
    3. Subordination ratio (subordinate conjunctions / total words)
    4. Punctuation diversity index (unique punctuation types / total punctuation)
    """
    subordinate_conjunctions = {'although', 'since', 'unless', 'because', 'while', 'whereas',
                                'though', 'although', 'until', 'after', 'before', 'if', 'when'}
    clause_conjunctions = {'and', 'but', 'or', 'because', 'which', 'that', 'while', 'although', 'since', 'unless'}

    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([1, 0, 0, 0])
            continue

        # Lowercase for analysis
        lower_text = text.lower()
        words = text.split()
        word_count = max(len(words), 1)

        # 1. Number of clauses: count clause conjunctions + 1 (base clause)
        clause_count = 1
        for conj in clause_conjunctions:
            clause_count += lower_text.count(' ' + conj + ' ')

        # 2. Average phrase length: split by punctuation and average length
        import re
        phrases = re.split(r'[.,;:!?]', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        if phrases:
            avg_phrase_length = sum(len(p.split()) for p in phrases) / len(phrases)
        else:
            avg_phrase_length = 0

        # 3. Subordination ratio: count subordinate conjunctions / word_count
        subordination_count = 0
        for conj in subordinate_conjunctions:
            subordination_count += lower_text.count(' ' + conj + ' ')
        subordination_ratio = subordination_count / word_count

        # 4. Punctuation diversity index: unique punctuation types / total punctuation
        punctuation_types = set()
        total_punctuation = 0
        for char in text:
            if char in '.,;:!?\'"()[]-':
                punctuation_types.add(char)
                total_punctuation += 1
        if total_punctuation > 0:
            punctuation_diversity = len(punctuation_types) / min(total_punctuation, 10)
            punctuation_diversity = min(punctuation_diversity, 1.0)  # cap at 1.0
        else:
            punctuation_diversity = 0

        features.append([clause_count, avg_phrase_length, subordination_ratio, punctuation_diversity])

    return np.array(features, dtype=np.float32)


# ==================== DATA LOADING ====================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode labels
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_authors = len(label_encoder.classes_)
author_names = label_encoder.classes_
print(f"Authors: {author_names}")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

train_texts = train_df["text"].tolist()
train_labels = train_df["author_encoded"].values
test_texts = test_df["text"].tolist()
test_ids = test_df["id"].values

# Extract and normalize syntactic features
print("Extracting syntactic features...")
train_syntactic_features = extract_syntactic_features(train_texts)
test_syntactic_features = extract_syntactic_features(test_texts)

# Fit StandardScaler on training data only
syntactic_scaler = StandardScaler()
train_syntactic_features = syntactic_scaler.fit_transform(train_syntactic_features)
test_syntactic_features = syntactic_scaler.transform(test_syntactic_features)

# Handle NaN values that may arise from constant features
train_syntactic_features = np.nan_to_num(train_syntactic_features, nan=0.0)
test_syntactic_features = np.nan_to_num(test_syntactic_features, nan=0.0)

print(f"Syntactic features shape - train: {train_syntactic_features.shape}, test: {test_syntactic_features.shape}")

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
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== DATASET CLASS ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, syntactic_features=None):
        self.texts = texts
        self.labels = labels
        self.syntactic_features = syntactic_features

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

        feat = torch.tensor(self.syntactic_features[idx], dtype=torch.float32) if self.syntactic_features is not None else torch.zeros(4, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, attention_mask, feat, label
        return input_ids, attention_mask, feat


# ==================== CUSTOM MODEL WITH SYNTACTIC FEATURES ====================
class CustomDebertaWithSyntacticFeatures(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import DebertaV2Model
        self.deberta = DebertaV2Model.from_pretrained(model_name)

        # Syntactic feature encoder: 4 -> 64
        self.syntactic_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # Classifier head: 1024 (CLS) + 64 (syntactic) = 1088
        self.classifier = nn.Linear(1024 + 64, num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, syntactic_features, num_samples=4):
        # Get DeBERTa encoder outputs
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token (first token) pooled output
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch, 1024]

        # Encode syntactic features
        syntactic_encoded = self.syntactic_encoder(syntactic_features)  # [batch, 64]

        # Concatenate
        combined = torch.cat([pooled_output, syntactic_encoded], dim=1)  # [batch, 1088]

        # Multi-Sample Dropout: run classifier head multiple times during training
        if self.training:
            logits_list = []
            for _ in range(num_samples):
                dropped = self.dropout(combined)
                logits_list.append(self.classifier(dropped))
            # Average logits across dropout samples
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            logits = self.classifier(combined)

        return logits


# ==================== TRAINING FUNCTION ====================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, syntactic_features, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        syntactic_features = syntactic_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask, syntactic_features, num_samples=4)
            loss = criterion(logits, labels)

            # Gradient penalty on classifier head weights
            if hasattr(model, 'classifier'):
                gp_lambda = 0.1
                gp = torch.norm(model.classifier.weight, p=2) ** 2
                loss = loss + gp_lambda * gp

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
        for input_ids, attention_mask, syntactic_features, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            syntactic_features = syntactic_features.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(input_ids, attention_mask, syntactic_features, num_samples=1)
                probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ==================== MAIN TRAINING LOOP ====================
print("Setting up stratified k-fold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []

# Use first fold for validation (fast approach)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    if fold > 0:  # Only use first fold for quick validation
        break

    print(f"\n===== Fold {fold+1} =====")

    fold_train_texts = [train_texts[i] for i in train_idx]
    fold_train_labels = train_labels[train_idx]
    fold_val_texts = [train_texts[i] for i in val_idx]
    fold_val_labels = train_labels[val_idx]

    # Get syntactic features for this fold
    fold_train_syntactic = train_syntactic_features[train_idx]
    fold_val_syntactic = train_syntactic_features[val_idx]

    # Create datasets and dataloaders
    train_dataset = TextDataset(fold_train_texts, fold_train_labels, fold_train_syntactic)
    val_dataset = TextDataset(fold_val_texts, fold_val_labels, fold_val_syntactic)

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

    # Initialize custom model
    epochs = 3  # Reduced epochs to prevent overfitting
    model = CustomDebertaWithSyntacticFeatures(model_name, num_authors)
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    # Cosine annealing scheduler with linear warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Linear warmup then cosine annealing
    class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.min_lr = min_lr
            super().__init__(optimizer)

        def get_lr(self):
            step = self._step_count
            if step < self.warmup_steps:
                # Linear warmup
                return [base_lr * (step / max(1, self.warmup_steps)) for base_lr in self.base_lrs]
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 1  # Early stopping patience

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device
        )

        val_preds, val_labels = evaluate(model, val_loader, device)

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
            patience_counter = 0
            # Save best model
            os.makedirs("./working", exist_ok=True)
            torch.save(model.state_dict(), f"./working/best_model_fold_{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    val_scores.append(best_val_loss)
    print(f"Fold {fold+1} best validation log loss: {best_val_loss:.4f}")

print(f"\nAverage validation log loss: {np.mean(val_scores):.4f}")

# ==================== LOAD BEST CHECKPOINT FOR TEST ====================
print("\n===== Loading best checkpoint for test predictions =====")
best_model_path = f"./working/best_model_fold_0.pt"
model = CustomDebertaWithSyntacticFeatures(model_name, num_authors)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
print(f"Loaded best model from {best_model_path}")

# ==================== TEST PREDICTIONS ====================
print("\nGenerating test predictions...")
test_dataset = TextDataset(test_texts, syntactic_features=test_syntactic_features)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_test_preds = []
with torch.no_grad():
    for input_ids, attention_mask, syntactic_features in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        syntactic_features = syntactic_features.to(device)

        with autocast():
            logits = model(input_ids, attention_mask, syntactic_features, num_samples=1)
            probs = torch.softmax(logits, dim=1)

        all_test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)

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
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

final_val_score = np.mean(val_scores)
print(f"Final Validation Score: {final_val_score}")