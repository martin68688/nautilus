import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
import re
import os
import gc
import warnings
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"\s+", " ", text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("—", "--").replace("–", "-")
        return text.strip()
    return str(text) if pd.notna(text) else ""


print("Cleaning text...")
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# ============================================================
# 3. STYLOMETRIC FEATURE ENGINEERING (REMOVED - not needed)
# ============================================================
# All stylometric, n-gram, and TF-IDF feature engineering has been removed
# as per improvement plan to reduce overfitting.
# Only 'text' and 'author' columns are used.

# ============================================================
# 4. N-GRAM FEATURES (REMOVED - not needed)
# ============================================================
# Character and word n-gram features not used in pure DeBERTa approach.

# ============================================================
# 5. COMBINE ALL FEATURES (REMOVED - not needed)
# ============================================================
# No feature combination needed. Model uses only tokenized text.

# ============================================================
# 6. TARGET ENCODING AND SPLIT
# ============================================================
print("Preparing target variable...")
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)
print(f"Classes: {label_encoder.classes_}, Encoded: {list(range(num_classes))}")

print("Creating train/validation split...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(
    train_df["clean_text"], train_df["author_encoded"]
):
    train_idx = train_idx
    val_idx = val_idx
    break

print(
    f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}, Test size: {len(test_df)}"
)

# ============================================================
# 7. SAVE INTERMEDIATE DATA (minimal)
# ============================================================
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/train_idx.npy", train_idx)
np.save(f"{output_dir}/val_idx.npy", val_idx)
np.save(f"{output_dir}/y_train.npy", train_df["author_encoded"].iloc[train_idx].values)
np.save(f"{output_dir}/y_val.npy", train_df["author_encoded"].iloc[val_idx].values)

joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")


# ============================================================
# 8. DATASET CLASS (with token augmentation)
# ============================================================
class AuthorshipDataset(Dataset):
    def __init__(
        self, texts, labels=None, tokenizer=None, max_length=256, augment=False
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # augment parameter kept for compatibility but augmentation is disabled per improvement plan
        self.augment = False  # Always disabled – no token augmentation

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        item = {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }
        if self.labels is not None:
            item["labels"] = torch.LongTensor([self.labels[idx]])[0]
        return item


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


# ============================================================
# 9. MODEL DEFINITIONS (Three diverse models for ensemble)
# ============================================================
class EnsembleModel(nn.Module):
    """Base class for ensemble models with mean pooling and dropout head."""
    pass

class DebertaV3Large(EnsembleModel):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(DebertaV3Large, self).__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        self.hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        logits = self.classifier(mean_pooled)
        return logits


class DebertaV3Small(EnsembleModel):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(DebertaV3Small, self).__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/deberta-v3-small")
        self.hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        logits = self.classifier(mean_pooled)
        return logits


class DistilBERT(EnsembleModel):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(DistilBERT, self).__init__()
        self.backbone = AutoModel.from_pretrained("distilbert-base-uncased")
        self.hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        logits = self.classifier(mean_pooled)
        return logits


# ============================================================
# 10. PREPARE DATA LOADERS
# ============================================================
print("Preparing data for training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
MAX_GRAD_NORM = 1.0

# ============================================================
# Model configurations for ensemble
# ============================================================
model_configs = [
    {
        "name": "deberta-v3-large",
        "model_class": DebertaV3Large,
        "model_kwargs": {"num_classes": 3, "dropout_rate": 0.3},
        "tokenizer_name": "microsoft/deberta-v3-large",
        "max_length": 256,
    },
    {
        "name": "deberta-v3-small",
        "model_class": DebertaV3Small,
        "model_kwargs": {"num_classes": 3, "dropout_rate": 0.3},
        "tokenizer_name": "microsoft/deberta-v3-small",
        "max_length": 256,
    },
    {
        "name": "distilbert-base-uncased",
        "model_class": DistilBERT,
        "model_kwargs": {"num_classes": 3, "dropout_rate": 0.3},
        "tokenizer_name": "distilbert-base-uncased",
        "max_length": 256,
    },
]


def compute_log_loss(y_true, y_pred):
    """Compute multiclass log loss."""
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    y_pred_normalized = y_pred_clipped / y_pred_clipped.sum(axis=1, keepdims=True)
    N = len(y_true)
    M = y_pred.shape[1]
    log_loss_val = 0.0
    for i in range(N):
        for j in range(M):
            y_ij = 1 if y_true[i] == j else 0
            log_loss_val += y_ij * np.log(y_pred_normalized[i, j])
    log_loss_val = -log_loss_val / N
    return log_loss_val


def train_single_model(config, train_texts, train_labels, val_texts, val_labels, test_texts, device):
    """Train a single model with the given configuration."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Datasets
    train_dataset = AuthorshipDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        augment=False,
    )
    val_dataset = AuthorshipDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        augment=False,
    )
    test_dataset = AuthorshipDataset(
        texts=test_texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        augment=False,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    model = config["model_class"](**config["model_kwargs"])
    model.to(device)

    # If tokenizer added special tokens, resize model embeddings
    if tokenizer.vocab_size != model.backbone.config.vocab_size:
        model.backbone.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
    )

    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    scaler = GradScaler()

    # Training loop
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Total optimization steps: {total_steps}")

    best_val_loss = float("inf")
    best_log_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                val_batches += 1
                probabilities = torch.softmax(logits, dim=1)
                all_val_preds.append(probabilities.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / val_batches
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_labels_np = np.concatenate(all_val_labels, axis=0)

        log_loss_val = compute_log_loss(val_labels_np, val_preds)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {log_loss_val:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_log_loss = log_loss_val
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # Load best model and get final predictions
    print(f"Loading best model for {name}...")
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # Validation predictions
    all_val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            all_val_preds.append(probabilities.cpu().numpy())

    val_preds_final = np.concatenate(all_val_preds, axis=0)

    # Test predictions
    all_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            all_test_preds.append(probabilities.cpu().numpy())

    test_preds_final = np.concatenate(all_test_preds, axis=0)

    final_log_loss = compute_log_loss(val_labels_np, val_preds_final)
    print(f"{name} - Final Validation Log Loss: {final_log_loss:.4f}")

    return val_preds_final, test_preds_final, best_log_loss, model


# ============================================================
# Train all models in the ensemble
# ============================================================
print("Starting ensemble training...")

all_val_preds = []
all_test_preds = []
all_model_scores = []

for config in model_configs:
    val_preds, test_preds, best_log_loss, _ = train_single_model(
        config,
        train_df["clean_text"].iloc[train_idx].values,
        train_df["author_encoded"].iloc[train_idx].values,
        train_df["clean_text"].iloc[val_idx].values,
        train_df["author_encoded"].iloc[val_idx].values,
        test_df["clean_text"].values,
        device,
    )
    all_val_preds.append(val_preds)
    all_test_preds.append(test_preds)
    all_model_scores.append(best_log_loss)

    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Ensemble weight optimization on validation set
# ============================================================
print("\n" + "="*60)
print("Optimizing ensemble weights on validation set")
print("="*60)

val_labels_np = train_df["author_encoded"].iloc[val_idx].values

# Simple averaging as baseline
ensemble_val_avg = np.mean(all_val_preds, axis=0)
ensemble_log_loss_avg = compute_log_loss(val_labels_np, ensemble_val_avg)
print(f"Simple average ensemble validation log loss: {ensemble_log_loss_avg:.4f}")

# Learned weights via grid search
print("Searching for optimal ensemble weights...")
best_ensemble_log_loss = float("inf")
best_weights = None

# Grid search over weights for 3 models
n_models = len(all_val_preds)
for i in range(11):
    w1 = i / 10.0
    for j in range(11 - i):
        w2 = j / 10.0
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue

        weights = np.array([w1, w2, w3])
        weighted_val = np.tensordot(weights, all_val_preds, axes=([0], [0]))
        log_loss_val = compute_log_loss(val_labels_np, weighted_val)

        if log_loss_val < best_ensemble_log_loss:
            best_ensemble_log_loss = log_loss_val
            best_weights = weights.copy()

print(f"Best ensemble weights: {best_weights}")
for idx, config in enumerate(model_configs):
    print(f"  {config['name']}: {best_weights[idx]:.4f}")
print(f"Best ensemble validation log loss: {best_ensemble_log_loss:.4f}")
print(f"Best individual model: min val log loss = {min(all_model_scores):.4f}")

# ============================================================
# Generate final test predictions with weighted ensemble
# ============================================================
print("\nGenerating final test predictions...")
ensemble_test = np.tensordot(best_weights, all_test_preds, axes=([0], [0]))

# ============================================================
# CREATE SUBMISSION FILE
# ============================================================
print("Creating submission file...")
test_ids = test_df["id"].values

# Clip and normalize predictions
epsilon = 1e-15
test_preds_clipped = np.clip(ensemble_test, epsilon, 1 - epsilon)
test_preds_normalized = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_normalized[:, 0],
        "HPL": test_preds_normalized[:, 1],
        "MWS": test_preds_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"First 5 rows:\n{submission_df.head()}")

print("\n" + "="*60)
print("Training Complete!")
print(f"Ensemble validation log loss: {best_ensemble_log_loss:.4f}")
print(f"Best individual model validation log loss: {min(all_model_scores):.4f}")
print("="*60)