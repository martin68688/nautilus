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
        self.augment = augment

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

        # With 50% probability, randomly mask 2% of tokens (replace with tokenizer.mask_token_id)
        if self.augment and np.random.random() < 0.5:
            input_ids = np.array(input_ids)
            mask_token_id = self.tokenizer.mask_token_id
            num_tokens = int(len(input_ids) * 0.02)
            if num_tokens > 0:
                valid_indices = [
                    i
                    for i in range(len(input_ids))
                    if attention_mask[i] == 1 and i != 0
                ]
                if valid_indices:
                    indices_to_mask = np.random.choice(
                        valid_indices,
                        size=min(num_tokens, len(valid_indices)),
                        replace=False,
                    )
                    input_ids[indices_to_mask] = mask_token_id

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
# 9. MODEL DEFINITION (Pure DeBERTa-v3-large with attention pool and multi-sample dropout)
# ============================================================
class DebertaOnly(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.4):
        super(DebertaOnly, self).__init__()
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        self.hidden_size = self.deberta.config.hidden_size

        # Attention pooling over last 4 hidden states
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )

        # Classifier with two dropout layers for multi-sample dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract last 4 hidden states and apply mean pooling over sequence dim
        hidden_states = outputs.hidden_states[-4:]  # tuple of 4 tensors
        pooled = []
        for hs in hidden_states:
            # Mean pooling over sequence dimension (excluding padding)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hs.size()).float()
            sum_embeddings = torch.sum(hs * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            pooled.append(mean_pooled)

        # Stack and apply learned weighted averaging via attention pool
        stacked = torch.stack(pooled, dim=2)  # (batch_size, hidden_size, 4)
        # Attention weights over the 4 layers
        attn_weights = torch.softmax(
            self.attention_pool(stacked.transpose(1, 2)).squeeze(-1), dim=1
        )  # (batch_size, 4)
        combined = (stacked @ attn_weights.unsqueeze(-1)).squeeze(-1)  # (batch_size, hidden_size)

        logits = self.classifier(combined)
        return logits

    def multi_sample_forward(self, input_ids, attention_mask, num_samples=4):
        """Forward pass with multi-sample dropout during training."""
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-4:]
        pooled = []
        for hs in hidden_states:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hs.size()).float()
            sum_embeddings = torch.sum(hs * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            pooled.append(mean_pooled)

        stacked = torch.stack(pooled, dim=2)
        attn_weights = torch.softmax(
            self.attention_pool(stacked.transpose(1, 2)).squeeze(-1), dim=1
        )
        combined = (stacked @ attn_weights.unsqueeze(-1)).squeeze(-1)

        # Multi-sample dropout: run classifier num_samples times
        log_probs_list = []
        for _ in range(num_samples):
            logits = self.classifier(combined)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs_list.append(log_probs)

        # Average log probabilities
        avg_log_probs = torch.stack(log_probs_list, dim=0).mean(dim=0)
        return avg_log_probs


# ============================================================
# 10. PREPARE DATA LOADERS
# ============================================================
print("Preparing data for training...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = AuthorshipDataset(
    texts=train_df["clean_text"].iloc[train_idx].values,
    labels=train_df["author_encoded"].iloc[train_idx].values,
    tokenizer=tokenizer,
    max_length=256,
    augment=True,
)

val_dataset = AuthorshipDataset(
    texts=train_df["clean_text"].iloc[val_idx].values,
    labels=train_df["author_encoded"].iloc[val_idx].values,
    tokenizer=tokenizer,
    max_length=256,
    augment=False,
)

test_dataset = AuthorshipDataset(
    texts=test_df["clean_text"].values,
    labels=None,
    tokenizer=tokenizer,
    max_length=256,
    augment=False,
)

print(
    f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
MAX_GRAD_NORM = 1.0

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

# ============================================================
# 11. MODEL, OPTIMIZER, LOSS, SCHEDULER SETUP
# ============================================================
model = DebertaOnly(num_classes=3, dropout_rate=0.4)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.05,
)

total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)

from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = GradScaler()

# ============================================================
# 12. TRAINING LOOP
# ============================================================
print("Starting training...")
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
            # Use multi-sample dropout during training
            avg_log_probs = model.multi_sample_forward(input_ids, attention_mask, num_samples=4)
            loss = criterion(avg_log_probs, labels)
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
                # Single forward pass during validation (no multi-sample dropout)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            val_loss += loss.item()
            val_batches += 1
            probabilities = torch.softmax(logits, dim=1)
            all_val_preds.append(probabilities.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)

    # Calculate multiclass log loss
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )
    N = len(val_labels)
    M = 3
    log_loss_val = 0.0
    for i in range(N):
        for j in range(M):
            y_ij = 1 if val_labels[i] == j else 0
            log_loss_val += y_ij * np.log(val_preds_normalized[i, j])
    log_loss_val = -log_loss_val / N

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

# ============================================================
# 13. LOAD BEST MODEL AND FINAL VALIDATION SCORE
# ============================================================
print("Loading best model...")
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        all_val_preds.append(probabilities.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_preds = np.concatenate(all_val_preds, axis=0)
val_labels = np.concatenate(all_val_labels, axis=0)

epsilon = 1e-15
val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
N = len(val_labels)
M = 3
final_log_loss = 0.0
for i in range(N):
    for j in range(M):
        y_ij = 1 if val_labels[i] == j else 0
        final_log_loss += y_ij * np.log(val_preds_normalized[i, j])
final_log_loss = -final_log_loss / N

print(f"Final Validation Score: {final_log_loss}")

# ============================================================
# 14. TEST INFERENCE
# ============================================================
print("Running test inference...")
model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        all_test_preds.append(probabilities.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)
print(f"Test predictions shape: {test_preds.shape}")

# ============================================================
# 15. CREATE SUBMISSION FILE
# ============================================================
print("Creating submission file...")
test_ids = test_df["id"].values

# Clip and normalize predictions as per competition rules
epsilon = 1e-15
test_preds_clipped = np.clip(test_preds, epsilon, 1 - epsilon)
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
