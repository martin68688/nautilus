import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import os
import math
import nlpaug.augmenter.word as naw
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

X_train_text = train_df["text"].values
y_author = train_df["author"].values
X_test_text = test_df["text"].values
test_ids = test_df["id"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_author)
num_classes = len(label_encoder.classes_)

# ============================================================
# 2. STRATIFIED SPLIT (70% train, 15% val, 15% test from train)
# ============================================================
X_tr_text, X_va_text, y_tr, y_va = train_test_split(
    X_train_text, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_va_text, X_te_text, y_va, y_te = train_test_split(
    X_va_text, y_va, test_size=0.5, stratify=y_va, random_state=42
)

print(f"Train: {len(X_tr_text)}, Val: {len(X_va_text)}, Test: {len(X_te_text)}")


# ============================================================
# 3. TRAINING DATA (no augmentation to avoid dependency issues)
# ============================================================
print("Using original training text (no augmentation)...")
# Keep original data only - synonym augmentation caused API incompatibility
X_tr_text_combined = X_tr_text
y_tr_combined = y_tr

# ============================================================
# 4. FEATURE ENGINEERING - DistilBERT Tokenization
# ============================================================
from transformers import DistilBertTokenizer

print("Tokenizing with DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
max_length = 256

def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

tokenized_tr = tokenize_texts(X_tr_text_combined)
tokenized_va = tokenize_texts(X_va_text)
tokenized_te = tokenize_texts(X_te_text)
tokenized_test = tokenize_texts(X_test_text)

print(f"Tokenized input ids shape: {tokenized_tr['input_ids'].shape}")

# ============================================================
# NOTE: TF-IDF features removed per evolution analysis.
# DistilBERT tokenizer replaces all handcrafted feature engineering.
# Tokenization is now handled in Section 4 above.
# ============================================================
print("TF-IDF feature extraction removed - using DistilBERT token embeddings instead.")

# ============================================================
# 6. PREPARE TOKENIZED DATA FOR DATALOADERS
# ============================================================
# Store tokenized tensors directly for DataLoader usage.
# y labels remain unchanged, just use torch tensors.
y_train_new = y_tr_combined

print(
    f"Tokenized data prepared: Train {tokenized_tr['input_ids'].shape}, "
    f"Val {tokenized_va['input_ids'].shape}, Test {tokenized_te['input_ids'].shape}"
)


# ============================================================
# MODEL DEFINITION: DistilBERT Sequence Classifier with Multi-Layer Gating
# ============================================================
from transformers import DistilBertModel, DistilBertConfig

class DistilBERTSequenceClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        # Load pretrained DistilBERT backbone
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Initially freeze all DistilBERT layers (Stage 1 training)
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Attention pooling over all 6 transformer layers' CLS embeddings
        # There are 6 hidden states corresponding to layers 1-6 (indices 1-6 of hidden_states)
        self.num_layers_pool = 6
        # Query network: single linear layer applied to top-layer CLS to compute attention scores
        self.attention_query = nn.Linear(self.distilbert.config.hidden_size, 1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
        nn.init.xavier_uniform_(self.attention_query.weight)
        if self.attention_query.bias is not None:
            nn.init.constant_(self.attention_query.bias, 0)

    def forward(self, input_ids, attention_mask):
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Collect [CLS] embeddings from all 6 transformer layers (hidden_states indices 1-6)
        # hidden_states[0] is the embedding layer, hidden_states[1..6] are transformer layers 1-6
        hidden_states = outputs.hidden_states  # tuple of 7 tensors: (batch, seq_len, 768)
        cls_embeddings = []
        for layer_idx in range(1, 7):  # indices 1 through 6 (transformer layers 1-6)
            layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, 768)
            cls_embeddings.append(layer_hidden[:, 0, :])  # (batch, 768)
        # Stack: (batch, 6, 768)
        stacked_cls = torch.stack(cls_embeddings, dim=1)

        # Compute attention scores using the top-layer CLS as query
        query_cls = hidden_states[-1][:, 0, :]  # (batch, 768) - last layer CLS
        # Apply query network: (batch, 768) -> (batch, 1) per layer? No, we need (batch, 6, 1)
        # Actually we use a single linear layer that projects 768 -> 1, and apply to each layer's CLS
        # Better approach: compute attention weights using the query network on the stacked CLS
        # We use the query_cls to attend over all layer CLS embeddings
        # Simpler: project each layer's CLS to a score using the same linear layer
        # stacked_cls: (batch, 6, 768) -> attention_logits: (batch, 6, 1) -> squeeze to (batch, 6)
        attention_logits = self.attention_query(stacked_cls).squeeze(-1)  # (batch, 6)
        attention_weights = torch.softmax(attention_logits, dim=1)  # (batch, 6)

        # Weighted sum: (batch, 768)
        weighted_cls = torch.sum(stacked_cls * attention_weights.unsqueeze(-1), dim=1)

        # Apply dropout and classify
        x = self.dropout(weighted_cls)
        logits = self.classifier(x)
        return logits


def create_model(num_classes=3):
    return DistilBERTSequenceClassifier(num_classes=num_classes, dropout=0.3)


# ============================================================
# LOSS FUNCTION WITH LABEL SMOOTHING
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def get_criterion_and_optimizer(model, learning_rate=2e-5, weight_decay=0.01):
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
    return criterion, optimizer


# ============================================================
# MULTICLASS LOG LOSS
# ============================================================
def multiclass_log_loss(y_true, y_pred_probs, eps=1e-15):
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    y_pred_probs = y_pred_probs / row_sums
    y_true_one_hot = np.zeros_like(y_pred_probs)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1
    return -np.sum(y_true_one_hot * np.log(y_pred_probs)) / len(y_true)


# ============================================================
# SETUP DATA LOADERS (using tokenized inputs)
# ============================================================
# Custom dataset that returns input_ids, attention_mask, and labels
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels=None):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

y_train_tensor = torch.LongTensor(y_train_new)
y_val_tensor = torch.LongTensor(y_va)
y_test_tensor = torch.LongTensor(y_te)

batch_size = 16  # Smaller batch size for transformer model

train_dataset = TokenizedDataset(tokenized_tr, labels=y_train_tensor)
val_dataset = TokenizedDataset(tokenized_va, labels=y_val_tensor)
test_dataset = TokenizedDataset(tokenized_te, labels=y_test_tensor)
final_test_dataset = TokenizedDataset(tokenized_test, labels=None)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
final_test_loader = DataLoader(
    final_test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ============================================================
# INSTANTIATE MODEL, CRITERION, OPTIMIZER (two-stage strategy)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = create_model(num_classes=num_classes).to(device)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

scaler = GradScaler()

patience = 5  # Early stopping patience for this training setup

best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

# ============================================================
# STAGE 1: Train only classification head and attention pooling (3 epochs)
# Freeze all DistilBERT layers, train classifier + attention_query
# ============================================================
print("=" * 60)
print("STAGE 1: Training classifier head and attention pooling (DistilBERT frozen)")
print("=" * 60)

# Ensure DistilBERT is frozen
for param in model.distilbert.parameters():
    param.requires_grad = False

# Only train classifier and attention_query
stage1_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        stage1_params.append(param)
# Also explicitly add attention_query and classifier params (they shouldn't be frozen)
for name, param in model.attention_query.named_parameters():
    param.requires_grad = True
for name, param in model.classifier.named_parameters():
    param.requires_grad = True

# Build optimizer for Stage 1 (only classifier + attention query are trainable)
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

stage1_epochs = 3
total_steps_stage1 = len(train_loader) * stage1_epochs
warmup_steps_stage1 = 1 * len(train_loader)  # 1 epoch warmup

from torch.optim.lr_scheduler import LambdaLR
import math

def get_linear_warmup_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0  # Constant after warmup for Stage 1
    return lr_lambda

scheduler = LambdaLR(optimizer, lr_lambda=get_linear_warmup_lambda(warmup_steps_stage1, total_steps_stage1))

print("Stage 1 training (frozen backbone)...")
for epoch in range(stage1_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(batch["labels"].numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_loss = multiclass_log_loss(val_labels, val_preds)
    print(
        f"Stage 1 Epoch {epoch+1}/{stage1_epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), "./working/best_model_stage1.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 1 early stopping triggered at epoch {epoch+1}")
            break

# ============================================================
# STAGE 2: Unfreeze all DistilBERT layers, train full model (12 epochs)
# Reduced learning rate with cosine annealing and 5-epoch warmup
# ============================================================
print("=" * 60)
print("STAGE 2: Full fine-tuning (all layers unfrozen, reduced LR)")
print("=" * 60)

# Unfreeze all DistilBERT layers
for param in model.distilbert.parameters():
    param.requires_grad = True

# Build optimizer for Stage 2 with weight decay groups
optimizer = AdamW(
    [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ],
    lr=5e-6,
    betas=(0.9, 0.999),
    eps=1e-8,
)

stage2_epochs = 12
total_steps_stage2 = len(train_loader) * stage2_epochs
warmup_steps_stage2 = 5 * len(train_loader)  # 5-epoch warmup

# Cosine annealing scheduler with warmup
def get_cosine_warmup_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine annealing from 1 to 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

scheduler = LambdaLR(optimizer, lr_lambda=get_cosine_warmup_lambda(warmup_steps_stage2, total_steps_stage2))

# Reset patience for Stage 2
patience_counter = 0

print("Stage 2 training (full fine-tuning)...")
for epoch in range(stage2_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(batch["labels"].numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_loss = multiclass_log_loss(val_labels, val_preds)
    print(
        f"Stage 2 Epoch {epoch+1}/{stage2_epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 2 early stopping triggered at epoch {epoch+1}")
            break

# ============================================================
# LOAD BEST MODEL AND COMPUTE FINAL VALIDATION SCORE
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        val_preds.append(probs.cpu().numpy())
val_preds = np.concatenate(val_preds, axis=0)
score = multiclass_log_loss(y_va, val_preds)
print(f"Best Validation LogLoss: {best_val_loss:.6f}")

# ============================================================
# FINAL TEST INFERENCE FOR SUBMISSION
# ============================================================
final_preds = []
with torch.no_grad():
    for batch in final_test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        final_preds.append(probs.cpu().numpy())
final_preds = np.concatenate(final_preds, axis=0)
final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_preds[:, 0],
        "HPL": final_preds[:, 1],
        "MWS": final_preds[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {score}")