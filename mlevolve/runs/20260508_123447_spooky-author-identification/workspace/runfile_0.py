import os
os.sched_setaffinity(0, {0, 1, 2, 3, 12})
import pandas as pd
import numpy as np
import re
import string
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ================================================
# LOAD DATA
# ================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ================================================
# TEXT CLEANING
# ================================================
def clean_text(text):
    """Basic text cleaning while preserving stylistic elements"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?.])\1+", r"\1", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)
    return text


# ================================================
# TRAIN/VALIDATION SPLIT (stratified by author)
# ================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_df, train_df["author"]))[0]

train_data = train_df.iloc[train_idx].reset_index(drop=True)
val_data = train_df.iloc[val_idx].reset_index(drop=True)

# Apply text cleaning AFTER split to prevent data leakage
train_data["clean_text"] = train_data["text"].apply(clean_text)
val_data["clean_text"] = val_data["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_data["author"])
y_val = le.transform(val_data["author"])
num_classes = 3

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ================================================
# Load DeBERTa-v3-base model and tokenizer
# ================================================
model_id = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_classes,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# ================================================
# Freeze strategy: progressive unfreezing
# ================================================
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

if hasattr(model, "deberta"):
    encoder_layers = model.deberta.encoder.layer
    for layer in encoder_layers[-6:]:
        for param in layer.parameters():
            param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,}")


# ================================================
# Tokenization with memory-efficient approach
# ================================================
def tokenize_texts(texts, tokenizer, max_length=192):
    """Tokenize texts in batches to save memory"""
    input_ids_list = []
    attention_mask_list = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch_texts = (
            texts[i : i + batch_size].tolist()
            if hasattr(texts, "tolist")
            else texts[i : i + batch_size]
        )
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])
        if i % (batch_size * 10) == 0 and i > 0:
            gc.collect()
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    return input_ids, attention_mask


print("Tokenizing training data...")
train_input_ids, train_attention_mask = tokenize_texts(
    train_data["text"].values, tokenizer
)
print("Tokenizing validation data...")
val_input_ids, val_attention_mask = tokenize_texts(val_data["text"].values, tokenizer)
print("Tokenizing test data...")
test_input_ids, test_attention_mask = tokenize_texts(test_df["text"].values, tokenizer)

# ================================================
# Create DataLoaders
# ================================================
train_dataset = TensorDataset(
    train_input_ids, train_attention_mask, torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    val_input_ids, val_attention_mask, torch.tensor(y_val, dtype=torch.long)
)
test_dataset = TensorDataset(test_input_ids, test_attention_mask)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

# ================================================
# Optimizer and Scheduler
# ================================================
# Layer-wise Learning Rate Decay (LLRD)
base_lr = 1e-5
param_groups = []

# Classifier layers - highest learning rate
classifier_params = {
    'params': model.classifier.parameters(),
    'lr': base_lr,
    'weight_decay': 0.1
}
param_groups.append(classifier_params)

# Encoder layers with decay
if hasattr(model, 'deberta'):
    encoder = model.deberta.encoder
    num_layers = len(encoder.layer)
    for i, layer in enumerate(encoder.layer):
        # depth: 0 at classifier, increasing for earlier layers
        depth = num_layers - i
        layer_lr = base_lr * (0.85 ** depth)

        # Separate attention output and feed-forward layers
        attention_params = []
        feed_forward_params = []
        for name, param in layer.named_parameters():
            if param.requires_grad:
                if 'attention' in name or 'output' in name:
                    attention_params.append(param)
                else:
                    feed_forward_params.append(param)

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': layer_lr,
                'weight_decay': 0.1
            })
        if feed_forward_params:
            param_groups.append({
                'params': feed_forward_params,
                'lr': layer_lr,
                'weight_decay': 0.01
            })

optimizer = AdamW(
    param_groups,
    lr=base_lr,
    weight_decay=0.02,
    betas=(0.9, 0.999),
    eps=1e-8,
)

total_steps = len(train_loader) * 8
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# ================================================
# FGM (Fast Gradient Method) for adversarial training
# ================================================
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


fgm = FGM(model, epsilon=0.5)

# ================================================
# Training Loop
# ================================================
scaler = GradScaler()
# Custom Focal Loss with adaptive gamma
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma_min=0.5, gamma_max=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        with torch.no_grad():
            avg_confidence = pt.mean().item()
            gamma = max(self.gamma_min, self.gamma_max * (1.0 - avg_confidence))
        focal_loss = (self.alpha * (1 - pt) ** gamma * ce_loss).mean()
        return focal_loss

criterion = FocalLoss(alpha=0.25, gamma_min=0.5, gamma_max=5.0)

best_val_loss = float("inf")
best_model_state = None
patience = 3
patience_counter = 0
num_epochs = 8

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)

        optimizer.zero_grad()

        # MixUp with probability 0.3
        mixup_applied = False
        mix_lambda = 1.0
        if np.random.random() < 0.3:
            mixup_applied = True
            mix_lambda = np.random.beta(0.2, 0.2)
            shuffle_idx = torch.randperm(input_ids.size(0), device=device)
            input_ids_mix = input_ids[shuffle_idx]
            attention_mask_mix = attention_mask[shuffle_idx]
            labels_mix = labels[shuffle_idx]

        with autocast():
            if mixup_applied:
                # Forward pass for original and mixed inputs
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                outputs_mix = model(
                    input_ids=input_ids_mix, attention_mask=attention_mask_mix
                )
                logits = outputs.logits
                logits_mix = outputs_mix.logits
                # Weighted combination of losses
                loss = mix_lambda * criterion(logits, labels) + (1 - mix_lambda) * criterion(logits_mix, labels_mix)
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()

        fgm.attack()
        with autocast():
            if mixup_applied:
                outputs_adv = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                outputs_adv_mix = model(
                    input_ids=input_ids_mix, attention_mask=attention_mask_mix
                )
                loss_adv = mix_lambda * criterion(outputs_adv.logits, labels) + (1 - mix_lambda) * criterion(outputs_adv_mix.logits, labels_mix)
            else:
                outputs_adv = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss_adv = criterion(outputs_adv.logits, labels)

        scaler.scale(loss_adv).backward()
        fgm.restore()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() + loss_adv.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches

    # Validation
    model.eval()
    val_preds = []
    val_true = []
    val_loss_total = 0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                probs = F.softmax(outputs.logits, dim=-1)

            val_loss_total += outputs.loss.item()
            val_batches += 1
            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())

    val_loss = val_loss_total / val_batches
    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

    val_logloss = log_loss(val_true, val_preds_clipped)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.6f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# ================================================
# Load best model and generate predictions
# ================================================
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

# Generate validation predictions
val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
        val_preds.append(probs.cpu().numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

final_val_score = log_loss(y_val, val_preds_clipped)
print(f"Final Validation Score: {final_val_score}")

# ================================================
# Generate test predictions and save submission
# ================================================
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_4de84b50a8954290a501e0e5596ca998.csv", index=False)
print(f"Submission saved to ./submission/submission_4de84b50a8954290a501e0e5596ca998.csv")
print(f"Submission shape: {submission.shape}")