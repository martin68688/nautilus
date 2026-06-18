import os
os.sched_setaffinity(0, {18, 19, 20, 21, 22})
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import warnings
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Extract labels
y = train_df["author"].values
label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
y_encoded = np.array([label_map[a] for a in y])

# ============================================================
# 2. SETUP MODERNBERT MODEL AND TOKENIZER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# ============================================================
# 3. CUSTOM MODEL WITH MULTI-POOLING
# ============================================================
class ModernBertMultiPool(nn.Module):
    def __init__(self, model_id, num_labels=3, hidden_size=1024, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.num_labels = num_labels
        # Multi-pooling: CLS + mean + max -> concatenated size = 3 * hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Linear(3 * hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last hidden state: [batch_size, seq_len, hidden_size]
        last_hidden = outputs.last_hidden_state

        # Multi-pooling
        # CLS token pooling
        cls_pool = last_hidden[:, 0, :]  # [batch, hidden]

        # Mean pooling (masked)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pool = sum_embeddings / sum_mask  # [batch, hidden]

        # Max pooling (masked)
        last_hidden_masked = last_hidden * input_mask_expanded
        max_pool = torch.max(last_hidden_masked, dim=1)[0]  # [batch, hidden]

        # Concatenate all poolings
        pooled = torch.cat([cls_pool, mean_pool, max_pool], dim=-1)  # [batch, 3*hidden]

        logits = self.classifier(pooled)
        return logits


# ============================================================
# 4. LABEL SMOOTHING LOSS
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -torch.sum(true_dist * log_probs, dim=-1).mean()
        return loss


# ============================================================
# 5. DATASET
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 6. TRAINING AND EVALUATION FUNCTIONS
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    eps = 1e-15
    all_preds = np.clip(all_preds, eps, 1 - eps)
    all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)

    ll = log_loss(all_labels, all_preds)
    return ll, all_preds


def get_grouped_parameters(model):
    """Group parameters for different learning rates."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    return backbone_params, head_params


def set_grad_freeze(model, freeze_level):
    """
    freeze_level: 0 = train only head (backbone frozen)
                  1 = train head + last 4 backbone layers
                  2 = train all
    """
    # First freeze everything
    for param in model.backbone.parameters():
        param.requires_grad = False

    if freeze_level >= 1:
        # Unfreeze last 4 layers of backbone
        # ModernBERT backbone layers are in model.backbone.layers
        num_layers = len(model.backbone.layers)
        for i in range(num_layers - 4, num_layers):
            for param in model.backbone.layers[i].parameters():
                param.requires_grad = True

    if freeze_level >= 2:
        # Unfreeze all backbone
        for param in model.backbone.parameters():
            param.requires_grad = True

    # Always train head
    for param in model.classifier.parameters():
        param.requires_grad = True


# ============================================================
# 7. SETUP CROSS-VALIDATION
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store test predictions for ensembling
test_preds_folds = []
fold_scores = []

print("Starting 5-fold fine-tuning cross-validation...")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_df)), y_encoded)):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/5")
    print(f"{'='*60}")

    # Create data loaders
    train_texts = train_df["text"].iloc[train_idx].tolist()
    train_labels = y_encoded[train_idx]
    val_texts = train_df["text"].iloc[val_idx].tolist()
    val_labels = y_encoded[val_idx]

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=512)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=512)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    model = ModernBertMultiPool(model_id, num_labels=3, hidden_size=1024, dropout=0.3)
    model.to(device)

    # Optimizer with different LR for backbone and head
    backbone_params, head_params = get_grouped_parameters(model)
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": 2e-5},
            {"params": head_params, "lr": 5e-4},
        ],
        weight_decay=0.01,
    )

    # Scheduler: cosine annealing with warm restarts
    total_steps = len(train_loader) * 10  # 10 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 3, T_mult=2, eta_min=1e-6
    )

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = GradScaler()

    # SWA
    swa_model = None
    swa_start_epoch = 7

    # Training loop with progressive unfreezing
    best_val_ll = float("inf")
    patience_counter = 0
    max_patience = 3

    for epoch in range(1, 11):  # 10 epochs per plan
        print(f"\nEpoch {epoch}/10")

        # Progressive unfreezing
        if epoch <= 2:
            set_grad_freeze(model, freeze_level=0)
            freeze_status = "head only"
        elif epoch <= 5:
            set_grad_freeze(model, freeze_level=1)
            freeze_status = "head + last4 layers"
        else:
            set_grad_freeze(model, freeze_level=2)
            freeze_status = "all layers"

        print(f"  Training mode: {freeze_status}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, grad_clip=1.0
        )
        val_ll, _ = evaluate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.6f}, Val Log Loss: {val_ll:.6f}")

        # Update SWA model
        if epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = model.state_dict()
            else:
                # Moving average
                n = epoch - swa_start_epoch + 1
                factor = 1.0 / n
                for key in swa_model:
                    swa_model[key] = (1 - factor) * swa_model[key] + factor * model.state_dict()[key]

        # Early stopping check
        if val_ll < best_val_ll:
            best_val_ll = val_ll
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping triggered after epoch {epoch}")
                break

        # Step scheduler
        scheduler.step()

    # Evaluate with SWA model
    if swa_model is not None:
        model.load_state_dict(swa_model)
        val_ll_swa, _ = evaluate(model, val_loader, device)
        print(f"\n  SWA Val Log Loss: {val_ll_swa:.6f}")
        if val_ll_swa < best_val_ll:
            best_val_ll = val_ll_swa
            best_model_state = model.state_dict()
    else:
        model.load_state_dict(best_model_state)

    fold_scores.append(best_val_ll)
    print(f"  Fold {fold_idx + 1} Best Val Log Loss: {best_val_ll:.6f}")

    # Test inference
    test_dataset = TextDataset(test_df["text"].tolist(), labels=None, tokenizer=tokenizer, max_length=512)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)

            fold_test_preds.append(probs.cpu().numpy())

    fold_test_preds = np.concatenate(fold_test_preds, axis=0)
    eps = 1e-15
    fold_test_preds = np.clip(fold_test_preds, eps, 1 - eps)
    fold_test_preds = fold_test_preds / fold_test_preds.sum(axis=1, keepdims=True)
    test_preds_folds.append(fold_test_preds)

    # Clean up to free memory
    del model
    torch.cuda.empty_cache()

# ============================================================
# 8. CALCULATE OVERALL CV SCORE
# ============================================================
mean_cv_score = np.mean(fold_scores)
std_cv_score = np.std(fold_scores)
print(f"\n{'='*60}")
print(f"Cross-Validation Results:")
print(f"Per-fold scores: {[f'{s:.6f}' for s in fold_scores]}")
print(f"Mean CV Log Loss: {mean_cv_score:.6f} (+/- {std_cv_score:.6f})")

# ============================================================
# 9. ENSEMBLE TEST PREDICTIONS ACROSS FOLDS
# ============================================================
print("\nEnsembling test predictions across folds...")
test_pred_proba = np.mean(test_preds_folds, axis=0)

eps = 1e-15
test_pred_proba = np.clip(test_pred_proba, eps, 1 - eps)
test_pred_proba = test_pred_proba / test_pred_proba.sum(axis=1, keepdims=True)

# ============================================================
# 10. CREATE SUBMISSION FILE
# ============================================================
print("Creating submission file...")
submission_dir = "./submission"
os.makedirs(submission_dir, exist_ok=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_pred_proba[:, 0],
        "HPL": test_pred_proba[:, 1],
        "MWS": test_pred_proba[:, 2],
    }
)

submission.to_csv(os.path.join(submission_dir, "submission_811f752c53f541119949ac2a9b434547.csv"), index=False)
print(f"Submission saved to {submission_dir}/submission_811f752c53f541119949ac2a9b434547.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# 11. PRINT FINAL VALIDATION SCORE
# ============================================================
print(f"Final Cross-Validation Score: {mean_cv_score}")
