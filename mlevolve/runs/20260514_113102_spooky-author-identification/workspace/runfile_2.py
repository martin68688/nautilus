import os
os.sched_setaffinity(0, {4, 5, 6, 7, 8, 9, 12, 13, 14})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# ENGINEERED FEATURES
# ============================================================
def get_engineered_features(texts):
    """
    Compute engineered features:
    - sentence_length: number of characters
    - punctuation_density: count of .!? / length
    - word_count: number of words
    """
    features = []
    for text in texts:
        text_str = str(text)
        length = len(text_str)
        punct_count = text_str.count('.') + text_str.count('!') + text_str.count('?')
        punct_density = punct_count / max(length, 1)
        word_count = len(text_str.split())
        features.append([length, punct_density, word_count])
    return np.array(features, dtype=np.float32)

# Compute engineered features for all texts
all_texts = np.concatenate([train_df["text"].values, test_df["text"].values], axis=0)
all_feats = get_engineered_features(all_texts)

# Fit scaler on training texts only
scaler = StandardScaler()
train_feats_raw = get_engineered_features(train_df["text"].values)
scaler.fit(train_feats_raw)

# Transform all features
train_feats_scaled = scaler.transform(get_engineered_features(train_df["text"].values))
test_feats_scaled = scaler.transform(get_engineered_features(test_df["text"].values))

# ============================================================
# MODEL DESIGN - DeBERTa-v3-base (more stable)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


from transformers import AutoModel


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-base",
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # Freeze all backbone layers first
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze last 4 encoder layers
        for layer in self.backbone.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(hidden_size, num_authors)
        self._init_weights(self.head)
        # Small feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, hidden_size),
        )
        self._init_weights(self.feature_proj[0])
        self._init_weights(self.feature_proj[3])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, extra_features=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use the [CLS] token from the last hidden state
        hidden_states = outputs.last_hidden_state
        cls_pool = hidden_states[:, 0, :]
        if extra_features is not None:
            feat_vec = self.feature_proj(extra_features)
            cls_pool = cls_pool + 0.1 * feat_vec
        cls_pool = self.dropout(cls_pool)
        logits = self.head(cls_pool)
        return logits

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels_all = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

batch_size = 16
max_length = 512
num_epochs = 30
patience = 5

# ============================================================
# DATASET AND DATALOADER (with engineered features)
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, extra_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extra_features = extra_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
        }
        if self.extra_features is not None:
            item["extra_features"] = torch.tensor(self.extra_features[idx], dtype=torch.float32)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ============================================================
# TRAINING FUNCTION FOR A GIVEN FOLD AND SEED
# ============================================================
def train_single_fold(train_idx, val_idx, seed, fold_id, model_save_dir):
    """Train a single fold with a given random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_texts_final = train_df["text"].values[train_idx]
    train_labels_final = train_labels_all[train_idx]
    val_texts_final = train_df["text"].values[val_idx]
    val_labels_final = train_labels_all[val_idx]

    train_feats_final = train_feats_scaled[train_idx]
    val_feats_final = train_feats_scaled[val_idx]

    train_dataset = SpookyDataset(
        train_texts_final, train_labels_final, tokenizer, max_length, extra_features=train_feats_final
    )
    val_dataset = SpookyDataset(
        val_texts_final, val_labels_final, tokenizer, max_length, extra_features=val_feats_final
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    scaler_grad = GradScaler()

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    def get_lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    best_val_score = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device)

            if torch.isnan(input_ids).any() or torch.isnan(attention_mask).any():
                continue

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask, extra_features)
                loss = criterion(logits, labels)

            scaler_grad.scale(loss).backward()
            # Unscale only once - clip gradients and then step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isnan(torch.tensor(grad_norm)):
                optimizer.zero_grad()
                continue

            scaler_grad.step(optimizer)
            scaler_grad.update()

            current_step = epoch * len(train_loader) + batch_idx
            lr_mult = get_lr_multiplier(current_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = 2e-5 * lr_mult

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(1, num_train_batches)

        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                extra_features = batch.get("extra_features", None)
                if extra_features is not None:
                    extra_features = extra_features.to(device)

                with autocast():
                    logits = model(input_ids, attention_mask, extra_features)
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)

                if torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)

                total_val_loss += loss.item()
                num_val_batches += 1
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / max(1, num_val_batches)

        val_probs = np.concatenate(all_val_probs, axis=0)
        val_true = np.concatenate(all_val_labels, axis=0)

        val_probs = np.nan_to_num(val_probs, nan=1.0/3.0)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

        val_score = log_loss(val_true, val_probs_clipped)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Seed {seed} Fold {fold_id} | Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}")

        if val_score < best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"best_model_seed{seed}_fold{fold_id}.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered for Seed {seed} Fold {fold_id} after {epoch+1} epochs")
                break

    # Load best model and compute out-of-fold predictions
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f"best_model_seed{seed}_fold{fold_id}.pt")))
    model.eval()

    val_probs_final = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device)
            with autocast():
                logits = model(input_ids, attention_mask, extra_features)
                probs = torch.softmax(logits, dim=1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            val_probs_final.append(probs.cpu().numpy())

    oof_probs = np.concatenate(val_probs_final, axis=0)
    oof_probs = np.nan_to_num(oof_probs, nan=1.0/3.0)

    return oof_probs, model

# ============================================================
# CROSS-VALIDATION ENSEMBLE TRAINING
# ============================================================
seeds = [42, 123, 456]
n_folds = 5
skf_list = []
for seed in seeds:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    skf_list.append(skf)

# Prepare arrays to store out-of-fold predictions
n_train = len(train_df)
n_authors = 3
oof_all = np.zeros((n_train, n_authors))  # Sum across all models
oof_counts = np.zeros(n_train, dtype=np.int32)

model_save_dir = "./working"
os.makedirs(model_save_dir, exist_ok=True)

# Also store individual oof predictions for meta-learner
all_oof_preds = []  # list of (oof_probs, val_indices) pairs
all_models = []  # list of trained models (for test inference)

for seed_idx, seed in enumerate(seeds):
    print(f"\n{'='*60}")
    print(f"Training with seed {seed} ({seed_idx+1}/{len(seeds)})")
    print(f"{'='*60}")
    skf = skf_list[seed_idx]
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["author"])):
        print(f"\n--- Seed {seed}, Fold {fold_id+1}/{n_folds} ---")
        oof_probs, trained_model = train_single_fold(train_idx, val_idx, seed, fold_id, model_save_dir)

        # Accumulate OOF predictions
        oof_all[val_idx] += oof_probs
        oof_counts[val_idx] += 1

        # Store for meta-learner
        all_oof_preds.append((oof_probs, val_idx))
        all_models.append(trained_model)

# Average OOF predictions
oof_all = oof_all / np.maximum(oof_counts, 1)[:, None]
oof_all = np.nan_to_num(oof_all, nan=1.0/n_authors)

# Compute baseline ensemble validation score
val_probs_clipped = np.clip(oof_all, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
baseline_val_score = log_loss(train_labels_all, val_probs_clipped)
print(f"\nBaseline Ensemble Validation LogLoss: {baseline_val_score:.6f}")

# ============================================================
# META-LEARNER: Softmax-weighted linear blending
# ============================================================
# Prepare meta-features: stack all OOF predictions
oof_stacked = np.zeros((n_train, len(all_oof_preds) * n_authors))
for i, (oof_probs, val_idx) in enumerate(all_oof_preds):
    oof_stacked[val_idx, i*n_authors:(i+1)*n_authors] = oof_probs

# Train softmax-weighted blending weights
def blend_loss(weights_flat, preds, y_true):
    """Blend with softmax weights and compute log loss."""
    n_models = len(all_oof_preds)
    weights = weights_flat.reshape(n_models, n_authors)
    # Apply softmax across models for each class
    weights_softmax = np.zeros_like(weights)
    for c in range(n_authors):
        exp_w = np.exp(weights[:, c] - np.max(weights[:, c]))
        weights_softmax[:, c] = exp_w / (exp_w.sum() + 1e-15)

    # Compute blended predictions
    blended = np.zeros((preds.shape[0], n_authors))
    for m in range(n_models):
        blended += preds[:, m*n_authors:(m+1)*n_authors] * weights_softmax[m, :]

    # Add small constant for stability
    blended = np.clip(blended, 1e-15, 1 - 1e-15)
    blended = blended / blended.sum(axis=1, keepdims=True)
    return log_loss(y_true, blended)

init_weights = np.zeros((len(all_oof_preds), n_authors)).flatten()
result = minimize(
    blend_loss,
    init_weights,
    args=(oof_stacked, train_labels_all),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'disp': False}
)

# Compute final softmax weights
n_models = len(all_oof_preds)
opt_weights = result.x.reshape(n_models, n_authors)
blend_weights = np.zeros_like(opt_weights)
for c in range(n_authors):
    exp_w = np.exp(opt_weights[:, c] - np.max(opt_weights[:, c]))
    blend_weights[:, c] = exp_w / (exp_w.sum() + 1e-15)

print(f"\nOptimization converged: {result.success}")
print(f"Optimal blend weights (shape {blend_weights.shape}):")
for i, w in enumerate(blend_weights):
    print(f"  Model {i}: {w}")

# Compute final meta-learner validation score
blended_val = np.zeros((n_train, n_authors))
for m in range(n_models):
    blended_val += oof_stacked[:, m*n_authors:(m+1)*n_authors] * blend_weights[m, :]
blended_val = np.clip(blended_val, 1e-15, 1 - 1e-15)
blended_val = blended_val / blended_val.sum(axis=1, keepdims=True)
final_val_score = log_loss(train_labels_all, blended_val)
print(f"\nMeta-Learner Validation LogLoss: {final_val_score:.6f}")

# ============================================================
# TEST INFERENCE with ensemble
# ============================================================
test_loader_ensemble = DataLoader(
    SpookyDataset(test_texts, None, tokenizer, max_length, extra_features=test_feats_scaled),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

all_test_preds = []
for model_idx, trained_model in enumerate(all_models):
    trained_model.eval()
    model_test_probs = []
    with torch.no_grad():
        for batch in test_loader_ensemble:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device)
            with autocast():
                logits = trained_model(input_ids, attention_mask, extra_features)
                probs = torch.softmax(logits, dim=1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            model_test_probs.append(probs.cpu().numpy())
    test_probs_model = np.concatenate(model_test_probs, axis=0)
    test_probs_model = np.nan_to_num(test_probs_model, nan=1.0/3.0)
    all_test_preds.append(test_probs_model)

# Blend test predictions using meta-learner weights
test_probs = np.zeros_like(all_test_preds[0])
for m in range(n_models):
    test_probs += all_test_preds[m] * blend_weights[m, :]

test_probs = np.nan_to_num(test_probs, nan=1.0/3.0)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission_342d81bff8f3491c956be6afc1f9d28d.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Meta-Learner Validation Score: {final_val_score}")