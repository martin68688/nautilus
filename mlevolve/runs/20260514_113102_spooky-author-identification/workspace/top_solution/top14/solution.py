import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-base (more stable)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


from transformers import AutoModel


class StochasticDepth(nn.Module):
    """Stochastic Depth (DropPath) regularization for Transformer layers."""
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        keep_prob = 1 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.4, stochastic_depth_rate=0.1):
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
        # Add stochastic depth to unfrozen layers
        self.stochastic_depth = StochasticDepth(stochastic_depth_rate)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.num_authors = num_authors
        self.head = nn.Linear(hidden_size, num_authors)
        self._init_weights(self.head)
        # Projection head for supervised contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        for module in self.projection:
            if isinstance(module, nn.Linear):
                self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, num_samples=4):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use the [CLS] token from the last hidden state
        hidden_states = outputs.last_hidden_state
        cls_pool = hidden_states[:, 0, :]

        # Apply stochastic depth to CLS embeddings in training
        cls_pool = self.stochastic_depth(cls_pool)

        # Projection for contrastive learning
        proj_embeddings = self.projection(cls_pool)

        # Multi-sample dropout: apply dropout K times and average
        logits_list = []
        for _ in range(num_samples):
            dropped = self.dropout(cls_pool)
            logits_list.append(self.head(dropped))
        # Stack and average
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        return logits, proj_embeddings


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.4, stochastic_depth_rate=0.1)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Supervised contrastive loss (NT-Xent)
def contrastive_loss(embeddings, labels, temperature=0.1):
    batch_size = embeddings.shape[0]
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Create mask for positive pairs (same author)
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()
    mask.fill_diagonal_(0)  # Remove self-pairs

    # Numerical stability
    max_sim = similarity.max(dim=1, keepdim=True)[0].detach()
    similarity_stable = similarity - max_sim

    # Compute exp and sum
    exp_sim = torch.exp(similarity_stable)
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)

    # Compute log prob
    log_prob = similarity_stable - torch.log(exp_sim_sum)

    # Mean of positive pairs
    pos_count = mask.sum(dim=1)
    pos_loss = -(log_prob * mask).sum(dim=1) / pos_count.clamp(min=1)

    return pos_loss.mean()

# Collect all trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(
    trainable_params,
    lr=2e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, aux_targets=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aux_targets = aux_targets

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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.aux_targets is not None:
            item["aux_targets"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)
        return item


# Get original texts for training
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Precompute top-50 starting words/bigrams for auxiliary task
from collections import Counter
import re

def get_starting_words(text, n=5):
    """Extract first n words from text."""
    words = str(text).lower().split()[:n]
    return words

# Build vocabulary of starting word combinations from training data
start_ngrams = []
for text in train_texts_orig:
    words = get_starting_words(text, n=5)
    # Use unigrams and bigrams of starting words
    for w in words:
        start_ngrams.append(('unigram', w))
    for i in range(len(words)-1):
        start_ngrams.append(('bigram', f"{words[i]}_{words[i+1]}"))

counter = Counter(start_ngrams)
top_start_features = [item for item, _ in counter.most_common(50)]
top_start_feature_set = set(top_start_features)
num_aux_features = len(top_start_features)

def get_aux_target(text):
    """Create binary vector indicating presence of top starting features."""
    words = get_starting_words(text, n=5)
    features = []
    for w in words:
        features.append(('unigram', w))
    for i in range(len(words)-1):
        features.append(('bigram', f"{words[i]}_{words[i+1]}"))
    bin_vec = np.zeros(num_aux_features, dtype=np.float32)
    for feat in features:
        if feat in top_start_feature_set:
            idx = top_start_features.index(feat)
            bin_vec[idx] = 1.0
    return bin_vec

train_aux_targets = np.array([get_aux_target(text) for text in train_texts_orig])
val_aux_targets = np.array([get_aux_target(text) for text in val_set["text"].values])

# Use previously computed indices for train/validation split
train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length, aux_targets=None
)
val_dataset = SpookyDataset(
    val_texts_final, val_labels_final, tokenizer, max_length, aux_targets=None
)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
num_epochs = 30
patience = 5
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

# OneCycleLR scheduler
total_steps = len(train_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-5,
    total_steps=total_steps,
    pct_start=0.1,
    final_div_factor=100,
    anneal_strategy='cos',
)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Check for NaN inputs
        if torch.isnan(input_ids).any() or torch.isnan(attention_mask).any():
            continue

        optimizer.zero_grad()
        with autocast():
            logits, proj_embeddings = model(input_ids, attention_mask, num_samples=4)
            loss_main = criterion(logits, labels)
            loss_contrastive = contrastive_loss(proj_embeddings, labels)
            loss = loss_main + 0.2 * loss_contrastive

        scaler_grad.scale(loss).backward()
        # Skip step if gradient is NaN (check scaled gradients)
        skip_step = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                skip_step = True
                break
        if skip_step:
            optimizer.zero_grad()
            continue

        # Unscale gradients ONCE before clipping/stepping
        scaler_grad.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler_grad.step(optimizer)
        scaler_grad.update()

        # OneCycleLR step
        scheduler.step()

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

            with autocast():
                # Use dropout in eval mode for multi-sample averaging
                model.train()
                # Multi-sample: run forward K times and average
                logits_list = []
                for _ in range(4):
                    logits, _ = model(input_ids, attention_mask, num_samples=4)
                    logits_list.append(logits)
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
                loss_main = criterion(logits, labels)
                # No auxiliary loss during validation (only CE)
                loss = loss_main
                model.eval()

                probs = torch.softmax(logits, dim=1)

            # Check for NaN in predictions
            if torch.isnan(probs).any():
                # Replace NaN with uniform distribution
                probs = torch.ones_like(probs) / probs.size(-1)

            total_val_loss += loss.item()
            num_val_batches += 1
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / max(1, num_val_batches)

    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    # Handle any remaining NaN values
    val_probs = np.nan_to_num(val_probs, nan=1.0/3.0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_score = log_loss(val_true, val_probs_clipped)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            # Enable dropout for multi-sample inference
            model.train()
            logits_list = []
            for _ in range(4):
                logits, _ = model(input_ids, attention_mask, num_samples=4)
                logits_list.append(logits)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            model.eval()
            probs = torch.softmax(logits, dim=1)
        # Check for NaN
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.size(-1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)
# Handle NaN
val_probs = np.nan_to_num(val_probs, nan=1.0/3.0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_true, val_probs_clipped)

# ============================================================
# TEST INFERENCE
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            # Enable dropout for multi-sample inference
            model.train()
            logits_list = []
            for _ in range(4):
                logits, _ = model(input_ids, attention_mask, num_samples=4)
                logits_list.append(logits)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            model.eval()
            probs = torch.softmax(logits, dim=1)
        # Check for NaN
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.size(-1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)
# Final NaN safeguard
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

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")