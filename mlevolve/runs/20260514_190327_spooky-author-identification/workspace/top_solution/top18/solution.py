import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING (no feature engineering)
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author_encoded"] = le.fit_transform(train_set["author"])
val_set["author_encoded"] = le.transform(val_set["author"])

print(f"Classes: {le.classes_}")
print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
print(f"Test size: {len(test_df)}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# MODEL DESIGN - Frozen DeBERTa-v3-large + TextCNN
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class FrozenBackboneTextCNN(nn.Module):
    def __init__(self, backbone, hidden_size=1024, num_labels=3):
        super().__init__()
        self.backbone = backbone
        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.hidden_size = hidden_size

        # TextCNN with kernel sizes 2, 3, 4
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=k, padding=k//2)
            for k in [2, 3, 4]
        ])

        # Total features = mean pooled (hidden_size) + 3 * 128 (CNN outputs)
        total_features = hidden_size + 3 * 128

        self.classifier = nn.Linear(total_features, num_labels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        # Get backbone embeddings (frozen, no grad)
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        # outputs.last_hidden_state: (batch, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state

        # Mean pooling over sequence dimension
        mean_pooled = last_hidden.mean(dim=1)  # (batch, hidden_size)

        # TextCNN branch: operate on mean pooled (unsqueeze for channel dim)
        # Shape: (batch, 1, hidden_size) for Conv1d
        cnn_input = mean_pooled.unsqueeze(1)  # (batch, 1, hidden_size)

        cnn_outputs = []
        for conv in self.convs:
            # Apply conv: (batch, 128, hidden_size)
            conv_out = conv(cnn_input)
            # Global max pooling over sequence (hidden_size) dimension
            pooled_out = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            cnn_outputs.append(pooled_out)

        # Concatenate CNN outputs: (batch, 3*128)
        cnn_concat = torch.cat(cnn_outputs, dim=1)

        # Combine mean pooling and CNN features
        combined = torch.cat([mean_pooled, cnn_concat], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        return logits


# Load backbone without classifier head (use AutoModel, not AutoModelForSequenceClassification)
backbone = AutoModel.from_pretrained(
    "microsoft/deberta-v3-large",
    output_hidden_states=False,
    output_attentions=False,
)
model = FrozenBackboneTextCNN(backbone, hidden_size=backbone.config.hidden_size, num_labels=3)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Only CNN and classifier params are trainable
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))

print(f"Total trainable params: {sum(p.numel() for p in trainable_params):,}")
print(f"Frozen backbone params: {sum(p.numel() for p in backbone.parameters()):,}")


# ============================================================
# DATASET AND DATALOADER (simplified)
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        return item


author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts = train_set["text"].values
train_labels = train_set["author_encoded"].values
val_texts = val_set["text"].values
val_labels = val_set["author_encoded"].values
test_texts = test_df["text"].values
test_ids = test_df["id"].values

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = SpookyDataset(val_texts, val_labels, tokenizer, max_length)
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
patience = 7
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# For checkpointing on val log-loss improvement
best_checkpoint_score = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        # Linear warmup then cosine annealing
        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = 2e-4 * warmup_factor

        total_train_loss += loss.item()
        num_train_batches += 1

    # Step scheduler after warmup
    if current_step >= warmup_steps:
        scheduler.step()

    avg_train_loss = total_train_loss / num_train_batches

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
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

            total_val_loss += loss.item()
            num_val_batches += 1
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_score = log_loss(val_true, val_probs_clipped)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    # Checkpoint on val log-loss improvement
    if val_score < best_checkpoint_score:
        best_checkpoint_score = val_score
        torch.save(model.state_dict(), "./working/best_model.pt")

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

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