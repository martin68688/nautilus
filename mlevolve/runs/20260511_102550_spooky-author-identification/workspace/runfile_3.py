import os
os.sched_setaffinity(0, {189})
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import warnings

warnings.filterwarnings("ignore")
import gc
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import log_loss

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
BATCH_SIZE = 8
ACCUM_STEPS = 1
N_EPOCHS = 20
N_FOLDS = 5
LR_BACKBONE = 5e-6
LR_HEAD = 2e-5
WARMUP_RATIO = 0.1
PATIENCE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_WORKERS = 2

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------
# Dataset
# ---------------------------
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------
# Model with multi-scale conv head + attention pooling
# ---------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        # x: (batch, seq_len, hidden), mask: (batch, seq_len)
        scores = self.attn(x).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class SpookyModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.attn_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        cls_output = sequence_output[:, 0, :]  # (batch, hidden)
        attn_output = self.attn_pool(sequence_output, attention_mask)  # (batch, hidden)
        combined = torch.cat([cls_output, attn_output], dim=-1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}


# ---------------------------
# Training function
# ---------------------------
def train_epoch(model, dataloader, optimizer, scheduler, stage="frozen"):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs["loss"]
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (batch_idx + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

        del input_ids, attention_mask, labels, outputs, loss
        if batch_idx % 50 == 0 and batch_idx > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs["loss"]
        total_loss += loss.item()
        probs = torch.softmax(outputs["logits"], dim=-1)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        del input_ids, attention_mask, labels, outputs

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    score = log_loss(all_labels, all_preds)
    return avg_loss, score, all_preds, all_labels


# ---------------------------
# Main execution
# ---------------------------
def main():
    print("Loading data...")
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(train_df["author"].values)

    # Split into train/val (20% validation)
    from sklearn.model_selection import train_test_split

    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, stratify=y, random_state=SEED
    )

    train_texts = train_df["text"].iloc[idx_train].values
    val_texts = train_df["text"].iloc[idx_val].values
    y_train = y[idx_train]
    y_val = y[idx_val]

    test_texts = test_df["text"].values

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_dataset = SpookyDataset(train_texts, y_train, tokenizer)
    val_dataset = SpookyDataset(val_texts, y_val, tokenizer)
    test_dataset = SpookyDataset(test_texts, labels=None, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    # Initialize model
    model = SpookyModel().to(DEVICE)

    # Separate params
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Progressive unfreezing stages
    stages = [("frozen", 5), ("unfreeze_last4", 4), ("full", 4)]

    best_val_score = float("inf")
    best_model_state = None
    patience_counter = 0
    total_epochs_run = 0

    # Training loop (no SWA to save memory)
    for stage_name, stage_epochs in stages:
        print(f"\nStage: {stage_name} for {stage_epochs} epochs")
        total_epochs_run += stage_epochs

        # Set gradient requirements
        if stage_name == "frozen":
            for p in model.backbone.parameters():
                p.requires_grad = False
            optimizer = AdamW([{"params": head_params, "lr": LR_HEAD}])
        elif stage_name == "unfreeze_last4":
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.backbone.encoder.layer[-4:].parameters():
                p.requires_grad = True
            backbone_params_stage = [p for p in model.backbone.parameters() if p.requires_grad]
            optimizer = AdamW(
                [
                    {"params": backbone_params_stage, "lr": LR_BACKBONE * 0.5},
                    {"params": head_params, "lr": LR_HEAD},
                ]
            )
        elif stage_name == "full":
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": LR_BACKBONE},
                    {"params": head_params, "lr": LR_HEAD},
                ]
            )

        # Clear cache at stage transition
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Scheduler
        num_training_steps = len(train_loader) * stage_epochs
        num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        for epoch in range(stage_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, stage_name
            )
            val_loss, val_score, val_preds, val_labels = validate(model, val_loader)

            print(
                f"Epoch {total_epochs_run - stage_epochs + epoch + 1}/{N_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_score:.4f}"
            )

            if val_score < best_val_score:
                best_val_score = val_score
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {total_epochs_run - stage_epochs + epoch + 1}"
                    )
                    break

        if patience_counter >= PATIENCE:
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    model.to(DEVICE)

    # Generate out-of-fold predictions for stacking on validation set
    print("Generating OOF predictions for stacking...")
    _, val_score, val_probs, val_labels = validate(model, val_loader)

    # Generate test predictions directly (no stacking to keep it simple and memory-efficient)
    print("Generating test predictions...")
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            test_preds_list.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_preds_list, axis=0)

    final_val_score = best_val_score

    # Create submission
    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(test_probs, columns=le.classes_)
    submission["id"] = test_df["id"].values
    submission = submission[["id", "EAP", "HPL", "MWS"]]
    submission.to_csv("./submission/submission_7e167850c7784d1ebc75136753d55881.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")
    print("Submission saved to ./submission/submission_7e167850c7784d1ebc75136753d55881.csv")


if __name__ == "__main__":
    main()