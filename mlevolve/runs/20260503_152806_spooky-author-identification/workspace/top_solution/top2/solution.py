import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import os
import copy
import re
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

# Configuration
MODEL_ID = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
NUM_EPOCHS = 6
N_FOLDS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MIXOUT_PROB = 0.1
NUM_LABELS = 3
DROPOUT_RATE = 0.1
LABEL_SMOOTHING = 0.1
WARMUP_STEPS = 100


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, augment=True):
        self.texts = texts.values if hasattr(texts, "values") else texts
        self.labels = labels.values if hasattr(labels, "values") else labels
        self.augment = augment
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Token masking augmentation: only apply during training
        if self.augment and torch.rand(1).item() < 0.3:
            # Identify non-special tokens (exclude [CLS]=101, [SEP]=102, and padding=0)
            non_special_mask = ~torch.isin(input_ids, torch.tensor([101, 102, 0]))
            non_special_indices = torch.where(non_special_mask)[0]
            if len(non_special_indices) > 0:
                num_to_mask = max(1, int(len(non_special_indices) * 0.1))
                mask_indices = non_special_indices[torch.randperm(len(non_special_indices))[:num_to_mask]]
                input_ids[mask_indices] = self.mask_token_id

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class MixoutLinear(nn.Module):
    def __init__(
        self, original_linear, pretrained_weight, pretrained_bias, mixout_prob=0.2
    ):
        super().__init__()
        self.original_linear = original_linear
        self.pretrained_weight = nn.Parameter(
            pretrained_weight.clone(), requires_grad=False
        )
        self.pretrained_bias = (
            nn.Parameter(pretrained_bias.clone(), requires_grad=False)
            if pretrained_bias is not None
            else None
        )
        self.mixout_prob = mixout_prob

    def forward(self, x):
        if not self.training or self.mixout_prob == 0:
            return self.original_linear(x)
        weight = self.original_linear.weight
        bias = self.original_linear.bias
        mask = torch.rand_like(weight) > self.mixout_prob
        mixed_weight = torch.where(mask, weight, self.pretrained_weight)
        if bias is not None and self.pretrained_bias is not None:
            bias_mask = torch.rand_like(bias) > self.mixout_prob
            mixed_bias = torch.where(bias_mask, bias, self.pretrained_bias)
        else:
            mixed_bias = bias
        return F.linear(x, mixed_weight, mixed_bias)


def apply_mixout_to_model(model, mixout_prob):
    pretrained_state = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            pretrained_state[name] = {
                "weight": module.weight.data.clone(),
                "bias": module.bias.data.clone() if module.bias is not None else None,
            }
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(
            x in name for x in ["classifier", "pooler"]
        ):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            mixout_layer = MixoutLinear(
                module,
                pretrained_state[name]["weight"],
                pretrained_state[name]["bias"],
                mixout_prob,
            )
            setattr(parent, child_name, mixout_layer)


def compute_log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


def train_epoch(
    model, dataloader, optimizer, scheduler, scaler, device, accumulation_steps=1
):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    y_true = np.eye(3)[all_labels]
    logloss = compute_log_loss(y_true, all_preds)
    return logloss, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    # Encode labels
    label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
    train_df["label"] = train_df["author"].map(label_map)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Store predictions
    test_predictions = np.zeros((len(test_df), 3))
    val_scores = []

    # Use a single train/val split to save time
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"], train_df["label"], test_size=0.15, random_state=42, stratify=train_df["label"]
    )

    print(f"\n{'='*50}")
    print(f"Single train/val split: {len(train_texts)} train, {len(val_texts)} val")
    print(f"{'='*50}")

    train_dataset = SpookyDataset(train_texts, train_labels, augment=True)
    val_dataset = SpookyDataset(val_texts, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_labels = 3
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": WEIGHT_DECAY,
            "lr": LEARNING_RATE,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": LEARNING_RATE,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    num_training_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler()

    best_val_loss = float("inf")
    best_model_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            ACCUMULATION_STEPS,
        )
        val_loss, _ = validate_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Log Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    val_loss, _ = validate_epoch(model, val_loader, device)
    val_scores.append(val_loss)
    print(f"Best Val Log Loss: {val_loss:.5f}")

    model.eval()
    test_dataset = SpookyDataset(test_df["text"], labels=None, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
            test_preds.append(probs.cpu().numpy())

    test_predictions = np.concatenate(test_preds, axis=0)

    final_val_score = val_loss
    print(f"\n{'='*50}")
    print(f"Cross-Validation Log Loss: {final_val_score:.5f}")

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_df["id"].values,
            "EAP": test_predictions[:, 0],
            "HPL": test_predictions[:, 1],
            "MWS": test_predictions[:, 2],
        }
    )

    eps = 1e-15
    for col in ["EAP", "HPL", "MWS"]:
        submission[col] = np.clip(submission[col], eps, 1 - eps)
    row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
    for col in ["EAP", "HPL", "MWS"]:
        submission[col] = submission[col] / row_sums

    submission.to_csv("./submission/submission.csv", index=False)
    print(f"\nSubmission saved to ./submission/submission.csv")
    print(f"Submission shape: {submission.shape}")
    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()
