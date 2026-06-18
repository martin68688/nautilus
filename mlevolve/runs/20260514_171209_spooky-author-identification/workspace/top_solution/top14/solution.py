import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import os
import random
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
N_FOLDS = 5
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LR_BASE = 2e-5
LR_LORA = 5e-4
WARMUP_RATIO = 0.15
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sub = pd.read_csv("./input/sample_submission.csv")

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_map)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        return self.dropout(x @ self.lora_a.T) @ self.lora_b.T * self.scale


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.1):
    lora_params = []
    for name, module in model.named_modules():
        if (
            "attention" in name
            and hasattr(module, "query")
            and hasattr(module.query, "weight")
        ):
            q_lora = LoRALayer(
                module.query.in_features,
                module.query.out_features,
                rank,
                alpha,
                dropout,
            ).to(device)
            v_lora = LoRALayer(
                module.value.in_features,
                module.value.out_features,
                rank,
                alpha,
                dropout,
            ).to(device)
            setattr(module, "lora_query", q_lora)
            setattr(module, "lora_value", v_lora)

            orig_forward = module.forward

            def make_lora_forward(q_lora, v_lora, orig_forward):
                def lora_forward(*args, **kwargs):
                    hidden_states = args[0]
                    out = orig_forward(*args, **kwargs)
                    if isinstance(out, tuple) and len(out) >= 2:
                        q_out, k_out, v_out = out[0], out[1], out[2]
                        q_out = q_out + q_lora(hidden_states)
                        v_out = v_out + v_lora(hidden_states)
                        return (q_out, k_out, v_out) + out[3:]
                    return out

                return lora_forward

            module.forward = make_lora_forward(q_lora, v_lora, orig_forward)
            lora_params.append(q_lora.parameters())
            lora_params.append(v_lora.parameters())

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return list(lora_params)


class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment and random.random() < 0.3:
            words = text.split()
            if len(words) > 10:
                idx1 = random.randint(0, len(words) - 1)
                idx2 = random.randint(0, len(words) - 1)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                text = " ".join(words)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DebertaWithLoRA(nn.Module):
    def __init__(self, num_labels=3, rank=8, alpha=16, lora_dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
        config.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large", config=config
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.lora_params = apply_lora_to_model(self.backbone, rank, alpha, lora_dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(
    model, loader, optimizer, scheduler, criterion, scaler, epoch, mixup_alpha=None
):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast():
            logits = model(input_ids, attention_mask)

            if mixup_alpha is not None and random.random() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                indices = torch.randperm(labels.size(0)).to(device)
                labels_a, labels_b = labels, labels[indices]
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_model(model, loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(loader)
    try:
        ll = log_loss(all_labels, all_preds)
    except:
        ll = avg_loss
    return ll, all_preds, all_labels


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
val_scores = []
fold_models = []
fold_predictions = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df["text"], train_df["label"])
):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

    train_texts = train_df["text"].iloc[train_idx].values
    train_labels = train_df["label"].iloc[train_idx].values
    val_texts = train_df["text"].iloc[val_idx].values
    val_labels = train_df["label"].iloc[val_idx].values

    train_dataset = AuthorDataset(
        train_texts, train_labels, tokenizer, MAX_LEN, augment=True
    )
    val_dataset = AuthorDataset(
        val_texts, val_labels, tokenizer, MAX_LEN, augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = DebertaWithLoRA(num_labels=3, rank=8, alpha=16, lora_dropout=0.1).to(device)

    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    lora_params = [p for n, p in model.named_parameters() if "lora" in n]
    backbone_params = [p for n, p in model.named_parameters() if "lora" not in n]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": LR_BASE},
            {"params": lora_params, "lr": LR_LORA},
        ],
        weight_decay=0.01,
    )

    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()

    best_val_score = float("inf")
    patience = 2
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            epoch,
            MIXUP_ALPHA,
        )
        val_score, _, _ = eval_model(model, val_loader, criterion)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val LogLoss: {val_score:.4f} - LR: {lr_now:.2e}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    val_score, _, _ = eval_model(model, val_loader, criterion)
    val_scores.append(val_score)
    fold_models.append(model)

    print(f"Fold {fold+1} best val score: {val_score:.4f}")

final_val_score = np.mean(val_scores)
print(f"\nAverage validation score across folds: {final_val_score:.4f}")

test_dataset = AuthorDataset(
    test_df["text"].values,
    labels=None,
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    augment=False,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

test_preds = np.zeros((len(test_df), 3))
for model in fold_models:
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            fold_preds.append(probs.cpu().numpy())
    fold_preds = np.concatenate(fold_preds)
    test_preds += fold_preds / N_FOLDS

sub["EAP"] = test_preds[:, 0]
sub["HPL"] = test_preds[:, 1]
sub["MWS"] = test_preds[:, 2]

os.makedirs("./submission", exist_ok=True)
sub.to_csv("./submission/submission.csv", index=False)
print(f"Saved submission to ./submission/submission.csv")

print(f"Final Validation Score: {final_val_score:.4f}")
