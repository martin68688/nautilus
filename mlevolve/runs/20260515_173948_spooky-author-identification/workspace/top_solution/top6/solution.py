import os, gc, sys, math, json, random, warnings, pickle, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
ROOT = "./input/"
OUT = "./submission/"
os.makedirs(OUT, exist_ok=True)
os.makedirs("./working/", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
MAX_LEN = 256
BATCH_SIZE = 8
ACCUM_STEPS = 4
EPOCHS = 5
LR = 2e-5
MODEL_NAME = "microsoft/deberta-v3-large"
NUM_LABELS = 3
CLASS_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}
INV_CLASS = {0: "EAP", 1: "HPL", 2: "MWS"}

train_df = pd.read_csv(ROOT + "train.csv")
test_df = pd.read_csv(ROOT + "test.csv")
sample_sub = pd.read_csv(ROOT + "sample_submission.csv")

train_df["label"] = train_df["author"].map(CLASS_MAP)
all_texts = pd.concat([train_df["text"], test_df["text"]]).values


# ---- Stylometric Feature Engineering ----
def get_stylometric_features(texts):
    features = []
    for text in texts:
        words = str(text).split()
        sentences = str(text).split(".")
        chars = len(str(text))
        word_len = len(words)
        sent_len = len(sentences)
        avg_word_len = chars / max(word_len, 1)
        avg_sent_len = word_len / max(sent_len, 1)
        punct_count = sum(1 for c in str(text) if c in ".,;:!?'\"-")
        comma_ratio = str(text).count(",") / max(word_len, 1)
        exclam_ratio = str(text).count("!") / max(word_len, 1)
        quest_ratio = str(text).count("?") / max(word_len, 1)
        colon_ratio = str(text).count(":") / max(word_len, 1)
        semicolon_ratio = str(text).count(";") / max(word_len, 1)
        quote_ratio = str(text).count('"') / max(word_len, 1)
        dash_ratio = str(text).count("—") / max(word_len, 1)
        cap_ratio = sum(1 for c in str(text) if c.isupper()) / max(chars, 1)
        digit_ratio = sum(1 for c in str(text) if c.isdigit()) / max(chars, 1)
        unique_word_ratio = len(set(w.lower() for w in words)) / max(word_len, 1)
        features.append(
            [
                avg_word_len,
                avg_sent_len,
                punct_count / max(chars, 1),
                comma_ratio,
                exclam_ratio,
                quest_ratio,
                colon_ratio,
                semicolon_ratio,
                quote_ratio,
                dash_ratio,
                cap_ratio,
                digit_ratio,
                unique_word_ratio,
                word_len,
                chars,
            ]
        )
    return np.array(features)


stylo_train = get_stylometric_features(train_df["text"].values)
stylo_test = get_stylometric_features(test_df["text"].values)


class SpookyDataset(Dataset):
    def __init__(
        self,
        texts,
        labels=None,
        tokenizer=None,
        max_len=MAX_LEN,
        trunc_strategy="first",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.trunc_strategy = trunc_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.trunc_strategy == "last" and len(text) > self.max_len * 4:
            text = text[-self.max_len * 4 :]
        elif self.trunc_strategy == "random" and len(text) > self.max_len * 4:
            start = random.randint(0, len(text) - self.max_len * 4)
            text = text[start : start + self.max_len * 4]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class DebertaForSequenceClassificationWithPooling(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.config.hidden_dropout_prob = 0.1
        self.attention_pool = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512), nn.Tanh(), nn.Linear(512, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        attn_weights = self.attention_pool(hidden)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (attn_weights * hidden).sum(dim=1)
        logits = self.classifier(pooled)
        if labels is not None:
            logits = torch.clamp(logits, min=-50.0, max=50.0)  # prevent extreme logits -> NaN softmax gradients
            loss = F.cross_entropy(logits, labels)
            return loss, logits, pooled
        return logits, pooled


def train_epoch(model, loader, optimizer, scaler, scheduler=None, accumulate=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        if DEVICE.type == "cuda":
            with autocast():
                loss, logits, _ = model(input_ids, attention_mask, labels)
                loss = loss / accumulate
                loss = torch.clamp(loss, max=10.0)  # prevent NaN from extreme loss
            scaler.scale(loss).backward()
            if (i + 1) % accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
        else:
            loss, logits, _ = model(input_ids, attention_mask, labels)
            loss = loss / accumulate
            loss = torch.clamp(loss, max=10.0)
            loss.backward()
            if (i + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
        total_loss += loss.item() * accumulate
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with autocast():
                loss, logits, _ = model(input_ids, attention_mask, labels)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    l = log_loss(all_labels, all_preds)
    return l, all_preds


def predict_test(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds)


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_oof = np.zeros((len(train_df), NUM_LABELS))
all_test_preds = np.zeros((len(test_df), NUM_LABELS))
val_ll_list = []

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train_df["text"], train_df["label"])
):
    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
    trn_texts = train_df["text"].iloc[trn_idx].values
    val_texts = train_df["text"].iloc[val_idx].values
    trn_labels = train_df["label"].iloc[trn_idx].values
    val_labels = train_df["label"].iloc[val_idx].values

    trn_ds = SpookyDataset(trn_texts, trn_labels, tokenizer, trunc_strategy="first")
    val_ds = SpookyDataset(val_texts, val_labels, tokenizer, trunc_strategy="first")
    test_ds = SpookyDataset(
        test_df["text"].values, None, tokenizer, trunc_strategy="first"
    )

    trn_loader = DataLoader(
        trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(
        DEVICE
    )
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(trn_loader) // ACCUM_STEPS * EPOCHS
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    best_ll = float("inf")
    best_model_state = None
    patience = 2
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model, trn_loader, optimizer, scaler, scheduler, ACCUM_STEPS
        )
        val_ll, _ = evaluate(model, val_loader)
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val LL: {val_ll:.4f}"
        )
        if val_ll < best_ll:
            best_ll = val_ll
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
            torch.save(best_model_state, f"./working/deberta_fold{fold}.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f"./working/deberta_fold{fold}.pt"))
    model.to(DEVICE)
    _, val_probs = evaluate(model, val_loader)
    test_probs = predict_test(model, test_loader)

    all_oof[val_idx] = val_probs
    all_test_preds += test_probs / N_FOLDS
    val_ll_list.append(best_ll)
    print(f"Fold {fold+1} best LL: {best_ll:.4f}")
    del model, trn_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

print(f'\nDeBERTa CV OOF LL: {log_loss(train_df["label"], all_oof):.4f}')


# ---- XGBoost with stylometric + DeBERTa embeddings ----
# Extract pooled embeddings from DeBERTa for meta-features
def extract_embeddings(model, loader):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                _, pooled = model(input_ids, attention_mask)
            all_embs.append(pooled.cpu().numpy())
    return np.concatenate(all_embs)


# Re-load a model to extract embeddings for meta-features
meta_model = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(
    DEVICE
)
meta_model.load_state_dict(torch.load(f"./working/deberta_fold0.pt"))
meta_model.to(DEVICE)
meta_model.eval()

full_train_ds = SpookyDataset(
    train_df["text"].values, None, tokenizer, trunc_strategy="first"
)
full_train_loader = DataLoader(
    full_train_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2
)
train_embs = extract_embeddings(meta_model, full_train_loader)

full_test_ds = SpookyDataset(
    test_df["text"].values, None, tokenizer, trunc_strategy="first"
)
full_test_loader = DataLoader(
    full_test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2
)
test_embs = extract_embeddings(meta_model, full_test_loader)

del meta_model
gc.collect()
torch.cuda.empty_cache()

# Combine stylometric + embedding features
X_train_meta = np.concatenate([stylo_train, train_embs], axis=1)
X_test_meta = np.concatenate([stylo_test, test_embs], axis=1)
y_train_meta = train_df["label"].values

# XGBoost with meta features
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    objective="multi:softprob",
    num_class=NUM_LABELS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

# KFold CV for XGBoost
skf_xgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
xgb_oof = np.zeros((len(train_df), NUM_LABELS))
xgb_test = np.zeros((len(test_df), NUM_LABELS))
xgb_scores = []

for fold, (trn_idx, val_idx) in enumerate(skf_xgb.split(X_train_meta, y_train_meta)):
    print(f"XGB Fold {fold+1}/5")
    X_tr, X_val = X_train_meta[trn_idx], X_train_meta[val_idx]
    y_tr, y_val = y_train_meta[trn_idx], y_train_meta[val_idx]
    eval_set = [(X_val, y_val)]
    xgb_model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
    val_probs = xgb_model.predict_proba(X_val)
    xgb_oof[val_idx] = val_probs
    xgb_test += xgb_model.predict_proba(X_test_meta) / 5
    score = log_loss(y_val, val_probs)
    xgb_scores.append(score)
    print(f"XGB Fold {fold+1} LL: {score:.4f}")

print(f"XGB CV OOF LL: {log_loss(y_train_meta, xgb_oof):.4f}")

# ---- Logistic Regression on embeddings only ----
lr_model = LogisticRegression(
    C=1.0, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=SEED
)
lr_oof = np.zeros((len(train_df), NUM_LABELS))
lr_test = np.zeros((len(test_df), NUM_LABELS))

for fold, (trn_idx, val_idx) in enumerate(skf_xgb.split(train_embs, y_train_meta)):
    X_tr, X_val = train_embs[trn_idx], train_embs[val_idx]
    y_tr, y_val = y_train_meta[trn_idx], y_train_meta[val_idx]
    lr_model.fit(X_tr, y_tr)
    val_probs = lr_model.predict_proba(X_val)
    lr_oof[val_idx] = val_probs
    lr_test += lr_model.predict_proba(test_embs) / 5

print(f"LR CV OOF LL: {log_loss(y_train_meta, lr_oof):.4f}")

# ---- Learned Weighted Ensemble via Logistic Regression blender ----
# Use meta-learner to find optimal weights
stacked_train = np.stack([all_oof, xgb_oof, lr_oof], axis=-1)  # (N, 3, 3)
stacked_test = np.stack([all_test_preds, xgb_test, lr_test], axis=-1)  # (N_test, 3, 3)

# Flatten: each model's predictions become features
stacked_train_flat = stacked_train.reshape(len(train_df), -1)  # (N, 9)
stacked_test_flat = stacked_test.reshape(len(test_df), -1)

blender = LogisticRegression(
    C=0.1, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=SEED
)
blender_ll = 0

for fold, (trn_idx, val_idx) in enumerate(
    skf_xgb.split(stacked_train_flat, y_train_meta)
):
    X_tr, X_val = stacked_train_flat[trn_idx], stacked_train_flat[val_idx]
    y_tr, y_val = y_train_meta[trn_idx], y_train_meta[val_idx]
    blender.fit(X_tr, y_tr)
    val_probs = blender.predict_proba(X_val)
    blender_ll += log_loss(y_val, val_probs)

blender_ll /= 5
print(f"Blender CV OOF LL: {blender_ll:.4f}")

# Final blending
blender.fit(stacked_train_flat, y_train_meta)
final_preds = blender.predict_proba(stacked_test_flat)

final_validation_score = log_loss(
    y_train_meta, blender.predict_proba(stacked_train_flat)
)
print(f"Final Validation Score: {final_validation_score:.4f}")

# Generate submission
sub = pd.DataFrame({"id": test_df["id"].values})
for j, name in enumerate(["EAP", "HPL", "MWS"]):
    sub[name] = final_preds[:, j]
sub.to_csv(OUT + "submission.csv", index=False)

# Cleanup
for f in os.listdir("./working/"):
    if f.endswith(".pt"):
        os.remove(os.path.join("./working/", f))

print("Submission saved successfully!")