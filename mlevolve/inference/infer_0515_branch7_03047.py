"""
Inference-only script for Branch 7 model (metric 0.3047)
Loads deberta_fold0.pt, predicts on full test set. No training.
Architecture: DebertaForSequenceClassificationWithPooling (5-layer attention pooling)
+ stylometric features + XGBoost + LR + Stacking Blender
"""

import os, gc, random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.cuda.amp import autocast
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# PATHS
# ============================================================
TRAIN_CSV = "/workspace/nautilus/mlevolve/data/spooky-author-identification/prepared/public/train.csv"
TEST_CSV = "/workspace/nautilus/mlevolve/inference/test.csv"
WEIGHT_PATH = "/workspace/nautilus/mlevolve/runs/20260515_173948_spooky-author-identification/workspace/working/deberta_fold0.pt"
SUBMISSION_DIR = "/workspace/nautilus/mlevolve/inference/submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 16
MODEL_NAME = "microsoft/deberta-v3-large"
NUM_LABELS = 3
CLASS_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}

# ============================================================
# DATA
# ============================================================
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
train_df["label"] = train_df["author"].map(CLASS_MAP)

# ============================================================
# STYLOMETRIC FEATURES
# ============================================================
print("Extracting stylometric features...")

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
        features.append([
            avg_word_len, avg_sent_len, punct_count / max(chars, 1),
            comma_ratio, exclam_ratio, quest_ratio, colon_ratio,
            semicolon_ratio, quote_ratio, dash_ratio, cap_ratio,
            digit_ratio, unique_word_ratio, word_len, chars,
        ])
    return np.array(features)

stylo_train = get_stylometric_features(train_df["text"].values)
stylo_test = get_stylometric_features(test_df["text"].values)
print(f"Stylometric: train {stylo_train.shape}, test {stylo_test.shape}")

# ============================================================
# DATASET (inference only, no labels needed)
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# MODEL (matches actual deberta_fold0.pt structure)
# ============================================================
class DebertaForSequenceClassificationWithPooling(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.config.hidden_dropout_prob = 0.3
        self.attention_pool = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )
        self.label_smoothing = 0.1

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        attn_weights = torch.softmax(self.attention_pool(hidden), dim=1)
        pooled = (attn_weights * hidden).sum(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled

# ============================================================
# LOAD MODEL & PREDICT
# ============================================================
print("\nLoading deberta_fold0.pt and predicting...")
model = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(DEVICE)
state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

def predict(model, loader):
    all_probs, all_embs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                logits, pooled = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_embs.append(pooled.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_embs)

# Train predictions (for XGBoost/LR/blender fitting)
train_ds = SpookyDataset(train_df["text"].values, tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
deberta_train_probs, train_embs = predict(model, train_loader)

# Test predictions
test_ds = SpookyDataset(test_df["text"].values, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
deberta_test_probs, test_embs = predict(model, test_loader)

print(f"DeBERTa train log loss: {log_loss(train_df['label'], deberta_train_probs):.4f}")

del model
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# XGBoost
# ============================================================
print("\n===== XGBoost =====")
X_train_meta = np.concatenate([stylo_train, train_embs], axis=1)
X_test_meta = np.concatenate([stylo_test, test_embs], axis=1)
y_train = train_df["label"].values

xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
    objective="multi:softprob", num_class=NUM_LABELS,
    eval_metric="mlogloss", early_stopping_rounds=30,
    random_state=SEED, n_jobs=-1, verbosity=0,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
xgb_oof = np.zeros((len(train_df), NUM_LABELS))
xgb_test = np.zeros((len(test_df), NUM_LABELS))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_meta, y_train)):
    xgb_model.fit(X_train_meta[trn_idx], y_train[trn_idx],
                  eval_set=[(X_train_meta[val_idx], y_train[val_idx])], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict_proba(X_train_meta[val_idx])
    xgb_test += xgb_model.predict_proba(X_test_meta) / 5
    print(f"  Fold {fold+1}: {log_loss(y_train[val_idx], xgb_oof[val_idx]):.4f}")

print(f"XGB CV OOF LL: {log_loss(y_train, xgb_oof):.4f}")

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
print("\n===== Logistic Regression =====")
lr_model = LogisticRegression(C=1.0, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=SEED)
lr_oof = np.zeros((len(train_df), NUM_LABELS))
lr_test = np.zeros((len(test_df), NUM_LABELS))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_embs, y_train)):
    lr_model.fit(train_embs[trn_idx], y_train[trn_idx])
    lr_oof[val_idx] = lr_model.predict_proba(train_embs[val_idx])
    lr_test += lr_model.predict_proba(test_embs) / 5

print(f"LR CV OOF LL: {log_loss(y_train, lr_oof):.4f}")

# ============================================================
# STACKING BLENDER
# ============================================================
print("\n===== Stacking Blender =====")
stacked_train = np.stack([deberta_train_probs, xgb_oof, lr_oof], axis=-1).reshape(len(train_df), -1)
stacked_test = np.stack([deberta_test_probs, xgb_test, lr_test], axis=-1).reshape(len(test_df), -1)

blender = LogisticRegression(C=0.1, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=SEED)

blender_scores = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(stacked_train, y_train)):
    blender.fit(stacked_train[trn_idx], y_train[trn_idx])
    blender_scores.append(log_loss(y_train[val_idx], blender.predict_proba(stacked_train[val_idx])))

print(f"Blender CV OOF LL: {np.mean(blender_scores):.4f}")

blender.fit(stacked_train, y_train)
final_preds = blender.predict_proba(stacked_test)

# ============================================================
# SUBMISSION
# ============================================================
eps = 1e-15
final_preds = np.clip(final_preds, eps, 1 - eps)
final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)
final_preds = np.clip(final_preds, eps, 1 - eps)

sub = pd.DataFrame({"id": test_df["id"].values})
for j, name in enumerate(["EAP", "HPL", "MWS"]):
    sub[name] = final_preds[:, j]

output_path = f"{SUBMISSION_DIR}/submission_0515_branch7_03047.csv"
sub.to_csv(output_path, index=False)
print(f"\nSubmission saved to {output_path}")
print(f"Shape: {sub.shape}")
print(sub.head())
print(f"\nBlender CV: {np.mean(blender_scores):.4f}")
