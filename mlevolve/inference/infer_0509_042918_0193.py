"""
Run 20260509_042918 Train+Inference Script
LogLoss: ~0.193 (真实 log_loss, 无 INDEX_BUG)
模型: 3个冻结Transformer (deberta-v3-base, roberta-base, distilbert-base-uncased) 提取[CLS]嵌入
      + XGBoost 5折交叉验证集成

用法: python infer_0509_042918_0193.py
"""

import numpy as np
import pandas as pd
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb

# ============================================================
# 路径配置
# ============================================================
INFERENCE_DIR = "/workspace/nautilus/mlevolve/inference"
DATA_DIR = "/workspace/nautilus/mlevolve/data/spooky-author-identification/prepared/public"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{INFERENCE_DIR}/test.csv"
OUTPUT_CSV = f"{INFERENCE_DIR}/submissions/run_0509_042918_logloss_0193_full_8392.csv"
WORKING_DIR = f"{INFERENCE_DIR}/working_0509_042918"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(f"{INFERENCE_DIR}/submissions", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = train_df["author"].map(author_mapping).values

# Define complementary frozen models for feature extraction
model_ids = [
    "microsoft/deberta-v3-base",
    "roberta-base",
    "distilbert-base-uncased",
]

max_length = 256

# Load tokenizers and models (frozen)
tokenizers = {}
models = {}
for model_id in model_ids:
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizers[model_id] = tokenizer
    models[model_id] = model
    print(f"  Loaded {model_id} with {sum(p.numel() for p in model.parameters()):,} parameters (frozen)")


def extract_embeddings(texts, tokenizer, model, batch_size=32):
    """Extract [CLS] token embeddings from a frozen transformer model."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist() if hasattr(texts, 'tolist') else texts[i:i+batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        del input_ids, attention_mask, outputs, cls_embeddings
        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


print("\nExtracting features from training texts...")
train_texts = train_df["text"].values
all_train_features = []
for model_id in model_ids:
    print(f"  Extracting {model_id} features...")
    emb = extract_embeddings(train_texts, tokenizers[model_id], models[model_id])
    all_train_features.append(emb)
    print(f"    Shape: {emb.shape}")

X_train = np.concatenate(all_train_features, axis=1)
print(f"Combined training feature shape: {X_train.shape}")

print("\nExtracting features from test texts...")
test_texts = test_df["text"].values
all_test_features = []
for model_id in model_ids:
    print(f"  Extracting {model_id} features...")
    emb = extract_embeddings(test_texts, tokenizers[model_id], models[model_id])
    all_test_features.append(emb)
    print(f"    Shape: {emb.shape}")

X_test = np.concatenate(all_test_features, axis=1)
print(f"Combined test feature shape: {X_test.shape}")

# Free up GPU memory
del models, tokenizers
gc.collect()
torch.cuda.empty_cache()

# Stratified 5-fold cross-validation with XGBoost
print("\nStarting stratified 5-fold cross-validation with XGBoost...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_val_scores = []
fold_oof_preds = np.zeros((len(X_train), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_labels)):
    print(f"\n--- Fold {fold+1}/5 ---")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = train_labels[train_idx], train_labels[val_idx]

    X_tr_train, X_tr_val, y_tr_train, y_tr_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        n_estimators=2000,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        early_stopping_rounds=20,
    )

    xgb_model.fit(
        X_tr_train, y_tr_train,
        eval_set=[(X_tr_val, y_tr_val)],
        verbose=False,
    )

    val_preds = xgb_model.predict_proba(X_val)
    fold_oof_preds[val_idx] = val_preds

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

    fold_log_loss = log_loss(y_val, val_preds_clipped)
    fold_val_scores.append(fold_log_loss)

    print(f"Fold {fold+1} Validation Log Loss: {fold_log_loss:.6f}")

    del X_tr, X_val, y_tr, y_val, X_tr_train, X_tr_val, y_tr_train, y_tr_val, xgb_model
    gc.collect()

print(f"\nCross-validation results:")
print(f"  Per-fold log losses: {[f'{s:.6f}' for s in fold_val_scores]}")
print(f"  Mean CV Log Loss: {np.mean(fold_val_scores):.6f}")
print(f"  Std CV Log Loss: {np.std(fold_val_scores):.6f}")

eps = 1e-15
oof_preds_clipped = np.clip(fold_oof_preds, eps, 1 - eps)
oof_preds_clipped = oof_preds_clipped / oof_preds_clipped.sum(axis=1, keepdims=True)
oof_log_loss = log_loss(train_labels, oof_preds_clipped)
print(f"Overall OOF Log Loss: {oof_log_loss:.6f}")

# Train final XGBoost model on all training data
print("\n--- Training final XGBoost model on all training data ---")

X_train_full, X_val_final, y_train_full, y_val_final = train_test_split(
    X_train, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

final_xgb = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    n_estimators=2000,
    eval_metric="mlogloss",
    use_label_encoder=False,
    verbosity=0,
    random_state=42,
    early_stopping_rounds=20,
)

final_xgb.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val_final, y_val_final)],
    verbose=False,
)

final_val_preds = final_xgb.predict_proba(X_val_final)
final_val_preds_clipped = np.clip(final_val_preds, eps, 1 - eps)
final_val_preds_clipped = final_val_preds_clipped / final_val_preds_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(y_val_final, final_val_preds_clipped)
print(f"Final held-out validation Log Loss: {final_val_score:.6f}")

best_n = getattr(final_xgb, "best_ntree_limit", final_xgb.best_iteration + 1)
print(f"Best iteration: {best_n}")

final_xgb_full = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    n_estimators=best_n,
    eval_metric="mlogloss",
    use_label_encoder=False,
    verbosity=0,
    random_state=42,
)

final_xgb_full.fit(X_train, train_labels)

test_preds = final_xgb_full.predict_proba(X_test)

test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSubmission saved to {OUTPUT_CSV} with {len(submission_df)} rows")
print("Sample predictions:")
print(submission_df.head())

del X_train, X_test, final_xgb, final_xgb_full
gc.collect()
torch.cuda.empty_cache()
