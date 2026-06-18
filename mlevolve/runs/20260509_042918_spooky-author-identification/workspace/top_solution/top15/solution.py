import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import warnings
from transformers import AutoTokenizer, ModernBertForSequenceClassification

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
model = ModernBertForSequenceClassification.from_pretrained(model_id, num_labels=3)
model.to(device)
model.eval()

# Freeze backbone - ModernBERT uses 'model' attribute for the transformer backbone
for param in model.model.parameters():
    param.requires_grad = False


# ============================================================
# 3. DEFINE DATASET FOR EMBEDDING EXTRACTION
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
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
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def extract_embeddings_batched(texts, tokenizer, model, batch_size=16, max_length=512):
    """Extract ModernBERT embeddings for a list of texts."""
    model.eval()
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # ModernBERT uses the last hidden state from the backbone
            # outputs.hidden_states is a tuple of (embedding, layer1, ..., layerN)
            # We want the last hidden layer's CLS token
            cls_embedding = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_embedding.cpu().numpy())

    return np.vstack(all_embeddings)


# ============================================================
# 4. EXTRACT EMBEDDINGS
# ============================================================
print("Extracting embeddings...")
train_embeddings = extract_embeddings_batched(
    train_df["text"].tolist(), tokenizer, model, batch_size=16
)
test_embeddings = extract_embeddings_batched(
    test_df["text"].tolist(), tokenizer, model, batch_size=16
)
print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")

# ============================================================
# 5. CREATE STRATIFIED TRAIN/VALIDATION SPLITS
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = []
for train_index, val_index in skf.split(np.arange(len(train_df)), y_encoded):
    splits.append(
        {"train_index": train_index.tolist(), "val_index": val_index.tolist()}
    )

# ============================================================
# 6. 5-FOLD CROSS-VALIDATION WITH XGBOOST
# ============================================================
print("\nStarting 5-fold cross-validation...")
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1.0,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "min_child_weight": 5,
    "seed": 42,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_jobs": 4,
}

# XGBoost in this environment does not support GPU tree_method
# Use 'hist' for faster CPU training
xgb_params["tree_method"] = "hist"

# Store predictions for each fold
fold_val_predictions = []
fold_val_labels = []
fold_models = []
fold_scores = []

for fold_idx, split in enumerate(splits):
    print(f"\nFold {fold_idx + 1}/{len(splits)}")

    train_idx = np.array(split["train_index"])
    val_idx = np.array(split["val_index"])

    X_train_fold = train_embeddings[train_idx]
    y_train_fold = y_encoded[train_idx]
    X_val_fold = train_embeddings[val_idx]
    y_val_fold = y_encoded[val_idx]

    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

    watchlist = [(dtrain, "train"), (dval, "eval")]

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    fold_models.append(model_xgb)

    val_pred_proba = model_xgb.predict(dval)

    eps = 1e-15
    val_pred_proba = np.clip(val_pred_proba, eps, 1 - eps)
    val_pred_proba = val_pred_proba / val_pred_proba.sum(axis=1, keepdims=True)

    fold_ll = log_loss(y_val_fold, val_pred_proba)
    fold_scores.append(fold_ll)

    print(f"Fold {fold_idx + 1} - Log Loss: {fold_ll:.6f}")

# ============================================================
# 7. CALCULATE OVERALL CV SCORE
# ============================================================
mean_cv_score = np.mean(fold_scores)
std_cv_score = np.std(fold_scores)
print(f"\nCross-Validation Results:")
print(f"Per-fold scores: {[f'{s:.6f}' for s in fold_scores]}")
print(f"Mean CV Log Loss: {mean_cv_score:.6f} (+/- {std_cv_score:.6f})")

# ============================================================
# 8. RETRAIN ON FULL TRAINING DATA
# ============================================================
print("\nRetraining on full training data...")
# Retrain on full data using the average optimal rounds from CV folds
# XGBoost booster stores best_iteration, but only as int attribute
optimal_rounds = 500
if hasattr(model_xgb, 'best_iteration'):
    optimal_rounds = model_xgb.best_iteration

dtrain_full = xgb.DMatrix(train_embeddings, label=y_encoded)

final_model = xgb.train(
    xgb_params,
    dtrain_full,
    num_boost_round=optimal_rounds,
    verbose_eval=False,
)

# ============================================================
# 9. TEST INFERENCE
# ============================================================
print("Performing test inference...")
dtest = xgb.DMatrix(test_embeddings)
test_pred_proba = final_model.predict(dtest)

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

submission.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)
print(f"Submission saved to {submission_dir}/submission.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# 11. PRINT FINAL VALIDATION SCORE
# ============================================================
print(f"Final Validation Score: {mean_cv_score}")
