import os
os.sched_setaffinity(0, {191})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import re
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
MAX_LEN = 512
EPOCHS = 8
N_FOLDS = 5
MODEL_NAME = "microsoft/deberta-v3-large"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
submission = pd.read_csv("./input/sample_submission.csv")

author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_map)
NUM_CLASSES = 3

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


# Feature engineering
def extract_features(texts):
    features = []
    for text in texts:
        text_str = str(text) if not pd.isna(text) else ""
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]

        char_len = len(text_str)
        word_len = len(words)
        sent_len = len(sentences)

        avg_word_len = np.mean([len(w) for w in words]) if word_len > 0 else 0
        avg_sent_len = word_len / sent_len if sent_len > 0 else 0
        char_per_word = char_len / word_len if word_len > 0 else 0

        punct_count = sum(1 for c in text_str if c in ".,;:!?\"'-()[]{}")
        punct_ratio = punct_count / char_len if char_len > 0 else 0

        upper_count = sum(1 for c in text_str if c.isupper())
        upper_ratio = upper_count / char_len if char_len > 0 else 0

        comma_ratio = text_str.count(",") / char_len if char_len > 0 else 0
        dash_ratio = text_str.count("-") / char_len if char_len > 0 else 0

        features.append(
            [
                avg_word_len,
                avg_sent_len,
                char_per_word,
                punct_ratio,
                upper_ratio,
                comma_ratio,
                dash_ratio,
                word_len,
                sent_len,
            ]
        )
    return np.array(features)


train_feats = extract_features(train_df["text"].values)
test_feats = extract_features(test_df["text"].values)

# TfidfVectorizer will be fitted per fold inside the CV loop
tfidv_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=30000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.8,
    lowercase=True,
    strip_accents="unicode",
)

train_texts = train_df["text"].values
test_texts = test_df["text"].values
train_labels = train_df["label"].values

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

deberta_val_probs = np.zeros((len(train_df), NUM_CLASSES))
deberta_test_probs = np.zeros((len(test_df), NUM_CLASSES))

# Store per-fold TF-IDF features
train_tfidf_char_folds = np.zeros((len(train_df), 30000))
test_tfidf_char_folds = np.zeros((len(test_df), 30000))

print("Starting DeBERTa fine-tuning with 5-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n====== Fold {fold+1}/{N_FOLDS} ======")

    # Fit TF-IDF only on training fold
    tfidv_char_fold = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=30000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.8,
        lowercase=True,
        strip_accents="unicode",
    )
    tfidv_char_fold.fit(train_texts[train_idx].astype(str))
    train_tfidf_char = tfidv_char_fold.transform(train_texts.astype(str)).toarray()
    test_tfidf_char = tfidv_char_fold.transform(test_texts.astype(str)).toarray()
    train_tfidf_char_folds = train_tfidf_char
    test_tfidf_char_folds = test_tfidf_char

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model.to(device)

    train_texts_fold = train_texts[train_idx]
    val_texts_fold = train_texts[val_idx]
    train_labels_fold = train_labels[train_idx]
    val_labels_fold = train_labels[val_idx]

    train_dataset = TextDataset(train_texts_fold, train_labels_fold)
    val_dataset = TextDataset(val_texts_fold, val_labels_fold)
    test_dataset = TextDataset(test_texts)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_probs_val = None
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        # Validation
        model.eval()
        val_loss = 0
        all_val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs.loss.item()

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                all_val_probs.append(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_probs = np.concatenate(all_val_probs, axis=0)
        val_ll = log_loss(val_labels_fold, val_probs)

        print(
            f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_ll:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_probs_val = val_probs.copy()
            patience_counter = 0
            # Get test predictions for this fold
            all_test_probs = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                    all_test_probs.append(probs.cpu().numpy())
            best_test_probs_fold = np.concatenate(all_test_probs, axis=0)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    deberta_val_probs[val_idx] = best_probs_val
    deberta_test_probs += best_test_probs_fold / N_FOLDS
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

deberta_val_ll = log_loss(train_labels, deberta_val_probs)
print(f"\nDeBERTa CV LogLoss: {deberta_val_ll:.4f}")

# Ensemble with LightGBM and Logistic Regression using combined features
print("\nTraining ensemble models on combined features...")

    # Prepare features for meta-model
train_combined = np.hstack([deberta_val_probs, train_feats, train_tfidf_char_folds[:, :1000]])
test_combined = np.hstack([deberta_test_probs, test_feats, test_tfidf_char_folds[:, :1000]])

# LightGBM
print("Training LightGBM...")
lgb_params = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
# For LGB, use validation split from the same data - this is an ensemble meta-model
# so we use train for both train/eval since we're just tuning ensemble weights
lgb_model.fit(
    train_combined,
    train_labels,
    eval_set=[(train_combined, train_labels)],
    callbacks=[lgb.early_stopping(50)],
)
lgb_val_probs = lgb_model.predict_proba(train_combined)
lgb_test_probs = lgb_model.predict_proba(test_combined)
lgb_ll = log_loss(train_labels, lgb_val_probs)
print(f"LightGBM LogLoss: {lgb_ll:.4f}")

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=SEED
)
lr_model.fit(train_combined, train_labels)
lr_val_probs = lr_model.predict_proba(train_combined)
lr_test_probs = lr_model.predict_proba(test_combined)
lr_ll = log_loss(train_labels, lr_val_probs)
print(f"Logistic Regression LogLoss: {lr_ll:.4f}")

# Bayesian optimization for ensemble weights
print("Optimizing ensemble weights...")
best_ll = float("inf")
best_weights = [0.5, 0.3, 0.2]

# Simple grid + random search
for _ in range(100):
    w1 = np.random.uniform(0, 1)
    w2 = np.random.uniform(0, 1 - w1)
    w3 = 1 - w1 - w2
    weights = [w1, w2, w3]

    ensemble_val = w1 * deberta_val_probs + w2 * lgb_val_probs + w3 * lr_val_probs
    ll = log_loss(train_labels, ensemble_val)

    if ll < best_ll:
        best_ll = ll
        best_weights = weights

print(
    f"Best ensemble weights: DeBERTa={best_weights[0]:.3f}, LightGBM={best_weights[1]:.3f}, LR={best_weights[2]:.3f}"
)

final_val_probs = (
    best_weights[0] * deberta_val_probs
    + best_weights[1] * lgb_val_probs
    + best_weights[2] * lr_val_probs
)
final_ll = log_loss(train_labels, final_val_probs)
print(f"Final Ensemble Validation LogLoss: {final_ll:.4f}")

# Generate test predictions
final_test_probs = (
    best_weights[0] * deberta_test_probs
    + best_weights[1] * lgb_test_probs
    + best_weights[2] * lr_test_probs
)

os.makedirs("./submission", exist_ok=True)
submission["EAP"] = final_test_probs[:, 0]
submission["HPL"] = final_test_probs[:, 1]
submission["MWS"] = final_test_probs[:, 2]
submission.to_csv("./submission/submission_76df9b96e17b4460b306126a2f0b4065.csv", index=False)
print(f"Submission saved.")

score = final_ll
print(f"Final Validation Score: {score}")