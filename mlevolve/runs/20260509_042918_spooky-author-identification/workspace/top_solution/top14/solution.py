import numpy as np
import pandas as pd
import torch
import gc
import os
import re
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
from torch.cuda.amp import autocast
import warnings
import textstat

warnings.filterwarnings("ignore")

# Load data
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

X_train_raw = train_df["text"].values
y_train_raw = train_df["author"].values
X_test_raw = test_df["text"].values
test_ids = test_df["id"].values

# Encode labels
label_enc = LabelEncoder()
y_train_encoded = label_enc.fit_transform(y_train_raw)
num_classes = len(label_enc.classes_)

# Stratified train/val split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_raw,
    y_train_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_train_encoded,
)

print(
    f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test_raw)}"
)

# =============================================
# Feature Engineering Functions (from Step 1)
# =============================================


def extract_stylometric_features(text_series):
    features = []
    for text in text_series:
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        char_count = len(text_str)
        word_count = len(words)
        sent_count = max(len(sentences), 1)
        punctuation = sum(1 for c in text_str if c in ".,;:!?\"'()[]-")
        punct_density = punctuation / max(char_count, 1)
        caps_count = sum(1 for c in text_str if c.isupper())
        caps_ratio = caps_count / max(char_count, 1)
        digit_count = sum(1 for c in text_str if c.isdigit())
        digit_ratio = digit_count / max(char_count, 1)
        word_lengths = [len(w) for w in words] if words else [0]
        avg_word_len = np.mean(word_lengths)
        max_word_len = max(word_lengths)
        long_words_ratio = sum(1 for w in words if len(w) > 6) / max(word_count, 1)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = np.mean(sent_lengths)
        max_sent_len = max(sent_lengths)
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        unique_words = len(set(w.lower() for w in words))
        unique_ratio = unique_words / max(word_count, 1)
        try:
            fk_grade = textstat.flesch_kincaid_grade(text_str)
        except:
            fk_grade = 0
        try:
            coleman_liau = textstat.coleman_liau_index(text_str)
        except:
            coleman_liau = 0
        features.append(
            [
                punct_density,
                caps_ratio,
                digit_ratio,
                avg_word_len,
                max_word_len,
                long_words_ratio,
                avg_sent_len,
                max_sent_len,
                sent_len_std,
                unique_ratio,
                fk_grade,
                coleman_liau,
            ]
        )
    return np.array(features)


def extract_pos_style_features(text_series):
    features = []
    for text in text_series:
        text_str = str(text)
        words = text_str.split()
        word_count = max(len(words), 1)
        pronoun_pattern = r"\b(I|you|he|she|it|we|they|me|him|her|us|them|my|your|his|her|its|our|their|mine|yours|hers|ours|theirs)\b"
        pronouns = len(re.findall(pronoun_pattern, text_str, re.IGNORECASE))
        pronoun_ratio = pronouns / word_count
        article_pattern = r"\b(a|an|the)\b"
        articles = len(re.findall(article_pattern, text_str, re.IGNORECASE))
        article_ratio = articles / word_count
        prep_pattern = r"\b(in|on|at|to|for|with|by|from|of|about|into|through|during|before|after|above|below|between|under|over)\b"
        prepositions = len(re.findall(prep_pattern, text_str, re.IGNORECASE))
        prep_ratio = prepositions / word_count
        conj_pattern = r"\b(and|but|or|nor|yet|so|for|because|although|while|since|unless|if|when|where|whether)\b"
        conjunctions = len(re.findall(conj_pattern, text_str, re.IGNORECASE))
        conj_ratio = conjunctions / word_count
        adverbs = sum(1 for w in words if w.endswith("ly"))
        adv_ratio = adverbs / word_count
        past_tense = sum(1 for w in words if w.endswith("ed") and len(w) > 3)
        present_part = sum(1 for w in words if w.endswith("ing") and len(w) > 4)
        verb_marker_ratio = (past_tense + present_part) / word_count
        question_marks = text_str.count("?")
        exclam_marks = text_str.count("!")
        punctuation_markers = (question_marks + exclam_marks) / max(len(text_str), 1)
        ellipsis = text_str.count("...") + text_str.count("--")
        punctuation_density = ellipsis / max(len(text_str), 1)
        features.append(
            [
                pronoun_ratio,
                article_ratio,
                prep_ratio,
                conj_ratio,
                adv_ratio,
                verb_marker_ratio,
                punctuation_markers,
                punctuation_density,
            ]
        )
    return np.array(features)


def create_tfidf_features(train_texts, val_texts, test_texts, max_features=2000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        analyzer="char_wb",
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
    )
    tfidf_train = vectorizer.fit_transform(train_texts)
    tfidf_val = vectorizer.transform(val_texts)
    tfidf_test = vectorizer.transform(test_texts)
    return tfidf_train, tfidf_val, tfidf_test, vectorizer


def create_word_ngram_features(train_texts, val_texts, test_texts, max_features=1500):
    vectorizer = CountVectorizer(
        max_features=max_features, ngram_range=(1, 4), min_df=3, max_df=0.95
    )
    ngram_train = vectorizer.fit_transform(train_texts)
    ngram_val = vectorizer.transform(val_texts)
    ngram_test = vectorizer.transform(test_texts)
    return ngram_train, ngram_val, ngram_test, vectorizer


# Extract classical features
print("Extracting classical features...")
styl_train = extract_stylometric_features(X_train)
styl_val = extract_stylometric_features(X_val)
styl_test = extract_stylometric_features(X_test_raw)

pos_train = extract_pos_style_features(X_train)
pos_val = extract_pos_style_features(X_val)
pos_test = extract_pos_style_features(X_test_raw)

char_tfidf_train, char_tfidf_val, char_tfidf_test, _ = create_tfidf_features(
    X_train, X_val, X_test_raw, max_features=2000
)
word_ngram_train, word_ngram_val, word_ngram_test, _ = create_word_ngram_features(
    X_train, X_val, X_test_raw, max_features=1500
)

# Scale features
scaler = StandardScaler()
styl_train_scaled = scaler.fit_transform(styl_train)
styl_val_scaled = scaler.transform(styl_val)
styl_test_scaled = scaler.transform(styl_test)

pos_scaler = StandardScaler()
pos_train_scaled = pos_scaler.fit_transform(pos_train)
pos_val_scaled = pos_scaler.transform(pos_val)
pos_test_scaled = pos_scaler.transform(pos_test)

X_train_features = hstack(
    [
        csr_matrix(styl_train_scaled),
        csr_matrix(pos_train_scaled),
        char_tfidf_train,
        word_ngram_train,
    ]
)
X_val_features = hstack(
    [
        csr_matrix(styl_val_scaled),
        csr_matrix(pos_val_scaled),
        char_tfidf_val,
        word_ngram_val,
    ]
)
X_test_features = hstack(
    [
        csr_matrix(styl_test_scaled),
        csr_matrix(pos_test_scaled),
        char_tfidf_test,
        word_ngram_test,
    ]
)

print(
    f"Classical features shape - Train: {X_train_features.shape}, Val: {X_val_features.shape}, Test: {X_test_features.shape}"
)

# =============================================
# Smaller Transformer Feature Extraction (distilbert)
# =============================================
print("Loading distilbert-base-uncased...")
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.cuda()
model.eval()
for param in model.parameters():
    param.requires_grad = False
print(
    f"distilbert-base-uncased loaded. Parameters frozen: {sum(p.numel() for p in model.parameters()):,}"
)


def extract_embeddings(texts, batch_size=8, max_length=256):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            (
                batch_texts.tolist()
                if hasattr(batch_texts, "tolist")
                else list(batch_texts)
            ),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
        # Aggressively free memory
        del inputs, outputs, cls_emb
        torch.cuda.empty_cache()
    result = np.vstack(embeddings)
    gc.collect()
    torch.cuda.empty_cache()
    return result


print("Extracting transformer embeddings...")
train_emb = extract_embeddings(X_train)
gc.collect()
torch.cuda.empty_cache()
val_emb = extract_embeddings(X_val)
gc.collect()
torch.cuda.empty_cache()
test_emb = extract_embeddings(X_test_raw)
gc.collect()
torch.cuda.empty_cache()

# Convert sparse classical features to dense numpy for stacking
X_train_dense = X_train_features.toarray()
X_val_dense = X_val_features.toarray()
X_test_dense = X_test_features.toarray()

# Combine all features
X_train_combined = np.hstack([train_emb, X_train_dense])
X_val_combined = np.hstack([val_emb, X_val_dense])
X_test_combined = np.hstack([test_emb, X_test_dense])

print(
    f"Combined feature shapes - Train: {X_train_combined.shape}, Val: {X_val_combined.shape}, Test: {X_test_combined.shape}"
)

# Free memory
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# =============================================
# XGBoost Training (Two-Stage with Hard Negative Mining)
# =============================================
# Free memory before XGBoost
gc.collect()
torch.cuda.empty_cache()
print("Setting up XGBoost...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
val_scores = []
val_preds_list = []
val_true_list = []

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.1,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

print("Training XGBoost with cross-validation (using combined train data)...")
all_train_combined = np.vstack([X_train_combined, X_val_combined])
all_y = np.concatenate([y_train, y_val])

for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_combined, all_y)):
    X_tr, X_val_fold = all_train_combined[train_idx], all_train_combined[val_idx]
    y_tr, y_val_fold = all_y[train_idx], all_y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred = model_xgb.predict(dval)
    fold_score = log_loss(y_val_fold, val_pred)
    val_scores.append(fold_score)
    val_preds_list.append(val_pred)
    val_true_list.append(y_val_fold)
    print(f"Fold {fold+1}/{n_splits} - Log Loss: {fold_score:.4f}")

all_val_preds = np.vstack(val_preds_list)
all_val_true = np.concatenate(val_true_list)
overall_val_score = log_loss(all_val_true, all_val_preds)
print(
    f"Cross-Validation Log Loss: {overall_val_score:.4f} (mean: {np.mean(val_scores):.4f})"
)

# Stage 1: Retrain on full data
print("Training Stage 1 XGBoost on full training data...")
dtrain_full = xgb.DMatrix(all_train_combined, label=all_y)
stage1_model = xgb.train(
    xgb_params, dtrain_full, num_boost_round=500, verbose_eval=False
)

# Stage 2: Hard Negative Mining (using Out-of-Fold predictions)
print("Stage 2: Hard Negative Mining...")
# For each sample, get predicted probability for the true class from OOF predictions
true_probs = all_val_preds[np.arange(len(all_val_true)), all_val_true]
# Identify hard examples (bottom 20% by predicted probability for the true class)
threshold = np.percentile(true_probs, 20)
hard_mask = true_probs < threshold
print(f"Hard examples identified: {hard_mask.sum()} out of {len(true_probs)}")

# Create sample weights: weight=2 for hard examples, weight=1 for others
sample_weights = np.ones(len(all_y))
sample_weights[hard_mask] = 2.0

# Retrain a second XGBoost model using instance weights on full training data
print("Training Stage 2 XGBoost with hard negative weights...")
dtrain_weighted = xgb.DMatrix(all_train_combined, label=all_y, weight=sample_weights)
stage2_model = xgb.train(
    xgb_params, dtrain_weighted, num_boost_round=500, verbose_eval=False
)

# Ensemble: average predictions from both models
print("Generating test predictions (ensemble)...")
dtest = xgb.DMatrix(X_test_combined)
stage1_probs = stage1_model.predict(dtest)
stage2_probs = stage2_model.predict(dtest)
test_probs = (stage1_probs + stage2_probs) / 2.0

# Create submission
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

# Normalize probabilities
row_sums = submission_df[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission_df[col] = submission_df[col] / row_sums

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")

print(f"Final Validation Score: {overall_val_score:.6f}")