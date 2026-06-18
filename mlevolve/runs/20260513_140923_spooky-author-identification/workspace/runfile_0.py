import os
os.sched_setaffinity(0, {120, 122, 123, 124, 125})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

NUM_AUTHORS = 3
MAX_LENGTH = 256
NUM_EPOCHS = 30
PATIENCE = 5
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)


def create_stylometric_features(df):
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["avg_word_len"] = df["char_count"] / (df["word_count"] + 1)
    df["sentence_count"] = df["text"].str.count("[.!?]") + 1
    df["avg_sentence_len"] = df["word_count"] / (df["sentence_count"] + 1)

    punct_counts = df["text"].apply(
        lambda x: pd.Series(
            {
                "comma_count": x.count(","),
                "semicolon_count": x.count(";"),
                "colon_count": x.count(":"),
                "dash_count": x.count("—") + x.count("-"),
                "exclamation_count": x.count("!"),
                "question_count": x.count("?"),
                "quote_count": x.count('"') + x.count("'"),
                "parenthesis_count": x.count("(") + x.count(")"),
                "period_count": x.count("."),
                "ellipsis_count": x.count("..."),
            }
        )
    )
    df = pd.concat([df, punct_counts], axis=1)

    df["punct_density"] = (
        df["comma_count"] + df["semicolon_count"] + df["colon_count"] + df["dash_count"]
    ) / (df["word_count"] + 1)
    df["stop_punct_ratio"] = (
        df["period_count"] + df["exclamation_count"] + df["question_count"]
    ) / (df["word_count"] + 1)
    df["complex_punct_ratio"] = (
        df["semicolon_count"] + df["colon_count"] + df["dash_count"]
    ) / (df["sentence_count"] + 1)
    df["capital_words"] = df["text"].str.findall(r"\b[A-Z][a-z]+\b").str.len()
    df["all_caps_words"] = df["text"].str.findall(r"\b[A-Z]{2,}\b").str.len()
    df["capital_ratio"] = df["capital_words"] / (df["word_count"] + 1)
    df["proper_noun_density"] = df["capital_words"] / (df["sentence_count"] + 1)

    archaic_words = [
        "thee",
        "thou",
        "thy",
        "thine",
        "hath",
        "doth",
        "whence",
        "thence",
        "hither",
        "thither",
        "wherefore",
        "therein",
        "wherein",
    ]
    df["archaic_word_count"] = (
        df["text"]
        .str.lower()
        .apply(lambda x: sum(1 for w in archaic_words if w in x.split()))
    )

    intensity_words = [
        "horror",
        "terror",
        "dread",
        "fear",
        "awful",
        "hideous",
        "ghastly",
        "monstrous",
        "unspeakable",
        "unutterable",
        "eldritch",
    ]
    df["intensity_word_count"] = (
        df["text"]
        .str.lower()
        .apply(lambda x: sum(1 for w in intensity_words if w in x.split()))
    )

    adjective_indicators = ["ing", "ous", "ive", "ful", "less", "able", "ible"]
    df["suffix_adj_count"] = (
        df["text"]
        .str.lower()
        .apply(
            lambda x: sum(
                1 for w in x.split() if any(w.endswith(s) for s in adjective_indicators)
            )
        )
    )

    subordinating_conjunctions = [
        "although",
        "because",
        "since",
        "while",
        "whereas",
        "unless",
        "whenever",
        "wherever",
        "before",
        "after",
    ]
    df["complex_sentence_markers"] = (
        df["text"]
        .str.lower()
        .apply(lambda x: sum(1 for w in subordinating_conjunctions if w in x.split()))
    )

    determiners = ["the", "a", "an", "this", "that", "these", "those", "each", "every"]
    df["determiner_density"] = df["text"].str.lower().apply(
        lambda x: sum(1 for w in x.split() if w in determiners)
    ) / (df["word_count"] + 1)

    prepositions = [
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "with",
        "by",
        "from",
        "through",
        "between",
        "among",
        "within",
        "without",
        "beyond",
    ]
    df["preposition_density"] = df["text"].str.lower().apply(
        lambda x: sum(1 for w in x.split() if w in prepositions)
    ) / (df["word_count"] + 1)

    df["possessive_count"] = df["text"].str.count("'s|'t|'ve|'re|'ll|'d")
    df["possessive_density"] = df["possessive_count"] / (df["word_count"] + 1)

    df["unique_words"] = (
        df["text"].str.split().apply(lambda x: len(set([w.lower() for w in x])))
    )
    df["type_token_ratio"] = df["unique_words"] / (df["word_count"] + 1)
    df["hapax_ratio"] = df["text"].str.split().apply(
        lambda x: sum(1 for w, c in Counter([w.lower() for w in x]).items() if c == 1)
    ) / (df["word_count"] + 1)

    df["short_words"] = df["text"].str.findall(r"\b\w{1,3}\b").str.len()
    df["long_words"] = df["text"].str.findall(r"\b\w{7,}\b").str.len()
    df["short_word_ratio"] = df["short_words"] / (df["word_count"] + 1)
    df["long_word_ratio"] = df["long_words"] / (df["word_count"] + 1)

    return df


def create_tfidf_features(train_texts, test_texts, max_features=500):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=3,
        max_df=0.7,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    return (
        train_tfidf.toarray(),
        test_tfidf.toarray(),
        vectorizer.get_feature_names_out(),
    )


def create_character_ngram_features(train_texts, test_texts, n=3, max_features=200):
    char_vectorizer = CountVectorizer(
        analyzer="char", ngram_range=(n, n), max_features=max_features, min_df=3
    )
    train_char = char_vectorizer.fit_transform(train_texts)
    test_char = char_vectorizer.transform(test_texts)
    return (
        train_char.toarray(),
        test_char.toarray(),
        char_vectorizer.get_feature_names_out(),
    )


print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Classes: {train_df['author'].unique()}")

train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["author"] = "unknown"
combined = pd.concat([train_df, test_df], ignore_index=True)

print("Creating stylometric features...")
combined = create_stylometric_features(combined)

train_processed = combined[combined["is_train"] == 1].copy()
test_processed = combined[combined["is_train"] == 0].copy()

train_texts = train_processed["text"].values
test_texts = test_processed["text"].values

le = LabelEncoder()
train_processed["author_encoded"] = le.fit_transform(train_processed["author"])

feature_cols = [
    col
    for col in train_processed.columns
    if col not in ["id", "text", "author", "is_train", "author_encoded"]
]
print(f"Total features created: {len(feature_cols)}")

X = train_processed[feature_cols].values
y = train_processed["author_encoded"].values

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
test_texts_list = test_df["text"].values.tolist()
test_encodings = tokenizer(
    test_texts_list,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=150, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        if num_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.head = nn.Linear(hidden_size + 64, num_authors)
        else:
            self.feature_proj = None
            self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_pool = outputs.last_hidden_state[:, 0, :]
        if self.feature_proj is not None and features is not None:
            feat_embed = self.feature_proj(features)
            combined = torch.cat([cls_pool, feat_embed], dim=1)
        else:
            combined = cls_pool
        logits = self.head(combined)
        return logits


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_test_probs = []
best_val_score = float("inf")
all_texts = train_df["text"].values
all_feature_rows = train_processed[feature_cols].values

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{N_FOLDS}")
    print(f"{'='*50}")

    fold_train_texts = all_texts[train_idx].tolist()
    fold_val_texts = all_texts[val_idx].tolist()

    # Fit vectorizers and scaler on fold training data only to prevent leakage
    tfidf_vec = TfidfVectorizer(
        max_features=500, ngram_range=(1, 3), sublinear_tf=True,
        min_df=3, max_df=0.7, stop_words="english"
    )
    train_fold_tfidf = tfidf_vec.fit_transform(fold_train_texts)
    val_fold_tfidf = tfidf_vec.transform(fold_val_texts)
    test_fold_tfidf = tfidf_vec.transform(test_texts)

    char_vec = CountVectorizer(
        analyzer="char", ngram_range=(3, 3), max_features=200, min_df=3
    )
    train_fold_char = char_vec.fit_transform(fold_train_texts)
    val_fold_char = char_vec.transform(fold_val_texts)
    test_fold_char = char_vec.transform(test_texts)

    # Build fold feature matrices - ONLY from fold training data for fit
    train_feat_list = [all_feature_rows[train_idx]]
    val_feat_list = [all_feature_rows[val_idx]]
    test_feat_list = [test_processed[feature_cols].values]

    for i in range(min(50, train_fold_tfidf.shape[1])):
        train_feat_list.append(train_fold_tfidf[:, i].toarray().ravel())
        val_feat_list.append(val_fold_tfidf[:, i].toarray().ravel())
        test_feat_list.append(test_fold_tfidf[:, i].toarray().ravel())
    for i in range(min(30, train_fold_char.shape[1])):
        train_feat_list.append(train_fold_char[:, i].toarray().ravel())
        val_feat_list.append(val_fold_char[:, i].toarray().ravel())
        test_feat_list.append(test_fold_char[:, i].toarray().ravel())

    X_fold_train = np.column_stack(train_feat_list)
    X_fold_val = np.column_stack(val_feat_list)
    X_fold_test = np.column_stack(test_feat_list)

    # Scale features on fold training data only
    fold_scaler = StandardScaler()
    X_fold_train = fold_scaler.fit_transform(X_fold_train)
    X_fold_val = fold_scaler.transform(X_fold_val)
    X_fold_test = fold_scaler.transform(X_fold_test)

    y_fold_train = y[train_idx]
    y_fold_val = y[val_idx]

    train_encodings = tokenizer(
        fold_train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        fold_val_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    model = SpookyClassifier(
        num_authors=NUM_AUTHORS, num_features=X_fold_train.shape[1], dropout_rate=0.3
    )
    model.to(device)

    backbone_params = [
        p
        for layer in model.backbone.deberta.encoder.layer[-8:]
        for n, p in layer.named_parameters()
        if "bias" not in n and "LayerNorm" not in n
    ]
    head_params = list(model.head.parameters()) + (
        list(model.feature_proj.parameters()) if model.feature_proj else []
    )

    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
            {
                "params": head_params,
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.98),
            },
        ]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    total_steps = (
        len(train_idx) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scaler = GradScaler()

    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(y_fold_train, dtype=torch.long),
        torch.tensor(X_fold_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(y_fold_val, dtype=torch.long),
        torch.tensor(X_fold_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    best_fold_score = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (input_ids, attention_mask, labels, features) in enumerate(
            train_loader
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            features = features.to(device)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                current_step = epoch * len(train_loader) + batch_idx
                if current_step < warmup_steps:
                    for pg in optimizer.param_groups:
                        pg["lr"] = initial_lrs[0] * (
                            current_step / max(1, warmup_steps)
                        )
                else:
                    scheduler.step(epoch + current_step / len(train_loader))

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for input_ids, attention_mask, labels, features in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                features = features.to(device)
                with autocast():
                    logits = model(input_ids, attention_mask, features)
                    probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_true.append(labels.numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        score = log_loss(val_true, val_probs)

        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Val Log Loss: {score:.6f}"
        )

        if score < best_fold_score:
            best_fold_score = score
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/best_model_fold_{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Fold {fold + 1} best validation log loss: {best_fold_score:.6f}")

    model.load_state_dict(torch.load(f"./working/best_model_fold_{fold}.pt"))
    model.eval()

    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(X_fold_test, dtype=torch.float32),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_probs = []
    with torch.no_grad():
        for input_ids, attention_mask, features in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            with autocast():
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)
            test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(test_probs)
    all_test_probs.append(fold_test_probs)

    if fold == 0:
        best_val_score = best_fold_score
    else:
        best_val_score = min(best_val_score, best_fold_score)

print(f"\n{'='*50}")
print(f"Cross-validation completed")
print(f"Best validation log loss: {best_val_score:.6f}")
print(f"{'='*50}")

final_test_probs = np.mean(all_test_probs, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

submission_df.to_csv("./submission/submission_e1199f6d05b5476f80a10f6ff24c4374.csv", index=False)
print(f"Submission saved to ./submission/submission_e1199f6d05b5476f80a10f6ff24c4374.csv")
print(f"Test predictions shape: {final_test_probs.shape}")
print(f"Submission file preview:")
print(submission_df.head())

score = best_val_score
print(f"Final Validation Score: {score}")