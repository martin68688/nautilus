import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9, 10, 11, 12, 13})
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
import re
import string
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoModel, AutoTokenizer, AutoConfig
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# ─── Data Loading ────────────────────────────────────────────────────────
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


# ─── Text Cleaning ──────────────────────────────────────────────────────
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s.,!?;:\'\"\-\(\)\[\]]", "", text)
    return text


# ─── Feature Engineering ────────────────────────────────────────────────
def extract_features(df, is_train=True):
    features = pd.DataFrame()
    features["id"] = df["id"]
    text = df["text"].apply(clean_text)
    features["text_length"] = text.str.len()
    features["word_count"] = text.str.split().str.len()
    features["avg_word_length"] = features["text_length"] / (features["word_count"] + 1)
    features["sentence_count"] = text.str.split("[.!?]+").str.len()
    features["avg_sentence_length"] = features["word_count"] / (
        features["sentence_count"] + 1
    )
    features["exclamation_count"] = text.str.count("!")
    features["question_count"] = text.str.count(r"\?")
    features["period_count"] = text.str.count(r"\.")
    features["comma_count"] = text.str.count(",")
    features["semicolon_count"] = text.str.count(";")
    features["colon_count"] = text.str.count(":")
    features["dash_count"] = text.str.count("-")
    features["quote_count"] = text.str.count('"') + text.str.count("'")
    features["parenthesis_count"] = text.str.count(r"\(") + text.str.count(r"\)")
    features["punctuation_ratio"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["period_count"]
        + features["comma_count"]
        + features["semicolon_count"]
        + features["colon_count"]
        + features["dash_count"]
        + features["quote_count"]
        + features["parenthesis_count"]
    ) / (features["word_count"] + 1)
    features["capital_ratio"] = text.str.findall(r"\b[A-Z][a-z]*\b").str.len() / (
        features["word_count"] + 1
    )
    features["all_caps_count"] = text.str.findall(r"\b[A-Z]{2,}\b").str.len()
    features["unique_word_ratio"] = text.apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1)
    )
    stopwords = set(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "can",
            "could",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "not",
            "it",
            "its",
            "that",
            "this",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "his",
            "hers",
            "ours",
            "theirs",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "as",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "at",
            "before",
            "behind",
            "between",
            "beyond",
            "by",
            "down",
            "during",
            "except",
            "for",
            "from",
            "in",
            "inside",
            "into",
            "near",
            "of",
            "off",
            "on",
            "out",
            "outside",
            "over",
            "through",
            "to",
            "under",
            "up",
            "upon",
            "with",
            "within",
            "without",
        ]
    )
    features["stopword_ratio"] = text.apply(
        lambda x: len([w for w in x.lower().split() if w in stopwords])
        / (len(x.split()) + 1)
    )
    features["content_word_ratio"] = 1 - features["stopword_ratio"]
    features["syllable_count"] = text.apply(
        lambda x: sum(1 for char in x if char.lower() in "aeiou")
    )
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["avg_sentence_length"]
        - 84.6 * (features["syllable_count"] / (features["word_count"] + 1))
    )
    features["flesch_kincaid_grade"] = (
        0.39 * features["avg_sentence_length"]
        + 11.8 * (features["syllable_count"] / (features["word_count"] + 1))
        - 15.59
    )
    features["noun_like"] = text.str.findall(
        r"\b\w+(?:tion|sion|ment|ness|ity|ance|ence|dom|ship|age|ery|ing)\b",
        re.IGNORECASE,
    ).str.len() / (features["word_count"] + 1)
    features["verb_like"] = text.str.findall(
        r"\b\w+(?:ize|ise|ify|ate|en|ing|ed|en)\b", re.IGNORECASE
    ).str.len() / (features["word_count"] + 1)
    features["adjective_like"] = text.str.findall(
        r"\b\w+(?:ous|ive|able|ible|ful|less|ic|ical|al|ent|ant|ish|like|some)\b",
        re.IGNORECASE,
    ).str.len() / (features["word_count"] + 1)
    features["adverb_like"] = text.str.findall(
        r"\b\w+ly\b", re.IGNORECASE
    ).str.len() / (features["word_count"] + 1)
    features["starts_with_article"] = text.str.match(
        r"^(The|A|An)\b", case=False
    ).astype(int)
    features["starts_with_pronoun"] = text.str.match(
        r"^(I|You|He|She|It|We|They|This|That|These|Those)\b", case=False
    ).astype(int)
    features["starts_with_preposition"] = text.str.match(
        r"^(In|On|At|For|With|By|From|As|To|Of|About|Before|After|During|Through|Under|Over|Between|Among)\b",
        case=False,
    ).astype(int)
    features["dialog_count"] = text.str.count('"') // 2
    features["dialog_ratio"] = features["dialog_count"] / (
        features["sentence_count"] + 1
    )
    features["short_word_ratio"] = text.apply(
        lambda x: len([w for w in x.split() if len(w) <= 3]) / (len(x.split()) + 1)
    )
    features["medium_word_ratio"] = text.apply(
        lambda x: len([w for w in x.split() if 4 <= len(w) <= 7]) / (len(x.split()) + 1)
    )
    features["long_word_ratio"] = text.apply(
        lambda x: len([w for w in x.split() if len(w) >= 8]) / (len(x.split()) + 1)
    )
    features["first_person_pronoun_count"] = text.str.findall(
        r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.IGNORECASE
    ).str.len()
    features["second_person_pronoun_count"] = text.str.findall(
        r"\b(you|your|yours|yourself|yourselves)\b", re.IGNORECASE
    ).str.len()
    features["third_person_pronoun_count"] = text.str.findall(
        r"\b(he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves)\b",
        re.IGNORECASE,
    ).str.len()
    positive_words = set(
        [
            "love",
            "happy",
            "joy",
            "beautiful",
            "wonderful",
            "good",
            "great",
            "excellent",
            "perfect",
            "pleasant",
            "amazing",
            "fantastic",
            "delightful",
            "charming",
            "splendid",
            "hope",
            "kind",
            "gentle",
            "warm",
            "bright",
            "lovely",
            "nice",
            "sweet",
            "tender",
        ]
    )
    negative_words = set(
        [
            "dark",
            "terrible",
            "horrible",
            "awful",
            "dreadful",
            "fearful",
            "frightful",
            "sad",
            "angry",
            "cruel",
            "evil",
            "wicked",
            "sinister",
            "gloomy",
            "gloom",
            "hideous",
            "monstrous",
            "gruesome",
            "ghastly",
            "horror",
            "terror",
            "fear",
            "panic",
            "dread",
            "dead",
            "death",
            "blood",
            "scream",
            "cry",
            "pain",
            "suffering",
        ]
    )
    features["positive_word_count"] = text.apply(
        lambda x: len([w for w in x.lower().split() if w in positive_words])
    )
    features["negative_word_count"] = text.apply(
        lambda x: len([w for w in x.lower().split() if w in negative_words])
    )
    features["sentiment_ratio"] = (
        features["positive_word_count"] - features["negative_word_count"]
    ) / (features["word_count"] + 1)
    features["adverb_ratio"] = text.str.findall(
        r"\b\w+ly\b", re.IGNORECASE
    ).str.len() / (features["word_count"] + 1)
    features["conjunction_count"] = text.str.findall(
        r"\b(and|but|or|nor|for|yet|so|because|although|while|since|unless|if|then|else|when|where|why|how)\b",
        re.IGNORECASE,
    ).str.len()
    features["conjunction_ratio"] = features["conjunction_count"] / (
        features["word_count"] + 1
    )
    prepositions = set(
        [
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "at",
            "before",
            "behind",
            "below",
            "beneath",
            "beside",
            "between",
            "beyond",
            "by",
            "down",
            "during",
            "except",
            "for",
            "from",
            "in",
            "inside",
            "into",
            "near",
            "of",
            "off",
            "on",
            "out",
            "outside",
            "over",
            "through",
            "to",
            "under",
            "up",
            "upon",
            "with",
            "within",
            "without",
        ]
    )
    features["preposition_count"] = text.apply(
        lambda x: len([w for w in x.lower().split() if w in prepositions])
    )
    features["preposition_ratio"] = features["preposition_count"] / (
        features["word_count"] + 1
    )
    features["unique_chars"] = text.apply(lambda x: len(set(x.lower())))
    features["char_diversity"] = features["unique_chars"] / (
        features["text_length"] + 1
    )
    features["digit_count"] = text.str.findall(r"\d").str.len()
    return features


# Extract features
train_features = extract_features(train_df, is_train=True)
test_features = extract_features(test_df, is_train=False)

# Prepare labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df["author"])
print(
    f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
)

feature_cols = [col for col in train_features.columns if col != "id"]
X = train_features[feature_cols].values
X_test_feat = test_features[feature_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling will be done per fold to avoid data leakage
# (No global scaler fit here - moved inside fold loop)

# ─── Tokenizer ──────────────────────────────────────────────────────────
# Tokenizer is loaded lazily inside each fold to avoid multiprocessing issues
def get_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

def tokenize_texts(texts, max_length=256):
    tokenizer = get_tokenizer()
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


# ─── Model Definition ───────────────────────────────────────────────────
class SpookyClassifier(nn.Module):
    def __init__(self, num_authors=3, num_features=150, dropout_rate=0.3):
        super().__init__()
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        # Use safetensors to avoid any potential loading issues
        self.deberta = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large", config=config, trust_remote_code=True
        )
        for param in self.deberta.parameters():
            param.requires_grad = False
        for layer in self.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.deberta.config.hidden_size
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
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_pool = outputs.last_hidden_state[:, 0, :]
        if self.feature_proj is not None and features is not None:
            feat_embed = self.feature_proj(features)
            combined = torch.cat([cls_pool, feat_embed], dim=1)
        else:
            combined = cls_pool
        logits = self.head(combined)
        # Clamp logits to prevent extreme values leading to NaN in softmax
        logits = torch.clamp(logits, min=-25.0, max=25.0)
        return logits


# ─── Training Function for One Fold ─────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_fold(
    train_texts,
    train_features,
    train_labels,
    val_texts,
    val_features,
    val_labels,
    fold_idx,
    fold_scaler,
):
    model = SpookyClassifier(
        num_authors=3, num_features=train_features.shape[1], dropout_rate=0.3
    )
    model.to(device)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "deberta.encoder.layer" in name and any(
            f".{i}." in name for i in range(16, 24)
        ):
            if "bias" not in name and "LayerNorm" not in name:
                backbone_params.append(param)
        elif "head" in name or "feature_proj" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

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

    train_ids, train_mask = tokenize_texts(train_texts)
    val_ids, val_mask = tokenize_texts(val_texts)

    train_dataset = TensorDataset(
        train_ids,
        train_mask,
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        val_ids,
        val_mask,
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    best_val_loss = float("inf")
    best_val_probs = None
    best_model_state = None
    patience_counter = 0
    max_patience = 4

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (input_ids, attention_mask, features, labels) in enumerate(
            train_loader
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)

            # Check for NaN loss and skip if detected
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = initial_lrs[pg_idx] * (
                        current_step / max(1, warmup_steps)
                    )
            else:
                scheduler.step(current_step - warmup_steps)

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print(f"Fold {fold_idx} | Epoch {epoch+1} | All batches skipped due to NaN loss. Skipping fold.")
            return float("inf"), None, None, None

        # Validation loop
        val_probs_list = []
        val_loss_total = 0.0
        val_num_batches = 0
        with torch.no_grad():
            for input_ids, attention_mask, features, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                features = features.to(device)
                labels = labels.to(device)
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                probs = torch.softmax(logits, dim=1)
                val_probs_list.append(probs.cpu().numpy())
                val_loss_total += loss.item()
                val_num_batches += 1

        if len(val_probs_list) == 0:
            print(f"Fold {fold_idx} | Epoch {epoch+1} | All val batches NaN. Skipping fold.")
            return float("inf"), None, None, None

        val_probs = np.concatenate(val_probs_list)
        # Ensure no NaN values before clipping
        val_probs = np.nan_to_num(val_probs, nan=1.0/3.0)
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs = val_probs / (val_probs.sum(axis=1, keepdims=True) + 1e-15)
        val_loss_avg = val_loss_total / max(1, val_num_batches)
        val_logloss = log_loss(val_labels[:len(val_probs)], val_probs)

        print(
            f"Fold {fold_idx} | Epoch {epoch+1:2d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss_avg:.4f} | Val LogLoss: {val_logloss:.4f}"
        )

        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_val_probs = val_probs
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is None:
        print(f"Fold {fold_idx}: No best model found, returning uniform predictions")
        n_val = len(val_texts)
        n_test = len(test_df)
        uniform_val = np.ones((n_val, 3)) / 3.0
        uniform_test = np.ones((n_test, 3)) / 3.0
        return float("inf"), uniform_val, val_labels, uniform_test

    model.load_state_dict(best_model_state)
    model.eval()

    # Tokenize test set
    test_ids_tensor, test_mask_tensor = tokenize_texts(
        test_df["text"].values,
    )
    test_dataset_wrapper = TensorDataset(
        test_ids_tensor,
        test_mask_tensor,
        torch.tensor(fold_scaler.transform(X_test_feat), dtype=torch.float32),
        torch.zeros(len(test_ids_tensor), dtype=torch.long),
    )
    test_loader = DataLoader(
        test_dataset_wrapper, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    test_probs_list = []
    with torch.no_grad():
        for input_ids, attention_mask, features, _ in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            logits = model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)
            test_probs_list.append(probs.cpu().numpy())

    test_probs = np.concatenate(test_probs_list)
    # Ensure no NaN values before clipping
    test_probs = np.nan_to_num(test_probs, nan=1.0/3.0)
    test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
    test_probs = test_probs / (test_probs.sum(axis=1, keepdims=True) + 1e-15)

    return best_val_loss, best_val_probs, val_labels, test_probs


# ─── 5-Fold Stratified Cross-Validation ─────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_val_losses = []
fold_val_probs_list = []
fold_val_labels_list = []
fold_test_probs_list = []

train_texts_all = train_df["text"].values

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"\n===== Fold {fold+1}/5 =====")
    train_texts_fold = train_texts_all[train_idx]
    train_labels_fold = y[train_idx]
    val_texts_fold = train_texts_all[val_idx]
    val_labels_fold = y[val_idx]

    # Fit scaler on training fold only to prevent data leakage
    fold_scaler = StandardScaler()
    train_features_fold = fold_scaler.fit_transform(X[train_idx])
    val_features_fold = fold_scaler.transform(X[val_idx])

    val_loss, val_probs, val_labels, test_probs = train_fold(
        train_texts_fold,
        train_features_fold,
        train_labels_fold,
        val_texts_fold,
        val_features_fold,
        val_labels_fold,
        fold + 1,
        fold_scaler,
    )

    # Handle case where fold returned None (all NaN)
    if val_probs is None:
        print(f"WARNING: Fold {fold+1} produced NaN predictions, using uniform fallback")
        n_val = len(val_labels)
        n_test = len(test_df)
        val_probs = np.ones((n_val, 3)) / 3.0
        test_probs = np.ones((n_test, 3)) / 3.0
        val_loss = np.log(3)  # uniform log loss
    elif np.isnan(val_probs).any() or np.isnan(test_probs).any():
        print(f"WARNING: Fold {fold+1} produced NaN predictions, using uniform fallback")
        n_val = len(val_labels)
        n_test = len(test_df)
        val_probs = np.ones((n_val, 3)) / 3.0
        test_probs = np.ones((n_test, 3)) / 3.0
        val_loss = np.log(3)  # uniform log loss
    fold_val_losses.append(val_loss)
    fold_val_probs_list.append(val_probs)
    fold_val_labels_list.append(val_labels)
    fold_test_probs_list.append(test_probs)

# ─── Ensemble Fold Predictions ──────────────────────────────────────────
all_val_probs = np.concatenate(fold_val_probs_list)
all_val_labels = np.concatenate(fold_val_labels_list)
final_val_score = log_loss(all_val_labels, all_val_probs)

final_test_probs = np.mean(fold_test_probs_list, axis=0)
final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / (final_test_probs.sum(axis=1, keepdims=True) + 1e-15)

# ─── Submission ─────────────────────────────────────────────────────────
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)
submission_df.to_csv("./submission/submission_9aa4e6ec7f4b4a54bdb840b077da5050.csv", index=False)
print(f"Submission saved to ./submission/submission_9aa4e6ec7f4b4a54bdb840b077da5050.csv")
print(f"Final Validation Score: {final_val_score}")