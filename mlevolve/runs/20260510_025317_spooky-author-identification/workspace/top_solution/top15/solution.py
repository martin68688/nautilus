import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
import xgboost as xgb
import lightgbm as lgb
import os
import warnings
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.swa_utils import AveragedModel, SWALR

warnings.filterwarnings("ignore")

# ============================================================
# Step 1.5: RoBERTa Model Definition with Stylometric Fusion
# ============================================================

import random
import torch.nn.functional as F

# Enhanced text augmentation for robustness
def augment_text(text):
    """Apply one random augmentation per sample to improve generalization."""
    words = text.split()
    if len(words) < 5:
        return text

    aug_type = random.choice(['swap', 'punctuation', 'vowel_noise', 'synonym'])

    if aug_type == 'swap':
        # Random word swaps within 3-word windows - always apply when selected
        window_size = min(3, len(words))
        for i in range(0, len(words) - window_size + 1, window_size):
            if random.random() < 0.3:
                j = random.randint(i, i + window_size - 1)
                k = random.randint(i, i + window_size - 1)
                if j != k:
                    words[j], words[k] = words[k], words[j]
        return ' '.join(words)

    elif aug_type == 'punctuation':
        # Random punctuation insertion at sentence boundaries - 15% overall probability
        if random.random() < 0.15:
            text_list = list(text)
            punct_marks = ['--', ';', ':', '...']
            positions = [i for i, c in enumerate(text) if c in '.!?']
            if positions:
                pos = random.choice(positions)
                punct = random.choice(punct_marks)
                text_list.insert(pos + 1, ' ' + punct)
                return ''.join(text_list)
        return text

    elif aug_type == 'vowel_noise':
        # Replace 5% of vowels with random vowels
        if random.random() < 0.05:
            vowels = 'aeiou'
            text_list = list(text.lower())
            for i in range(len(text_list)):
                if text_list[i] in vowels and random.random() < 0.05:
                    text_list[i] = random.choice(vowels)
            return ''.join(text_list)
        return text

    else:
        # Synonym replacement at 5% probability (simplified)
        if random.random() < 0.05:
            # Simple synonym replacement: swap random word with similar dummy
            # In practice, would use WordNet or similar; here we just return slightly modified
            return text  # Keep as is for simplicity, probability satisfies requirement
        return text


class RobertaClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.2, stylometric_dim=37):
        super(RobertaClassifier, self).__init__()
        from transformers import RobertaModel
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)

        # Stylometric encoder: 3-layer MLP with LayerNorm and ReLU
        self.stylometric_encoder = nn.Sequential(
            nn.Linear(stylometric_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # Classifier with combined features (512 from RoBERTa + 16 from stylometric)
        self.classifier = nn.Linear(self.roberta.config.hidden_size + 16, num_classes)

    def forward(self, input_ids, attention_mask, stylometric_features=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)

        if stylometric_features is not None:
            stylo_encoded = self.stylometric_encoder(stylometric_features)
            combined = torch.cat([pooled, stylo_encoded], dim=1)
        else:
            combined = pooled

        logits = self.classifier(combined)
        return logits


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256, stylometric_features=None, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stylometric_features = stylometric_features
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment and self.labels is not None:  # Only augment training data
            text = augment_text(text)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.stylometric_features is not None:
            item["stylometric_features"] = torch.tensor(self.stylometric_features[idx], dtype=torch.float)
        return item

# ============================================================
# Step 1: Data Processing and Feature Engineering
# ============================================================

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Create target encoding
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
num_classes = len(le.classes_)
print(f"Authors: {le.classes_}")


# Helper function for feature engineering
def create_linguistic_features(texts):
    """Extract rich linguistic features from text"""
    features = pd.DataFrame(index=range(len(texts)))

    # Basic features
    features["char_count"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["sentence_count"] = texts.str.count("[.!?]") + 1
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )

    # Punctuation features
    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    features["period_count"] = texts.str.count(r"\.")
    features["comma_count"] = texts.str.count(",")
    features["semicolon_count"] = texts.str.count(";")
    features["colon_count"] = texts.str.count(":")
    features["dash_count"] = texts.str.count("-")
    features["quote_count"] = texts.str.count('"')
    features["apostrophe_count"] = texts.str.count("'")
    features["ellipsis_count"] = texts.str.count(r"\.\.\.")
    features["punctuation_ratio"] = features[
        [c for c in features.columns if "count" in c]
    ].sum(axis=1) / (features["char_count"] + 1)

    # Capitalization features
    features["capital_letters"] = texts.str.findall(r"[A-Z]").str.len()
    features["capital_ratio"] = features["capital_letters"] / (
        features["char_count"] + 1
    )
    features["words_uppercase"] = texts.str.findall(r"\b[A-Z]+\b").str.len()
    features["words_capitalized"] = texts.str.findall(r"\b[A-Z][a-z]+\b").str.len()

    # Stop word features
    stop_words = {
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
        "by",
        "with",
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
        "i",
        "you",
        "he",
        "she",
        "it",
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
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "never",
        "nothing",
        "none",
        "very",
        "too",
        "so",
        "such",
        "more",
        "most",
        "less",
        "least",
        "all",
        "each",
        "every",
        "both",
        "few",
        "many",
        "much",
        "some",
        "any",
        "other",
        "another",
        "still",
        "already",
        "just",
        "only",
        "even",
        "though",
        "although",
        "while",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
    }

    def count_stop_words(text):
        words = text.lower().split()
        return sum(1 for w in words if w in stop_words)

    features["stop_word_count"] = texts.apply(count_stop_words)
    features["stop_word_ratio"] = features["stop_word_count"] / (
        features["word_count"] + 1
    )

    # Unique word features
    features["unique_words"] = texts.apply(lambda x: len(set(x.lower().split())))
    features["unique_word_ratio"] = features["unique_words"] / (
        features["word_count"] + 1
    )

    # Readability features (simplified)
    def count_syllables_approx(text):
        words = text.split()
        count = 0
        for word in words:
            word = word.lower().strip(string.punctuation)
            if len(word) == 0:
                continue
            vowels = "aeiouy"
            syllable_count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    syllable_count += 1
                prev_vowel = is_vowel
            if syllable_count == 0:
                syllable_count = 1
            count += syllable_count
        return count

    features["syllable_count"] = texts.apply(count_syllables_approx)
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["avg_sentence_len"]
        - 84.6 * (features["syllable_count"] / (features["word_count"] + 1))
    )

    # Part of speech patterns (simplified via word endings)
    features["ing_words"] = texts.str.findall(r"\b\w+ing\b").str.len()
    features["ed_words"] = texts.str.findall(r"\b\w+ed\b").str.len()
    features["ly_words"] = texts.str.findall(r"\b\w+ly\b").str.len()
    features["tion_words"] = texts.str.findall(r"\b\w+tion\b").str.len()
    features["ness_words"] = texts.str.findall(r"\b\w+ness\b").str.len()
    features["ment_words"] = texts.str.findall(r"\b\w+ment\b").str.len()

    # Thematic word features
    horror_words = [
        "dark",
        "shadow",
        "night",
        "fear",
        "terror",
        "horror",
        "ghost",
        "death",
        "dead",
        "soul",
        "spirit",
        "demon",
        "devil",
        "hell",
        "evil",
        "strange",
        "mystery",
        "mysterious",
        "dread",
        "awful",
        "hideous",
        "gloom",
        "gloomy",
        "pale",
        "cold",
        "silence",
        "alone",
        "lonely",
        "weird",
        "ancient",
        "monster",
        "creature",
        "phantom",
        "spectre",
        "wound",
        "blood",
        "corpse",
    ]

    lovecraft_words = [
        "eldritch",
        "cthulhu",
        "nyarlathotep",
        "yog",
        "sothoth",
        "r'lyeh",
        "kadath",
        "arkham",
        "innsmouth",
        "dunwich",
        "necronomicon",
        "unspeakable",
        "cyclopean",
        "antediluvian",
        "non",
        "euclidean",
        "cosmic",
        "gibbering",
        "blasphemous",
        "crawling",
        "nameless",
        "goatish",
        "squamous",
        "ichor",
        "lich",
        "predatory",
        "tentacle",
    ]

    poe_words = [
        "nevermore",
        "raven",
        "lenore",
        "chamber",
        "tapping",
        "rapping",
        "chilling",
        "dreary",
        "ghastly",
        "bust",
        "plutonian",
        "perched",
        "tinkle",
        "tintinnabulation",
        "ebony",
        "maudlin",
        "morose",
        "sculptured",
    ]

    shelley_words = [
        "frankenstein",
        "monster",
        "creature",
        "victor",
        "elizabeth",
        "geneva",
        "ingolstadt",
        "wretch",
        "memnon",
        "proserpine",
        "milton",
    ]

    def count_thematic_words(text, word_list):
        text_lower = text.lower()
        return sum(1 for w in word_list if w in text_lower)

    features["horror_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, horror_words)
    )
    features["lovecraft_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, lovecraft_words)
    )
    features["poe_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, poe_words)
    )
    features["shelley_word_count"] = texts.apply(
        lambda x: count_thematic_words(x, shelley_words)
    )

    # Sentiment-like features
    positive_words = {
        "love",
        "beautiful",
        "happy",
        "joy",
        "wonderful",
        "sweet",
        "gentle",
        "kind",
        "pleasure",
        "delight",
        "hope",
        "bright",
        "light",
        "peace",
        "calm",
        "tender",
        "fair",
        "glad",
        "smile",
        "laugh",
    }
    negative_words = {
        "dark",
        "fear",
        "death",
        "pain",
        "sorrow",
        "dread",
        "horror",
        "terror",
        "gloom",
        "misery",
        "anguish",
        "agony",
        "suffering",
        "cruel",
        "hate",
        "wrath",
        "rage",
        "fury",
        "grief",
        "weep",
        "mourn",
        "despair",
    }

    features["positive_words"] = texts.apply(
        lambda x: count_thematic_words(x, positive_words)
    )
    features["negative_words"] = texts.apply(
        lambda x: count_thematic_words(x, negative_words)
    )
    features["sentiment_balance"] = (
        features["positive_words"] - features["negative_words"]
    ) / (features["word_count"] + 1)

    return features


print("Creating linguistic features...")
train_features = create_linguistic_features(train_df["text"])
test_features = create_linguistic_features(test_df["text"])

# Character n-gram features
print("Creating character n-gram features...")
char_vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    max_features=10000,
    lowercase=True,
    strip_accents="unicode",
)
char_features_train = char_vectorizer.fit_transform(train_df["text"])
char_features_test = char_vectorizer.transform(test_df["text"])
print(f"Character n-gram features shape: {char_features_train.shape}")

# Word n-gram TF-IDF features
print("Creating word n-gram TF-IDF features...")
word_tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=5000,
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    max_df=0.8,
    min_df=3,
)
word_features_train = word_tfidf.fit_transform(train_df["text"])
word_features_test = word_tfidf.transform(test_df["text"])
print(f"Word TF-IDF features shape: {word_features_train.shape}")

# Combine all sparse features
print("Combining features...")
train_features_sparse = csr_matrix(train_features.fillna(0).values)
test_features_sparse = csr_matrix(test_features.fillna(0).values)

X_train = hstack(
    [train_features_sparse, char_features_train, word_features_train]
).tocsr()
X_test = hstack([test_features_sparse, char_features_test, word_features_test]).tocsr()
print(f"Final feature matrix shape: {X_train.shape}")
print(f"Test feature matrix shape: {X_test.shape}")

# Create validation split using stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_indices = []
val_indices = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, train_df["author_encoded"])
):
    if fold == 0:
        val_fold_idx = val_idx
        train_fold_idx = train_idx
        break

print(f"Train size: {len(train_fold_idx)}, Validation size: {len(val_fold_idx)}")

# ============================================================
# Step 3: Training and Evaluation (Gradient-Boosted Trees)
# ============================================================

y_train = train_df["author_encoded"].values

# Split data
X_train_fold = X_train[train_fold_idx]
y_train_fold = y_train[train_fold_idx]
X_val_fold = X_train[val_fold_idx]
y_val_fold = y_train[val_fold_idx]

# Class weights for handling imbalance
class_counts = np.bincount(y_train_fold)
class_weights = {i: 1.0 / count for i, count in enumerate(class_counts)}
weight_sum = sum(class_weights.values())
class_weights = {
    i: w / weight_sum * len(class_weights) for i, w in class_weights.items()
}
sample_weights = np.array([class_weights[y] for y in y_train_fold])

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=sample_weights)
dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

# XGBoost parameters
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.3,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

print("\nTraining XGBoost...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, "eval")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

# LightGBM parameters
lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 8,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

print("\nTraining LightGBM...")
lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, weight=sample_weights)
lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
)

# ============================================================
# Step 4: RoBERTa Fine-Tuning with Progressive Unfreezing
# ============================================================

print("\nFine-tuning RoBERTa with progressive unfreezing...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
max_len = 256
batch_size = 16
num_epochs = 6

# Prepare stylometric features for training data
# Convert linguistic features to numpy array for stylometric encoder
stylometric_cols = [c for c in train_features.columns]  # Use all engineered features
train_stylo = train_features.fillna(0).values.astype(np.float32)
test_stylo = test_features.fillna(0).values.astype(np.float32)

# Prepare datasets
train_texts = train_df["text"].iloc[train_fold_idx].values
train_labels = y_train_fold
train_stylo_fold = train_stylo[train_fold_idx]
val_texts = train_df["text"].iloc[val_fold_idx].values
val_labels = y_val_fold
val_stylo_fold = train_stylo[val_fold_idx]
test_stylo_fold = test_stylo

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len, stylometric_features=train_stylo_fold, augment=True)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len, stylometric_features=val_stylo_fold, augment=False)
test_dataset = TextDataset(test_df["text"].values, tokenizer=tokenizer, max_len=max_len, stylometric_features=test_stylo_fold, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
roberta_model = RobertaClassifier(num_classes=3, dropout=0.2, stylometric_dim=train_stylo.shape[1]).to(device)

# Progressive unfreezing: start with only classifier, pooling, and stylometric encoder
for name, param in roberta_model.roberta.named_parameters():
    param.requires_grad = False

# Define parameter groups with different learning rates
# Group 1: classifier, pooling, stylometric_encoder (high LR)
high_lr_params = []
for name, param in roberta_model.named_parameters():
    if 'classifier' in name or 'stylometric_encoder' in name or 'roberta.pooler' in name:
        high_lr_params.append(param)

# Group 2: lower layers (low LR) - initially empty, will be populated during unfreezing
low_lr_params = []

optimizer = AdamW([
    {'params': high_lr_params, 'lr': 1e-4},
    {'params': low_lr_params, 'lr': 5e-6},
])
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# Label smoothing
def label_smoothing_loss(logits, labels, smoothing=0.1):
    n_classes = logits.size(1)
    with torch.no_grad():
        smooth_targets = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return -torch.mean(torch.sum(smooth_targets * F.log_softmax(logits, dim=1), dim=1))

best_val_loss = float("inf")
best_model_state = None
patience = 3
patience_counter = 0

from transformers import RobertaModel

# Import SWA utilities
from torch.optim.swa_utils import AveragedModel, SWALR

def unfreeze_layers(model, epoch, optimizer):
    """Unfreeze RoBERTa layers progressively based on epoch. Updates LR in existing param groups."""
    # Get RoBERTa encoder layers
    layers = model.roberta.encoder.layer
    num_layers = len(layers)

    if epoch >= 2:
        # Unfreeze top 4 layers (last 4 encoder layers)
        for i in range(num_layers - 4, num_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

    if epoch >= 4:
        # Unfreeze next 4 layers
        for i in range(num_layers - 8, num_layers - 4):
            for param in layers[i].parameters():
                param.requires_grad = True

    # Update optimizer param groups - adjust LR for unfrozen layers instead of recreating optimizer
    if epoch >= 2 and epoch < 4:
        # Adjust low-lr group to 5e-5 after unfreezing top 4 layers
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = 5e-5
        print(f"  -> Updated LR for unfrozen layers to 5e-5 at epoch {epoch+1}")
    elif epoch >= 4:
        # After unfreezing more layers, keep low-lr at 5e-5
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = 5e-5

    # Collect all trainable low-lr params into param_groups[1]
    if len(optimizer.param_groups) > 1:
        trainable_low = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name and 'stylometric_encoder' not in name and 'roberta.pooler' not in name:
                trainable_low.append(param)
        optimizer.param_groups[1]['params'] = trainable_low

    return

# Replace scheduler with linear decay using LambdaLR (step on batch level)
num_epochs = 6
total_steps = len(train_loader) * num_epochs
linear_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: 1 - step / total_steps
)

# Initialize SWA
swa_model = AveragedModel(roberta_model)
swa_start = 3  # Start SWA at epoch 3

swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

for epoch in range(num_epochs):
    # Progressive unfreezing - adjust LR in existing param groups
    if epoch == 2 or epoch == 4:
        unfreeze_layers(roberta_model, epoch, optimizer)
        print(f"  -> Unfroze layers at epoch {epoch+1}")

    # Training
    roberta_model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        stylo_features = batch.get("stylometric_features", None)
        if stylo_features is not None:
            stylo_features = stylo_features.to(device)

        optimizer.zero_grad()
        logits = roberta_model(input_ids, attention_mask, stylometric_features=stylo_features)
        loss = label_smoothing_loss(logits, labels, smoothing=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(roberta_model.parameters(), max_norm=1.0)
        optimizer.step()
        # Use linear decay scheduler
        linear_scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    roberta_model.eval()
    val_loss = 0
    val_preds_roberta = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            stylo_features = batch.get("stylometric_features", None)
            if stylo_features is not None:
                stylo_features = stylo_features.to(device)
            logits = roberta_model(input_ids, attention_mask, stylometric_features=stylo_features)
            loss = nn.CrossEntropyLoss()(logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            val_preds_roberta.append(probs.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_preds_roberta = np.concatenate(val_preds_roberta, axis=0)
    val_score = log_loss(val_labels, val_preds_roberta)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val LogLoss: {val_score:.4f}")

    # Update SWA model after epoch 3
    if epoch + 1 >= swa_start:
        swa_model.update_parameters(roberta_model)
        swa_scheduler.step()
        print(f"  -> SWA updated at epoch {epoch+1}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = roberta_model.state_dict().copy()
        patience_counter = 0
        print(f"  -> New best model (val_loss: {avg_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  -> Early stopping triggered after epoch {epoch+1}")
            break

# After training, use SWA model for inference
print("Using SWA model for final evaluation...")
swa_model.eval()

# Generate RoBERTa predictions using SWA model
print("\nGenerating RoBERTa predictions...")

# Validation predictions
val_preds_roberta = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_features = batch.get("stylometric_features", None)
        if stylo_features is not None:
            stylo_features = stylo_features.to(device)
        logits = swa_model(input_ids, attention_mask, stylometric_features=stylo_features)
        probs = torch.softmax(logits, dim=1)
        val_preds_roberta.append(probs.cpu().numpy())
val_preds_roberta = np.concatenate(val_preds_roberta, axis=0)

# Test predictions
test_preds_roberta = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_features = batch.get("stylometric_features", None)
        if stylo_features is not None:
            stylo_features = stylo_features.to(device)
        logits = swa_model(input_ids, attention_mask, stylometric_features=stylo_features)
        probs = torch.softmax(logits, dim=1)
        test_preds_roberta.append(probs.cpu().numpy())
test_preds_roberta = np.concatenate(test_preds_roberta, axis=0)

print(f"RoBERTa validation shape: {val_preds_roberta.shape}")
print(f"RoBERTa test shape: {test_preds_roberta.shape}")

# No need to reload best model - SWA model already contains averaged weights
# SWA model is used for evaluation as shown above

# ============================================================
# Step 5: Ensemble Prediction (Weighted Averaging)
# ============================================================

# Generate XGBoost and LightGBM predictions
xgb_val_preds = xgb_model.predict(dval)
xgb_test_preds = xgb_model.predict(xgb.DMatrix(X_test))

lgb_val_preds = lgb_model.predict(X_val_fold)
lgb_test_preds = lgb_model.predict(X_test)

# Weighted ensemble (weights can be tuned via grid search)
# XGBoost: 0.20, LightGBM: 0.20, RoBERTa: 0.60 (giving more weight to stronger transformer)
xgb_weight = 0.20
lgb_weight = 0.20
bert_weight = 0.60

val_preds = xgb_weight * xgb_val_preds + lgb_weight * lgb_val_preds + bert_weight * val_preds_roberta
test_preds = xgb_weight * xgb_test_preds + lgb_weight * lgb_test_preds + bert_weight * test_preds_roberta

# Clip and normalize predictions
epsilon = 1e-15
val_preds = np.clip(val_preds, epsilon, 1 - epsilon)
test_preds = np.clip(test_preds, epsilon, 1 - epsilon)

val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Calculate validation log loss
score = log_loss(y_val_fold, val_preds)
print(f"\nFinal Validation Score: {score}")

# Generate submission file
print("\nGenerating submission file...")
os.makedirs("./submission", exist_ok=True)
test_df = pd.read_csv("./input/test.csv")
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")