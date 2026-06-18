import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import os
import joblib
import warnings

warnings.filterwarnings("ignore")


def create_stylometric_features(df):
    """Create comprehensive stylometric features from text data"""
    features = pd.DataFrame(index=df.index)

    # Basic text statistics
    features["char_count"] = df["text"].str.len()
    features["word_count"] = df["text"].str.split().str.len()
    features["sentence_count"] = df["text"].str.split("[.!?]+").str.len()
    features["avg_word_length"] = features["char_count"] / (features["word_count"] + 1)
    features["avg_sentence_length"] = features["word_count"] / (
        features["sentence_count"] + 1
    )

    # Vocabulary richness
    features["unique_words_count"] = df["text"].apply(
        lambda x: len(set(x.lower().split()))
    )
    features["type_token_ratio"] = features["unique_words_count"] / (
        features["word_count"] + 1
    )

    # Punctuation features
    features["exclamation_count"] = df["text"].str.count("!")
    features["question_count"] = df["text"].str.count(r"\?")
    features["period_count"] = df["text"].str.count(r"\.")
    features["comma_count"] = df["text"].str.count(",")
    features["semicolon_count"] = df["text"].str.count(";")
    features["colon_count"] = df["text"].str.count(":")
    features["dash_count"] = df["text"].str.count("-")
    features["quote_count"] = df["text"].str.count('"') + df["text"].str.count("'")
    features["paren_count"] = df["text"].str.count(r"\(|\)")

    # Punctuation density
    features["punctuation_density"] = df["text"].str.count(
        r"[{}]".format(re.escape(string.punctuation))
    ) / (features["char_count"] + 1)

    # Capitalization features
    features["capital_letters_ratio"] = df["text"].str.findall(r"[A-Z]").str.len() / (
        features["char_count"] + 1
    )
    features["uppercase_words_ratio"] = df["text"].str.findall(
        r"\b[A-Z]+\b"
    ).str.len() / (features["word_count"] + 1)

    # Word length distribution
    features["short_words_ratio"] = df["text"].str.findall(r"\b\w{1,3}\b").str.len() / (
        features["word_count"] + 1
    )
    features["medium_words_ratio"] = df["text"].str.findall(
        r"\b\w{4,6}\b"
    ).str.len() / (features["word_count"] + 1)
    features["long_words_ratio"] = df["text"].str.findall(r"\b\w{7,}\b").str.len() / (
        features["word_count"] + 1
    )

    # Stop word frequency (using common English stop words)
    stop_words = set(
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
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "then",
            "than",
            "that",
            "this",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "where",
            "when",
            "why",
            "how",
        ]
    )

    features["stopword_ratio"] = df["text"].apply(
        lambda x: sum(
            1 for w in x.lower().split() if w.strip(string.punctuation) in stop_words
        )
        / (len(x.split()) + 1)
    )

    # Readability proxy features
    features["syllable_count"] = features["word_count"] * 1.5  # Approximation
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["avg_sentence_length"]
        - 84.6 * (features["syllable_count"] / (features["word_count"] + 1))
    )

    # Special character patterns
    features["ellipsis_count"] = df["text"].str.count(r"\.{3,}")
    features["camel_case_words"] = (
        df["text"].str.findall(r"\b[A-Z][a-z]+[A-Z][a-z]+\b").str.len()
    )

    # Emotion/psycholinguistic markers
    emotion_words = {
        "fear": [
            "fear",
            "terrify",
            "horror",
            "dread",
            "fright",
            "terror",
            "panic",
            "alarm",
            "scare",
            "horrible",
            "gloom",
            "shadow",
            "dark",
            "ghost",
            "haunt",
            "spirit",
            "phantom",
            "specter",
            "dismal",
            "dreary",
        ],
        "sadness": [
            "sad",
            "grief",
            "sorrow",
            "mourn",
            "weep",
            "tear",
            "melancholy",
            "despair",
            "wretched",
            "miserable",
        ],
        "surprise": [
            "surprise",
            "astonish",
            "amaze",
            "wonder",
            "sudden",
            "abrupt",
            "startle",
            "unexpected",
        ],
        "positive": [
            "beautiful",
            "wonderful",
            "splendid",
            "glorious",
            "happy",
            "joy",
            "delight",
            "pleasure",
            "love",
            "hope",
            "bright",
            "light",
            "warm",
            "kind",
            "gentle",
            "peace",
            "calm",
        ],
    }

    for emotion, words in emotion_words.items():
        pattern = r"\b(?:" + "|".join(words) + r")\b"
        features[f"{emotion}_word_count"] = (
            df["text"].str.findall(pattern, flags=re.IGNORECASE).str.len()
        )
        features[f"{emotion}_word_ratio"] = features[f"{emotion}_word_count"] / (
            features["word_count"] + 1
        )

    # Gothic literature markers
    gothic_words = [
        "mystery",
        "secret",
        "ancient",
        "forbidden",
        "supernatural",
        "unholy",
        "sacred",
        "curse",
        "crypt",
        "tomb",
        "coffin",
        "grave",
        "cemetery",
        "blood",
        "scream",
        "shadow",
        "strange",
        "peculiar",
        "singular",
        "unusual",
        "extraordinary",
        "remarkable",
    ]
    gothic_pattern = r"\b(?:" + "|".join(gothic_words) + r")\b"
    features["gothic_word_count"] = (
        df["text"].str.findall(gothic_pattern, flags=re.IGNORECASE).str.len()
    )
    features["gothic_word_ratio"] = features["gothic_word_count"] / (
        features["word_count"] + 1
    )

    # Scientific/technical marker (Shelley's style)
    science_words = [
        "scientific",
        "experiment",
        "natural",
        "philosophy",
        "principle",
        "science",
        "study",
        "theory",
        "reason",
        "discover",
        "know",
        "knowledge",
        "understand",
        "explain",
        "phenomena",
    ]
    science_pattern = r"\b(?:" + "|".join(science_words) + r")\b"
    features["science_word_count"] = (
        df["text"].str.findall(science_pattern, flags=re.IGNORECASE).str.len()
    )
    features["science_word_ratio"] = features["science_word_count"] / (
        features["word_count"] + 1
    )

    # Archaic language marker (Lovecraft's style)
    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "art",
        "thou",
        "thyself",
        "whence",
        "hence",
        "thence",
        "wherefore",
        "therein",
        "thereof",
        "thereto",
        "ere",
        "unto",
        "nay",
        "yea",
        "prithee",
        "forsooth",
        "methinks",
        "perchance",
        "dread",
        "eldritch",
        "cyclopean",
        "non-euclidean",
        "cosmic",
        "antique",
        "primordial",
    ]
    archaic_pattern = r"\b(?:" + "|".join(archaic_words) + r")\b"
    features["archaic_word_count"] = (
        df["text"].str.findall(archaic_pattern, flags=re.IGNORECASE).str.len()
    )
    features["archaic_word_ratio"] = features["archaic_word_count"] / (
        features["word_count"] + 1
    )

    # Repeat word patterns (stylistic)
    features["word_repetition"] = (
        df["text"].str.findall(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE).str.len()
    )

    # Handle NaN and infinity values
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    return features


def create_tfidf_features(df, vectorizer=None, is_train=True, max_features=1000):
    """Create TF-IDF features from text"""
    if is_train:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            stop_words="english",
            min_df=5,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(df["text"])
    else:
        tfidf_matrix = vectorizer.transform(df["text"])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
    )
    return tfidf_df, vectorizer


def load_and_process_data():
    """Main function to load and process data"""
    print("Loading data...")
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    # Encode target labels
    label_encoder = LabelEncoder()
    train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Authors distribution:\n{train_df['author'].value_counts()}")

    # Create stylometric features for both train and test
    print("\nCreating stylometric features...")
    train_stylo = create_stylometric_features(train_df)
    test_stylo = create_stylometric_features(test_df)

    # Create TF-IDF features
    print("Creating TF-IDF features...")
    train_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_df, is_train=True, max_features=500
    )
    test_tfidf, _ = create_tfidf_features(
        test_df, vectorizer=tfidf_vectorizer, is_train=False, max_features=500
    )

    # Combine all features
    train_features = pd.concat([train_stylo, train_tfidf], axis=1)
    test_features = pd.concat([test_stylo, test_tfidf], axis=1)

    # Scale numerical features
    print("Scaling features...")
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    train_features[numeric_cols] = scaler.fit_transform(train_features[numeric_cols])
    test_features[numeric_cols] = scaler.transform(test_features[numeric_cols])

    # Create stratified train/validation split FIRST (before any feature fitting)
    print("\nCreating train/validation split...")
    train_df_split, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=42,
        stratify=train_df["author_encoded"],
    )

    # Now create features separately for train and val
    print("\nCreating stylometric features for train...")
    train_stylo = create_stylometric_features(train_df_split)
    print("Creating stylometric features for val...")
    val_stylo = create_stylometric_features(val_df)

    print("Creating TF-IDF features...")
    train_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_df_split, is_train=True, max_features=500
    )
    val_tfidf, _ = create_tfidf_features(
        val_df, vectorizer=tfidf_vectorizer, is_train=False, max_features=500
    )

    # Combine features for train and val
    X_train = pd.concat([train_stylo, train_tfidf], axis=1)
    X_val = pd.concat([val_stylo, val_tfidf], axis=1)

    # Now create test features (these were already created above but let's re-use them)
    # Re-create test features using the same process
    print("Creating stylometric features for test...")
    test_stylo = create_stylometric_features(test_df)
    print("Creating TF-IDF features for test...")
    test_tfidf, _ = create_tfidf_features(
        test_df, vectorizer=tfidf_vectorizer, is_train=False, max_features=500
    )
    test_features = pd.concat([test_stylo, test_tfidf], axis=1)

    # Scale numerical features - fit ONLY on train
    print("Scaling features...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    test_features[numeric_cols] = scaler.transform(test_features[numeric_cols])

    y_train = train_df_split["author_encoded"]
    y_val = val_df["author_encoded"]

    # Create data dictionary for modeling step
    data_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": test_features,
        "test_ids": test_df["id"],
        "label_encoder": label_encoder,
        "scaler": scaler,
        "train_text": train_df_split["text"],
        "val_text": val_df["text"],
        "test_text": test_df["text"],
        "original_train": train_df,
        "original_test": test_df,
    }

    print(f"\nFinal feature shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {test_features.shape}")

    # Save processed data for later use
    print("\nSaving processed data...")
    os.makedirs("./working", exist_ok=True)
    joblib.dump(data_dict, "./working/processed_data.pkl")

    # Also save individual objects for easy access
    X_train.to_pickle("./working/X_train.pkl")
    X_val.to_pickle("./working/X_val.pkl")
    y_train.to_pickle("./working/y_train.pkl")
    y_val.to_pickle("./working/y_val.pkl")
    test_features.to_pickle("./working/X_test.pkl")
    test_df["id"].to_pickle("./working/test_ids.pkl")
    joblib.dump(label_encoder, "./working/label_encoder.pkl")
    joblib.dump(scaler, "./working/scaler.pkl")

    return data_dict


class SpookyDataset(Dataset):
    def __init__(self, texts, features, labels=None, max_len=256):
        self.texts = texts
        self.features = (
            features.values if isinstance(features, pd.DataFrame) else features
        )
        self.labels = labels.values if isinstance(labels, pd.Series) else labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = (
            str(self.texts.iloc[idx])
            if hasattr(self.texts, "iloc")
            else str(self.texts[idx])
        )
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
        }
        if self.labels is not None:
            label = (
                self.labels[idx]
                if hasattr(self.labels, "__getitem__")
                else self.labels[idx]
            )
            item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class StyleDropout(nn.Module):
    """Dropout that randomly masks entire feature groups (e.g., all TF-IDF or all punctuation features)"""
    def __init__(self, num_feature_groups, drop_rate=0.15):
        super().__init__()
        self.num_groups = num_feature_groups
        self.drop_rate = drop_rate

    def forward(self, x):
        """x shape: (batch, dim) where dim is the original stylometric feature dimension"""
        if not self.training or self.drop_rate == 0:
            return x
        # Create group index mapping: we treat every 10 consecutive features as a group (simplified)
        device = x.device
        group_size = max(1, x.shape[1] // self.num_groups)
        num_groups = (x.shape[1] + group_size - 1) // group_size
        # Randomly drop entire groups
        bernoulli = torch.distributions.Bernoulli(probs=1 - self.drop_rate)
        mask = bernoulli.sample(sample_shape=torch.Size([x.shape[0], num_groups])).to(device)
        # Expand mask to feature dimension
        mask_expanded = mask[:, :num_groups].repeat_interleave(group_size, dim=1)
        if mask_expanded.shape[1] > x.shape[1]:
            mask_expanded = mask_expanded[:, :x.shape[1]]
        else:
            mask_expanded = torch.nn.functional.pad(mask_expanded, (0, x.shape[1] - mask_expanded.shape[1]), value=1.0)
        return x * mask_expanded / (1 - self.drop_rate + 1e-8)


class SpookyAuthorModel(nn.Module):
    def __init__(self, num_authors=3, num_stylometric_features=None, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size

        # StyleDropout: randomly mask entire feature groups (e.g., TF-IDF or punctuation)
        self.style_dropout = StyleDropout(num_feature_groups=10, drop_rate=0.15)

        self.feature_proj = nn.Sequential(
            nn.Linear(num_stylometric_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Cross-attention: BERT [CLS] (1024-dim) as query, stylometric features (64-dim) as key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=2,
            batch_first=True,
            dropout=dropout_rate,
        )
        # Project style features to the same dimension as CLS for cross-attention
        self.style_to_kv = nn.Linear(64, hidden_size)
        # Gated residual: learn a gate to blend attended style with original CLS
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_sigmoid = nn.Sigmoid()

        self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask, stylometric_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_pool = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        if stylometric_features is not None:
            # Apply StyleDropout (masks entire feature groups during training)
            stylometric_features = self.style_dropout(stylometric_features)

            feat_embed = self.feature_proj(stylometric_features)  # (batch, 64)

            # Project style to key/value for cross-attention
            style_kv = self.style_to_kv(feat_embed).unsqueeze(1)  # (batch, 1, hidden_size)

            # Cross-attention: CLS is query, style is key/value
            attn_output, _ = self.cross_attn(
                query=cls_pool.unsqueeze(1),  # (batch, 1, hidden_size)
                key=style_kv,                  # (batch, 1, hidden_size)
                value=style_kv,                # (batch, 1, hidden_size)
            )
            attn_output = attn_output.squeeze(1)  # (batch, hidden_size)

            # Gated residual connection
            gate_input = torch.cat([cls_pool, attn_output], dim=1)
            gate_value = self.gate_sigmoid(self.gate(gate_input))
            combined = gate_value * attn_output + (1 - gate_value) * cls_pool
        else:
            combined = cls_pool

        logits = self.head(combined)
        return logits


# Main execution
print("Starting Spooky Author Identification Pipeline")
print("=" * 50)

# Step 1: Data Processing and Feature Engineering
data_dict = load_and_process_data()

X_train = data_dict["X_train"]
X_val = data_dict["X_val"]
y_train = data_dict["y_train"]
y_val = data_dict["y_val"]
X_test = data_dict["X_test"]
test_ids = data_dict["test_ids"]
train_text = data_dict["train_text"]
val_text = data_dict["val_text"]
test_text = data_dict["test_text"]

NUM_FEATURES = X_train.shape[1]

# Step 2: Model Design and Training
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

# Create datasets and dataloaders
train_dataset = SpookyDataset(train_text, X_train, y_train)
val_dataset = SpookyDataset(val_text, X_val, y_val)
test_dataset = SpookyDataset(test_text, X_test)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SpookyAuthorModel(
    num_authors=3, num_stylometric_features=NUM_FEATURES, dropout_rate=0.3
)
model.to(device)

# Setup optimizer with differentiated learning rates
backbone_params = []
for layer in model.backbone.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_params.append(param)

head_params = list(model.head.parameters()) + list(model.feature_proj.parameters())

optimizer = AdamW(
    [
        {
            "params": backbone_params,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01, "betas": (0.9, 0.98)},
    ]
)

# Consistency loss: force model predictions to be invariant to word-level perturbations
class ConsistencyLoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, original_logits, perturbed_logits, labels, cross_entropy_loss):
        # Original and perturbed probabilities
        orig_probs = torch.softmax(original_logits.detach(), dim=1)  # detach to stop gradient
        perturbed_log_probs = torch.log_softmax(perturbed_logits, dim=1)
        # KL divergence: D(original || perturbed)
        kl_loss = self.kl_div(perturbed_log_probs, orig_probs)
        # Combined loss
        return cross_entropy_loss + self.weight * kl_loss

criterion_ce = nn.CrossEntropyLoss()
consistency_loss_fn = ConsistencyLoss(weight=0.5)

# Setup scheduler
num_epochs = 30
patience = 5
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
)
initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

# Mixed precision scaler
scaler_amp = GradScaler()

# Training loop
best_val_loss = float("inf")
best_val_score = float("inf")
epochs_no_improve = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features = batch["features"].to(device)

        optimizer.zero_grad()

        # Ensure features are valid (no NaN)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

        with autocast():
            # Original forward pass
            logits = model(input_ids, attention_mask, features)
            loss_ce = criterion_ce(logits, labels)

            # Create perturbed version: apply word-level dropout (mask tokens at rate 0.15)
            with torch.no_grad():
                # Create a random mask for input_ids
                mask_token_id = tokenizer.mask_token_id
                if mask_token_id is None:
                    mask_token_id = tokenizer.pad_token_id  # fallback
                # Randomly mask 15% of tokens (excluding padding and special tokens)
                special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                special_tokens_mask[:, 0] = True  # [CLS]
                # Find padding positions
                padding_mask = (input_ids == tokenizer.pad_token_id)
                special_tokens_mask = special_tokens_mask | padding_mask
                # Find [SEP] positions
                sep_token_id = tokenizer.sep_token_id
                special_tokens_mask = special_tokens_mask | (input_ids == sep_token_id)

                # Random masking only on non-special tokens
                random_mask = torch.rand_like(input_ids.float()) < 0.15
                random_mask = random_mask & (~special_tokens_mask)
                perturbed_input_ids = input_ids.clone()
                perturbed_input_ids[random_mask] = mask_token_id

            # Forward pass for perturbed input
            perturbed_logits = model(perturbed_input_ids, attention_mask, features)

            # Compute consistency loss
            loss = consistency_loss_fn(logits, perturbed_logits, labels, loss_ce)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        # Warmup + Cosine scheduling
        current_step = epoch * len(train_loader) + batch_idx
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            for pg_idx, pg in enumerate(optimizer.param_groups):
                pg["lr"] = initial_lrs[pg_idx] * warmup_factor
        else:
            scheduler.step(epoch + current_step / len(train_loader))

        total_train_loss += loss.item()
        train_batches += 1

    avg_train_loss = total_train_loss / train_batches

    # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device)
            # Ensure features are valid (no NaN)
            features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

            with autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion_ce(logits, labels)
                probs = torch.softmax(logits, dim=1)

            # Check for NaN in predictions and replace if needed
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
                # Renormalize
                probs = probs / probs.sum(dim=1, keepdim=True)

            val_loss += loss.item()
            val_batches += 1
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_probs = np.concatenate(all_val_probs, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)

    # Apply probability clipping and normalization as per competition rules
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

    # Compute multi-class log loss
    val_score = log_loss(val_labels, val_probs)

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Log Loss: {val_score:.6f} - LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Save best model based on log loss
    if val_score < best_val_score:
        best_val_score = val_score
        best_val_loss = avg_val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
        torch.save(best_model_state, "./working/best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Compute final validation score with best model
all_val_probs = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features = batch["features"].to(device)
        # Ensure features are valid (no NaN)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

        with autocast():
            logits = model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)

        # Check for NaN in predictions and replace if needed
        if torch.isnan(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0)
            # Renormalize
            probs = probs / probs.sum(dim=1, keepdim=True)

        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_labels = np.concatenate(all_val_labels, axis=0)

# Apply same clipping and normalization as test inference
val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
val_score = log_loss(val_labels, val_probs)

print(f"Final Validation Score: {val_score}")

# Test inference
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["features"].to(device)
        # Ensure features are valid (no NaN)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

        with autocast():
            logits = model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)

        # Check for NaN in predictions and replace if needed
        if torch.isnan(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0)
            # Renormalize
            probs = probs / probs.sum(dim=1, keepdim=True)

        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

# Apply probability clipping and normalization (same as validation)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# Create submission dataframe
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)
print(f"Sample predictions:\n{submission_df.head()}")