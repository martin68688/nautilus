import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from transformers import AutoModel, AutoTokenizer
import random
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_sub.shape}")


# ============ FEATURE ENGINEERING ============
def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2:
        count += 1
    if count == 0:
        count += 1
    return count


def extract_stylometric_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_length"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.count("[.!?]") + 1
    features["avg_sentence_length"] = features["word_count"] / (
        features["sentence_count"] + 1
    )
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["period_count"] = text_series.str.count(r"\.")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["dash_count"] = text_series.str.count("-")
    features["quote_count"] = text_series.str.count('"')
    features["apostrophe_count"] = text_series.str.count("'")
    features["parenthesis_count"] = text_series.str.count(r"[()]")
    features["punctuation_ratio"] = features[
        [
            "exclamation_count",
            "question_count",
            "comma_count",
            "semicolon_count",
            "colon_count",
            "period_count",
        ]
    ].sum(axis=1) / (features["char_count"] + 1)
    features["capitalized_words"] = text_series.str.findall(
        r"\b[A-Z][a-z]*\b"
    ).str.len()
    features["all_caps_words"] = text_series.str.findall(r"\b[A-Z]{2,}\b").str.len()
    features["capital_ratio"] = features["capitalized_words"] / (
        features["word_count"] + 1
    )
    features["unique_words"] = text_series.apply(lambda x: len(set(x.lower().split())))
    features["lexical_diversity"] = features["unique_words"] / (
        features["word_count"] + 1
    )
    features["long_words"] = text_series.apply(
        lambda x: len([w for w in x.split() if len(w) > 6])
    )
    features["long_word_ratio"] = features["long_words"] / (features["word_count"] + 1)
    features["very_long_words"] = text_series.apply(
        lambda x: len([w for w in x.split() if len(w) > 10])
    )
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
            "was",
            "had",
            "have",
            "has",
            "not",
            "that",
            "this",
            "it",
            "its",
            "he",
            "she",
            "they",
            "we",
            "you",
            "i",
            "my",
            "me",
            "his",
            "her",
            "their",
            "our",
            "your",
            "be",
            "been",
            "being",
            "is",
            "are",
            "were",
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
            "so",
            "if",
            "then",
            "than",
            "too",
            "very",
            "just",
            "about",
            "up",
            "out",
            "no",
            "nor",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "some",
            "any",
            "such",
            "only",
            "own",
            "same",
            "here",
            "there",
            "where",
            "when",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "after",
            "before",
            "between",
            "through",
            "during",
            "above",
            "below",
            "over",
            "under",
            "again",
            "further",
            "once",
            "now",
            "then",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "both",
            "each",
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
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "down",
            "up",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
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
            "until",
            "while",
        ]
    )
    features["stopword_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in stop_words)
    )
    features["stopword_ratio"] = features["stopword_count"] / (
        features["word_count"] + 1
    )
    features["the_ratio"] = text_series.str.lower().str.count(r"\bthe\b") / (
        features["word_count"] + 1
    )
    features["and_ratio"] = text_series.str.lower().str.count(r"\band\b") / (
        features["word_count"] + 1
    )
    features["of_ratio"] = text_series.str.lower().str.count(r"\bof\b") / (
        features["word_count"] + 1
    )
    features["to_ratio"] = text_series.str.lower().str.count(r"\bto\b") / (
        features["word_count"] + 1
    )
    features["in_ratio"] = text_series.str.lower().str.count(r"\bin\b") / (
        features["word_count"] + 1
    )
    features["was_ratio"] = text_series.str.lower().str.count(r"\bwas\b") / (
        features["word_count"] + 1
    )
    features["had_ratio"] = text_series.str.lower().str.count(r"\bhad\b") / (
        features["word_count"] + 1
    )
    features["with_ratio"] = text_series.str.lower().str.count(r"\bwith\b") / (
        features["word_count"] + 1
    )
    pronouns = {
        "i": r"\bi\b",
        "we": r"\bwe\b",
        "you": r"\byou\b",
        "he": r"\bhe\b",
        "she": r"\bshe\b",
        "they": r"\bthey\b",
        "it": r"\bit\b",
        "my": r"\bmy\b",
        "his": r"\bhis\b",
        "her": r"\bher\b",
        "our": r"\bour\b",
        "their": r"\btheir\b",
    }
    for pronoun, pattern in pronouns.items():
        features[f"pronoun_{pronoun}"] = text_series.str.lower().str.count(pattern)
    features["first_person_pronouns"] = features[
        ["pronoun_i", "pronoun_we", "pronoun_my", "pronoun_our"]
    ].sum(axis=1)
    features["third_person_pronouns"] = features[
        [
            "pronoun_he",
            "pronoun_she",
            "pronoun_they",
            "pronoun_his",
            "pronoun_her",
            "pronoun_their",
        ]
    ].sum(axis=1)
    features["short_words"] = text_series.apply(
        lambda x: len([w for w in x.split() if len(w) <= 3])
    )
    features["medium_words"] = text_series.apply(
        lambda x: len([w for w in x.split() if 4 <= len(w) <= 6])
    )
    features["short_word_ratio"] = features["short_words"] / (
        features["word_count"] + 1
    )
    features["medium_word_ratio"] = features["medium_words"] / (
        features["word_count"] + 1
    )
    features["number_count"] = text_series.str.count(r"\d+")
    features["spaces_per_char"] = text_series.str.count(" ") / (
        features["char_count"] + 1
    )
    features["positive_words"] = text_series.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w
            in [
                "love",
                "happy",
                "beautiful",
                "wonderful",
                "great",
                "good",
                "joy",
                "hope",
                "peace",
                "light",
                "glad",
                "gentle",
                "kind",
                "sweet",
                "fair",
                "faith",
                "grace",
                "bless",
                "bright",
                "golden",
            ]
        )
    )
    features["negative_words"] = text_series.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w
            in [
                "dark",
                "death",
                "fear",
                "terror",
                "horror",
                "pain",
                "sorrow",
                "grief",
                "shadow",
                "gloom",
                "dread",
                "doom",
                "ghost",
                "grave",
                "corpse",
                "agony",
                "anguish",
                "woe",
                "curse",
                "damned",
            ]
        )
    )
    features["syllables_est"] = text_series.apply(
        lambda x: sum(count_syllables(w) for w in x.split())
    )
    features["readability"] = (
        206.835
        - 1.015 * features["avg_sentence_length"]
        - 84.6 * (features["syllables_est"] / (features["word_count"] + 1))
    )
    features["dialogue_markers"] = text_series.str.count(r'["\'“”‘’]')
    return features


print("Extracting stylometric features...")
stylo_train = extract_stylometric_features(train_df["text"])
stylo_test = extract_stylometric_features(test_df["text"])
num_stylometric_features = stylo_train.shape[1]
print(f"Number of stylometric features: {num_stylometric_features}")


# ============ MODEL ARCHITECTURE ============
class StylometricCNN(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=64, num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in [2, 3, 4, 5]]
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * 4, 128)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class StylometricMLP(nn.Module):
    def __init__(self, num_stylometric_features, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_stylometric_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Projections for query (from BERT), key/value (from CNN+MLP)
        self.query_proj = nn.Linear(768, hidden_dim)
        self.key_proj = nn.Linear(128 + 64, hidden_dim)
        self.value_proj = nn.Linear(128 + 64, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temperature parameter for gating
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, bert_feat, cnn_feat, mlp_feat):
        # bert_feat: (batch, 768)
        # cnn_feat: (batch, 128)
        # mlp_feat: (batch, 64)
        batch_size = bert_feat.size(0)

        # Concatenate CNN and MLP features
        fused_custom = torch.cat([cnn_feat, mlp_feat], dim=1)  # (batch, 192)

        # Projections
        Q = self.query_proj(bert_feat)  # (batch, hidden_dim)
        K = self.key_proj(fused_custom)  # (batch, hidden_dim)
        V = self.value_proj(fused_custom)  # (batch, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, 1, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, 1, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, 1, head_dim)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, 1, 1)
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (batch, heads, 1, 1)

        # Weighted sum
        attn_output = torch.matmul(attn_probs, V)  # (batch, heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)  # (batch, 1, hidden_dim)
        attn_output = attn_output.squeeze(1)  # (batch, hidden_dim)

        # Output projection
        fused_repr = self.out_proj(attn_output)  # (batch, hidden_dim)

        # Gating mechanism with temperature
        gate_scores = self.gate(fused_repr)  # (batch, hidden_dim)
        temperature_scaled = torch.sigmoid(gate_scores / self.temperature)  # (batch, hidden_dim)

        # Residual connection: scale original bert features and add attended features
        combined_repr = temperature_scaled * fused_repr + (1 - temperature_scaled) * self.query_proj(bert_feat)

        return combined_repr


class MultiViewAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, num_stylometric_features=100):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.bert_hidden_size = 768
        self.char_cnn = StylometricCNN(vocab_size=128, embed_dim=64)
        self.stylo_mlp = StylometricMLP(
            num_stylometric_features=num_stylometric_features
        )
        self.fusion = CrossAttentionFusion(hidden_dim=256, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_authors),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, attention_mask, stylo_features=None, char_ids=None):
        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        bert_pooled = bert_outputs.last_hidden_state[:, 0, :]
        if char_ids is None:
            char_ids = input_ids
        char_features = self.char_cnn(char_ids)
        if stylo_features is not None:
            stylo_out = self.stylo_mlp(stylo_features)
        else:
            stylo_out = torch.zeros((input_ids.size(0), 64), device=input_ids.device)
        fused_repr = self.fusion(bert_pooled, char_features, stylo_out)
        logits = self.classifier(fused_repr)
        return logits


# ============ DATA PREPARATION ============
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_map)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(train_df["text"], train_df["label"]))

train_texts = train_df.iloc[train_idx]["text"].tolist()
train_labels = train_df.iloc[train_idx]["label"].values
val_texts = train_df.iloc[val_idx]["text"].tolist()
val_labels = train_df.iloc[val_idx]["label"].values
test_texts = test_df["text"].tolist()
test_ids = test_df["id"].tolist()

stylo_train_split = stylo_train.iloc[train_idx].values.astype(np.float32)
stylo_val_split = stylo_train.iloc[val_idx].values.astype(np.float32)
stylo_test_arr = stylo_test.values.astype(np.float32)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# Tokenizer
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_texts(texts, max_length=256):
    return text_tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

def tokenize_characters(texts, max_length=512, vocab=128):
    """Convert text to character-level token IDs (0-127 for ASCII, 0 for unknown, 1 for padding)."""
    ids_list = []
    for text in texts:
        ids = [ord(c) % vocab for c in text[:max_length]]
        # Pad to max_length
        if len(ids) < max_length:
            ids = ids + [1] * (max_length - len(ids))
        ids_list.append(ids)
    return torch.tensor(ids_list, dtype=torch.long)

train_char_ids = tokenize_characters(train_texts)
val_char_ids = tokenize_characters(val_texts)
test_char_ids = tokenize_characters(test_texts)


train_encodings = tokenize_texts(train_texts)
val_encodings = tokenize_texts(val_texts)
test_encodings = tokenize_texts(test_texts)

# Datasets
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_char_ids,
    torch.tensor(train_labels, dtype=torch.long),
    torch.tensor(stylo_train_split),
)

val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    val_char_ids,
    torch.tensor(val_labels, dtype=torch.long),
    torch.tensor(stylo_val_split),
)

test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
    test_char_ids,
    torch.tensor(stylo_test_arr),
)

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)

# ============ MODEL SETUP ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultiViewAuthorClassifier(
    num_authors=3, num_stylometric_features=num_stylometric_features
)
model.to(device)

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

def gradient_normalize(model, bert_scale=1.0, custom_scale=1.0):
    """Scale gradients so that BERT and custom contributions have equal norm."""
    bert_norm = 0.0
    custom_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'bert_model' in name:
                bert_norm += param.grad.norm().item() ** 2
            else:
                custom_norm += param.grad.norm().item() ** 2
    bert_norm = bert_norm ** 0.5
    custom_norm = custom_norm ** 0.5
    if bert_norm > 0 and custom_norm > 0:
        total_norm = (bert_norm + custom_norm) / 2.0
        bert_scale_factor = total_norm / bert_norm
        custom_scale_factor = total_norm / custom_norm
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'bert_model' in name:
                    param.grad.mul_(bert_scale_factor * bert_scale)
                else:
                    param.grad.mul_(custom_scale_factor * custom_scale)

bert_params = list(model.bert_model.parameters())
custom_params = (
    list(model.char_cnn.parameters())
    + list(model.stylo_mlp.parameters())
    + list(model.fusion.parameters())
    + list(model.classifier.parameters())
)

# Phase 1: freeze BERT, train heads and fusion
for param in bert_params:
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    custom_params, lr=2e-4, weight_decay=0.001
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)
scaler = GradScaler()

# ============ TRAINING ============
num_epochs = 25
best_val_loss = float("inf")
best_model_state = None
patience = 7
patience_counter = 0
phase1_epochs = 3

for epoch in range(1, num_epochs + 1):
    # Phase transition: unfreeze BERT at epoch 4
    if epoch == phase1_epochs + 1:
        print("Phase 2: Unfreezing BERT with lower learning rate and gradient surgery")
        for param in bert_params:
            param.requires_grad = True
        # Recreate optimizer with both parameter groups
        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": 2e-5, "weight_decay": 0.01},
                {"params": custom_params, "lr": 2e-4, "weight_decay": 0.001},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        # Reset scaler for new optimizer
        scaler = GradScaler()

    model.train()
    total_train_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, char_ids, labels, stylo_feats = [b.to(device) for b in batch]

        optimizer.zero_grad()
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylo_features=stylo_feats,
                char_ids=char_ids,
            )
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Apply gradient clipping with max_norm=0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        # Apply gradient normalization for balanced update
        gradient_normalize(model, bert_scale=1.0, custom_scale=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, char_ids, labels, stylo_feats = [
                b.to(device) for b in batch
            ]
            with autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stylo_features=stylo_feats,
                    char_ids=char_ids,
                )
                loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            total_val_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_val_preds.extend(probs)
            all_val_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    all_val_preds = np.array(all_val_preds)
    all_val_labels = np.array(all_val_labels)
    eps = 1e-15
    all_val_preds_clipped = np.clip(all_val_preds, eps, 1 - eps)
    all_val_preds_clipped /= all_val_preds_clipped.sum(axis=1, keepdims=True)
    val_log_loss = log_loss(all_val_labels, all_val_preds_clipped)

    print(
        f"Epoch {epoch:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Log Loss: {val_log_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "./working/best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

# ============ INFERENCE ============
print("Loading best model for inference...")
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

# Validation predictions
all_val_preds = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, char_ids, labels, stylo_feats = [b.to(device) for b in batch]
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylo_features=stylo_feats,
                char_ids=char_ids,
            )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_val_preds.extend(probs)
        all_val_labels.extend(labels.cpu().numpy())

all_val_preds = np.array(all_val_preds)
all_val_labels = np.array(all_val_labels)
eps = 1e-15
all_val_preds_clipped = np.clip(all_val_preds, eps, 1 - eps)
all_val_preds_clipped /= all_val_preds_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(all_val_labels, all_val_preds_clipped)
print(f"Final Validation Score: {final_val_score}")

# Test predictions
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, char_ids, stylo_feats = [b.to(device) for b in batch]
        with autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylo_features=stylo_feats,
                char_ids=char_ids,
            )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_preds.extend(probs)

all_test_preds = np.array(all_test_preds)
all_test_preds = np.clip(all_test_preds, eps, 1 - eps)
all_test_preds /= all_test_preds.sum(axis=1, keepdims=True)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {final_val_score}")
