"""
Merged script: Spooky Author Identification using Multi-Level Neural Network
Combines character CNNs, word BiLSTM, and stylometric features
"""

import pandas as pd
import numpy as np
import os
import re
import gc
import warnings
import string
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import save_npz
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
NUM_AUTHORS = 3
MAX_WORD_LENGTH = 300
MAX_CHAR_LENGTH = 1500
CHAR_VOCAB_SIZE = 100
WORD_VOCAB_SIZE = 50000
EMBEDDING_DIM_CHAR = 32
EMBEDDING_DIM_WORD = 128
HIDDEN_DIM_LSTM = 256
CNN_FILTER_SIZES = [2, 3, 4, 5, 6]
CNN_NUM_FILTERS = 128
NUM_STYLOMETRIC_FEATURES = 48
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 100
PATIENCE = 7
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1
RANDOM_STATE = 42

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# PATH CONFIGURATION
# ============================================================
DATA_DIR = "./input"
WORKING_DIR = "./working"
OUTPUT_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(train_df["author"])
author_classes = label_encoder.classes_
author_mapping = dict(zip(author_classes, range(len(author_classes))))
print(f"Author mapping: {author_mapping}")

# ============================================================
# STRATIFIED SPLIT
# ============================================================
print("Creating stratified train/validation split...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, val_idx = next(skf.split(train_df, y_full))

train_texts = train_df["text"].values[train_idx]
train_labels = y_full[train_idx]
val_texts = train_df["text"].values[val_idx]
val_labels = y_full[val_idx]
test_texts = test_df["text"].values

assert (
    len(set(train_idx) & set(val_idx)) == 0
), "INDEX BUG: Train and validation overlap!"
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# ============================================================
# BUILD VOCABULARIES
# ============================================================
print("Building vocabularies from training data only...")

all_train_text_chars = "".join(train_texts)
char_counts = Counter(all_train_text_chars)
most_common_chars = [c for c, _ in char_counts.most_common(CHAR_VOCAB_SIZE - 4)]
char2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
char2idx.update({c: i + 4 for i, c in enumerate(most_common_chars)})
print(f"Character vocabulary size: {len(char2idx)}")

all_train_words = []
for text in train_texts:
    all_train_words.extend(text.lower().split())
word_counts = Counter(all_train_words)
most_common_words = [w for w, _ in word_counts.most_common(WORD_VOCAB_SIZE - 4)]
word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
word2idx.update({w: i + 4 for i, w in enumerate(most_common_words)})
print(f"Word vocabulary size: {len(word2idx)}")

def text_to_char_ids(text, max_len=MAX_CHAR_LENGTH):
    ids = [char2idx.get(c, char2idx["<UNK>"]) for c in str(text)[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids[:max_len]

def text_to_word_ids(text, max_len=MAX_WORD_LENGTH):
    words = str(text).lower().split()[:max_len]
    ids = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids[:max_len]

def create_word_mask(text, max_len=MAX_WORD_LENGTH):
    words = str(text).lower().split()[:max_len]
    mask = [1] * len(words) + [0] * (max_len - len(words))
    return mask[:max_len]

# ============================================================
# DISTILROBERTA TOKENIZATION
# ============================================================
def tokenize_for_roberta(texts, tokenizer, max_length=128):
    encoding = tokenizer(
        texts.tolist() if isinstance(texts, np.ndarray) else list(texts),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return encoding['input_ids'], encoding['attention_mask']

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def extract_stylometric_features(texts):
    features = []
    archaic_words = set(
        [
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "whence",
            "thence",
            "wherefore",
            "therein",
            "herein",
            "wherewith",
            "therewith",
        ]
    )
    emotional_words = set(
        [
            "nevermore",
            "horror",
            "terror",
            "dread",
            "fear",
            "ghastly",
            "hideous",
            "dismal",
            "dreary",
            "melancholy",
            "gloomy",
            "solemn",
            "shadow",
            "phantom",
        ]
    )
    lovecraft_words = set(
        [
            "cyclopean",
            "eldritch",
            "antediluvian",
            "ichor",
            "non-euclidean",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "nyarlathotep",
            "azathoth",
        ]
    )
    function_words = set(
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
            "was",
            "were",
            "had",
            "have",
            "has",
            "been",
            "being",
            "shall",
            "should",
            "will",
            "would",
            "can",
            "could",
            "may",
            "might",
            "do",
            "does",
            "did",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "then",
        ]
    )

    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features.append([0] * 30)
            continue
        text_lower = text.lower()
        total_chars = len(text)
        words = text.split()
        total_words = max(1, len(words))
        sents = re.split(r"[.!?]+", text)
        total_sents = max(1, len([s for s in sents if len(s.strip()) > 0]))

        avg_word_len = total_chars / total_words
        avg_sent_len = total_words / total_sents
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, total_chars)
        lower_ratio = sum(1 for c in text if c.islower()) / max(1, total_chars)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, total_chars)
        space_ratio = sum(1 for c in text if c.isspace()) / max(1, total_chars)

        punct_counts = Counter(text)
        dash_ratio = punct_counts.get("-", 0) / max(1, total_chars)
        semi_ratio = punct_counts.get(";", 0) / max(1, total_chars)
        colon_ratio = punct_counts.get(":", 0) / max(1, total_chars)
        excl_ratio = punct_counts.get("!", 0) / max(1, total_chars)
        comma_ratio = punct_counts.get(",", 0) / max(1, total_chars)
        quote_ratio = (punct_counts.get('"', 0) + punct_counts.get("'", 0)) / max(
            1, total_chars
        )

        char_diversity = len(set(text_lower)) / max(1, total_chars)
        long_words = sum(1 for w in words if len(w) > 8) / max(1, total_words)
        capitalized = sum(1 for w in words if w and w[0].isupper()) / max(
            1, total_words
        )
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1) / max(
            1, total_words
        )

        sent_lengths = [len(s.split()) for s in sents if len(s.strip()) > 0]
        sent_len_std = np.std(sent_lengths) if sent_lengths else 0

        function_word_ratio = sum(
            1 for w in words if w.lower() in function_words
        ) / max(1, total_words)
        archaic_ratio = sum(1 for w in words if w.lower() in archaic_words) / max(
            1, total_words
        )
        emotional_ratio = sum(1 for w in words if w.lower() in emotional_words) / max(
            1, total_words
        )
        lovecraft_ratio = sum(1 for w in words if w.lower() in lovecraft_words) / max(
            1, total_words
        )

        feat = [
            total_chars,
            total_words,
            total_sents,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            space_ratio,
            dash_ratio,
            semi_ratio,
            colon_ratio,
            excl_ratio,
            comma_ratio,
            quote_ratio,
            char_diversity,
            long_words,
            capitalized,
            all_caps,
            sent_len_std,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
        ]
        feat += [
            punct_counts.get(p, 0) / max(1, total_chars) for p in string.punctuation
        ][:6]
        features.append(feat[:30])
    return np.array(features)

def create_readability_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features.append([0] * 4)
            continue
        words = text.split()
        total_words = max(1, len(words))
        sents = re.split(r"[.!?]+", text)
        total_sents = max(1, len([s for s in sents if len(s.strip()) > 0]))
        total_chars = len(text.replace(" ", ""))

        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            if word and word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i - 1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if word.endswith("le") and len(word) > 2:
                count += 1
            return max(1, count)

        syllables = sum(count_syllables(w) for w in words)
        avg_syllables = syllables / total_words
        complex_words = sum(1 for w in words if count_syllables(w) > 2)
        complex_ratio = complex_words / total_words

        flesch = (
            206.835
            - 1.015 * (total_words / total_sents)
            - 84.6 * (syllables / total_words)
        )
        flesch = max(0, min(100, flesch))
        ari = (
            4.71 * (total_chars / total_words)
            + 0.5 * (total_words / total_sents)
            - 21.43
        )
        ari = max(0, ari)

        features.append([flesch, ari, avg_syllables, complex_ratio])
    return np.array(features)

def create_pos_tag_approximation(texts):
    features = []
    noun_suffixes = [
        "tion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "sion",
        "ship",
        "dom",
        "hood",
    ]
    verb_suffixes = ["ed", "ing", "ate", "ize", "ify", "en", "ish", "ise"]
    adj_suffixes = [
        "able",
        "ible",
        "ous",
        "ive",
        "ful",
        "less",
        "ic",
        "al",
        "ent",
        "ant",
        "ish",
        "ive",
    ]
    adv_suffixes = ["ly", "ward", "wise"]

    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features.append([0] * 5)
            continue
        words = text.split()
        total_words = max(1, len(words))
        noun_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in noun_suffixes)
        )
        verb_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in verb_suffixes)
        )
        adj_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in adj_suffixes)
        )
        adv_count = sum(
            1 for w in words if any(w.lower().endswith(s) for s in adv_suffixes)
        )
        content_words = sum(1 for w in words if len(w) > 4)
        content_ratio = content_words / total_words
        features.append(
            [
                noun_count / total_words,
                verb_count / total_words,
                adj_count / total_words,
                adv_count / total_words,
                content_ratio,
            ]
        )
    return np.array(features)

# ============================================================
# EXTRACT ENGINEERED FEATURES
# ============================================================
print("Extracting engineered features...")
train_stylo = extract_stylometric_features(train_texts)
val_stylo = extract_stylometric_features(val_texts)
test_stylo = extract_stylometric_features(test_texts)

train_read = create_readability_features(train_texts)
val_read = create_readability_features(val_texts)
test_read = create_readability_features(test_texts)

train_pos = create_pos_tag_approximation(train_texts)
val_pos = create_pos_tag_approximation(val_texts)
test_pos = create_pos_tag_approximation(test_texts)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

vt = VarianceThreshold(threshold=0.001)
train_stylo_filtered = vt.fit_transform(train_stylo_scaled)
val_stylo_filtered = vt.transform(val_stylo_scaled)
test_stylo_filtered = vt.transform(test_stylo_scaled)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

train_engineered = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled]
)
val_engineered = np.hstack([val_stylo_filtered, val_read_scaled, val_pos_scaled])
test_engineered = np.hstack([test_stylo_filtered, test_read_scaled, test_pos_scaled])

NUM_STYLOMETRIC_FEATURES = train_engineered.shape[1]
print(f"Total engineered features: {NUM_STYLOMETRIC_FEATURES}")

# ============================================================
# PREPARE TOKENIZED DATA
# ============================================================
print("Preparing tokenized data...")
train_char_ids = np.array([text_to_char_ids(t) for t in train_texts])
val_char_ids = np.array([text_to_char_ids(t) for t in val_texts])
test_char_ids = np.array([text_to_char_ids(t) for t in test_texts])

train_word_ids = np.array([text_to_word_ids(t) for t in train_texts])
val_word_ids = np.array([text_to_word_ids(t) for t in val_texts])
test_word_ids = np.array([text_to_word_ids(t) for t in test_texts])

train_word_mask = np.array([create_word_mask(t) for t in train_texts], dtype=bool)
val_word_mask = np.array([create_word_mask(t) for t in val_texts], dtype=bool)
test_word_mask = np.array([create_word_mask(t) for t in test_texts], dtype=bool)

# ============================================================
# DISTILROBERTA TOKENIZATION
# ============================================================
print("Tokenizing for DistilRoBERTa...")
from transformers import AutoTokenizer
roberta_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
train_input_ids, train_attention_mask = tokenize_for_roberta(train_texts, roberta_tokenizer)
val_input_ids, val_attention_mask = tokenize_for_roberta(val_texts, roberta_tokenizer)
test_input_ids, test_attention_mask = tokenize_for_roberta(test_texts, roberta_tokenizer)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class CharCNNEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, filter_sizes, num_filters, dropout_rate
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    padding=0,
                )
                for fs in filter_sizes
            ]
        )
        self.output_dim = len(filter_sizes) * num_filters
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        embedded = self.char_embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            # Use adaptive pooling to ensure consistent output size regardless of input length
            pooled_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            conv_outputs.append(pooled_out)
        out = torch.cat(conv_outputs, dim=1)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out

class DistilRoBERTaWordEncoder(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        self.roberta = AutoModel.from_pretrained('distilroberta-base')
        # Freeze first 4 layers (layers 0-3)
        for i in range(4):
            self.roberta.encoder.layer[i].requires_grad_(False)
        # Fine-tune top 2 layers (layers 4-5) with stochastic depth
        self.output_dim = 256
        self.projection = nn.Linear(768, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(256)
        # Stochastic depth survival probabilities
        self.p4 = 0.9
        self.p5 = 0.8
        self.training = True

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len] already tokenized
        # attention_mask: [batch_size, seq_len] with 1 for valid tokens, 0 for padding
        # Get hidden states from all layers
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (embedding + 6 layer outputs)

        # Apply stochastic depth to layers 4 and 5 during training
        if self.training:
            batch_size = input_ids.size(0)
            device = input_ids.device
            # Layer 4 (index 4 in hidden_states, corresponds to encoder.layer[3] output)
            layer4_out = hidden_states[4]  # [batch_size, seq_len, 768]
            survival_mask4 = torch.rand(batch_size, 1, 1, device=device) < self.p4
            layer4_out = layer4_out * survival_mask4.float() / self.p4
            hidden_states = list(hidden_states)
            hidden_states[4] = layer4_out

            # Layer 5 (index 5 in hidden_states, corresponds to encoder.layer[4] output)
            layer5_out = hidden_states[5]  # [batch_size, seq_len, 768]
            survival_mask5 = torch.rand(batch_size, 1, 1, device=device) < self.p5
            layer5_out = layer5_out * survival_mask5.float() / self.p5
            hidden_states[5] = layer5_out

        # Use CLS token representation (first token) from last hidden state
        last_hidden = hidden_states[-1] if isinstance(hidden_states, tuple) else outputs.last_hidden_state
        cls_output = last_hidden[:, 0, :]  # [batch_size, 768]
        projected = self.projection(cls_output)  # [batch_size, 256]
        out = self.dropout(projected)
        out = self.layer_norm(out)
        return out

class StylometricEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.output_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MultiLevelAuthorClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.char_encoder = CharCNNEncoder(
            vocab_size=config["char_vocab_size"],
            embedding_dim=config["embedding_dim_char"],
            filter_sizes=config["cnn_filter_sizes"],
            num_filters=config["cnn_num_filters"],
            dropout_rate=config["dropout_rate"],
        )
        self.word_encoder = DistilRoBERTaWordEncoder(
            dropout_rate=config["dropout_rate"],
        )
        self.stylometric_encoder = StylometricEncoder(
            input_dim=config["num_stylometric_features"],
            hidden_dim=64,
            dropout_rate=config["dropout_rate"],
        )
        total_features = (
            len(config["cnn_filter_sizes"]) * config["cnn_num_filters"]
            + self.word_encoder.output_dim
            + self.stylometric_encoder.output_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(128, config["num_classes"]),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, char_ids, input_ids, attention_mask=None, stylometric_features=None):
        char_features = self.char_encoder(char_ids)
        word_features = self.word_encoder(input_ids=input_ids, attention_mask=attention_mask)
        stylo_features = self.stylometric_encoder(stylometric_features)
        combined = torch.cat([char_features, word_features, stylo_features], dim=1)
        logits = self.classifier(combined)
        return logits

model_config = {
    "char_vocab_size": len(char2idx),
    "embedding_dim_char": EMBEDDING_DIM_CHAR,
    "embedding_dim_word": EMBEDDING_DIM_WORD,
    "hidden_dim_lstm": HIDDEN_DIM_LSTM,
    "cnn_filter_sizes": CNN_FILTER_SIZES,
    "cnn_num_filters": CNN_NUM_FILTERS,
    "num_stylometric_features": NUM_STYLOMETRIC_FEATURES,
    "dropout_rate": DROPOUT_RATE,
    "num_classes": NUM_AUTHORS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
}

model = MultiLevelAuthorClassifier(model_config)
model.to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================
# SET UP OPTIMIZER, LOSS, SCHEDULER
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

no_decay = ["bias", "LayerNorm.weight", "BatchNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

# ============================================================
# CREATE DATALOADERS
# ============================================================
train_dataset = TensorDataset(
    torch.tensor(train_char_ids, dtype=torch.long),
    train_input_ids,
    train_attention_mask,
    torch.tensor(train_engineered, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(val_char_ids, dtype=torch.long),
    val_input_ids,
    val_attention_mask,
    torch.tensor(val_engineered, dtype=torch.float32),
    torch.tensor(val_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MULTI-LEVEL CLASSIFIER")
print("=" * 60)

def compute_log_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(NUM_AUTHORS):
            if y_true[i] == j:
                loss -= np.log(y_pred_proba[i, j])
    return loss / n

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            char_ids, input_ids, attention_mask, stylo_feat, labels = [
                b.to(device) for b in batch
            ]
            with autocast():
                logits = model(char_ids, input_ids, attention_mask, stylo_feat)
                probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

scaler = GradScaler()
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
# Linear warmup followed by linear decay
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        char_ids, input_ids, attention_mask, stylo_feat, labels = [
            b.to(device) for b in batch
        ]
        optimizer.zero_grad()
        with autocast():
            logits = model(char_ids, input_ids, attention_mask, stylo_feat)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, _ = evaluate(model, val_loader)

    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), f"{WORKING_DIR}/best_model.pt")
        print(f"  -> New best model saved (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(
    f"\nTraining complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
)

# ============================================================
# LOAD BEST MODEL AND COMPUTE FINAL METRICS
# ============================================================
print("Loading best model...")
model.load_state_dict(torch.load(f"{WORKING_DIR}/best_model.pt", map_location=device))
model.eval()

val_logloss, val_accuracy, val_probs = evaluate(model, val_loader)
print(f"Final Validation - Log Loss: {val_logloss:.6f}, Accuracy: {val_accuracy:.4f}")

# ============================================================
# TEST INFERENCE
# ============================================================
print("Performing test inference...")
test_dataset = TensorDataset(
    torch.tensor(test_char_ids, dtype=torch.long),
    test_input_ids,
    test_attention_mask,
    torch.tensor(test_engineered, dtype=torch.float32),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        char_ids, input_ids, attention_mask, stylo_feat = [b.to(device) for b in batch]
        with autocast():
            logits = model(char_ids, input_ids, attention_mask, stylo_feat)
            probs = F.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)

eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("Generating submission file...")
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

submission_df.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
print(f"Submission saved to {OUTPUT_DIR}/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# CLEANUP
# ============================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {val_logloss:.6f}")
