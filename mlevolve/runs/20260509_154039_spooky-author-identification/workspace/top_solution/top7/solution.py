import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
import re
import string
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import log_loss
import os
import gc
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# ============================================================
# 2. AUTHOR LABEL ENCODING
# ============================================================
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["label"] = train_df["author"].map(author_map)
y = train_df["label"].values

# ============================================================
# 3. STRATIFIED SPLIT (train 80%, val 10%, test 10%)
# ============================================================
X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
    train_df["text"].values, y, test_size=0.2, stratify=y, random_state=42
)

X_val_texts, X_test_texts, y_val, y_test = train_test_split(
    X_temp_texts, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


# ============================================================
# 4. FEATURE ENGINEERING FUNCTIONS
# ============================================================
def extract_sentence_stats(texts):
    """Extract numerical sentence-level statistics that capture authorial style."""
    features = []
    for text in texts:
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        char_count = len(text)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Punctuation frequencies (per character)
        comma_freq = text.count(",") / max(char_count, 1)
        period_freq = text.count(".") / max(char_count, 1)
        exclaim_freq = text.count("!") / max(char_count, 1)
        question_freq = text.count("?") / max(char_count, 1)
        semicolon_freq = text.count(";") / max(char_count, 1)
        colon_freq = text.count(":") / max(char_count, 1)
        dash_freq = text.count("—") + text.count("–") + text.count("-")
        dash_freq /= max(char_count, 1)
        quote_freq = text.count('"') + text.count("'") / max(char_count, 1)

        # Capitalization patterns
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(
            len([c for c in text if c.isalpha()]), 1
        )
        first_word_cap_ratio = sum(1 for s in sentences if s and s[0].isupper()) / max(
            sentence_count, 1
        )

        # Function word density (common author-specific patterns)
        function_words = set(
            [
                "the",
                "and",
                "of",
                "to",
                "a",
                "in",
                "that",
                "was",
                "is",
                "it",
                "with",
                "as",
                "for",
                "on",
                "but",
                "by",
                "his",
                "had",
                "not",
                "be",
                "at",
                "from",
                "were",
                "are",
                "so",
                "have",
                "has",
                "been",
                "or",
                "he",
                "you",
                "I",
                "my",
                "me",
                "she",
                "her",
                "they",
                "them",
                "their",
                "we",
                "our",
                "us",
                "all",
                "an",
                "no",
                "do",
                "did",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "will",
                "can",
                "must",
                "now",
                "then",
                "there",
                "here",
                "very",
                "such",
                "some",
                "any",
                "every",
                "each",
                "both",
                "own",
                "this",
                "that",
                "these",
                "those",
                "who",
                "whom",
                "whose",
                "which",
                "what",
                "when",
                "where",
                "why",
                "how",
                "if",
                "though",
                "while",
                "because",
                "since",
                "until",
                "before",
                "after",
                "about",
                "between",
                "through",
                "during",
                "without",
                "within",
                "upon",
                "into",
                "over",
                "under",
                "above",
                "below",
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
                "also",
                "moreover",
                "furthermore",
                "however",
                "therefore",
                "thus",
                "nevertheless",
                "meanwhile",
                "afterward",
            ]
        )
        token_lower = [w.lower().strip(string.punctuation) for w in words]
        fn_words = [w for w in token_lower if w in function_words]
        func_word_ratio = len(fn_words) / max(len(token_lower), 1)

        # Unique word ratio (vocabulary richness)
        unique_words = len(set(token_lower))
        vocab_richness = unique_words / max(word_count, 1)

        # Stopword ratio
        stop_words = set(
            [
                "the",
                "and",
                "of",
                "to",
                "a",
                "in",
                "that",
                "is",
                "was",
                "it",
                "with",
                "as",
                "for",
                "on",
                "but",
                "by",
                "his",
                "had",
                "not",
                "be",
                "at",
                "from",
                "were",
                "are",
                "so",
                "have",
                "has",
                "been",
                "or",
                "he",
                "you",
                "i",
                "my",
                "me",
                "she",
                "her",
                "they",
                "them",
                "their",
                "we",
                "our",
                "us",
                "all",
                "an",
                "no",
                "do",
                "did",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "will",
                "can",
                "must",
                "now",
                "then",
                "there",
                "here",
                "very",
                "such",
                "some",
                "any",
                "every",
                "each",
                "both",
                "own",
                "this",
                "that",
                "these",
                "those",
                "who",
                "whom",
                "whose",
                "which",
                "what",
                "when",
                "where",
                "why",
                "how",
                "if",
                "though",
                "while",
                "because",
                "since",
                "until",
                "before",
                "after",
                "about",
                "between",
                "through",
                "during",
                "without",
                "within",
                "upon",
                "into",
                "over",
                "under",
                "above",
                "below",
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
                "also",
                "moreover",
                "furthermore",
                "however",
                "therefore",
                "thus",
                "nevertheless",
                "meanwhile",
                "afterward",
            ]
        )
        token_lower_words = [
            w.lower().strip(string.punctuation) for w in words if w.strip()
        ]
        stopword_count = sum(1 for w in token_lower_words if w in stop_words)
        stopword_ratio = stopword_count / max(len(token_lower_words), 1)

        features.append(
            [
                char_count,
                word_count,
                sentence_count,
                avg_word_len,
                avg_sent_len,
                comma_freq,
                period_freq,
                exclaim_freq,
                question_freq,
                semicolon_freq,
                colon_freq,
                dash_freq,
                quote_freq,
                uppercase_ratio,
                first_word_cap_ratio,
                func_word_ratio,
                vocab_richness,
                stopword_ratio,
            ]
        )
    return np.array(features)


def extract_char_ngrams(texts, vectorizer=None, fit=False):
    """Extract character n-grams (2-5 grams) that capture morphology and punctuation patterns."""
    if fit:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=15000,
            sublinear_tf=True,
            min_df=5,
            max_df=0.8,
        )
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    else:
        return vectorizer.transform(texts)


def extract_word_ngrams(texts, vectorizer=None, fit=False):
    """Extract word n-grams (1-3 grams) that capture vocabulary and phrase patterns."""
    if fit:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=15000,
            sublinear_tf=True,
            min_df=5,
            max_df=0.8,
            stop_words=None,
        )
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    else:
        return vectorizer.transform(texts)


def extract_pos_patterns(texts, vectorizer=None, fit=False):
    """
    Extract POS tag n-grams to capture syntactic patterns.
    Using a simple heuristic: tag words based on suffixes and patterns.
    """

    def tag_sentence(text):
        words = text.split()
        tags = []
        for w in words:
            w_clean = w.lower().strip(string.punctuation)
            if not w_clean:
                tags.append("PUNCT")
            elif w_clean in ["the", "a", "an"]:
                tags.append("DET")
            elif w_clean in ["is", "am", "are", "was", "were", "be", "been", "being"]:
                tags.append("VERB_BE")
            elif w_clean in ["have", "has", "had", "having"]:
                tags.append("VERB_HAVE")
            elif w_clean in ["do", "does", "did", "doing", "done"]:
                tags.append("VERB_DO")
            elif w_clean in [
                "will",
                "would",
                "can",
                "could",
                "shall",
                "should",
                "may",
                "might",
                "must",
            ]:
                tags.append("MODAL")
            elif w_clean.endswith("ing"):
                tags.append("VBG")
            elif w_clean.endswith("ed"):
                tags.append("VBD")
            elif w_clean.endswith("ly"):
                tags.append("ADV")
            elif (
                w_clean.endswith("tion")
                or w_clean.endswith("sion")
                or w_clean.endswith("ment")
            ):
                tags.append("NN")
            elif (
                w_clean.endswith("ness")
                or w_clean.endswith("ity")
                or w_clean.endswith("ship")
            ):
                tags.append("NN")
            elif (
                w_clean.endswith("able")
                or w_clean.endswith("ible")
                or w_clean.endswith("ful")
            ):
                tags.append("ADJ")
            elif (
                w_clean.endswith("ous")
                or w_clean.endswith("ive")
                or w_clean.endswith("al")
            ):
                tags.append("ADJ")
            elif w_clean.endswith("s") and len(w_clean) > 2:
                tags.append("NNS")
            elif len(w_clean) <= 3 and w_clean.isalpha():
                tags.append("FUNC")
            elif w_clean[0].isupper() and len(w_clean) > 1:
                tags.append("NNP")
            else:
                tags.append("OTHER")
        return " ".join(tags)

    pos_texts = [tag_sentence(t) for t in texts]

    if fit:
        vectorizer = CountVectorizer(
            ngram_range=(2, 4), max_features=5000, min_df=3, max_df=0.9
        )
        features = vectorizer.fit_transform(pos_texts)
        return features, vectorizer
    else:
        return vectorizer.transform(pos_texts)


# ============================================================
# 5. EXTRACT FEATURES (FIT ONLY ON TRAINING DATA)
# ============================================================
# Sentence-level statistics
train_stats = extract_sentence_stats(X_train_texts)
val_stats = extract_sentence_stats(X_val_texts)
test_stats = extract_sentence_stats(X_test_texts)

# Character n-grams (fit on training only)
char_features_train, char_vectorizer = extract_char_ngrams(X_train_texts, fit=True)
char_features_val = extract_char_ngrams(X_val_texts, vectorizer=char_vectorizer)
char_features_test = extract_char_ngrams(X_test_texts, vectorizer=char_vectorizer)

# Word n-grams (fit on training only)
word_features_train, word_vectorizer = extract_word_ngrams(X_train_texts, fit=True)
word_features_val = extract_word_ngrams(X_val_texts, vectorizer=word_vectorizer)
word_features_test = extract_word_ngrams(X_test_texts, vectorizer=word_vectorizer)

# POS pattern features (fit on training only)
pos_features_train, pos_vectorizer = extract_pos_patterns(X_train_texts, fit=True)
pos_features_val = extract_pos_patterns(X_val_texts, vectorizer=pos_vectorizer)
pos_features_test = extract_pos_patterns(X_test_texts, vectorizer=pos_vectorizer)

# ============================================================
# 6. COMBINE ALL FEATURES INTO SPARSE MATRIX
# ============================================================
train_stats_sparse = csr_matrix(train_stats)
val_stats_sparse = csr_matrix(val_stats)
test_stats_sparse = csr_matrix(test_stats)

X_train = hstack(
    [train_stats_sparse, char_features_train, word_features_train, pos_features_train]
)
X_val = hstack(
    [val_stats_sparse, char_features_val, word_features_val, pos_features_val]
)
X_test = hstack(
    [test_stats_sparse, char_features_test, word_features_test, pos_features_test]
)

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")
print(f"Test features shape: {X_test.shape}")

# ============================================================
# 7. PREPARE TEST DATA FOR SUBMISSION (ALL TEST SAMPLES)
# ============================================================
test_texts_full = test_df["text"].values
test_stats_full = csr_matrix(extract_sentence_stats(test_texts_full))
test_char_full = extract_char_ngrams(test_texts_full, vectorizer=char_vectorizer)
test_word_full = extract_word_ngrams(test_texts_full, vectorizer=word_vectorizer)
test_pos_full = extract_pos_patterns(test_texts_full, vectorizer=pos_vectorizer)

X_test_full = hstack([test_stats_full, test_char_full, test_word_full, test_pos_full])

# ============================================================
# 8. SAVE PROCESSED DATA FOR NEXT STEPS
# ============================================================
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)
np.save("./working/y_test.npy", y_test)

save_npz("./working/X_train.npz", X_train)
save_npz("./working/X_val.npz", X_val)
save_npz("./working/X_test.npz", X_test)
save_npz("./working/X_test_full.npz", X_test_full)

joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
joblib.dump(pos_vectorizer, "./working/pos_vectorizer.pkl")

test_ids = test_df["id"].values
np.save("./working/test_ids.npy", test_ids)

np.save("./working/X_train_texts.npy", X_train_texts)
np.save("./working/X_val_texts.npy", X_val_texts)
np.save("./working/X_test_texts.npy", X_test_texts)
np.save("./working/test_texts_full.npy", test_texts_full)

print("Data processing and feature engineering completed successfully.")
print(
    f"Train samples: {len(y_train)}, Val samples: {len(y_val)}, Test samples: {len(y_test)}"
)
print(f"Feature dimensions: {X_train.shape[1]}")

# ============================================================
# 9. IMPORT TRANSFORMERS AND SET UP DEVICE
# ============================================================
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Setting up transformer model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 10. TRANSFORMER + SPARSE FUSION CLASSIFIER DEFINITION
# ============================================================
num_classes = 3
sparse_dim = X_train.shape[1]  # 35018

# Use a smaller transformer model to reduce memory usage
TRANSFORMER_MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"  # 384 hidden dim vs 768 for DistilBERT


class MiniLMFusionClassifier(nn.Module):
    def __init__(self, num_classes, sparse_dim, fusion_dim=128, dropout_rate=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
        self.fusion_dim = fusion_dim

        # Enable gradient checkpointing to save memory
        self.transformer.gradient_checkpointing_enable()

        # Sparse feature projection: 35018 -> 128
        self.sparse_proj = nn.Linear(sparse_dim, fusion_dim)

        # [CLS] token projection: 384 -> 128
        self.cls_proj = nn.Linear(384, fusion_dim)

        # Classification head after concatenation
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fusion_dim * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'transformer' not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, sparse_features):
        # Transformer forward
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token, shape: (batch, 384)

        # Project [CLS] token
        cls_proj = self.dropout(torch.relu(self.cls_proj(cls_output)))  # (batch, 128)

        # Project sparse features
        sparse_proj = self.dropout(torch.relu(self.sparse_proj(sparse_features)))  # (batch, 128)

        # Concatenate fused representations
        fused = torch.cat([cls_proj, sparse_proj], dim=1)  # (batch, 256)

        # Classification head
        logits = self.classifier(self.dropout(fused))  # (batch, 3)

        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# ============================================================
# 11. PREPARE SPARSE FEATURES FOR ALL DATA
# ============================================================
# Keep sparse representation and convert to dense on-the-fly to save memory

# Re-load all texts for stratified k-fold
all_train_texts = train_df["text"].values
all_train_labels = y
all_train_ids = train_df["id"].values
test_texts_full = test_df["text"].values
test_ids = test_df["id"].values

# Precompute sparse features for all training data (for fusion) - keep sparse!
print("Precomputing sparse features for all training data...")
all_train_stats = extract_sentence_stats(all_train_texts)
all_train_char = extract_char_ngrams(all_train_texts, vectorizer=char_vectorizer)
all_train_word = extract_word_ngrams(all_train_texts, vectorizer=word_vectorizer)
all_train_pos = extract_pos_patterns(all_train_texts, vectorizer=pos_vectorizer)

X_all_train_sparse = hstack([
    csr_matrix(all_train_stats),
    all_train_char,
    all_train_word,
    all_train_pos
])

# Precompute sparse features for test data - keep sparse!
test_stats_full = extract_sentence_stats(test_texts_full)
test_char_full = extract_char_ngrams(test_texts_full, vectorizer=char_vectorizer)
test_word_full = extract_word_ngrams(test_texts_full, vectorizer=word_vectorizer)
test_pos_full = extract_pos_patterns(test_texts_full, vectorizer=pos_vectorizer)

X_test_full_sparse = hstack([
    csr_matrix(test_stats_full),
    test_char_full,
    test_word_full,
    test_pos_full
])

# ============================================================
# 12. TOKENIZER AND DATA COLLATION
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")


def tokenize_texts(texts, max_length=512):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ============================================================
# 13. STRATIFIED 5-FOLD CROSS-VALIDATION TRAINING
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_val_scores = []
fold_test_probs = []
best_fold_val_loss = float("inf")
best_fold_val_score = None

print("\nStarting 5-fold cross-validation training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_texts, all_train_labels)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*50}")

    # Split data for this fold
    fold_train_texts = all_train_texts[train_idx]
    fold_val_texts = all_train_texts[val_idx]
    fold_train_labels = torch.tensor(all_train_labels[train_idx], dtype=torch.long)
    fold_val_labels = torch.tensor(all_train_labels[val_idx], dtype=torch.long)
    # Convert sparse to dense in mini-batches instead of loading all at once
    fold_train_sparse_indices = train_idx  # Store indices for on-the-fly conversion
    fold_val_sparse_indices = val_idx

    # Tokenize texts
    fold_train_tokens = tokenize_texts(fold_train_texts)
    fold_val_tokens = tokenize_texts(fold_val_texts)

    # Create custom dataset that converts sparse to dense on-the-fly to save memory
    class SparseToDenseDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, sparse_matrix, indices, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.sparse_matrix = sparse_matrix
            self.indices = indices
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            dense_row = torch.tensor(self.sparse_matrix[self.indices[idx]].toarray().flatten().astype(np.float32))
            return (
                self.input_ids[idx],
                self.attention_mask[idx],
                dense_row,
                self.labels[idx],
            )

    train_dataset = SparseToDenseDataset(
        fold_train_tokens["input_ids"],
        fold_train_tokens["attention_mask"],
        X_all_train_sparse,
        fold_train_sparse_indices,
        fold_train_labels,
    )
    val_dataset = SparseToDenseDataset(
        fold_val_tokens["input_ids"],
        fold_val_tokens["attention_mask"],
        X_all_train_sparse,
        fold_val_sparse_indices,
        fold_val_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Reduced further for memory efficiency (effective batch 24 with grad accumulation)
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    # Initialize model for this fold
    model = MiniLMFusionClassifier(num_classes, sparse_dim)
    model = model.to(device)

    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)

    # Warmup scheduler (10% warmup steps, linear decay)
    total_steps = len(train_loader) * 8  # 8 epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop for this fold
    epochs = 8
    patience = 3
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    gradient_accumulation_steps = 3  # Effective batch = 8 * 3 = 24

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        optimizer.zero_grad()

        for step, (input_ids, attention_mask, sparse_feat, batch_labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            sparse_feat = sparse_feat.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(input_ids, attention_mask, sparse_feat)
            loss = criterion(logits, batch_labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * gradient_accumulation_steps
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        all_val_probs = []
        all_val_targets = []

        with torch.no_grad():
            for input_ids, attention_mask, sparse_feat, batch_labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                sparse_feat = sparse_feat.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(input_ids, attention_mask, sparse_feat)
                loss = criterion(logits, batch_labels)

                total_val_loss += loss.item()
                num_val_batches += 1

                probs = F.softmax(logits, dim=1)
                all_val_probs.append(probs.cpu().numpy())
                all_val_targets.append(batch_labels.cpu().numpy())

        avg_val_loss = total_val_loss / num_val_batches

        val_probs = np.concatenate(all_val_probs, axis=0)
        val_targets = np.concatenate(all_val_targets, axis=0)

        epsilon = 1e-15
        val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
        row_sums = val_probs_clipped.sum(axis=1, keepdims=True)
        val_probs_normalized = val_probs_clipped / row_sums

        val_log_loss = log_loss(val_targets, val_probs_normalized)

        print(
            f"Fold {fold+1} | Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_log_loss = val_log_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"./working/distilbert_fold_{fold+1}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for this fold
    model.load_state_dict(best_model_state)
    fold_val_scores.append(best_val_log_loss)
    print(f"Fold {fold+1} best validation log-loss: {best_val_log_loss:.4f}")

    # Predict on full test set with this fold's model
    model.eval()
    test_tokens = tokenize_texts(test_texts_full)

    # Test dataset with on-the-fly sparse-to-dense conversion
    class TestSparseToDenseDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, sparse_matrix, test_indices):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.sparse_matrix = sparse_matrix
            self.test_indices = test_indices

        def __len__(self):
            return len(self.test_indices)

        def __getitem__(self, idx):
            dense_row = torch.tensor(self.sparse_matrix[self.test_indices[idx]].toarray().flatten().astype(np.float32))
            return (self.input_ids[idx], self.attention_mask[idx], dense_row)

    test_indices_arr = np.arange(X_test_full_sparse.shape[0])
    test_dataset_fold = TestSparseToDenseDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        X_test_full_sparse,
        test_indices_arr,
    )
    test_loader_fold = DataLoader(
        test_dataset_fold,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    fold_test_probs_list = []
    with torch.no_grad():
        for input_ids, attention_mask, sparse_feat in test_loader_fold:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            sparse_feat = sparse_feat.to(device)

            logits = model(input_ids, attention_mask, sparse_feat)
            probs = F.softmax(logits, dim=1)
            fold_test_probs_list.append(probs.cpu().numpy())

    fold_test_probs.append(np.concatenate(fold_test_probs_list, axis=0))

    # Track best fold
    if best_val_log_loss < best_fold_val_loss:
        best_fold_val_loss = best_val_log_loss

    # Clean up to free memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 14. ENSEMBLE PREDICTIONS (AVERAGE SOFTMAX PROBS)
# ============================================================
print("\n" + "=" * 50)
print("ENSEMBLING PREDICTIONS")
print("=" * 50)

# Average the probabilities from all folds
test_probs_ensemble = np.mean(fold_test_probs, axis=0)

# Normalize to ensure probabilities sum to 1
epsilon = 1e-15
test_probs_clipped = np.clip(test_probs_ensemble, epsilon, 1 - epsilon)
row_sums = test_probs_clipped.sum(axis=1, keepdims=True)
test_probs_normalized = test_probs_clipped / row_sums

# ============================================================
# 15. SUBMISSION FILE
# ============================================================
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs_normalized[:, 0],
        "HPL": test_probs_normalized[:, 1],
        "MWS": test_probs_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# ============================================================
# 16. PRINT FINAL SCORES
# ============================================================
print(f"\nFold validation log-loss scores: {fold_val_scores}")
print(f"Mean cross-validation log-loss: {np.mean(fold_val_scores):.4f}")
print(f"Best fold log-loss: {best_fold_val_loss:.4f}")

# Clean up
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()