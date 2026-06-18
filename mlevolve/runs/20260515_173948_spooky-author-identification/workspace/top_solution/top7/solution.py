import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW, lr_scheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, save_npz, load_npz
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch_size, seq_len, hidden_size)
        attn_weights = self.attention(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        sentence_embedding = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return sentence_embedding


class CustomDebertaV2ForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3, weight_decay=0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.pooling = AttentionPooling(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.weight_decay = weight_decay
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, return_embedding=False):
        # Freeze-thaw: if encoder requires_grad is False, no gradients flow
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        sentence_embedding = self.pooling(last_hidden, attention_mask)
        sentence_embedding = self.dropout(sentence_embedding)
        logits = self.classifier(sentence_embedding)
        loss = None
        if labels is not None:
            # Compute class frequencies for alpha balancing
            if not hasattr(self, "_focal_loss"):
                self._focal_loss = FocalLoss(gamma=2.0, alpha=None)
            loss = self._focal_loss(logits, labels)
        if return_embedding:
            return logits, sentence_embedding
        return type("Output", (object,), {"logits": logits, "loss": loss})()


os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED K-FOLD SPLIT
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_data = []
all_texts = train_df["text"].values
all_labels = y_train_full

print("Creating fold splits...")
for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
    fold_data.append({
        "train_texts": all_texts[train_idx],
        "val_texts": all_texts[val_idx],
        "train_labels": all_labels[train_idx],
        "val_labels": all_labels[val_idx],
    })
    print(f"Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")

# Extract fold 0 data for feature engineering
X_train_texts_0 = fold_data[0]["train_texts"]
X_val_texts_0 = fold_data[0]["val_texts"]
y_train_labels_0 = fold_data[0]["train_labels"]
y_val_labels_0 = fold_data[0]["val_labels"]

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================


def extract_stylometric_features(texts):
    features = []
    archaic_words = set(
        [
            "thou",
            "thee",
            "thy",
            "thine",
            "ye",
            "hence",
            "thence",
            "whence",
            "hath",
            "doth",
            "art",
            "wilt",
            "dost",
            "doth",
            "ere",
            "whilst",
            "betwixt",
            "anon",
            "perchance",
            "methinks",
            "forsooth",
            "verily",
            "hither",
            "thither",
            "yon",
            "yonder",
            "alas",
            "lo",
            "behold",
            "nay",
            "aye",
            "oft",
            "ofttimes",
            "ne'er",
            "o'er",
            "e'en",
            "tis",
            "twas",
            "twill",
            "twere",
            "couldst",
            "wouldst",
            "shouldst",
            "durst",
            "wast",
            "wert",
            "knowest",
            "thinkest",
            "madst",
            "didst",
        ]
    )
    emotional_words = set(
        [
            "horror",
            "terror",
            "dread",
            "fear",
            "fright",
            "panic",
            "alarm",
            "gloom",
            "darkness",
            "shadow",
            "ghost",
            "spectre",
            "phantom",
            "apparition",
            "demon",
            "devil",
            "satan",
            "hell",
            "damned",
            "corpse",
            "death",
            "dead",
            "dying",
            "grave",
            "tomb",
            "coffin",
            "sorrow",
            "grief",
            "anguish",
            "agony",
            "torment",
            "torture",
            "madness",
            "insanity",
            "lunacy",
            "frenzy",
            "rage",
            "fury",
            "despair",
            "hopeless",
            "doom",
            "fate",
            "curse",
            "omen",
            "supernatural",
            "unearthly",
            "mysterious",
            "secret",
            "hidden",
            "ancient",
            "forgotten",
            "lost",
            "eternal",
            "immortal",
        ]
    )
    lovecraft_words = set(
        [
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "azathoth",
            "nyarlathotep",
            "shoggoth",
            "necronomicon",
            "arkham",
            "innsmouth",
            "dunwich",
            "kadath",
            "ulthar",
            "dreamlands",
            "eldritch",
            "cyclopean",
            "non-euclidean",
            "blasphemous",
            "unspeakable",
            "nameless",
            "indescribable",
            "unutterable",
            "abysmal",
            "primordial",
            "antediluvian",
            "cyclopean",
            "rugose",
            "squamous",
            "ichor",
            "gibbering",
            "loathsome",
            "noisome",
            "putrid",
            "foul",
            "accursed",
            "monstrous",
            "hideous",
            "ghastly",
            "ghoulish",
            "sepulchral",
            "tenebrous",
            "Stygian",
            "nefarious",
            "malignant",
            "cosmic",
            "aeon",
            "immemorial",
            "dimensional",
            "infernal",
        ]
    )
    function_words = set(
        [
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "of",
            "and",
            "or",
            "but",
            "if",
            "when",
            "while",
            "where",
            "there",
            "their",
            "them",
            "they",
            "this",
            "that",
            "these",
            "those",
            "who",
            "whom",
            "which",
            "what",
            "its",
            "it",
            "i",
            "you",
            "he",
            "she",
            "we",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "him",
            "us",
            "not",
            "no",
            "nor",
            "very",
            "too",
            "so",
            "such",
            "just",
            "only",
            "also",
            "as",
            "than",
            "then",
            "now",
            "here",
            "there",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "many",
            "some",
            "any",
            "much",
            "more",
            "most",
            "other",
            "another",
            "one",
        ]
    )
    subordinate_conjunctions = set(
        [
            "because",
            "since",
            "as",
            "for",
            "so",
            "although",
            "though",
            "while",
            "whereas",
            "if",
            "unless",
            "when",
            "where",
            "whether",
            "after",
            "before",
            "until",
            "once",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "when",
            "where",
            "why",
            "how",
            "whatever",
            "whenever",
            "wherever",
            "whichever",
            "whoever",
        ]
    )
    punct_marks = [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "..."]

    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0] * 30)
            continue
        text_str = str(text)
        text_lower = text_str.lower()
        words = text_lower.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_chars = len(text_str)
        n_words = len(words)
        n_sentences = len(sentences) if sentences else 1
        avg_word_len = np.mean([len(w) for w in words]) if n_words > 0 else 0
        avg_sent_len = n_words / n_sentences if n_sentences > 0 else 0
        upper_ratio = sum(1 for c in text_str if c.isupper()) / max(n_chars, 1)
        lower_ratio = sum(1 for c in text_str if c.islower()) / max(n_chars, 1)
        digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(n_chars, 1)
        whitespace_ratio = sum(1 for c in text_str if c.isspace()) / max(n_chars, 1)
        punct_ratios = []
        for p in punct_marks:
            count = text_str.count(p)
            punct_ratios.append(count / max(n_chars, 1))
        unique_chars = len(set(text_str.lower()))
        char_diversity = unique_chars / max(n_chars, 1)
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / max(n_words, 1)
        capitalized = sum(1 for w in words if w[0].isupper()) / max(n_words, 1)
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1) / max(n_words, 1)
        sent_lengths = [len(s.split()) for s in sentences]
        if n_sentences > 1:
            sent_len_std = np.std(sent_lengths)
            sent_len_var = np.var(sent_lengths)
        else:
            sent_len_std = 0
            sent_len_var = 0
        function_word_ratio = sum(1 for w in words if w in function_words) / max(
            n_words, 1
        )
        archaic_ratio = sum(1 for w in words if w in archaic_words) / max(n_words, 1)
        emotional_ratio = sum(1 for w in words if w in emotional_words) / max(
            n_words, 1
        )
        lovecraft_ratio = sum(1 for w in words if w in lovecraft_words) / max(
            n_words, 1
        )
        sub_conj_ratio = sum(1 for w in words if w in subordinate_conjunctions) / max(
            n_words, 1
        )
        feature_vector = [
            n_chars,
            n_words,
            n_sentences,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            whitespace_ratio,
            *punct_ratios,
            char_diversity,
            long_word_ratio,
            capitalized,
            all_caps,
            sent_len_std,
            sent_len_var,
            function_word_ratio,
            archaic_ratio,
            emotional_ratio,
            lovecraft_ratio,
            sub_conj_ratio,
        ]
        features.append(feature_vector)
    return np.array(features)


def create_readability_features(texts):
    features = []
    vowels = set("aeiouy")
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0] * 4)
            continue
        text_str = str(text)
        words = text_str.lower().split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_words = len(words)
        n_sentences = len(sentences) if sentences else 1
        n_chars = len(text_str.replace(" ", ""))
        syllable_counts = []
        for word in words:
            count = 0
            prev_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
            if count == 0:
                count = 1
            syllable_counts.append(count)
        avg_syllables = np.mean(syllable_counts) if syllable_counts else 0
        flesch = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * avg_syllables
        flesch = max(0, min(100, flesch))
        if n_words > 0 and n_sentences > 0:
            ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sentences) - 21.43
        else:
            ari = 0
        ari = max(0, ari)
        complex_words = sum(1 for c in syllable_counts if c > 2)
        complex_word_ratio = complex_words / max(n_words, 1)
        features.append([flesch, ari, avg_syllables, complex_word_ratio])
    return np.array(features)


def create_pos_tag_approximation(texts):
    noun_suffixes = [
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "dom",
        "hood",
        "ship",
        "ism",
        "ist",
        "logy",
        "ment",
        "ure",
        "age",
        "al",
        "cy",
        "ry",
        "ty",
        "th",
        "ee",
    ]
    verb_suffixes = ["ate", "ify", "ize", "ise", "en", "ify", "ish", "ed", "ing", "en"]
    adj_suffixes = [
        "able",
        "ible",
        "al",
        "ial",
        "ical",
        "ful",
        "less",
        "ous",
        "ious",
        "ive",
        "ative",
        "ic",
        "ish",
        "like",
        "ly",
        "ward",
        "ern",
        "ile",
    ]
    adv_suffixes = ["ly", "ward", "wise", "way"]
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0] * 5)
            continue
        text_lower = str(text).lower()
        words = text_lower.split()
        n_words = len(words)
        noun_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in noun_suffixes)
        )
        verb_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in verb_suffixes)
        )
        adj_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in adj_suffixes)
        )
        adv_count = sum(
            1 for w in words if any(w.endswith(suf) for suf in adv_suffixes)
        )
        content_words = noun_count + verb_count + adj_count + adv_count
        noun_ratio = noun_count / max(n_words, 1)
        verb_ratio = verb_count / max(n_words, 1)
        adj_ratio = adj_count / max(n_words, 1)
        adv_ratio = adv_count / max(n_words, 1)
        content_ratio = content_words / max(n_words, 1)
        features.append([noun_ratio, verb_ratio, adj_ratio, adv_ratio, content_ratio])
    return np.array(features)


# ============================================================
# HANDCRAFTED FEATURES EXTRACTION
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING HANDCRAFTED FEATURES (on fold 0 data)")
print("=" * 60)

print("Extracting stylometric features...", end=" ")
train_stylo = extract_stylometric_features(X_train_texts_0)
val_stylo = extract_stylometric_features(X_val_texts_0)
test_stylo = extract_stylometric_features(test_df["text"].values)
print(f"Done. Shape: train={train_stylo.shape}")

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)
print(f"Stylometric features after variance filter: {train_stylo_filtered.shape[1]}")

print("Extracting readability features...", end=" ")
train_read = create_readability_features(X_train_texts_0)
val_read = create_readability_features(X_val_texts_0)
test_read = create_readability_features(test_df["text"].values)
print(f"Done. Shape: train={train_read.shape}")

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

print("Extracting POS approximation features...", end=" ")
train_pos = create_pos_tag_approximation(X_train_texts_0)
val_pos = create_pos_tag_approximation(X_val_texts_0)
test_pos = create_pos_tag_approximation(test_df["text"].values)
print(f"Done. Shape: train={train_pos.shape}")

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# N-GRAM FEATURES
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING N-GRAM FEATURES")
print("=" * 60)

print("Extracting character n-grams (2-4)...", end=" ")
char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
)
train_char_short = char_vectorizer_short.fit_transform(X_train_texts_0)
val_char_short = char_vectorizer_short.transform(X_val_texts_0)
test_char_short = char_vectorizer_short.transform(test_df["text"].values)
print(f"Done. Shape: {train_char_short.shape}")

print("Extracting character n-grams (4-6)...", end=" ")
char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=2,
)
train_char_med = char_vectorizer_med.fit_transform(X_train_texts_0)
val_char_med = char_vectorizer_med.transform(X_val_texts_0)
test_char_med = char_vectorizer_med.transform(test_df["text"].values)
print(f"Done. Shape: {train_char_med.shape}")

print("Extracting character n-grams (5-7)...", end=" ")
char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=2,
)
train_char_long = char_vectorizer_long.fit_transform(X_train_texts_0)
val_char_long = char_vectorizer_long.transform(X_val_texts_0)
test_char_long = char_vectorizer_long.transform(test_df["text"].values)
print(f"Done. Shape: {train_char_long.shape}")

print("Extracting word n-grams (1-3)...", end=" ")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
    stop_words="english",
)
train_word = word_vectorizer.fit_transform(X_train_texts_0)
val_word = word_vectorizer.transform(X_val_texts_0)
test_word = word_vectorizer.transform(test_df["text"].values)
print(f"Done. Shape: {train_word.shape}")

print("Extracting punctuation sequence features...", end=" ")


def extract_punctuation_sequence(text):
    if not isinstance(text, str):
        return ""
    return "".join([c for c in text if c in string.punctuation])


all_texts_for_punct = np.concatenate(
    [X_train_texts_0, X_val_texts_0, test_df["text"].values]
)
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(X_train_texts_0)
n_val = len(X_val_texts_0)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + n_val]
test_punct = punct_features_all[n_train + n_val :]
print(f"Done. Shape: {train_punct.shape}")

print("\nCombining sparse features...", end=" ")
train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Done. Combined sparse shape: {train_sparse.shape}")

# ============================================================
# COMBINE DENSE FEATURES
# ============================================================
train_dense = np.hstack([train_stylo_filtered, train_read_scaled, train_pos_scaled])
val_dense = np.hstack([val_stylo_filtered, val_read_scaled, val_pos_scaled])
test_dense = np.hstack([test_stylo_filtered, test_read_scaled, test_pos_scaled])

print(
    f"Dense features - Train: {train_dense.shape}, Val: {val_dense.shape}, Test: {test_dense.shape}"
)

# Save for later use
np.save("./working/train_dense.npy", train_dense)
np.save("./working/val_dense.npy", val_dense)
np.save("./working/test_dense.npy", test_dense)
np.save("./working/y_train_labels.npy", y_train_labels_0)
np.save("./working/y_val_labels.npy", y_val_labels_0)
save_npz("./working/train_sparse.npz", train_sparse)
save_npz("./working/val_sparse.npz", val_sparse)
save_npz("./working/test_sparse.npz", test_sparse)

# ============================================================
# CUSTOM COLLATE WITH STYLISTIC AUGMENTATION
# ============================================================
def collate_with_augmentation(batch):
    input_ids, attention_mask, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)

    # Apply augmentation to a random subset of samples
    batch_size = input_ids.size(0)
    # For simplicity, we skip heavy augmentation on token ids directly
    # as word-level augmentation is complex with BPE tokenization.
    # The plan mentions augmentation via collate; we implement a lightweight version:
    # randomly swap tokens with [MASK] for 5% of tokens (simple dropout at token level)
    if np.random.rand() < 0.5:
        mask_token_id = tokenizer.mask_token_id
        mask = torch.rand(input_ids.shape, device=input_ids.device) < 0.05
        input_ids = torch.where(mask & (attention_mask == 1), mask_token_id, input_ids)

    return input_ids, attention_mask, labels


# ============================================================
# DEBERTA FINE-TUNING WITH 5-FOLD CV AND FREEZE-THAW
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING DEBERTA-V3-LARGE WITH 5-FOLD CV")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pre-tokenize test data once
test_encodings = tokenizer(
    list(test_df["text"].values),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

NUM_FREEZE_EPOCHS = 3
NUM_UNFREEZE_EPOCHS = NUM_EPOCHS - NUM_FREEZE_EPOCHS
fold_val_probs = []
fold_test_probs = []
fold_val_labels = []

for fold_idx, fold in enumerate(fold_data):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx+1}/5")
    print(f"{'='*60}")

    X_train_texts = fold["train_texts"]
    X_val_texts = fold["val_texts"]
    y_train_labels = fold["train_labels"]
    y_val_labels = fold["val_labels"]

    # Tokenize fold data
    train_encodings = tokenizer(
        list(X_train_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        list(X_val_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(y_train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(y_val_labels, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_with_augmentation,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model with attention pooling
    model = CustomDebertaV2ForSequenceClassification(
        MODEL_NAME,
        NUM_AUTHORS,
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
    )
    model.to(device)

    # Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Optimizer with different LR for head vs encoder
    head_params = list(model.pooling.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW([
        {"params": head_params, "lr": 1e-4, "weight_decay": WEIGHT_DECAY},
        {"params": model.encoder.parameters(), "lr": 0, "weight_decay": WEIGHT_DECAY},
    ], lr=1e-4, eps=1e-8)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    patience_counter = 0
    freeze_epochs_done = 0

    for epoch in range(NUM_EPOCHS):
        # Unfreeze after freeze epochs
        if epoch == NUM_FREEZE_EPOCHS:
            print("Unfreezing encoder...")
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Reset optimizer with lr for encoder
            optimizer = AdamW([
                {"params": head_params, "lr": 1e-4, "weight_decay": WEIGHT_DECAY},
                {"params": model.encoder.parameters(), "lr": 2e-5, "weight_decay": WEIGHT_DECAY},
            ], lr=2e-5, eps=1e-8)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            freeze_epochs_done = 1

        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # Evaluate on unaugmented validation data
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        all_val_preds = np.vstack(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        # Compute log loss using numpy (no sklearn dependency needed for manual impl)
        eps = 1e-15
        p = np.clip(all_val_preds, eps, 1 - eps)
        # Normalize rows to sum to 1
        row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums
        p = np.clip(p, eps, 1 - eps)
        # One-hot encoding of true labels
        num_classes = p.shape[1]
        y_one_hot = np.eye(num_classes)[all_val_labels]
        val_loss = -np.mean(np.sum(y_one_hot * np.log(p), axis=1))
        val_acc = np.mean(np.argmax(all_val_preds, axis=1) == all_val_labels)

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/best_deberta_fold{fold_idx+1}.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break

        scheduler.step()

    # Load best model for this fold
    model.load_state_dict(torch.load(f"./working/best_deberta_fold{fold_idx+1}.pt", map_location=device))
    model.eval()

    # Get validation predictions (for tracking, though we use out-of-fold for final)
    val_loader_eval = DataLoader(
        TensorDataset(
            val_encodings["input_ids"],
            val_encodings["attention_mask"],
            torch.tensor(y_val_labels, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    all_val_preds = []
    all_val_labels_list = []
    with torch.no_grad():
        for batch in val_loader_eval:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels_list.append(labels.cpu().numpy())

    fold_val_probs.append(np.vstack(all_val_preds))
    fold_val_labels.append(np.concatenate(all_val_labels_list))

    # Get test predictions
    all_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_test_preds.append(probs.cpu().numpy())

    fold_test_probs.append(np.vstack(all_test_preds))
    print(f"Fold {fold_idx+1} best val loss: {best_val_loss:.4f}")

    # Clean up
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# After all folds, average predictions
print("\n" + "="*60)
print("AGGREGATING FOLD PREDICTIONS")
print("="*60)

# Average test predictions across folds
deberta_test_probs = np.mean(fold_test_probs, axis=0)
print(f"Test predictions averaged across {len(fold_test_probs)} folds.")

# For validation, we use out-of-fold predictions (stack them)
# But for the ensemble with XGBoost/LR, we need a single validation set.
# Since we changed to CV, we'll use fold 0's validation for weight optimization if needed.
val_labels_for_ensemble = fold_val_labels[0]
deberta_val_probs = fold_val_probs[0]  # Use fold 0 as representative val set

def compute_log_loss(y_true, y_pred, eps=1e-15):
    """Compute multi-class log loss."""
    p = np.clip(y_pred, eps, 1 - eps)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / row_sums
    p = np.clip(p, eps, 1 - eps)
    num_classes = p.shape[1]
    y_one_hot = np.eye(num_classes)[y_true]
    return -np.mean(np.sum(y_one_hot * np.log(p), axis=1))

# Also compute mean val loss across folds
# Define compute_log_loss if not already defined
def compute_log_loss(y_true, y_pred, eps=1e-15):
    """Compute multi-class log loss."""
    p = np.clip(y_pred, eps, 1 - eps)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / row_sums
    p = np.clip(p, eps, 1 - eps)
    num_classes = p.shape[1]
    y_one_hot = np.eye(num_classes)[y_true]
    return -np.mean(np.sum(y_one_hot * np.log(p), axis=1))

mean_val_loss = np.mean([compute_log_loss(fold_val_labels[i], fold_val_probs[i]) for i in range(5)])
print(f"Mean OOF log loss: {mean_val_loss:.4f}")

# ============================================================
# EXTRACT DEBERTA EMBEDDINGS FOR XGBOOST (using fold 0 model)
# ============================================================
print("\nExtracting DeBERTa embeddings for XGBoost/LR ensemble...")

# Reload fold 0 model for embedding extraction
model = CustomDebertaV2ForSequenceClassification(
    MODEL_NAME, NUM_AUTHORS, dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
)
model.load_state_dict(torch.load("./working/best_deberta_fold1.pt", map_location=device))
model.to(device)
model.eval()


def extract_embeddings(model, loader):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                _, embeddings = model(input_ids=input_ids, attention_mask=attention_mask, return_embedding=True)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


# Use fold 0's data splits (stored in fold_data[0])
X_train_texts_0 = fold_data[0]["train_texts"]
X_val_texts_0 = fold_data[0]["val_texts"]
y_train_labels_0 = fold_data[0]["train_labels"]
y_val_labels_0 = fold_data[0]["val_labels"]

train_encodings_0 = tokenizer(
    list(X_train_texts_0), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
)
val_encodings_0 = tokenizer(
    list(X_val_texts_0), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
)
test_encodings = tokenizer(
    list(test_df["text"].values), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
)

train_loader_0 = DataLoader(
    TensorDataset(train_encodings_0["input_ids"], train_encodings_0["attention_mask"]),
    batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True,
)
val_loader_0 = DataLoader(
    TensorDataset(val_encodings_0["input_ids"], val_encodings_0["attention_mask"]),
    batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True,
)
test_loader_no_labels = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True,
)

train_embeddings = extract_embeddings(model, train_loader_0)
val_embeddings = extract_embeddings(model, val_loader_0)
test_embeddings = extract_embeddings(model, test_loader_no_labels)
print(f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")

# ============================================================
# XGBOOST
# ============================================================
print("\nTraining XGBoost classifier...")
xgb_train_features = np.hstack(
    [train_stylo_filtered, train_read_scaled, train_pos_scaled, train_embeddings]
)
xgb_val_features = np.hstack(
    [val_stylo_filtered, val_read_scaled, val_pos_scaled, val_embeddings]
)
xgb_test_features = np.hstack(
    [test_stylo_filtered, test_read_scaled, test_pos_scaled, test_embeddings]
)
print(f"XGBoost train features: {xgb_train_features.shape}")

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="multi:softprob",
    num_class=NUM_AUTHORS,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    xgb_train_features,
    y_train_labels_0,
    eval_set=[(xgb_val_features, y_val_labels_0)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
print(
    f"XGBoost validation log loss: {compute_log_loss(y_val_labels_0, xgb_val_probs):.4f}"
)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
lr_model.fit(train_sparse, y_train_labels_0)

lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)
print(
    f"Logistic Regression validation log loss: {compute_log_loss(y_val_labels_0, lr_val_probs):.4f}"
)

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "deberta": deberta_val_probs,
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
}

def compute_log_loss(y_true, y_pred, eps=1e-15):
    """Compute multi-class log loss."""
    p = np.clip(y_pred, eps, 1 - eps)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / row_sums
    p = np.clip(p, eps, 1 - eps)
    num_classes = p.shape[1]
    y_one_hot = np.eye(num_classes)[y_true]
    return -np.mean(np.sum(y_one_hot * np.log(p), axis=1))

best_ll = float("inf")
best_weights = None
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = (
            w1 * val_probas["deberta"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(val_labels_for_ensemble, ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"deberta": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
test_probas = {
    "deberta": deberta_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
ensemble_test_probs = (
    best_weights["deberta"] * test_probas["deberta"]
    + best_weights["xgboost"] * test_probas["xgboost"]
    + best_weights["lr"] * test_probas["lr"]
)

eps = 1e-15
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)
row_sums = ensemble_test_probs.sum(axis=1, keepdims=True)
ensemble_test_probs = ensemble_test_probs / row_sums
ensemble_test_probs = np.clip(ensemble_test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": ensemble_test_probs[:, 0],
        "HPL": ensemble_test_probs[:, 1],
        "MWS": ensemble_test_probs[:, 2],
    }
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
print(f"\nFinal Validation Score: {best_ll:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
