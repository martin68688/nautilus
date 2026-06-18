import pandas as pd
import numpy as np
import re
import string
import os
import gc
import math
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import hstack, csr_matrix, save_npz
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
MAX_SEQ_LENGTH = 384
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 6
WARMUP_RATIO = 0.1
LEARNING_RATE_BACKBONE = 1e-5
LEARNING_RATE_NEW = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MODEL_NAME = "microsoft/deberta-v3-large"

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. DATA LOADING & STRATIFIED SPLIT (NO INDEX_BUG)
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
print(
    f"Classes: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
)

# Stratified split using direct numpy indexing (NO INDEX_BUG)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["author_encoded"].values,
)

train_texts = train_df["text"].values[train_idx]
train_labels = train_df["author_encoded"].values[train_idx]
val_texts = train_df["text"].values[val_idx]
val_labels = train_df["author_encoded"].values[val_idx]
test_texts = test_df["text"].values
test_ids = test_df["id"].values

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
assert len(set(train_idx) & set(val_idx)) == 0, "INDEX OVERLAP DETECTED!"

# ============================================================
# 2. HELPER FUNCTIONS FOR FEATURE ENGINEERING
# ============================================================
def tokenize_words(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    return re.findall(r"[a-zA-Z']+|[.,!?;:\"'()-]", text.lower())

def tokenize_sentences(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if len(s.strip()) > 0]

# ============================================================
# 3. NOVEL FEATURE 1: NARRATIVE RHYTHM
# ============================================================
def extract_narrative_rhythm_features(texts):
    coord_conj = {"and", "but", "or", "nor", "for", "yet", "so"}
    features_list = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features_list.append([0.0] * 8)
            continue
        sents = tokenize_sentences(text)
        if len(sents) == 0:
            features_list.append([0.0] * 8)
            continue
        all_rel_positions = []
        for sent in sents:
            if len(sent) == 0:
                continue
            for i, ch in enumerate(sent):
                if ch in string.punctuation:
                    rel_pos = i / max(len(sent), 1)
                    all_rel_positions.append(rel_pos)
        punct_entropy = 0.0
        if len(all_rel_positions) >= 2:
            hist, _ = np.histogram(all_rel_positions, bins=10, range=(0.0, 1.0))
            hist = hist / max(hist.sum(), 1e-10)
            punct_entropy = entropy(hist + 1e-10) / math.log(10)
        sent_lengths = [len(s.split()) for s in sents]
        if len(sent_lengths) >= 2:
            hist_len, _ = np.histogram(sent_lengths, bins="auto")
            hist_len = hist_len / max(hist_len.sum(), 1e-10)
            sent_len_entropy = entropy(hist_len + 1e-10) / math.log(
                max(len(hist_len), 2)
            )
        else:
            sent_len_entropy = 0.0
        comma_count = text.count(",") + text.count(";")
        clause_density = comma_count / max(len(sents), 1)
        first_words = []
        for sent in sents:
            words = sent.split()
            if words:
                first_words.append(words[0].lower().strip(string.punctuation))
        if len(first_words) >= 2:
            start_diversity = len(set(first_words)) / max(len(first_words), 1)
        else:
            start_diversity = 0.0
        coord_count = sum(1 for w in first_words if w in coord_conj)
        coord_ratio = coord_count / max(len(first_words), 1)
        all_word_lengths = []
        for sent in sents:
            words = sent.split()
            for w in words:
                all_word_lengths.append(len(w))
        if len(all_word_lengths) >= 2:
            word_len_std = np.std(all_word_lengths)
        else:
            word_len_std = 0.0
        punct_count = sum(1 for ch in text if ch in string.punctuation)
        punct_density = punct_count / max(len(text), 1)
        excl_quest = text.count("!") + text.count("?")
        excl_quest_ratio = excl_quest / max(punct_count, 1)
        features_list.append(
            [
                punct_entropy,
                sent_len_entropy,
                clause_density,
                start_diversity,
                coord_ratio,
                word_len_std,
                punct_density,
                excl_quest_ratio,
            ]
        )
    return np.array(features_list, dtype=np.float32)

print("Extracting narrative rhythm features...")
train_rhythm = extract_narrative_rhythm_features(train_texts)
val_rhythm = extract_narrative_rhythm_features(val_texts)
test_rhythm = extract_narrative_rhythm_features(test_texts)
rhythm_scaler = StandardScaler()
train_rhythm_scaled = rhythm_scaler.fit_transform(train_rhythm)
val_rhythm_scaled = rhythm_scaler.transform(val_rhythm)
test_rhythm_scaled = rhythm_scaler.transform(test_rhythm)

# ============================================================
# 4. NOVEL FEATURE 2: VOCABULARY FRESHNESS
# ============================================================
def extract_vocabulary_freshness_features(texts):
    features_list = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features_list.append([0.0] * 6)
            continue
        sents = tokenize_sentences(text)
        if len(sents) < 2:
            features_list.append([0.0] * 6)
            continue
        all_words = [w for w in tokenize_words(text) if w.isalpha()]
        if len(all_words) < 5:
            features_list.append([0.0] * 6)
            continue
        ttr_overall = len(set(all_words)) / max(len(all_words), 1)
        word_counts = Counter(all_words)
        hapax_count = sum(1 for v in word_counts.values() if v == 1)
        hapax_ratio = hapax_count / max(len(all_words), 1)
        per_sentence_ttr = []
        for sent in sents:
            sent_words = [w for w in tokenize_words(sent) if w.isalpha()]
            if len(sent_words) >= 3:
                ttr = len(set(sent_words)) / max(len(sent_words), 1)
                per_sentence_ttr.append(ttr)
        if per_sentence_ttr:
            avg_ttr = np.mean(per_sentence_ttr)
            ttr_std = np.std(per_sentence_ttr)
        else:
            avg_ttr = 0.0
            ttr_std = 0.0
        new_words_per_sent = []
        cum_set = set()
        for sent in sents:
            sent_words = [w for w in tokenize_words(sent) if w.isalpha()]
            if len(sent_words) == 0:
                continue
            before = len(cum_set)
            cum_set.update(sent_words)
            after = len(cum_set)
            new_words_per_sent.append(after - before)
        if new_words_per_sent:
            growth_rate = np.mean(new_words_per_sent)
            growth_std = np.std(new_words_per_sent)
        else:
            growth_rate = 0.0
            growth_std = 0.0
        rare_count = sum(1 for v in word_counts.values() if v < 3)
        rare_ratio = rare_count / max(len(set(all_words)), 1)
        features_list.append(
            [ttr_overall, hapax_ratio, avg_ttr, ttr_std, growth_rate, rare_ratio]
        )
    return np.array(features_list, dtype=np.float32)

print("Extracting vocabulary freshness features...")
train_vocab_fresh = extract_vocabulary_freshness_features(train_texts)
val_vocab_fresh = extract_vocabulary_freshness_features(val_texts)
test_vocab_fresh = extract_vocabulary_freshness_features(test_texts)
vocab_scaler = StandardScaler()
train_vocab_fresh_scaled = vocab_scaler.fit_transform(train_vocab_fresh)
val_vocab_fresh_scaled = vocab_scaler.transform(val_vocab_fresh)
test_vocab_fresh_scaled = vocab_scaler.transform(test_vocab_fresh)

# ============================================================
# 5. TRADITIONAL STYLOMETRIC FEATURES
# ============================================================
def extract_stylometric_features(texts):
    features_list = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features_list.append([0.0] * 20)
            continue
        words = text.split()
        chars = len(text)
        word_count = len(words) if words else 1
        char_count = chars
        sent_count = max(len(tokenize_sentences(text)), 1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
        avg_sent_len = word_count / sent_count
        upper_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        lower_ratio = sum(1 for c in text if c.islower()) / max(char_count, 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
        punct_counts = {
            ".": text.count("."),
            ",": text.count(","),
            ";": text.count(";"),
            ":": text.count(":"),
            "!": text.count("!"),
            "?": text.count("?"),
            '"': text.count('"'),
            "'": text.count("'"),
            "-": text.count("-"),
            "(": text.count("("),
            ")": text.count(")"),
            "—": text.count("—"),
        }
        features = [
            char_count,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            upper_ratio,
            lower_ratio,
            digit_ratio,
            punct_counts["."] / max(char_count, 1),
            punct_counts[","] / max(char_count, 1),
            punct_counts[";"] / max(char_count, 1),
            punct_counts[":"] / max(char_count, 1),
            punct_counts["!"] / max(char_count, 1),
            punct_counts["?"] / max(char_count, 1),
            punct_counts['"'] / max(char_count, 1),
            punct_counts["'"] / max(char_count, 1),
            punct_counts["-"] / max(char_count, 1),
            punct_counts["("] / max(char_count, 1),
            punct_counts[")"] / max(char_count, 1),
            punct_counts["—"] / max(char_count, 1),
        ]
        features_list.append(features)
    return np.array(features_list, dtype=np.float32)

print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(train_texts)
val_stylo = extract_stylometric_features(val_texts)
test_stylo = extract_stylometric_features(test_texts)
stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)
variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)

# ============================================================
# 6. N-GRAM FEATURES (CHAR + WORD + PUNCTUATION)
# ============================================================
print("Extracting n-gram features...")
char_vectorizer_short = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_short = char_vectorizer_short.fit_transform(train_texts)
val_char_short = char_vectorizer_short.transform(val_texts)
test_char_short = char_vectorizer_short.transform(test_texts)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(train_texts)
val_char_med = char_vectorizer_med.transform(val_texts)
test_char_med = char_vectorizer_med.transform(test_texts)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(train_texts)
val_char_long = char_vectorizer_long.transform(val_texts)
test_char_long = char_vectorizer_long.transform(test_texts)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    min_df=3,
    max_df=0.85,
)
train_word = word_vectorizer.fit_transform(train_texts)
val_word = word_vectorizer.transform(val_texts)
test_word = word_vectorizer.transform(test_texts)

def extract_punctuation_pattern(text):
    if not isinstance(text, str):
        return ""
    return "".join([c for c in text if c in string.punctuation])

all_texts_for_punct = np.concatenate([train_texts, val_texts, test_texts])
punct_sequences = [extract_punctuation_pattern(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(train_texts)
n_val = len(val_texts)
train_punct = punct_features_all[:n_train]
val_punct = punct_features_all[n_train : n_train + n_val]
test_punct = punct_features_all[n_train + n_val :]

train_sparse = hstack(
    [train_char_short, train_char_med, train_char_long, train_word, train_punct]
).tocsr()
val_sparse = hstack(
    [val_char_short, val_char_med, val_char_long, val_word, val_punct]
).tocsr()
test_sparse = hstack(
    [test_char_short, test_char_med, test_char_long, test_word, test_punct]
).tocsr()
print(f"Sparse train shape: {train_sparse.shape}")

# ============================================================
# 7. COMBINE ALL DENSE FEATURES
# ============================================================
train_dense_features = np.hstack(
    [train_stylo_filtered, train_rhythm_scaled, train_vocab_fresh_scaled]
)
val_dense_features = np.hstack(
    [val_stylo_filtered, val_rhythm_scaled, val_vocab_fresh_scaled]
)
test_dense_features = np.hstack(
    [test_stylo_filtered, test_rhythm_scaled, test_vocab_fresh_scaled]
)
print(
    f"Dense features: Train {train_dense_features.shape}, Val {val_dense_features.shape}, Test {test_dense_features.shape}"
)

# ============================================================
# 8. CACHE DATA FOR TRAINING
# ============================================================
np.save("./working/train_labels.npy", train_labels)
np.save("./working/val_labels.npy", val_labels)
np.save("./working/train_texts.npy", train_texts, allow_pickle=True)
np.save("./working/val_texts.npy", val_texts, allow_pickle=True)
np.save("./working/test_texts.npy", test_texts, allow_pickle=True)
np.save("./working/train_dense_features.npy", train_dense_features)
np.save("./working/val_dense_features.npy", val_dense_features)
np.save("./working/test_dense_features.npy", test_dense_features)
save_npz("./working/train_sparse.npz", train_sparse)
save_npz("./working/val_sparse.npz", val_sparse)
save_npz("./working/test_sparse.npz", test_sparse)
np.save("./working/test_ids.npy", test_ids)
np.save("./working/label_classes.npy", label_encoder.classes_)

os.makedirs("./working/scalers", exist_ok=True)
joblib.dump(rhythm_scaler, "./working/scalers/rhythm_scaler.pkl")
joblib.dump(vocab_scaler, "./working/scalers/vocab_scaler.pkl")
joblib.dump(stylo_scaler, "./working/scalers/stylo_scaler.pkl")
joblib.dump(variance_selector, "./working/scalers/variance_selector.pkl")
joblib.dump(char_vectorizer_short, "./working/scalers/char_vectorizer_short.pkl")
joblib.dump(char_vectorizer_med, "./working/scalers/char_vectorizer_med.pkl")
joblib.dump(char_vectorizer_long, "./working/scalers/char_vectorizer_long.pkl")
joblib.dump(word_vectorizer, "./working/scalers/word_vectorizer.pkl")
joblib.dump(punct_vectorizer, "./working/scalers/punct_vectorizer.pkl")

print("Feature engineering complete. All data cached.")

# ============================================================
# 9. CONTRASTIVE MODEL ARCHITECTURE
# ============================================================
class AuthorContrastiveModel(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_authors=3,
        hidden_size=1024,
        projection_dim=256,
        dropout=0.15,
        temperature=0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_hidden = self.backbone.config.hidden_size
        self.fingerprint_projector = nn.Sequential(
            nn.Linear(backbone_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.author_prototypes = nn.Parameter(
            torch.randn(num_authors, projection_dim) * 0.02
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_authors),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.fingerprint_projector:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.author_prototypes, std=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        fingerprint_embeddings = self.fingerprint_projector(cls_embeddings)
        fingerprint_norm = F.normalize(fingerprint_embeddings, dim=1)
        prototypes_norm = F.normalize(self.author_prototypes, dim=1)
        prototype_similarities = torch.mm(fingerprint_norm, prototypes_norm.t())
        temp_scaled_similarities = prototype_similarities / (
            self.temperature.abs() + 1e-8
        )
        logits = self.classifier(fingerprint_embeddings)
        return {
            "logits": logits,
            "fingerprints": fingerprint_embeddings,
            "prototype_similarities": temp_scaled_similarities,
            "normalized_fingerprints": fingerprint_norm,
        }

class ContrastiveLossComputer:
    def __init__(self, temperature=0.1, alpha=0.3, beta=0.2, gamma=0.1):
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def supervised_contrastive_loss(self, fingerprints, labels):
        batch_size = fingerprints.shape[0]
        fingerprints = F.normalize(fingerprints, dim=1)
        sim_matrix = torch.mm(fingerprints, fingerprints.t())
        labels = labels.unsqueeze(1).expand(batch_size, batch_size)
        same_author_mask = (labels == labels.t()).float()
        self_mask = torch.eye(batch_size, device=fingerprints.device)
        same_author_mask = same_author_mask - self_mask
        pos_sim = sim_matrix * same_author_mask
        neg_sim = sim_matrix * (1 - same_author_mask - self_mask)
        pos_sum = torch.exp(pos_sim / self.temperature).sum(dim=1)
        neg_sum = torch.exp(neg_sim / self.temperature).sum(dim=1)
        denominator = pos_sum + neg_sum + 1e-8
        contrastive_loss = -torch.log(pos_sum / denominator + 1e-8)
        contrastive_loss = contrastive_loss[same_author_mask.sum(dim=1) > 0].mean()
        return contrastive_loss

    def prototype_alignment_loss(self, fingerprints, prototype_similarities, labels):
        alignment_probs = F.softmax(prototype_similarities, dim=1)
        nll_loss = F.nll_loss(torch.log(alignment_probs + 1e-10), labels)
        return nll_loss

    def entropy_regularization(self, prototype_similarities):
        probs = F.softmax(prototype_similarities, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        return -entropy

    def compute_loss(self, model_outputs, labels):
        fingerprints = model_outputs["fingerprints"]
        prototype_similarities = model_outputs["prototype_similarities"]
        logits = model_outputs["logits"]
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        contrastive_loss = self.supervised_contrastive_loss(fingerprints, labels)
        proto_loss = self.prototype_alignment_loss(
            fingerprints, prototype_similarities, labels
        )
        entropy_loss = self.entropy_regularization(prototype_similarities)
        total_loss = (
            ce_loss
            + self.alpha * contrastive_loss
            + self.beta * proto_loss
            + self.gamma * entropy_loss
        )
        loss_components = {
            "cross_entropy": ce_loss.item(),
            "contrastive": (
                contrastive_loss.item() if torch.is_tensor(contrastive_loss) else 0.0
            ),
            "prototype_alignment": proto_loss.item(),
            "entropy_reg": entropy_loss.item(),
        }
        return total_loss, loss_components

# ============================================================
# 10. DATASETS AND DATALOADERS
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ContrastiveAuthorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.author_indices = {}
        for i, lbl in enumerate(labels):
            if lbl not in self.author_indices:
                self.author_indices[lbl] = []
            self.author_indices[lbl].append(i)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        anchor_text = str(self.texts[idx])
        anchor_label = self.labels[idx]
        if torch.rand(1).item() < 0.5 and len(self.author_indices[anchor_label]) > 1:
            pos_indices = [i for i in self.author_indices[anchor_label] if i != idx]
            pos_idx = np.random.choice(pos_indices)
            paired_text = str(self.texts[pos_idx])
            is_positive = 1
        else:
            neg_labels = [l for l in self.author_indices.keys() if l != anchor_label]
            neg_label = np.random.choice(neg_labels)
            neg_idx = np.random.choice(self.author_indices[neg_label])
            paired_text = str(self.texts[neg_idx])
            is_positive = 0
        encodings = self.tokenizer(
            [anchor_text, paired_text],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "paired_input_ids": encodings["input_ids"][1],
            "paired_attention_mask": encodings["attention_mask"][1],
            "label": torch.tensor(anchor_label, dtype=torch.long),
            "is_positive": torch.tensor(is_positive, dtype=torch.float32),
        }

class SimpleDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        result = {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
        }
        if self.labels is not None:
            result["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return result

train_dataset = ContrastiveAuthorDataset(
    train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH
)
val_dataset = SimpleDataset(val_texts, val_labels)
test_dataset = SimpleDataset(test_texts)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# 11. INITIALIZE MODEL AND OPTIMIZERS
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING MODEL")
print("=" * 60)

model = AuthorContrastiveModel(
    model_name=MODEL_NAME,
    num_authors=3,
    hidden_size=1024,
    projection_dim=256,
    dropout=0.15,
    temperature=0.1,
)
model.to(device)

loss_computer = ContrastiveLossComputer(
    temperature=0.1, alpha=0.3, beta=0.2, gamma=0.05
)

backbone_params = []
new_params = []
for name, param in model.named_parameters():
    if "backbone" in name:
        backbone_params.append(param)
    else:
        new_params.append(param)

optimizer = AdamW(
    [
        {
            "params": backbone_params,
            "lr": LEARNING_RATE_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
        },
        {"params": new_params, "lr": LEARNING_RATE_NEW, "weight_decay": 0.001},
    ],
    eps=1e-8,
)

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)
scaler = GradScaler() if torch.cuda.is_available() else None

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# 12. TRAINING AND EVALUATION FUNCTIONS
# ============================================================
def compute_log_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(3):
            if y_true[i] == j:
                loss -= np.log(y_pred_proba[i, j])
    return loss / n

def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("label", None)
            with autocast():
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
            all_logits.append(logits.cpu().numpy())
            if labels is not None:
                all_labels.append(labels.cpu().numpy())
    all_logits = np.vstack(all_logits)
    all_probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    if all_labels:
        all_labels = np.concatenate(all_labels)
        log_loss = compute_log_loss(all_labels, all_probs)
        acc = np.mean(np.argmax(all_probs, axis=1) == all_labels)
        return log_loss, acc, all_probs
    return None, None, all_probs

# ============================================================
# 13. TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print("TRAINING CONTRASTIVE MODEL")
print("=" * 60)

best_val_loss = float("inf")
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        paired_input_ids = batch["paired_input_ids"].to(device)
        paired_attention_mask = batch["paired_attention_mask"].to(device)
        labels = batch["label"].to(device)
        is_positive = batch["is_positive"].to(device)
        optimizer.zero_grad()
        with autocast():
            anchor_outputs = model(input_ids, attention_mask, labels)
            paired_outputs = model(paired_input_ids, paired_attention_mask, labels)
            loss, loss_components = loss_computer.compute_loss(anchor_outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, val_probs = evaluate(model, val_loader)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "./working/best_contrastive_model.pt")
        print(f"  -> Saved new best model (val_logloss: {val_loss:.4f})")

print(
    f"\nBest model from epoch {best_epoch} with validation log loss: {best_val_loss:.4f}"
)

# ============================================================
# 14. LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(
    torch.load("./working/best_contrastive_model.pt", map_location=device)
)
final_val_loss, final_val_acc, final_val_probs = evaluate(model, val_loader)
print(
    f"\nFinal Validation - Log Loss: {final_val_loss:.4f}, Accuracy: {final_val_acc:.4f}"
)

# ============================================================
# 15. TEST INFERENCE
# ============================================================
print("\nPerforming test inference...")
_, _, test_probs = evaluate(model, test_loader)

# ============================================================
# 16. GENERATE SUBMISSION
# ============================================================
eps = 1e-15
test_probs = np.clip(test_probs, eps, 1 - eps)
row_sums = test_probs.sum(axis=1, keepdims=True)
test_probs = test_probs / row_sums
test_probs = np.clip(test_probs, eps, 1 - eps)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

print(f"Final Validation Score: {final_val_loss:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
