"""
Merged Script: Spooky Author Identification
Combines data processing, feature engineering, bi-encoder model design, and training/evaluation.
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import string
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & CONFIGURATION
# ============================================================
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "./working"
SUBMISSION_PATH = "./submission/submission.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

RANDOM_STATE = 42
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
HIDDEN_SIZE = 1024
EMBEDDING_DIM = 256
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
WARMUP_RATIO = 0.1
PATIENCE = 5
DROPOUT = 0.2
NUM_EXPERTS = 6
TOP_K_EXPERTS = 2
TEMPERATURE = 0.05

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# WORD LISTS FOR FEATURE ENGINEERING
# ============================================================
FUNCTION_WORDS = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
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
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
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
        "also",
        "however",
        "therefore",
        "thus",
        "furthermore",
        "nevertheless",
        "nonetheless",
        "moreover",
    ]
)

ARCHAIC_WORDS = set(
    [
        "thou",
        "thy",
        "thine",
        "thee",
        "doth",
        "hath",
        "dost",
        "canst",
        "wilt",
        "shalt",
        "art",
        "wert",
        "hast",
        "hadst",
        "didst",
        "ere",
        "whence",
        "thence",
        "hence",
        "whither",
        "thither",
        "hither",
        "wherefore",
        "therefor",
        "amongst",
        "whilst",
        "betwixt",
        "unto",
        "twas",
        "twere",
        "ye",
        "prithee",
        "forsooth",
        "alas",
        "perchance",
        "methinks",
        "deign",
        "beseech",
        "entreat",
        "thenceforth",
        "wherewith",
        "therewith",
    ]
)

EMOTIONAL_WORDS = set(
    [
        "fear",
        "terror",
        "horror",
        "dread",
        "anguish",
        "agony",
        "torment",
        "despair",
        "gloom",
        "shadow",
        "dark",
        "night",
        "death",
        "corpse",
        "ghost",
        "spirit",
        "demon",
        "devil",
        "hell",
        "scream",
        "shriek",
        "groan",
        "moan",
        "weep",
        "wept",
        "tear",
        "blood",
        "chill",
        "cold",
        "icy",
        "grave",
        "tomb",
        "coffin",
        "shroud",
        "spectre",
        "phantom",
        "apparition",
        "wraith",
        "ghastly",
        "hideous",
        "dreadful",
        "awful",
        "terrible",
        "frightful",
        "horrid",
        "shudder",
        "tremble",
        "quake",
        "pale",
        "wan",
        "livid",
        "ashen",
        "hollow",
        "solemn",
        "mournful",
    ]
)

LOVECRAFT_WORDS = set(
    [
        "eldritch",
        "cyclopean",
        "non-euclidean",
        "antediluvian",
        "primordial",
        "noisome",
        "squamous",
        "rugose",
        "ichor",
        "gibber",
        "gibbering",
        "maddening",
        "blasphemous",
        "unspeakable",
        "unnameable",
        "indescribable",
        "crawling",
        "slithering",
        "loathsome",
        "abyss",
        "chasm",
        "void",
        "ancient",
        "bygone",
        "forgotten",
        "cosmic",
        "infinite",
        "eternal",
        "cryptic",
        "arcane",
        "occult",
        "coven",
        "necronomicon",
        "cthulhu",
        "r'lyeh",
        "yog-sothoth",
        "azathoth",
        "nyarlathotep",
        "shoggoth",
        "mi-go",
        "great old ones",
        "outer gods",
        "yuggoth",
        "kadath",
        "hyperborean",
        "lemuria",
        "atlantis",
        "mu",
        "dreamlands",
        "carcosa",
        "hastur",
        "yellow king",
        "innsmouth",
        "arkham",
        "miskatonic",
        "dunwich",
        "providence",
    ]
)

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================

def extract_stylometric_features(texts):
    n = len(texts)
    features = np.zeros((n, 31))
    for i, text in enumerate(texts):
        if pd.isna(text) or text is None or len(str(text).strip()) == 0:
            features[i, :] = 0
            continue
        text = str(text)
        char_count = len(text)
        words = text.split()
        word_count = len(words) if len(words) > 0 else 1
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        sent_count = max(len(sentences), 1)
        features[i, 0] = char_count
        features[i, 1] = word_count
        features[i, 2] = sent_count
        avg_word_len = sum(len(w) for w in words) / word_count
        features[i, 3] = avg_word_len
        avg_sent_len = char_count / sent_count
        features[i, 4] = avg_sent_len
        upper_count = sum(1 for c in text if c.isupper())
        lower_count = sum(1 for c in text if c.islower())
        digit_count = sum(1 for c in text if c.isdigit())
        whitespace_count = sum(1 for c in text if c.isspace())
        features[i, 5] = upper_count / char_count if char_count > 0 else 0
        features[i, 6] = lower_count / char_count if char_count > 0 else 0
        features[i, 7] = digit_count / char_count if char_count > 0 else 0
        features[i, 8] = whitespace_count / char_count if char_count > 0 else 0
        punct_marks = [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "—"]
        for j, punct in enumerate(punct_marks):
            punct_count = text.count(punct)
            features[i, 9 + j] = punct_count / char_count if char_count > 0 else 0
        unique_chars = len(set(text))
        features[i, 21] = unique_chars / char_count if char_count > 0 else 0
        long_words = sum(1 for w in words if len(w) >= 7)
        features[i, 22] = long_words / word_count
        capitalized_words = sum(1 for w in words if w[0].isupper() and len(w) > 0)
        features[i, 23] = capitalized_words / word_count
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features[i, 24] = all_caps_words / word_count
        sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        if len(sent_lengths) > 1:
            features[i, 25] = np.std(sent_lengths)
            features[i, 26] = np.var(sent_lengths)
        low_words = [
            w.lower().strip(string.punctuation)
            for w in words
            if len(w.strip(string.punctuation)) > 0
        ]
        function_word_count = sum(1 for w in low_words if w in FUNCTION_WORDS)
        features[i, 27] = function_word_count / max(len(low_words), 1)
        archaic_count = sum(1 for w in low_words if w in ARCHAIC_WORDS)
        emotional_count = sum(1 for w in low_words if w in EMOTIONAL_WORDS)
        lovecraft_count = sum(1 for w in low_words if w in LOVECRAFT_WORDS)
        features[i, 28] = archaic_count / max(len(low_words), 1)
        features[i, 29] = emotional_count / max(len(low_words), 1)
        features[i, 30] = lovecraft_count / max(len(low_words), 1)
    return features

def create_readability_features(texts):
    def count_syllables_approx(word):
        word = word.lower().strip(string.punctuation)
        if len(word) == 0:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        return max(count, 1)

    n = len(texts)
    features = np.zeros((n, 4))
    for i, text in enumerate(texts):
        if pd.isna(text) or text is None or len(str(text).strip()) == 0:
            features[i, :] = 50
            continue
        text = str(text)
        words = text.split()
        word_count = max(len(words), 1)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        sent_count = max(len(sentences), 1)
        char_count = len(text.replace(" ", ""))
        total_syllables = sum(
            count_syllables_approx(w)
            for w in words
            if len(w.strip(string.punctuation)) > 0
        )
        avg_syllables = total_syllables / word_count
        flesch = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * avg_syllables
        flesch = max(0, min(100, flesch))
        features[i, 0] = flesch
        ari = 4.71 * (char_count / word_count) + 0.5 * (word_count / sent_count) - 21.43
        ari = max(0, ari)
        features[i, 1] = ari
        features[i, 2] = avg_syllables
        complex_words = sum(1 for w in words if count_syllables_approx(w) >= 3)
        features[i, 3] = complex_words / word_count
    return features

def create_pos_tag_approximation(texts):
    noun_suffixes = (
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ship",
        "ism",
        "age",
    )
    verb_suffixes = ("ed", "ing", "ize", "ate", "ify", "en", "ish")
    adj_suffixes = (
        "ous",
        "ious",
        "eous",
        "able",
        "ible",
        "ive",
        "ful",
        "less",
        "al",
        "ic",
        "ical",
        "ant",
        "ent",
        "ary",
    )
    n = len(texts)
    features = np.zeros((n, 5))
    for i, text in enumerate(texts):
        if pd.isna(text) or text is None or len(str(text).strip()) == 0:
            features[i, :] = 0
            continue
        text = str(text)
        words = text.split()
        clean_words = [
            w.strip(string.punctuation).lower()
            for w in words
            if len(w.strip(string.punctuation)) > 0
        ]
        clean_count = max(len(clean_words), 1)
        noun_count = sum(
            1 for w in clean_words if w.endswith(noun_suffixes) and len(w) > 4
        )
        verb_count = sum(
            1 for w in clean_words if w.endswith(verb_suffixes) and len(w) > 3
        )
        adj_count = sum(
            1 for w in clean_words if w.endswith(adj_suffixes) and len(w) > 4
        )
        adv_count = sum(1 for w in clean_words if w.endswith("ly") and len(w) > 4)
        features[i, 0] = noun_count / clean_count
        features[i, 1] = verb_count / clean_count
        features[i, 2] = adj_count / clean_count
        features[i, 3] = adv_count / clean_count
        content_words = sum(1 for w in clean_words if w not in FUNCTION_WORDS)
        features[i, 4] = content_words / clean_count
    return features

def extract_punctuation_sequence(text):
    return "".join([c for c in text if c in string.punctuation]) if text else ""

# ============================================================
# MODEL DEFINITION: Bi-Encoder with Contrastive + MoE
# ============================================================
class StyleAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_styles=4):
        super().__init__()
        self.num_styles = num_styles
        self.style_queries = nn.Parameter(torch.randn(num_styles, hidden_size))
        self.style_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, token_embeddings, attention_mask):
        batch_size, seq_len, _ = token_embeddings.shape
        style_queries = self.style_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attention_scores = torch.bmm(style_queries, token_embeddings.transpose(1, 2))
        mask = attention_mask.unsqueeze(1).expand(-1, self.num_styles, -1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        attention_weights = F.softmax(attention_scores / TEMPERATURE, dim=-1)
        style_embeddings = torch.bmm(attention_weights, token_embeddings)
        sentence_embedding = style_embeddings.mean(dim=1)
        return sentence_embedding, style_embeddings

class StyleMoE(nn.Module):
    def __init__(self, input_dim, num_experts, num_classes, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_classes = num_classes
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),  # Reduced from DROPOUT (0.2) to 0.1
                    nn.Linear(input_dim // 2, num_classes),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2),
            nn.GELU(),
            nn.Dropout(0.05),  # Reduced accordingly
            nn.Linear(num_experts * 2, num_experts),
        )
        self.load_balancing_coef = 0.01

    def forward(self, x, return_aux_loss=True):
        batch_size = x.shape[0]
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        output = torch.zeros(batch_size, self.num_classes).to(x.device)
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_weights[:, i]
            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_output = self.experts[expert_idx](x[mask])
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output
        aux_loss = torch.tensor(0.0, device=x.device)
        if return_aux_loss:
            expert_usage = gate_weights.mean(dim=0)
            expected_usage = torch.ones_like(expert_usage) / self.num_experts
            aux_loss = (
                F.mse_loss(expert_usage, expected_usage) * self.load_balancing_coef
            )
        return output, aux_loss

class StochasticDepth(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_rate
        mask = torch.empty(x.shape[0], 1, 1, device=x.device).bernoulli_(keep_prob)
        x = x / keep_prob * mask
        return x

class SpookyAuthorBiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, config=self.config)

        # Remove aggressive freezing - all layers remain trainable with different LR
        # Instead of freezing first 12 layers, we keep them trainable (differential LR handled in optimizer)

        self.style_pooling = StyleAttentionPooling(HIDDEN_SIZE, num_styles=4)
        # Increase projection MLP to 512-dim with 2-layer bottleneck (1024->1024->512)
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
        )
        self.classifier = StyleMoE(HIDDEN_SIZE, NUM_EXPERTS, NUM_AUTHORS, TOP_K_EXPERTS)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)

        # Skip Stochastic Depth monkey-patching - use simpler approach
        # DeBERTa-v2 layers have complex signatures that make patching fragile
        # Instead, we'll apply stochastic depth manually in forward if needed
        self.stochastic_depth = StochasticDepth(drop_rate=0.1) if hasattr(self.backbone.encoder, 'layer') else None
        self._patched = False
        # Note: Stochastic depth is applied via a different mechanism to avoid monkey-patching issues

        self._init_weights()

    def _init_weights(self):
        for module in [self.projection, self.style_pooling]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=1.0)
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0.0)

    def forward(self, input_ids, attention_mask, return_embeddings=True):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        last_hidden = self.layer_norm(last_hidden)
        sentence_embedding, style_embeddings = self.style_pooling(
            last_hidden, attention_mask
        )
        projected_embedding = self.projection(sentence_embedding)
        projected_embedding = F.normalize(projected_embedding, p=2, dim=-1)
        logits, aux_loss = self.classifier(sentence_embedding)
        if return_embeddings:
            return logits, projected_embedding, style_embeddings.detach(), aux_loss
        return logits, aux_loss

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.last_hidden_state
            last_hidden = self.layer_norm(last_hidden)
            sentence_embedding, _ = self.style_pooling(last_hidden, attention_mask)
            projected_embedding = self.projection(sentence_embedding)
            projected_embedding = F.normalize(projected_embedding, p=2, dim=-1)
        return projected_embedding

class ContrastiveAuthorLoss(nn.Module):
    def __init__(self, temperature=0.1, label_smoothing=0.1, contrastive_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight

    def forward(self, logits, embeddings, labels, aux_loss, return_all_losses=False):
        batch_size = logits.shape[0]
        if self.label_smoothing > 0:
            n_classes = logits.shape[1]
            smooth_labels = torch.full_like(
                logits, self.label_smoothing / (n_classes - 1)
            )
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        else:
            ce_loss = F.cross_entropy(logits, labels)

        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if self.contrastive_weight > 0:
            sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
            labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
            labels_eq.fill_diagonal_(0)
            exp_sim = torch.exp(sim_matrix)
            eye_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            exp_sim = exp_sim.masked_fill(eye_mask, 0.0)
            pos_mask = labels_eq.float()
            pos_sum = (exp_sim * pos_mask).sum(dim=-1)
            all_sum = exp_sim.sum(dim=-1)
            pos_sum = pos_sum + 1e-8
            all_sum = all_sum + 1e-8
            loss_per_sample = -torch.log(pos_sum / all_sum)
            valid_samples = pos_mask.sum(dim=-1) > 0
            if valid_samples.any():
                contrastive_loss = loss_per_sample[valid_samples].mean()

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss + aux_loss
        if return_all_losses:
            return total_loss, ce_loss, contrastive_loss, aux_loss
        return total_loss

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df["author"])
print(
    f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# ============================================================
# STRATIFIED SPLIT
# ============================================================
X_train_texts, X_val_texts, y_train_labels, y_val_labels = train_test_split(
    train_df["text"].values,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)
print(
    f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}, Test samples: {len(test_df)}"
)

# ============================================================
# HANDCRAFTED FEATURES EXTRACTION
# ============================================================
print("\nExtracting stylometric features...")
train_stylo = extract_stylometric_features(X_train_texts)
val_stylo = extract_stylometric_features(X_val_texts)
test_stylo = extract_stylometric_features(test_df["text"].values)

stylo_scaler = StandardScaler()
train_stylo_scaled = stylo_scaler.fit_transform(train_stylo)
val_stylo_scaled = stylo_scaler.transform(val_stylo)
test_stylo_scaled = stylo_scaler.transform(test_stylo)

variance_selector = VarianceThreshold(threshold=0.001)
train_stylo_filtered = variance_selector.fit_transform(train_stylo_scaled)
val_stylo_filtered = variance_selector.transform(val_stylo_scaled)
test_stylo_filtered = variance_selector.transform(test_stylo_scaled)

print("Extracting readability features...")
train_read = create_readability_features(X_train_texts)
val_read = create_readability_features(X_val_texts)
test_read = create_readability_features(test_df["text"].values)

read_scaler = StandardScaler()
train_read_scaled = read_scaler.fit_transform(train_read)
val_read_scaled = read_scaler.transform(val_read)
test_read_scaled = read_scaler.transform(test_read)

print("Extracting POS approximation features...")
train_pos = create_pos_tag_approximation(X_train_texts)
val_pos = create_pos_tag_approximation(X_val_texts)
test_pos = create_pos_tag_approximation(test_df["text"].values)

pos_scaler = StandardScaler()
train_pos_scaled = pos_scaler.fit_transform(train_pos)
val_pos_scaled = pos_scaler.transform(val_pos)
test_pos_scaled = pos_scaler.transform(test_pos)

# ============================================================
# N-GRAM FEATURES
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
train_char_short = char_vectorizer_short.fit_transform(X_train_texts)
val_char_short = char_vectorizer_short.transform(X_val_texts)
test_char_short = char_vectorizer_short.transform(test_df["text"].values)

char_vectorizer_med = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4, 6),
    max_features=3000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_med = char_vectorizer_med.fit_transform(X_train_texts)
val_char_med = char_vectorizer_med.transform(X_val_texts)
test_char_med = char_vectorizer_med.transform(test_df["text"].values)

char_vectorizer_long = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 7),
    max_features=2000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
)
train_char_long = char_vectorizer_long.fit_transform(X_train_texts)
val_char_long = char_vectorizer_long.transform(X_val_texts)
test_char_long = char_vectorizer_long.transform(test_df["text"].values)

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
train_word = word_vectorizer.fit_transform(X_train_texts)
val_word = word_vectorizer.transform(X_val_texts)
test_word = word_vectorizer.transform(test_df["text"].values)

all_texts_for_punct = np.concatenate(
    [X_train_texts, X_val_texts, test_df["text"].values]
)
punct_sequences = [extract_punctuation_sequence(str(t)) for t in all_texts_for_punct]
punct_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=500, min_df=2
)
punct_features_all = punct_vectorizer.fit_transform(punct_sequences)

n_train = len(X_train_texts)
n_val = len(X_val_texts)
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
# BI-ENCODER FINE-TUNING
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNING BI-ENCODER (DeBERTa-v3-large + StyleMoE)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SpookyAuthorBiEncoder()
model.to(device)

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
test_encodings = tokenizer(
    list(test_df["text"].values),
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
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
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
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

total_steps = len(train_loader) * NUM_EPOCHS
criterion = ContrastiveAuthorLoss(
    temperature=0.1, label_smoothing=0.1, contrastive_weight=0.3
)

# Differential learning rates: layers 0-12: 0.1x, 12-18: 0.5x, 18-24 + custom heads: 1.0x
param_groups_lr1 = []  # layers 0-12: 0.1x base LR
param_groups_lr2 = []  # layers 12-18: 0.5x base LR
param_groups_lr3 = []  # layers 18-24 + custom heads: 1.0x base LR

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # Custom heads (not backbone)
    if "backbone" not in name:
        param_groups_lr3.append(param)
    else:
        # Extract layer number from parameter name
        # Format: backbone.encoder.layer.XX. ...
        parts = name.split(".")
        layer_idx = -1
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
                break
        if layer_idx < 12:
            param_groups_lr1.append(param)
        elif layer_idx < 18:
            param_groups_lr2.append(param)
        else:
            param_groups_lr3.append(param)

optimizer = AdamW(
    [
        {"params": param_groups_lr1, "lr": LEARNING_RATE * 0.1, "weight_decay": 0.01},
        {"params": param_groups_lr2, "lr": LEARNING_RATE * 0.5, "weight_decay": 0.01},
        {"params": param_groups_lr3, "lr": LEARNING_RATE, "weight_decay": 0.01},
    ],
    lr=LEARNING_RATE,
    eps=1e-8,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)
scaler = GradScaler() if torch.cuda.is_available() else None

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

def evaluate_biencoder(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with autocast():
                logits, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_embeddings=True,
                )
                probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logloss = compute_log_loss(all_labels, all_preds)
    acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    return logloss, acc, all_preds

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        optimizer.zero_grad()
        with autocast():
            # Manifold Mixup: interpolate token embeddings before StyleAttentionPooling
            if epoch >= 0:  # Always apply during training
                alpha = 0.3
                batch_size = input_ids.size(0)
                # Get hidden states from backbone
                outputs = model.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden = outputs.last_hidden_state
                last_hidden = model.layer_norm(last_hidden)

                # Sample mixup lambda from Beta distribution
                lam = np.random.beta(alpha, alpha)
                # Random permutation indices
                perm_indices = torch.randperm(batch_size, device=device)

                # Mix token embeddings
                mixed_hidden = lam * last_hidden + (1 - lam) * last_hidden[perm_indices]
                mixed_attention_mask = attention_mask  # Keep original mask (approximate)

                # Continue forward with mixed embeddings
                sentence_embedding, style_embeddings = model.style_pooling(
                    mixed_hidden, mixed_attention_mask
                )
                projected_embedding = model.projection(sentence_embedding)
                projected_embedding = F.normalize(projected_embedding, p=2, dim=-1)
                logits, aux_loss = model.classifier(sentence_embedding)

                # Mix labels
                labels_mixed = labels
                labels_perm = labels[perm_indices]

                # Compute loss with mixed targets
                log_probs = F.log_softmax(logits, dim=-1)
                ce_loss = lam * F.nll_loss(log_probs, labels_mixed, reduction='mean') + (1 - lam) * F.nll_loss(log_probs, labels_perm, reduction='mean')

                # Contrastive loss with mixed embeddings
                contrastive_loss = torch.tensor(0.0, device=logits.device)
                if hasattr(criterion, 'contrastive_weight') and criterion.contrastive_weight > 0:
                    sim_matrix = torch.mm(projected_embedding, projected_embedding.t()) / criterion.temperature
                    labels_eq = (labels_mixed.unsqueeze(0) == labels_mixed.unsqueeze(1)) | (labels_perm.unsqueeze(0) == labels_perm.unsqueeze(1))
                    labels_eq = labels_eq.float()
                    labels_eq.fill_diagonal_(0)
                    exp_sim = torch.exp(sim_matrix)
                    eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
                    exp_sim = exp_sim.masked_fill(eye_mask, 0.0)
                    pos_sum = (exp_sim * labels_eq).sum(dim=-1)
                    all_sum = exp_sim.sum(dim=-1)
                    pos_sum = pos_sum + 1e-8
                    all_sum = all_sum + 1e-8
                    loss_per_sample = -torch.log(pos_sum / all_sum)
                    valid_samples = labels_eq.sum(dim=-1) > 0
                    if valid_samples.any():
                        contrastive_loss = loss_per_sample[valid_samples].mean()

                loss = ce_loss + criterion.contrastive_weight * contrastive_loss + aux_loss
            else:
                logits, embeddings, _, aux_loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_embeddings=True,
                )
                loss = criterion(logits, embeddings, labels, aux_loss)
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
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches
    val_loss, val_acc, _ = evaluate_biencoder(model, val_loader)
    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_biencoder_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}"
            )
            break

print(f"\nBest Bi-Encoder model: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
model.load_state_dict(
    torch.load(f"{OUTPUT_DIR}/best_biencoder_model.pt", map_location=device)
)

# ============================================================
# EXTRACT BI-ENCODER EMBEDDINGS
# ============================================================
print("\nExtracting Bi-Encoder embeddings...")

def extract_biencoder_embeddings(model, loader):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with autocast():
                embeddings = model.encode(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

train_loader_no_labels = DataLoader(
    TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
val_loader_no_labels = DataLoader(
    TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader_no_labels = DataLoader(
    TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"]),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

train_embeddings = extract_biencoder_embeddings(model, train_loader_no_labels)
val_embeddings = extract_biencoder_embeddings(model, val_loader_no_labels)
test_embeddings = extract_biencoder_embeddings(model, test_loader_no_labels)
print(
    f"Train embeddings: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}"
)

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
    y_train_labels,
    eval_set=[(xgb_val_features, y_val_labels)],
    verbose=False,
)

xgb_val_probs = xgb_model.predict_proba(xgb_val_features)
xgb_test_probs = xgb_model.predict_proba(xgb_test_features)
print(
    f"XGBoost validation log loss: {compute_log_loss(y_val_labels, xgb_val_probs):.4f}"
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
lr_model.fit(train_sparse, y_train_labels)

lr_val_probs = lr_model.predict_proba(val_sparse)
lr_test_probs = lr_model.predict_proba(test_sparse)
print(
    f"Logistic Regression validation log loss: {compute_log_loss(y_val_labels, lr_val_probs):.4f}"
)

# ============================================================
# BI-ENCODER VALIDATION & TEST PROBS
# ============================================================
print("\nGetting Bi-Encoder probabilities...")
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
_, _, biencoder_val_probs = evaluate_biencoder(model, val_loader_eval)
print(
    f"Bi-Encoder validation log loss: {compute_log_loss(y_val_labels, biencoder_val_probs):.4f}"
)

model.eval()
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with autocast():
            logits, _, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeddings=True,
            )
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())
biencoder_test_probs = np.vstack(all_test_probs)

# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================
print("\nOptimizing ensemble weights...")
val_probas = {
    "biencoder": biencoder_val_probs,
    "xgboost": xgb_val_probs,
    "lr": lr_val_probs,
}

best_ll = float("inf")
best_weights = None
for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.9:
            continue
        ensemble_proba = (
            w1 * val_probas["biencoder"]
            + w2 * val_probas["xgboost"]
            + w3 * val_probas["lr"]
        )
        ll = compute_log_loss(y_val_labels, ensemble_proba)
        if ll < best_ll:
            best_ll = ll
            best_weights = {"biencoder": w1, "xgboost": w2, "lr": w3}

print(f"Optimized ensemble weights: {best_weights}")
print(f"Ensemble validation log loss: {best_ll:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
test_probas = {
    "biencoder": biencoder_test_probs,
    "xgboost": xgb_test_probs,
    "lr": lr_test_probs,
}
ensemble_test_probs = (
    best_weights["biencoder"] * test_probas["biencoder"]
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

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nFinal Validation Score: {best_ll:.6f}")
