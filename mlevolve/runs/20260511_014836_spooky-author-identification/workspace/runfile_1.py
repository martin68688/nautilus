import os
os.sched_setaffinity(0, {77, 78, 79, 80, 81})
import pandas as pd
import numpy as np
import re
import os
import gc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()


train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)


# ============================================================
# 2. STYLISTIC FEATURE ENGINEERING
# ============================================================
def extract_stylistic_features(text):
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    avg_sentence_length = num_words / num_sentences
    num_commas = text.count(",")
    num_semicolons = text.count(";")
    num_colons = text.count(":")
    num_dashes = text.count("—") + text.count("–") + text.count("-")
    num_exclamations = text.count("!")
    num_questions = text.count("?")
    punct_ratio = (num_commas + num_semicolons + num_colons + num_dashes) / max(
        num_words, 1
    )
    exclamation_ratio = num_exclamations / max(num_words, 1)
    avg_word_length = num_chars / max(num_words, 1)
    capital_words = sum(1 for w in words if w[0].isupper() if len(w) > 0)
    capital_ratio = capital_words / max(num_words, 1)
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    all_caps_ratio = all_caps_words / max(num_words, 1)
    contraction_pattern = r"\b(can't|don't|won't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|didn't|couldn't|shouldn't|wouldn't|mightn't|mustn't|needn't|daren't|'ll|'re|'ve|'d|'m|'s)\b"
    num_contractions = len(re.findall(contraction_pattern, text.lower()))
    contraction_ratio = num_contractions / max(num_words, 1)
    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "shalt",
        "wilt",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "anon",
        "betwixt",
        "forsooth",
        "perchance",
        "durst",
        "methinks",
        "prithee",
        "wherein",
        "wherewith",
        "therewith",
    ]
    num_archaic = sum(1 for w in words if w.lower() in archaic_words)
    archaic_ratio = num_archaic / max(num_words, 1)
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / max(num_words, 1)

    def count_syllables(word):
        word = word.lower()
        syllable_count = 0
        vowels = "aeiouy"
        if len(word) <= 3:
            return 1
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                syllable_count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e"):
            syllable_count -= 1
        if word.endswith("le") and len(word) > 2:
            syllable_count += 1
        return max(1, syllable_count)

    total_syllables = sum(count_syllables(w) for w in words) if words else 1
    syllables_per_word = total_syllables / max(num_words, 1)
    fk_grade = 0.39 * avg_sentence_length + 11.8 * syllables_per_word - 15.59
    unique_chars_ratio = len(set(text)) / max(num_chars, 1)
    whitespace_ratio = text.count(" ") / max(num_chars, 1)
    punct_chars = "".join(re.findall(r"[^\w\s]", text))
    punct_diversity = (
        len(set(punct_chars)) / max(len(punct_chars), 1) if len(punct_chars) > 0 else 0
    )
    em_dashes = text.count("—") + text.count("--")
    features = {
        "num_words": num_words,
        "num_chars": num_chars,
        "num_sentences": num_sentences,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "num_commas": num_commas,
        "num_semicolons": num_semicolons,
        "num_colons": num_colons,
        "num_dashes": num_dashes,
        "num_exclamations": num_exclamations,
        "num_questions": num_questions,
        "punct_ratio": punct_ratio,
        "exclamation_ratio": exclamation_ratio,
        "capital_ratio": capital_ratio,
        "all_caps_ratio": all_caps_ratio,
        "contraction_ratio": contraction_ratio,
        "archaic_ratio": archaic_ratio,
        "long_word_ratio": long_word_ratio,
        "syllables_per_word": syllables_per_word,
        "fk_grade": fk_grade,
        "unique_chars_ratio": unique_chars_ratio,
        "whitespace_ratio": whitespace_ratio,
        "punct_diversity": punct_diversity,
        "em_dashes": em_dashes,
    }
    return features


print("Extracting stylistic features...")
train_features_list = [extract_stylistic_features(t) for t in train_df["text_clean"]]
train_stylistic_df = pd.DataFrame(train_features_list)
test_features_list = [extract_stylistic_features(t) for t in test_df["text_clean"]]
test_stylistic_df = pd.DataFrame(test_features_list)
print(
    f"Stylistic features shape: train={train_stylistic_df.shape}, test={test_stylistic_df.shape}"
)

# ============================================================
# 3. N-GRAM FEATURES
# ============================================================
print("Extracting character n-gram features...")
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=2000, min_df=5, max_df=0.95
)
char_ngrams_train = char_vectorizer.fit_transform(train_df["text_clean"]).toarray()
char_ngrams_test = char_vectorizer.transform(test_df["text_clean"]).toarray()
print(
    f"Char n-grams shape: train={char_ngrams_train.shape}, test={char_ngrams_test.shape}"
)

print("Extracting word n-gram TF-IDF features...")
word_tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=3000,
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
)
word_tfidf_train = word_tfidf.fit_transform(train_df["text_clean"]).toarray()
word_tfidf_test = word_tfidf.transform(test_df["text_clean"]).toarray()
print(
    f"Word TF-IDF shape: train={word_tfidf_train.shape}, test={word_tfidf_test.shape}"
)

# ============================================================
# 4. FUNCTION WORDS FEATURES
# ============================================================
function_words = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "were",
    "are",
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
    "no",
    "nor",
    "neither",
    "so",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
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
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "theirs",
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "ourselves",
    "yourselves",
    "themselves",
    "who",
    "whom",
    "which",
    "what",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "many",
    "much",
    "some",
    "any",
    "only",
    "own",
    "same",
    "such",
    "very",
    "too",
    "quite",
    "rather",
    "here",
    "there",
    "now",
    "then",
    "still",
    "already",
    "always",
    "never",
    "often",
    "sometimes",
    "generally",
    "usually",
    "almost",
    "just",
    "about",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "away",
    "back",
    "around",
    "along",
    "through",
    "throughout",
    "across",
    "into",
    "upon",
    "within",
    "without",
    "between",
    "among",
    "amongst",
    "above",
    "below",
    "beneath",
    "beside",
    "beyond",
    "toward",
    "towards",
    "before",
    "after",
    "during",
    "since",
    "until",
    "till",
    "once",
    "while",
    "whilst",
    "although",
    "though",
    "whereas",
    "because",
    "since",
    "unless",
    "except",
    "like",
    "as",
]


def extract_function_word_features(text, fw_list):
    words = text.lower().split()
    word_count = len(words) if len(words) > 0 else 1
    return {f"fw_{fw}": sum(1 for w in words if w == fw) / word_count for fw in fw_list}


print("Extracting function word features...")
train_fw = train_df["text_clean"].apply(
    lambda x: extract_function_word_features(x, function_words)
)
train_fw_df = pd.DataFrame(train_fw.tolist())
test_fw = test_df["text_clean"].apply(
    lambda x: extract_function_word_features(x, function_words)
)
test_fw_df = pd.DataFrame(test_fw.tolist())
print(
    f"Function word features shape: train={train_fw_df.shape}, test={test_fw_df.shape}"
)


# ============================================================
# 5. SENTENCE BOUNDARY FEATURES
# ============================================================
def extract_sentence_boundary_features(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) == 0:
        return {
            "first_word_avg_len": 0,
            "last_word_avg_len": 0,
            "the_start_ratio": 0,
            "conj_start_ratio": 0,
        }
    first_words = [s.split()[0] for s in sentences if len(s.split()) > 0]
    last_words = [s.split()[-1] for s in sentences if len(s.split()) > 0]
    first_word_avg_len = np.mean([len(w) for w in first_words]) if first_words else 0
    last_word_avg_len = np.mean([len(w) for w in last_words]) if last_words else 0
    the_starts = sum(1 for w in first_words if w.lower() == "the")
    the_start_ratio = the_starts / max(len(sentences), 1)
    conj_starts_words = ["and", "but", "or", "nor", "for", "yet", "so"]
    conj_starts = sum(1 for w in first_words if w.lower() in conj_starts_words)
    conj_start_ratio = conj_starts / max(len(sentences), 1)
    return {
        "first_word_avg_len": first_word_avg_len,
        "last_word_avg_len": last_word_avg_len,
        "the_start_ratio": the_start_ratio,
        "conj_start_ratio": conj_start_ratio,
    }


print("Extracting sentence boundary features...")
train_boundary = train_df["text_clean"].apply(extract_sentence_boundary_features)
train_boundary_df = pd.DataFrame(train_boundary.tolist())
test_boundary = test_df["text_clean"].apply(extract_sentence_boundary_features)
test_boundary_df = pd.DataFrame(test_boundary.tolist())
print(
    f"Sentence boundary features shape: train={train_boundary_df.shape}, test={test_boundary_df.shape}"
)


# ============================================================
# 6. STYLE PATTERN FEATURES
# ============================================================
def extract_style_patterns(text):
    num_quotes = len(re.findall(r'["""]', text))
    dialogue_flag = 1 if num_quotes >= 2 else 0
    num_parens = len(re.findall(r"[\(\)\[\]\{\}]", text))
    num_ellipsis = len(re.findall(r"\.{3,}", text))
    num_emdash = len(re.findall(r"—|--", text))
    colon_count = text.count(":")
    semicolon_count = text.count(";")
    colon_semicolon_ratio = colon_count / max(semicolon_count, 1)
    question_count = text.count("?")
    exclam_count = text.count("!")
    interj_excl_ratio = (question_count + exclam_count) / max(len(text), 1) * 1000
    num_numbers = len(re.findall(r"\d+", text))
    return {
        "dialogue_flag": dialogue_flag,
        "num_parens": num_parens,
        "num_ellipsis": num_ellipsis,
        "num_emdash": num_emdash,
        "colon_semicolon_ratio": colon_semicolon_ratio,
        "interj_excl_ratio": interj_excl_ratio,
        "num_numbers": num_numbers,
    }


print("Extracting style pattern features...")
train_style = train_df["text_clean"].apply(extract_style_patterns)
train_style_df = pd.DataFrame(train_style.tolist())
test_style = test_df["text_clean"].apply(extract_style_patterns)
test_style_df = pd.DataFrame(test_style.tolist())
print(
    f"Style pattern features shape: train={train_style_df.shape}, test={test_style_df.shape}"
)

# ============================================================
# 7. LABEL ENCODING
# ============================================================
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["author"])
print(f"Classes: {label_encoder.classes_}")
print(f"Class distribution: {np.bincount(y_train)}")

# Store raw texts and features for later use inside folds
os.makedirs("./working", exist_ok=True)
train_df["text"].to_csv("./working/train_texts.csv", index=False)
test_df["text"].to_csv("./working/test_texts.csv", index=False)
np.save("./working/y_train.npy", y_train)
np.save("./working/label_classes.npy", label_encoder.classes_)
test_ids = test_df["id"].values
np.save("./working/test_ids.npy", test_ids)

# Combine all features into unified feature matrices
print("Combining all features...")

# Concatenate all train features
train_all_features = np.concatenate([
    train_stylistic_df.values,
    char_ngrams_train,
    word_tfidf_train,
    train_fw_df.values,
    train_boundary_df.values,
    train_style_df.values
], axis=1)

# Concatenate all test features
test_all_features = np.concatenate([
    test_stylistic_df.values,
    char_ngrams_test,
    word_tfidf_test,
    test_fw_df.values,
    test_boundary_df.values,
    test_style_df.values
], axis=1)

print(f"Combined train features shape: {train_all_features.shape}")
print(f"Combined test features shape: {test_all_features.shape}")

# NOTE: Scaling will be done per-fold inside run_training to prevent data leakage.
# Save unscaled features; fold-level scaling happens later.

# Save combined features
np.save("./working/X_train.npy", train_all_features)
np.save("./working/X_test.npy", test_all_features)
np.save("./working/y_train.npy", y_train)
np.save("./working/label_classes.npy", label_encoder.classes_)
np.save("./working/test_ids.npy", test_ids)

# Also save original train text labels for fold indices creation
# Create stratified fold indices and save them
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = [(train_idx, val_idx) for train_idx, val_idx in skf.split(train_all_features, y_train)]
np.save("./working/fold_indices.npy", np.array(fold_indices, dtype=object))

# Save train and test texts
train_df["text"].to_csv("./working/train_texts.csv", index=False)
test_df["text"].to_csv("./working/test_texts.csv", index=False)

print("Preprocessed data saved to ./working/")


# ============================================================
# 8. MODEL ARCHITECTURE
# ============================================================
class StylometricGatingNetwork(nn.Module):
    def __init__(self, stylometric_dim, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(stylometric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, stylometric_dim),
            nn.Sigmoid(),
        )

    def forward(self, stylometric_features):
        gate_weights = self.gate(stylometric_features)
        return stylometric_features * gate_weights


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.scale_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(3)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 3 * (hidden_size // 2), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, hidden_states, attention_mask):
        batch_size = hidden_states.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        attn_output, attn_weights = self.multihead_attn(
            query,
            hidden_states,
            hidden_states,
            key_padding_mask=(
                ~attention_mask.bool() if attention_mask is not None else None
            ),
        )
        multi_scale_features = []
        for proj in self.scale_projections:
            scale_attn = F.softmax(
                attn_weights / (0.5 + 0.5 * torch.rand(1).item()), dim=-1
            )
            scale_output = torch.bmm(scale_attn, hidden_states)
            scale_features = proj(scale_output.squeeze(1))
            multi_scale_features.append(scale_features)
        pooled = attn_output.squeeze(1)
        multi_scale = torch.cat(multi_scale_features, dim=-1)
        fused = self.fusion(torch.cat([pooled, multi_scale], dim=-1))
        return fused


class HierarchicalAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads // 2,
            batch_first=True,
            dropout=dropout,
        )
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        window_size = min(128, seq_len)

        # Create local mask of shape (1, seq_len, seq_len) and expand to batch
        local_mask = torch.zeros(1, seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            local_mask[0, i, start:end] = 1

        # Expand to batch and also mask out padding tokens
        local_mask = local_mask.expand(batch_size, -1, -1).bool()
        if attention_mask is not None:
            # attention_mask has shape (batch_size, seq_len) -> expand to (batch_size, seq_len, seq_len)
            extended_mask = attention_mask.unsqueeze(1).float()
            local_mask = local_mask & (extended_mask.bool())

        # For MultiheadAttention with batch_first, attn_mask should be (batch_size*num_heads, L, S)
        # We need to repeat for each head
        num_heads_local = self.local_attention.num_heads
        local_mask = local_mask.repeat_interleave(num_heads_local, dim=0)

        local_out, _ = self.local_attention(x, x, x, attn_mask=local_mask)
        x = self.layer_norm1(x + self.dropout(local_out))

        # Global attention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        global_out, _ = self.global_attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
        )
        x = self.layer_norm2(x + self.dropout(global_out))
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return x


class AuthorEnsembleHead(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=3, num_heads=5, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.LayerNorm(hidden_size // 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 4, num_classes),
                )
                for _ in range(num_heads)
            ]
        )
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_heads),
            nn.Softmax(dim=-1),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, return_weights=False):
        weights = self.weight_net(x)
        head_outputs = [head(x) for head in self.heads]
        head_logits = torch.stack(head_outputs, dim=-1)
        weighted_logits = torch.sum(head_logits * weights.unsqueeze(1), dim=-1)
        scaled_logits = weighted_logits / self.temperature
        if return_weights:
            return scaled_logits, weights
        return scaled_logits


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_classes=3, stylometric_dim=150, dropout=0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        self.hidden_size = self.deberta.config.hidden_size
        for param in self.deberta.parameters():
            param.requires_grad = False
        for param in self.deberta.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.deberta.embeddings.parameters():
            param.requires_grad = True
        self.stylometric_gate = StylometricGatingNetwork(stylometric_dim, hidden_dim=64)
        self.stylometric_proj = nn.Sequential(
            nn.Linear(stylometric_dim, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.multi_scale_extractor = MultiScaleFeatureExtractor(
            hidden_size=self.hidden_size, num_heads=8
        )
        self.hierarchical_blocks = nn.ModuleList(
            [
                HierarchicalAttentionBlock(
                    self.hidden_size, num_heads=8, dropout=dropout
                )
                for _ in range(2)
            ]
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size // 4, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = AuthorEnsembleHead(
            hidden_size=self.hidden_size // 2,
            num_classes=num_classes,
            num_heads=5,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, attention_mask, stylometric_features=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        for block in self.hierarchical_blocks:
            hidden_states = block(hidden_states, attention_mask)
        pooled_features = self.multi_scale_extractor(hidden_states, attention_mask)
        if stylometric_features is not None:
            gated_stylo = self.stylometric_gate(stylometric_features)
            stylo_projected = self.stylometric_proj(gated_stylo)
            combined = torch.cat([pooled_features, stylo_projected], dim=-1)
        else:
            stylo_padding = torch.zeros(
                pooled_features.shape[0],
                self.hidden_size // 4,
                device=pooled_features.device,
            )
            combined = torch.cat([pooled_features, stylo_padding], dim=-1)
        fused_features = self.feature_fusion(combined)
        logits = self.classifier(fused_features)
        return logits


class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        pt = torch.sum(probs * true_dist, dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        nll_loss = torch.sum(-true_dist * log_probs, dim=-1)
        if self.alpha is not None:
            alpha_weight = (
                self.alpha[target] if hasattr(self.alpha, "__getitem__") else self.alpha
            )
            loss = focal_weight * alpha_weight * nll_loss
        else:
            loss = focal_weight * nll_loss
        return torch.mean(loss)


# ============================================================
# 9. DATASET CLASS
# ============================================================
class SpookyTextDataset(Dataset):
    def __init__(
        self, texts, tokenizer, stylometric_features=None, labels=None, max_length=512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.stylometric_features = stylometric_features
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.stylometric_features is not None:
            item["stylometric_features"] = torch.FloatTensor(
                self.stylometric_features[idx]
            )
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 10. TRAINING AND EVALUATION FUNCTIONS
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0
    accum_steps = config.get("gradient_accumulation_steps", 1)
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        stylo_feat = batch.get("stylometric_features", None)
        if stylo_feat is not None:
            stylo_feat = stylo_feat.to(device)
        with autocast(enabled=config.get("mixed_precision", True)):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylo_feat,
            )
            loss = criterion(logits, labels)
            loss = loss / accum_steps
        if config.get("mixed_precision", True):
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * accum_steps
        if (batch_idx + 1) % accum_steps == 0:
            if config.get("mixed_precision", True):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("gradient_clip_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("gradient_clip_norm", 1.0)
                )
                optimizer.step()
            optimizer.zero_grad()
    return total_loss / len(dataloader)


def validate(model, dataloader, device, config):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            stylo_feat = batch.get("stylometric_features", None)
            if stylo_feat is not None:
                stylo_feat = stylo_feat.to(device)
            with autocast(enabled=config.get("mixed_precision", True)):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stylometric_features=stylo_feat,
                )
                probs = F.softmax(logits, dim=-1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    eps = 1e-15
    all_preds = np.clip(all_preds, eps, 1 - eps)
    all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)
    N = len(all_labels)
    log_loss_val = 0
    for i in range(N):
        for j in range(3):
            y_ij = 1 if all_labels[i] == j else 0
            log_loss_val += y_ij * np.log(all_preds[i, j])
    log_loss_val = -log_loss_val / N
    return log_loss_val, all_preds


# ============================================================
# 11. MAIN TRAINING LOOP
# ============================================================
def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train = np.load("./working/X_train.npy", allow_pickle=True)
    X_test = np.load("./working/X_test.npy", allow_pickle=True)
    y_train = np.load("./working/y_train.npy", allow_pickle=True)
    fold_indices = np.load("./working/fold_indices.npy", allow_pickle=True)
    label_classes = np.load("./working/label_classes.npy", allow_pickle=True)
    train_texts = pd.read_csv("./working/train_texts.csv")["text"].values
    test_texts = pd.read_csv("./working/test_texts.csv")["text"].values
    test_ids = np.load("./working/test_ids.npy", allow_pickle=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    tokenizer.model_max_length = 512
    # stylometric_dim is the full feature dimension (all engineered features)
    stylometric_dim = X_train.shape[1]

    test_preds_folds = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{len(fold_indices)}")
        print(f"{'='*50}")

        train_texts_fold = train_texts[train_idx]
        val_texts_fold = train_texts[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]
        # Fit scaler on training fold only to prevent data leakage
        fold_scaler = StandardScaler()
        train_features_fold = fold_scaler.fit_transform(X_train[train_idx])
        val_features_fold = fold_scaler.transform(X_train[val_idx])

        model = SpookyAuthorClassifier(
            num_classes=config["num_classes"],
            stylometric_dim=stylometric_dim,
            dropout=config["dropout_rate"],
        )
        model = model.to(device)

        criterion = FocalLossWithLabelSmoothing(
            smoothing=config["label_smoothing"],
            gamma=config["focal_gamma"],
            num_classes=config["num_classes"],
        )

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "deberta" in n and "encoder" in n
                ],
                "lr": config["learning_rate"] * 0.5,
                "weight_decay": config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "deberta" in n and "embeddings" in n
                ],
                "lr": config["learning_rate"] * 0.3,
                "weight_decay": config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "deberta" not in n
                ],
                "lr": config["learning_rate"],
                "weight_decay": config["weight_decay"] * 0.5,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=config["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=config["learning_rate"] * 0.01
        )
        scaler = GradScaler() if config.get("mixed_precision", True) else None

        train_dataset = SpookyTextDataset(
            texts=train_texts_fold,
            tokenizer=tokenizer,
            stylometric_features=train_features_fold,
            labels=y_train_fold,
            max_length=config["max_length"],
        )
        val_dataset = SpookyTextDataset(
            texts=val_texts_fold,
            tokenizer=tokenizer,
            stylometric_features=val_features_fold,
            labels=y_val_fold,
            max_length=config["max_length"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"] * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        best_fold_val_score = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(config["num_epochs"]):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config
            )
            val_score, _ = validate(model, val_loader, device, config)
            scheduler.step()
            print(
                f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {train_loss:.4f} - Val Log Loss: {val_score:.4f} - LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            if val_score < best_fold_val_score:
                best_fold_val_score = val_score
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        model.load_state_dict(best_model_state)
        model = model.to(device)
        val_score, _ = validate(model, val_loader, device, config)
        val_scores.append(val_score)
        print(f"Fold {fold + 1} Validation Log Loss: {val_score:.6f}")

        test_dataset = SpookyTextDataset(
            texts=test_texts,
            tokenizer=tokenizer,
            stylometric_features=fold_scaler.transform(X_test),
            max_length=config["max_length"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"] * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                stylo_feat = batch.get("stylometric_features", None)
                if stylo_feat is not None:
                    stylo_feat = stylo_feat.to(device)
                with autocast(enabled=config.get("mixed_precision", True)):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        stylometric_features=stylo_feat,
                    )
                    probs = F.softmax(logits, dim=-1)
                fold_test_preds.append(probs.cpu().numpy())
        fold_test_preds = np.concatenate(fold_test_preds, axis=0)
        test_preds_folds.append(fold_test_preds)

        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    test_preds = np.mean(test_preds_folds, axis=0)
    eps = 1e-15
    test_preds = np.clip(test_preds, eps, 1 - eps)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

    mean_val_score = np.mean(val_scores)
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results:")
    for i, score in enumerate(val_scores):
        print(f"Fold {i+1}: {score:.6f}")
    print(f"Mean Validation Log Loss: {mean_val_score:.6f}")
    print(f"Std Validation Log Loss: {np.std(val_scores):.6f}")

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "EAP": test_preds[:, 0],
            "HPL": test_preds[:, 1],
            "MWS": test_preds[:, 2],
        }
    )
    submission.to_csv("./submission/submission_a52a09a3557446969605770a02538b4b.csv", index=False)
    print(f"Submission saved to ./submission/submission_a52a09a3557446969605770a02538b4b.csv")
    print(f"Submission shape: {submission.shape}")

    score = mean_val_score
    print(f"Final Validation Score: {score}")
    return score


# ============================================================
# 12. CONFIGURATION AND EXECUTION
# ============================================================
config = {
    "model_name": "microsoft/deberta-v3-large",
    "num_classes": 3,
    "max_length": 512,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_epochs": 25,
    "patience": 5,
    "label_smoothing": 0.1,
    "focal_gamma": 2.0,
    "dropout_rate": 0.15,
    "mixed_precision": True,
    "gradient_clip_norm": 1.0,
}

if __name__ == "__main__":
    score = run_training(config)