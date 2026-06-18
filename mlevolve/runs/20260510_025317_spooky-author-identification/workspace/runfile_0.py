import os
os.sched_setaffinity(0, {0, 1, 2, 3, 4})
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, DistilBertModel, DistilBertConfig
import xgboost as xgb
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Encode target
le = LabelEncoder()
train["author_encoded"] = le.fit_transform(train["author"])
num_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

# Create train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train["text"], train["author_encoded"])))

train_df = train.iloc[train_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def extract_features(text_series, is_train=True):
    features = pd.DataFrame(index=text_series.index)

    # Basic text stats
    features["char_count"] = text_series.apply(len)
    features["word_count"] = text_series.apply(lambda x: len(str(x).split()))
    features["sent_count"] = text_series.apply(
        lambda x: len(re.split(r"[.!?]+", str(x))) - 1
    )
    features["avg_word_len"] = text_series.apply(
        lambda x: (
            np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
    )

    # Punctuation features
    features["excl_count"] = text_series.apply(lambda x: str(x).count("!"))
    features["quest_count"] = text_series.apply(lambda x: str(x).count("?"))
    features["period_count"] = text_series.apply(lambda x: str(x).count("."))
    features["comma_count"] = text_series.apply(lambda x: str(x).count(","))
    features["semi_count"] = text_series.apply(lambda x: str(x).count(";"))
    features["colon_count"] = text_series.apply(lambda x: str(x).count(":"))
    features["quote_count"] = text_series.apply(
        lambda x: str(x).count('"') + str(x).count("'")
    )
    features["dash_count"] = text_series.apply(lambda x: str(x).count("-"))

    # Caps and special patterns
    features["caps_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    features["caps_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    features["digit_count"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isdigit())
    )

    # Readability features
    features["syllable_est"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).split()
            for c in ["a", "e", "i", "o", "u", "y"]
            if c in w.lower()
        )
    )
    features["complex_words"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 6)
    )

    # Quote density
    features["quote_density"] = text_series.apply(
        lambda x: str(x).count('"') / max(len(str(x)), 1)
    )

    # Stopword-based features
    stopwords = {
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
    }

    features["stopword_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in stopwords)
    )
    features["stopword_ratio"] = features["stopword_count"] / features[
        "word_count"
    ].replace(0, 1)

    # Word length distribution
    features["long_word_ratio"] = features["complex_words"] / features[
        "word_count"
    ].replace(0, 1)
    features["short_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) <= 3)
    ) / features["word_count"].replace(0, 1)

    # Vocabulary diversity
    features["unique_words"] = text_series.apply(
        lambda x: len(set(str(x).lower().split()))
    )
    features["type_token_ratio"] = features["unique_words"] / features[
        "word_count"
    ].replace(0, 1)

    # Historical/archaic word indicators
    archaic_words = {
        "thee",
        "thou",
        "thy",
        "thine",
        "ye",
        "hath",
        "doth",
        "art",
        "hast",
        "whence",
        "thence",
        "hither",
        "thither",
        "whither",
        "ere",
        "nay",
        "forsooth",
        "perchance",
        "anon",
        "prithee",
    }
    lovecraft_words = {
        "eldritch",
        "cyclopean",
        "non-euclidean",
        "ichor",
        "gibbering",
        "maddening",
        "cosmic",
        "carcosa",
        "yog-sothoth",
        "cthulhu",
    }
    poe_words = {
        "nevermore",
        "chamber",
        "tapping",
        "rapping",
        "sepulchre",
        "ghoul",
        "pallid",
        "dreary",
        "weary",
        "ebony",
    }

    features["archaic_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in archaic_words)
    )
    features["lovecraft_score"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in lovecraft_words)
    )
    features["poe_score"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in poe_words)
    )

    # First person pronoun usage
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    features["first_person_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in first_person)
    )

    # Sentence complexity
    features["avg_sent_length"] = features["word_count"] / features[
        "sent_count"
    ].replace(0, 1)

    # Character diversity
    features["unique_chars"] = text_series.apply(lambda x: len(set(str(x).lower())))
    features["char_diversity"] = features["unique_chars"] / features[
        "char_count"
    ].replace(0, 1)

    # Negation count
    negation_words = {
        "not",
        "no",
        "never",
        "nothing",
        "nowhere",
        "none",
        "neither",
        "nor",
        "cannot",
        "can't",
        "don't",
        "won't",
        "doesn't",
        "isn't",
    }
    features["negation_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in negation_words)
    )

    # Emotion/affect words
    positive_words = set(
        [
            "love",
            "beautiful",
            "wonderful",
            "happy",
            "joy",
            "delight",
            "pleasure",
            "gentle",
            "sweet",
            "tender",
            "bright",
            "hope",
        ]
    )
    negative_words = set(
        [
            "dark",
            "dread",
            "fear",
            "horror",
            "terror",
            "death",
            "shadow",
            "gloom",
            "pain",
            "sorrow",
            "misery",
            "agony",
            "anguish",
            "doom",
        ]
    )

    features["positive_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in positive_words)
    )
    features["negative_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in negative_words)
    )
    features["sentiment_ratio"] = (
        features["positive_count"] - features["negative_count"]
    ) / (features["positive_count"] + features["negative_count"] + 1)

    return features


# Extract features
train_features = extract_features(train_df["text"], is_train=True)
val_features = extract_features(val_df["text"], is_train=False)
test_features = extract_features(test["text"], is_train=False)

# Handle NaN and inf values
train_features = train_features.replace([np.inf, -np.inf], 0).fillna(0)
val_features = val_features.replace([np.inf, -np.inf], 0).fillna(0)
test_features = test_features.replace([np.inf, -np.inf], 0).fillna(0)

# Add percentile features
for df_features, df in [
    (train_features, train_df),
    (val_features, val_df),
    (test_features, test),
]:
    df_features["word_count_rank"] = df_features["word_count"].rank(pct=True)
    df_features["char_count_rank"] = df_features["char_count"].rank(pct=True)


# N-gram features
def create_ngram_count(text, n=2, ngram_type="char"):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    if ngram_type == "char":
        common_ngrams = [
            "th",
            "he",
            "in",
            "er",
            "an",
            "re",
            "nd",
            "at",
            "on",
            "nt",
            "ha",
            "ou",
            "it",
            "hi",
            "es",
            "st",
            "en",
            "ea",
            "to",
            "or",
            "ed",
            "te",
            "ar",
            "al",
            "le",
            "ve",
            "ti",
            "ra",
            "ur",
            "me",
        ]
        return sum(text.count(ng) for ng in common_ngrams) / max(len(text), 1)
    else:
        words = text.split()
        common_bigrams = [
            ("of", "the"),
            ("in", "the"),
            ("to", "the"),
            ("and", "the"),
            ("it", "was"),
            ("i", "was"),
            ("there", "was"),
            ("this", "was"),
        ]
        count = 0
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            if bigram in common_bigrams:
                count += 1
        return count / max(len(words), 1)


for prefix, df in [("train", train_df), ("val", val_df), ("test", test)]:
    text_data = df["text"]
    ngram_char = text_data.apply(lambda x: create_ngram_count(x, 2, "char"))
    ngram_word = text_data.apply(lambda x: create_ngram_count(x, 2, "word"))
    if prefix == "train":
        train_features["ngram_char_density"] = ngram_char.values
        train_features["ngram_word_density"] = ngram_word.values
    elif prefix == "val":
        val_features["ngram_char_density"] = ngram_char.values
        val_features["ngram_word_density"] = ngram_word.values
    else:
        test_features["ngram_char_density"] = ngram_char.values
        test_features["ngram_word_density"] = ngram_word.values

# Scale features
scaler = StandardScaler()
feature_columns = train_features.columns

train_features_scaled = pd.DataFrame(
    scaler.fit_transform(train_features),
    columns=feature_columns,
    index=train_features.index,
)
val_features_scaled = pd.DataFrame(
    scaler.transform(val_features), columns=feature_columns, index=val_features.index
)
test_features_scaled = pd.DataFrame(
    scaler.transform(test_features), columns=feature_columns, index=test_features.index
)

# Combine with original data
train_processed = pd.concat(
    [
        train_df[["id", "text", "author", "author_encoded"]].reset_index(drop=True),
        train_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)
val_processed = pd.concat(
    [
        val_df[["id", "text", "author", "author_encoded"]].reset_index(drop=True),
        val_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)
test_processed = pd.concat(
    [
        test[["id", "text"]].reset_index(drop=True),
        test_features_scaled.reset_index(drop=True),
    ],
    axis=1,
)

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed val shape: {val_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# Extract arrays
X_train_tab = train_processed[feature_columns].values
y_train = train_processed["author_encoded"].values
X_val_tab = val_processed[feature_columns].values
y_val = val_processed["author_encoded"].values
X_test_tab = test_processed[feature_columns].values

train_texts = train_processed["text"].tolist()
val_texts = val_processed["text"].tolist()
test_texts = test_processed["text"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# TRAIN XGBoost
# ============================================================
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)
xgb_val_probs = xgb_model.predict_proba(X_val_tab)
xgb_test_probs = xgb_model.predict_proba(X_test_tab)
print(f"XGBoost val log loss: {log_loss(y_val, xgb_val_probs):.6f}")

# ============================================================
# TRAIN LightGBM
# ============================================================
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_leaves=31,
    objective="multiclass",
    metric="multi_logloss",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_tab,
    y_train,
    eval_set=[(X_val_tab, y_val)],
    callbacks=[lgb.early_stopping(30)],
)
lgb_val_probs = lgb_model.predict_proba(X_val_tab)
lgb_test_probs = lgb_model.predict_proba(X_test_tab)
print(f"LightGBM val log loss: {log_loss(y_val, lgb_val_probs):.6f}")


# ============================================================
# TEXT ENCODERS (DistilBERT and RoBERTa)
# ============================================================
class WordAttention(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, mask=None):
        # hidden_states: (batch, seq_len, hidden_size)
        # mask: (batch, seq_len)
        attn_weights = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -65500)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)
        return context, attn_weights


class SentenceAttention(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, sentence_vectors, mask=None):
        # sentence_vectors: (batch, num_sentences, hidden_size)
        # mask: (batch, num_sentences)
        attn_weights = self.attention(sentence_vectors).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -65500)
        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), sentence_vectors).squeeze(1)
        return context, attn_weights


class HANLayer(nn.Module):
    def __init__(self, input_size, hidden_size=256, max_sentences=16, max_words_per_sentence=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence

        self.word_gru = nn.GRU(input_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.word_attention = WordAttention(hidden_size)
        self.word_fc = nn.Linear(hidden_size, hidden_size)

        self.sent_gru = nn.GRU(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.sent_attention = SentenceAttention(hidden_size)
        self.sent_fc = nn.Linear(hidden_size, hidden_size)

    def split_into_sentences(self, token_ids, attention_mask):
        # token_ids: (batch, seq_len)
        # attention_mask: (batch, seq_len)
        batch_size, seq_len = token_ids.shape

        # Use period (token 1012 in BERT/DistilBERT) or [SEP] as sentence boundary
        period_token = 1012  # BERT period token
        sep_token = 102  # [SEP]

        # Find sentence boundaries
        all_sentences = []
        all_sent_masks = []
        all_word_masks = []

        for b in range(batch_size):
            sentences = []
            sent_masks = []
            word_masks = []
            current_start = 1  # Skip [CLS]

            # Find sentence boundaries
            boundaries = [current_start]
            for pos in range(current_start, seq_len):
                if token_ids[b, pos] in [period_token, sep_token] or pos == seq_len - 1:
                    if pos > current_start:
                        boundaries.append(pos)
                        current_start = pos + 1
            if current_start < seq_len and token_ids[b, current_start-1] != sep_token:
                boundaries.append(seq_len)

            # Create sentence chunks
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                words = token_ids[b, start:min(end, start + self.max_words_per_sentence)]
                word_mask = attention_mask[b, start:min(end, start + self.max_words_per_sentence)]

                # Pad words
                if len(words) < self.max_words_per_sentence:
                    pad_len = self.max_words_per_sentence - len(words)
                    words = F.pad(words, (0, pad_len), value=0)
                    word_mask = F.pad(word_mask, (0, pad_len), value=0)

                sentences.append(words)
                sent_masks.append(1)
                word_masks.append(word_mask)

            # Pad sentences
            while len(sentences) < self.max_sentences:
                sentences.append(torch.zeros(self.max_words_per_sentence, dtype=torch.long, device=token_ids.device))
                sent_masks.append(0)
                word_masks.append(torch.zeros(self.max_words_per_sentence, dtype=torch.long, device=token_ids.device))

            sentences = torch.stack(sentences[:self.max_sentences])
            sent_masks = torch.tensor(sent_masks[:self.max_sentences], device=token_ids.device)
            word_masks = torch.stack(word_masks[:self.max_sentences])

            all_sentences.append(sentences)
            all_sent_masks.append(sent_masks)
            all_word_masks.append(word_masks)

        sentences = torch.stack(all_sentences)  # (batch, max_sentences, max_words)
        sent_masks = torch.stack(all_sent_masks)  # (batch, max_sentences)
        word_masks = torch.stack(all_word_masks)  # (batch, max_sentences, max_words)

        return sentences, sent_masks, word_masks

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch, seq_len)
        # attention_mask: (batch, seq_len)

        # Split into sentences (use input_ids directly)
        sentences, sent_masks, word_masks = self.split_into_sentences(input_ids, attention_mask)
        batch_size, num_sentences, max_words = sentences.shape

        # Word-level processing (using embeddings is done in the encoder, here we use token IDs as placeholder)
        # The actual word embeddings come from the BERT encoder, so this layer is applied after BERT
        # For now, return a placeholder - the actual HAN processing will be done in the encoder forward
        return sentences, sent_masks, word_masks


class HierarchicalAttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size=256, max_sentences=12, max_words_per_sentence=20):
        super().__init__()
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence
        self.hidden_size = hidden_size
        self.period_token = 1012  # Default BERT period token
        self.sep_token = 102      # Default BERT [SEP] token

        self.word_gru = nn.GRU(input_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.word_attention = WordAttention(hidden_size)
        self.word_proj = nn.Linear(hidden_size, hidden_size)

        self.sent_gru = nn.GRU(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.sent_attention = SentenceAttention(hidden_size)

    def split_into_sentences(self, token_ids, attention_mask):
        batch_size, seq_len = token_ids.shape

        period_token = self.period_token
        sep_token = self.sep_token

        all_sentences = []
        all_sent_masks = []
        all_word_masks = []
        all_sentence_ranges = []

        for b in range(batch_size):
            sentence_ranges = []
            current_start = 1

            for pos in range(current_start, seq_len):
                if token_ids[b, pos] in [period_token, sep_token] or (attention_mask[b, pos] == 0 and pos > current_start):
                    if pos > current_start:
                        sentence_ranges.append((current_start, pos))
                    current_start = pos + 1

            # Handle last sentence
            if current_start < seq_len and attention_mask[b, seq_len-1].item() > 0:
                last_end = seq_len
                for pos in range(seq_len-1, current_start, -1):
                    if attention_mask[b, pos] == 0:
                        last_end = pos
                    else:
                        break
                if last_end > current_start:
                    sentence_ranges.append((current_start, last_end))

            all_sentence_ranges.append(sentence_ranges)

        return all_sentence_ranges

    def forward(self, hidden_states, input_ids, attention_mask):
        # hidden_states: (batch, seq_len, input_size)
        # input_ids: (batch, seq_len)
        # attention_mask: (batch, seq_len)
        batch_size, seq_len, input_size = hidden_states.shape

        sentence_ranges = self.split_into_sentences(input_ids, attention_mask)

        # Process each sentence with word-level attention
        sentence_vectors = []
        sentence_mask = []

        for b in range(batch_size):
            sentences_b = sentence_ranges[b]
            sentence_vecs_b = []

            for start, end in sentences_b:
                if end - start > self.max_words_per_sentence:
                    end = start + self.max_words_per_sentence

                word_hidden = hidden_states[b, start:end, :].unsqueeze(0)  # (1, sent_len, input_size)
                word_mask = attention_mask[b, start:end].unsqueeze(0)  # (1, sent_len)

                # Word-level GRU
                word_output, _ = self.word_gru(word_hidden)  # (1, sent_len, hidden_size)

                # Word-level attention
                word_context, _ = self.word_attention(word_output, word_mask)  # (1, hidden_size)
                word_context = self.word_proj(word_context)
                sentence_vecs_b.append(word_context)

            # Pad sentences to max_sentences
            while len(sentence_vecs_b) < self.max_sentences:
                sentence_vecs_b.append(torch.zeros(1, self.hidden_size, device=hidden_states.device))

            sentence_vecs_b = torch.cat(sentence_vecs_b[:self.max_sentences], dim=0)  # (max_sentences, hidden_size)
            sentence_vectors.append(sentence_vecs_b)

            sent_mask = torch.zeros(self.max_sentences, device=hidden_states.device)
            sent_mask[:min(len(sentences_b), self.max_sentences)] = 1
            sentence_mask.append(sent_mask)

        sentence_vectors = torch.stack(sentence_vectors, dim=0)  # (batch, max_sentences, hidden_size)
        sentence_mask = torch.stack(sentence_mask, dim=0)  # (batch, max_sentences)

        # Sentence-level GRU
        sent_output, _ = self.sent_gru(sentence_vectors)  # (batch, max_sentences, hidden_size)

        # Sentence-level attention
        doc_vector, _ = self.sent_attention(sent_output, sentence_mask)  # (batch, hidden_size)

        return doc_vector


class TextFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        hidden_dim=512,
        num_classes=3,
        dropout=0.5,
    ):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.encoder_hidden = self.distilbert.config.hidden_size

        # Hierarchical Attention Pooling
        self.han_pooling = HierarchicalAttentionPooling(
            input_size=self.encoder_hidden,
            hidden_size=hidden_dim,
            max_sentences=12,
            max_words_per_sentence=20
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # For tracking layers for progressive unfreezing
        self.frozen_encoder = True
        self.frozen_attention = True

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # Apply HAN pooling
        doc_vector = self.han_pooling(last_hidden, input_ids, attention_mask)

        logits = self.classifier(doc_vector)
        return logits


class RoBERTaEncoder(nn.Module):
    def __init__(
        self,
        model_name="roberta-base",
        hidden_dim=384,
        num_classes=3,
        dropout=0.5,
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.encoder_hidden = self.roberta.config.hidden_size

        # Hierarchical Attention Pooling - RoBERTa uses different token IDs
        # RoBERTa period token is 298, SEP is 2
        self.han_pooling = HierarchicalAttentionPooling(
            input_size=self.encoder_hidden,
            hidden_size=hidden_dim,
            max_sentences=12,
            max_words_per_sentence=20
        )
        self.period_token_id = 298  # RoBERTa period token
        self.sep_token_id = 2       # RoBERTa </s> token

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # For tracking layers for progressive unfreezing
        self.frozen_encoder = True
        self.frozen_attention = True

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # Apply HAN pooling - override period token for RoBERTa
        self.han_pooling.period_token = self.period_token_id
        self.han_pooling.sep_token = self.sep_token_id
        doc_vector = self.han_pooling(last_hidden, input_ids, attention_mask)

        logits = self.classifier(doc_vector)
        return logits


tokenizer_distilbert = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
text_encoder = TextFeatureEncoder(num_classes=num_classes, dropout=0.5).to(device)
roberta_encoder = RoBERTaEncoder(num_classes=num_classes, dropout=0.5).to(device)


# Dataset
# ============================================================
# AUGMENTATION
# ============================================================
def augment_text(text, prob=0.5):
    """Apply enhanced augmentation targeting author-specific stylistic markers."""
    if np.random.random() > prob:
        return text

    import random as rnd

    text = str(text)
    words = text.split()

    # Choose one random augmentation
    aug_type = np.random.choice(['word_swap', 'punctuation_insert', 'vowel_noise', 'synonym_replace'])

    if aug_type == 'word_swap' and len(words) > 3:
        # Random word swaps within 3-word windows at 10% probability
        if np.random.random() < 0.10:
            for i in range(len(words) - 1):
                if np.random.random() < 0.1:
                    window_size = min(3, len(words) - i)
                    if window_size >= 2:
                        idx1 = i
                        idx2 = i + np.random.randint(1, window_size)
                        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    elif aug_type == 'punctuation_insert':
        # Random insertion of punctuation at sentence boundaries at 15% probability
        if np.random.random() < 0.15:
            punct_choices = ['--', ';', ':', '...']
            sentences = re.split(r'([.!?])', text)
            result = []
            for i, sent in enumerate(sentences):
                result.append(sent)
                if sent in ['.', '!', '?'] and i < len(sentences) - 1:
                    if np.random.random() < 0.15:
                        result.append(np.random.choice(punct_choices))
            return ''.join(result)
        return text

    elif aug_type == 'vowel_noise':
        # Character-level vowel noise (replace 5% of vowels with random vowels)
        vowels = 'aeiou'
        text_list = list(text)
        for i in range(len(text_list)):
            if text_list[i].lower() in vowels and np.random.random() < 0.05:
                if text_list[i].isupper():
                    text_list[i] = np.random.choice(list(vowels.upper()))
                else:
                    text_list[i] = np.random.choice(list(vowels))
        return ''.join(text_list)

    elif aug_type == 'synonym_replace':
        # Keep synonym replacement at 5% probability
        # Simple synonym dictionary for common words
        synonym_dict = {
            'very': 'extremely', 'big': 'large', 'small': 'little',
            'happy': 'glad', 'sad': 'unhappy', 'good': 'excellent',
            'bad': 'terrible', 'beautiful': 'lovely', 'ugly': 'hideous',
            'quick': 'fast', 'slow': 'gradual', 'old': 'ancient',
            'new': 'novel', 'strange': 'odd', 'dark': 'gloomy',
            'light': 'bright', 'deep': 'profound', 'cold': 'chilly',
            'hot': 'scorching', 'great': 'magnificent', 'strong': 'powerful',
            'weak': 'feeble', 'rich': 'wealthy', 'poor': 'destitute',
            'nice': 'pleasant', 'awful': 'dreadful', 'sure': 'certain',
            'true': 'genuine', 'false': 'deceitful', 'wise': 'sagacious',
            'foolish': 'absurd', 'kind': 'benevolent', 'cruel': 'ruthless',
            'brave': 'courageous', 'cowardly': 'timid', 'proud': 'haughty',
            'humble': 'modest', 'calm': 'tranquil', 'angry': 'furious',
            'sad': 'melancholy', 'happy': 'joyful', 'weary': 'exhausted',
            'dear': 'beloved', 'dreadful': 'horrible', 'dreary': 'gloomy',
            'strange': 'peculiar', 'fearful': 'terrifying', 'ghastly': 'gruesome',
            'horrible': 'abominable', 'frightful': 'appalling', 'mournful': 'sorrowful',
            'solemn': 'grave', 'trembling': 'shuddering', 'wild': 'frenzied',
            'awful': 'appalling', 'hideous': 'repulsive', 'mysterious': 'enigmatic',
            'shadowy': 'spectral', 'gloomy': 'tenebrous', 'terrible': 'horrendous'
        }
        if np.random.random() < 0.05:
            new_words = []
            for w in words:
                w_lower = w.lower()
                if w_lower in synonym_dict and np.random.random() < 0.3:
                    if w[0].isupper():
                        new_words.append(synonym_dict[w_lower].capitalize())
                    else:
                        new_words.append(synonym_dict[w_lower])
                else:
                    new_words.append(w)
            return ' '.join(new_words)
        return text

    return text


def augment_batch(texts, labels, prob=0.5):
    """Augment a batch of texts."""
    augmented_texts = [augment_text(t, prob) for t in texts]
    return augmented_texts, labels


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, tokenizer, train_texts, train_labels, val_texts, val_labels, test_texts, model_name, lr=2e-5, num_epochs=10, patience=3):
    # Datasets
    class TextDataset(Dataset):
        def __init__(self, texts, labels=None, augment=False):
            self.texts = texts
            self.labels = labels
            self.augment = augment

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            if self.augment and np.random.random() < 0.5:
                text = augment_text(text, prob=0.8)
            encoded = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            item = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = TextDataset(train_texts, train_labels, augment=True)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(test_texts)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Setup optimizer with separate learning rates for different components
    # Classifier and attention layers get higher LR, encoder layers lower
    def is_encoder_param(name):
        return 'distilbert' in name or 'roberta' in name

    def is_attention_param(name):
        return 'han_pooling' in name or 'word_attention' in name or 'sent_attention' in name or 'word_gru' in name or 'sent_gru' in name or 'word_proj' in name or 'sent_fc' in name or 'word_fc' in name

    classifier_params = []
    attention_params = []
    encoder_params = []

    for name, param in model.named_parameters():
        if is_encoder_param(name):
            encoder_params.append(param)
        elif is_attention_param(name):
            attention_params.append(param)
        else:
            classifier_params.append(param)

    optimizer = AdamW([
        {'params': classifier_params, 'lr': 1e-4},
        {'params': attention_params, 'lr': 5e-6},
        {'params': encoder_params, 'lr': 5e-6},
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch/num_epochs)
    scaler_grad = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Freeze encoder initially for progressive unfreezing
    for name, param in model.named_parameters():
        if is_encoder_param(name):
            param.requires_grad = False

    print(f"Training {model_name}...")
    for epoch in range(num_epochs):
        # Progressive unfreezing
        if epoch == 0:
            # First 2 epochs: only train classifier and attention
            for name, param in model.named_parameters():
                if is_encoder_param(name):
                    param.requires_grad = False
                    print(f"  [Epoch 0-1] Freezing encoder layer: {name.split('.')[1] if '.' in name else name}")
                else:
                    param.requires_grad = True
            # Update optimizer param groups for this phase
            optimizer.param_groups[2]['lr'] = 0  # encoder
        elif epoch == 2:
            # Unfreeze top 4 encoder layers
            for name, param in model.named_parameters():
                if is_encoder_param(name):
                    param.requires_grad = True
            # Update LR for newly unfrozen layers
            optimizer.param_groups[2]['lr'] = 5e-5  # encoder layers now trainable
            print(f"  [Epoch 2] Unfreezing ALL encoder layers, LR set to 5e-5")
        elif epoch == 4:
            # Unfreeze more layers - ensure all are trainable
            for name, param in model.named_parameters():
                param.requires_grad = True
            # Update LR for previously unfrozen layers to lower rate
            optimizer.param_groups[2]['lr'] = 5e-6  # lower LR for stability
            print(f"  [Epoch 4] All layers unfrozen, encoder LR set to 5e-6")

        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Apply augmentation to input texts
            # (Note: we can't easily augment tokenized data, so we rely on the dataset to handle it)

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            scaler_grad.scale(loss).backward()
            scaler_grad.step(optimizer)
            scaler_grad.update()
            total_loss += loss.item()
            num_batches += 1
        avg_train_loss = total_loss / num_batches

        # Step scheduler after each epoch
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_num_batches = 0
        all_val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_num_batches += 1
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
        avg_val_loss = val_loss / val_num_batches
        val_probs = np.concatenate(all_val_preds, axis=0)
        val_log_loss_val = log_loss(val_labels, val_probs)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Log Loss: {val_log_loss_val:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  -> New best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best {model_name} with val_loss={best_val_loss:.6f}")

    # Generate predictions
    model.eval()
    all_val_probs = []
    all_test_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu().numpy())

    val_probs = np.concatenate(all_val_probs, axis=0)
    test_probs = np.concatenate(all_test_probs, axis=0)
    return val_probs, test_probs, best_val_loss

# Train DistilBERT
distilbert_val_probs, distilbert_test_probs, distilbert_val_loss = train_model(
    text_encoder, tokenizer_distilbert, train_texts, y_train, val_texts, y_val, test_texts,
    model_name="DistilBERT", lr=2e-5
)

# Train RoBERTa
roberta_val_probs, roberta_test_probs, roberta_val_loss = train_model(
    roberta_encoder, tokenizer_roberta, train_texts, y_train, val_texts, y_val, test_texts,
    model_name="RoBERTa", lr=2e-5
)

# ============================================================
# WEIGHTED ENSEMBLE
# ============================================================
print("\nOptimizing ensemble weights...")
best_weight = 0.1
best_ensemble_val_loss = float("inf")
weights = np.arange(0.1, 1.0, 0.1)
for w in weights:
    ensemble_val_probs = w * distilbert_val_probs + (1 - w) * roberta_val_probs
    ensemble_val_loss = log_loss(y_val, ensemble_val_probs)
    if ensemble_val_loss < best_ensemble_val_loss:
        best_ensemble_val_loss = ensemble_val_loss
        best_weight = w
    print(f"  Weight DistilBERT={w:.1f}, RoBERTa={1-w:.1f} -> Val Log Loss: {ensemble_val_loss:.6f}")

print(f"Best weight: DistilBERT={best_weight:.1f}, RoBERTa={1-best_weight:.1f} with Val Log Loss: {best_ensemble_val_loss:.6f}")

# Final test predictions
test_probs_final = best_weight * distilbert_test_probs + (1 - best_weight) * roberta_test_probs

# Clip and normalize
eps = 1e-15
test_probs_final = np.clip(test_probs_final, eps, 1 - eps)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# Create submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_processed["id"].values,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
submission.to_csv("./submission/submission_a921d24b690c4693b80de5a5fc84936e.csv", index=False)
print(f"Submission saved to ./submission/submission_a921d24b690c4693b80de5a5fc84936e.csv")
print(f"Test predictions shape: {submission.shape}")
print(submission.head())
