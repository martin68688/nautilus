import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import lightgbm as lgb
import joblib
import os
import re
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Map authors to numeric labels
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_label"] = train_df["author"].map(author_mapping)

# Create stratified train/validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author_label"]))
train_texts = train_df.iloc[train_idx]["text"].values
val_texts = train_df.iloc[val_idx]["text"].values
test_texts = test_df["text"].values

train_data = train_df.iloc[train_idx].reset_index(drop=True)
val_data = train_df.iloc[val_idx].reset_index(drop=True)


# =============== STYLOMETRIC FEATURES ===============
def extract_stylometric_features(texts):
    features = []
    for text in texts:
        text_str = str(text) if isinstance(text, str) else ""
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        feat = {}
        feat["num_words"] = len(words)
        feat["num_chars"] = len(text_str)
        feat["num_sentences"] = max(len(sentences), 1)
        feat["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        feat["avg_sentence_length"] = feat["num_words"] / feat["num_sentences"]
        feat["num_unique_words"] = len(set(w.lower() for w in words))
        feat["lexical_diversity"] = feat["num_unique_words"] / max(feat["num_words"], 1)
        feat["num_commas"] = text_str.count(",")
        feat["num_periods"] = text_str.count(".")
        feat["num_exclamations"] = text_str.count("!")
        feat["num_questions"] = text_str.count("?")
        feat["num_semicolons"] = text_str.count(";")
        feat["num_colons"] = text_str.count(":")
        feat["num_quotes"] = (
            text_str.count('"') + text_str.count('"') + text_str.count("'")
        )
        feat["num_dashes"] = text_str.count("-") + text_str.count("—")
        feat["num_parentheses"] = text_str.count("(") + text_str.count(")")
        feat["comma_per_word"] = feat["num_commas"] / max(feat["num_words"], 1)
        feat["exclamation_per_sentence"] = (
            feat["num_exclamations"] / feat["num_sentences"]
        )
        word_lengths = [len(w) for w in words]
        feat["short_words_pct"] = sum(1 for wl in word_lengths if wl <= 3) / max(
            len(word_lengths), 1
        )
        feat["medium_words_pct"] = sum(1 for wl in word_lengths if 4 <= wl <= 7) / max(
            len(word_lengths), 1
        )
        feat["long_words_pct"] = sum(1 for wl in word_lengths if wl >= 8) / max(
            len(word_lengths), 1
        )
        feat["capitalized_words"] = sum(1 for w in words if w and w[0].isupper()) / max(
            len(words), 1
        )
        feat["all_caps_words"] = sum(
            1 for w in words if w.isupper() and len(w) > 1
        ) / max(len(words), 1)
        feat["vowel_ratio"] = sum(1 for c in text_str.lower() if c in "aeiou") / max(
            len(text_str), 1
        )
        feat["consonant_ratio"] = sum(
            1 for c in text_str.lower() if c in "bcdfghjklmnpqrstvwxyz"
        ) / max(len(text_str), 1)
        feat["digit_count"] = sum(1 for c in text_str if c.isdigit())
        common_stopwords = {
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
        }
        feat["stopword_ratio"] = sum(
            1 for w in words if w.lower() in common_stopwords
        ) / max(len(words), 1)
        feat["first_person_singular"] = sum(
            1 for w in words if w.lower() in ["i", "me", "my", "mine", "myself"]
        ) / max(len(words), 1)
        feat["first_person_plural"] = sum(
            1 for w in words if w.lower() in ["we", "us", "our", "ours", "ourselves"]
        ) / max(len(words), 1)
        feat["rare_words_ratio"] = sum(1 for w in words if len(w) > 10) / max(
            len(words), 1
        )

        features.append(feat)
    return pd.DataFrame(features)


train_stylo = extract_stylometric_features(train_texts)
val_stylo = extract_stylometric_features(val_texts)
test_stylo = extract_stylometric_features(test_texts)

# =============== N-GRAM TF-IDF FEATURES ===============
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=3000,
    sublinear_tf=True,
    lowercase=True,
)
train_char_ngrams = char_vectorizer.fit_transform(train_texts)
val_char_ngrams = char_vectorizer.transform(val_texts)
test_char_ngrams = char_vectorizer.transform(test_texts)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=2000,
    sublinear_tf=True,
    stop_words="english",
)
train_word_ngrams = word_vectorizer.fit_transform(train_texts)
val_word_ngrams = word_vectorizer.transform(val_texts)
test_word_ngrams = word_vectorizer.transform(test_texts)


# =============== READABILITY METRICS ===============
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
    return max(count, 1)


def compute_readability(texts):
    features = []
    for text in texts:
        text_str = str(text) if isinstance(text, str) else ""
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        num_syllables = sum(count_syllables(w) for w in words)
        num_chars = sum(len(w) for w in words)
        if num_words > 0 and num_sentences > 0:
            flesch = (
                206.835
                - 1.015 * (num_words / num_sentences)
                - 84.6 * (num_syllables / num_words)
            )
        else:
            flesch = 0
        if num_words > 0 and num_sentences > 0:
            ari = (
                4.71 * (num_chars / num_words)
                + 0.5 * (num_words / num_sentences)
                - 21.43
            )
        else:
            ari = 0
        features.append(
            {
                "flesch_reading_ease": flesch,
                "automated_readability_index": ari,
                "syllables_per_word": num_syllables / max(num_words, 1),
                "chars_per_word": num_chars / max(num_words, 1),
            }
        )
    return pd.DataFrame(features)


train_readability = compute_readability(train_texts)
val_readability = compute_readability(val_texts)
test_readability = compute_readability(test_texts)

# =============== COMBINE ALL FEATURES ===============
stylo_columns = [col for col in train_stylo.columns if col not in ["text"]]
scaler = StandardScaler()
train_stylo_scaled = scaler.fit_transform(train_stylo[stylo_columns])
val_stylo_scaled = scaler.transform(val_stylo[stylo_columns])
test_stylo_scaled = scaler.transform(test_stylo[stylo_columns])

readability_scaler = StandardScaler()
train_readability_scaled = readability_scaler.fit_transform(train_readability)
val_readability_scaled = readability_scaler.transform(val_readability)
test_readability_scaled = readability_scaler.transform(test_readability)

train_features_sparse = hstack(
    [
        train_char_ngrams,
        train_word_ngrams,
        csr_matrix(train_stylo_scaled),
        csr_matrix(train_readability_scaled),
    ]
)
val_features_sparse = hstack(
    [
        val_char_ngrams,
        val_word_ngrams,
        csr_matrix(val_stylo_scaled),
        csr_matrix(val_readability_scaled),
    ]
)
test_features_sparse = hstack(
    [
        test_char_ngrams,
        test_word_ngrams,
        csr_matrix(test_stylo_scaled),
        csr_matrix(test_readability_scaled),
    ]
)

train_labels = train_data["author_label"].values
val_labels = val_data["author_label"].values
test_ids = test_df["id"].values

# Extract dense stylometric features
num_char = 3000
num_word = 2000
num_stylo = 30
num_read = 4
train_dense = np.concatenate(
    [
        train_features_sparse[
            :, num_char + num_word : (num_char + num_word + num_stylo)
        ].toarray(),
        train_features_sparse[:, (num_char + num_word + num_stylo) :].toarray(),
    ],
    axis=1,
)
val_dense = np.concatenate(
    [
        val_features_sparse[
            :, num_char + num_word : (num_char + num_word + num_stylo)
        ].toarray(),
        val_features_sparse[:, (num_char + num_word + num_stylo) :].toarray(),
    ],
    axis=1,
)
test_dense = np.concatenate(
    [
        test_features_sparse[
            :, num_char + num_word : (num_char + num_word + num_stylo)
        ].toarray(),
        test_features_sparse[:, (num_char + num_word + num_stylo) :].toarray(),
    ],
    axis=1,
)


# =============== MODEL DEFINITION ===============
class MultiHeadAttentivePooling(nn.Module):
    def __init__(self, hidden_size, num_heads=4, output_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.attn_weights = nn.Linear(hidden_size, num_heads)
        self.proj = nn.Linear(hidden_size * num_heads, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, T, H], attention_mask: [B, T]
        scores = self.attn_weights(hidden_states)  # [B, T, num_heads]
        # Apply mask (replace padded positions with -inf)
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        scores = scores.masked_fill(mask == 0, -float("inf"))
        attn_weights = torch.softmax(scores, dim=1)  # [B, T, num_heads]
        attn_weights = self.dropout(attn_weights)
        # Weighted sum for each head
        head_outputs = []
        for h in range(self.num_heads):
            head_w = attn_weights[:, :, h:h+1]  # [B, T, 1]
            head_out = (head_w * hidden_states).sum(dim=1)  # [B, H]
            head_outputs.append(head_out)
        concat_output = torch.cat(head_outputs, dim=1)  # [B, H * num_heads]
        pooled = self.proj(concat_output)  # [B, output_dim]
        return pooled


class HybridAuthorshipModel(nn.Module):
    def __init__(self, num_labels=3, num_stylometric_features=150):
        super().__init__()
        config = AutoConfig.from_pretrained(
            "microsoft/deberta-v3-large", output_hidden_states=False
        )
        self.deberta = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large", config=config
        )
        self.hidden_size = config.hidden_size
        # BiLSTM to capture sequential patterns
        self.bilstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        lstm_output_dim = 256 * 2  # bidirectional
        # Multi-head attentive pooling over BiLSTM outputs
        self.attentive_pooling = MultiHeadAttentivePooling(
            hidden_size=lstm_output_dim, num_heads=4, output_dim=256
        )
        self.stylo_proj = nn.Sequential(
            nn.Linear(num_stylometric_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, input_ids, attention_mask, stylometric_features, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        # Run BiLSTM on top of DeBERTa embeddings
        lstm_out, _ = self.bilstm(last_hidden)  # [B, T, 512]
        # Apply multi-head attentive pooling
        pooled = self.attentive_pooling(lstm_out, attention_mask)  # [B, 256]
        stylo_embeds = self.stylo_proj(stylometric_features)  # [B, 256]
        combined = torch.cat([pooled, stylo_embeds], dim=1)  # [B, 512]
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss

    def freeze_embeddings_and_layers(self, num_layers_to_freeze):
        """Freeze embeddings and first num_layers_to_freeze transformer layers."""
        # Freeze embeddings
        for param in self.deberta.embeddings.parameters():
            param.requires_grad = False
        # Freeze specified number of encoder layers
        encoder_layers = self.deberta.encoder.layer
        for i in range(min(num_layers_to_freeze, len(encoder_layers))):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreeze all DeBERTa parameters."""
        for param in self.deberta.parameters():
            param.requires_grad = True

    def apply_progressive_unfreezing(self, epoch):
        """Unfreeze layers progressively based on epoch number."""
        if epoch == 0 or epoch == 1:
            # Epochs 1-2: freeze embeddings and first 8 layers
            self.freeze_embeddings_and_layers(8)
        elif epoch == 2:
            # Unfreeze 4 more layers (now 4 frozen)
            self.freeze_embeddings_and_layers(4)
        elif epoch == 3:
            # Unfreeze another 4 (now 0 frozen, all unfrozen)
            self.freeze_embeddings_and_layers(0)
            self.unfreeze_all_layers()


# =============== TEXT AUGMENTATION ===============
# Simple synonym dictionary for augmentation
SYNONYM_DICT = {
    "terrible": ["awful", "dreadful", "horrible", "frightful"],
    "strange": ["odd", "weird", "peculiar", "bizarre"],
    "dark": ["gloomy", "dim", "shadowy", "somber"],
    "fear": ["dread", "terror", "fright", "horror"],
    "great": ["vast", "immense", "enormous", "massive"],
    "small": ["tiny", "little", "minute", "compact"],
    "said": ["spoke", "uttered", "murmured", "whispered"],
    "went": ["walked", "moved", "proceeded", "traveled"],
    "thought": ["believed", "considered", "imagined", "reflected"],
    "looked": ["gazed", "stared", "glanced", "peered"],
    "felt": ["sensed", "perceived", "experienced", "noticed"],
    "saw": ["observed", "viewed", "noticed", "perceived"],
    "old": ["aged", "ancient", "elderly", "antique"],
    "new": ["fresh", "novel", "recent", "modern"],
    "good": ["fine", "excellent", "superior", "admirable"],
    "bad": ["evil", "wicked", "vile", "corrupt"],
    "beautiful": ["lovely", "splendid", "magnificent", "gorgeous"],
    "sad": ["gloomy", "melancholy", "sorrowful", "mournful"],
    "happy": ["glad", "joyful", "cheerful", "delighted"],
    "big": ["large", "huge", "grand", "substantial"],
    "long": ["extended", "lengthy", "prolonged", "extensive"],
    "short": ["brief", "concise", "limited", "compact"],
    "cold": ["chilly", "frigid", "icy", "freezing"],
    "hot": ["burning", "scorching", "fiery", "blazing"],
    "strong": ["powerful", "robust", "vigorous", "mighty"],
    "weak": ["feeble", "fragile", "frail", "delicate"],
    "quick": ["rapid", "swift", "hasty", "speedy"],
    "slow": ["sluggish", "leisurely", "gradual", "deliberate"],
    "thin": ["slender", "slim", "narrow", "sparse"],
    "thick": ["dense", "heavy", "solid", "stocky"],
    "hard": ["difficult", "tough", "stiff", "rigid"],
    "soft": ["gentle", "mild", "tender", "smooth"],
    "deep": ["profound", "intense", "extreme", "bottomless"],
    "high": ["elevated", "lofty", "towering", "supreme"],
    "low": ["below", "beneath", "inferior", "minor"],
    "true": ["real", "genuine", "authentic", "faithful"],
    "false": ["fake", "deceptive", "fraudulent", "spurious"],
    "open": ["unlocked", "unsealed", "exposed", "yawning"],
    "closed": ["shut", "sealed", "locked", "fastened"],
    "light": ["bright", "luminous", "radiant", "brilliant"],
    "heavy": ["weighty", "massive", "burdensome", "ponderous"],
    "alone": ["solitary", "isolated", "lonely", "unaccompanied"],
    "together": ["united", "joint", "combined", "collective"],
    "above": ["over", "higher", "upward", "aloft"],
    "below": ["under", "lower", "beneath", "underneath"],
    "before": ["prior", "previous", "earlier", "preceding"],
    "after": ["following", "subsequent", "later", "ensuing"],
    "inside": ["within", "interior", "internal", "inner"],
    "outside": ["exterior", "external", "outdoor", "outer"],
    "sudden": ["abrupt", "unexpected", "swift", "instantaneous"],
    "silent": ["quiet", "still", "hushed", "noiseless"],
    "loud": ["noisy", "thunderous", "booming", "resounding"],
    "empty": ["vacant", "void", "hollow", "deserted"],
    "full": ["filled", "crowded", "packed", "stuffed"],
    "bright": ["vivid", "brilliant", "shining", "glowing"],
    "dull": ["dim", "faint", "blurred", "obscure"],
    "sharp": ["keen", "acute", "pointed", "piercing"],
    "smooth": ["even", "level", "polished", "sleek"],
    "rough": ["coarse", "harsh", "rugged", "uneven"],
    "wet": ["damp", "moist", "humid", "soaked"],
    "dry": ["arid", "parched", "dehydrated", "barren"],
    "dangerous": ["perilous", "hazardous", "risky", "treacherous"],
    "safe": ["secure", "protected", "shielded", "guarded"],
    "certain": ["sure", "positive", "confident", "assured"],
    "doubtful": ["uncertain", "skeptical", "questionable", "ambiguous"],
    "rich": ["wealthy", "affluent", "prosperous", "opulent"],
    "poor": ["destitute", "impoverished", "needy", "deprived"],
    "wide": ["broad", "expansive", "spacious", "extensive"],
    "narrow": ["tight", "restricted", "confined", "slender"],
    "clear": ["plain", "obvious", "transparent", "evident"],
    "vague": ["obscure", "unclear", "ambiguous", "nebulous"],
    "bold": ["brave", "courageous", "fearless", "daring"],
    "timid": ["shy", "meek", "cautious", "hesitant"],
    "calm": ["serene", "tranquil", "peaceful", "placid"],
    "fierce": ["savage", "ferocious", "violent", "intense"],
    "gentle": ["tender", "mild", "soothing", "delicate"],
    "rough": ["harsh", "uneven", "coarse", "rugged"],
    "sweet": ["pleasant", "delightful", "agreeable", "charming"],
    "bitter": ["harsh", "sour", "acrid", "pungent"],
    "full": ["complete", "entire", "total", "whole"],
    "empty": ["vacant", "void", "deserted", "barren"],
    "near": ["close", "adjacent", "nearby", "neighboring"],
    "far": ["distant", "remote", "faraway", "secluded"],
    "young": ["youthful", "juvenile", "immature", "adolescent"],
    "old": ["aged", "elderly", "ancient", "senior"],
}


def synonym_replacement(text, prob=0.3):
    """Replace words with synonyms with given probability."""
    words = text.split()
    new_words = words.copy()
    for i, w in enumerate(words):
        w_lower = w.lower()
        # Remove punctuation for matching
        w_clean = w_lower.strip(".,!?;:'\"()-")
        if w_clean in SYNONYM_DICT and np.random.random() < prob:
            synonyms = SYNONYM_DICT[w_clean]
            chosen = np.random.choice(synonyms)
            # Preserve capitalization
            if w[0].isupper():
                chosen = chosen.capitalize()
            new_words[i] = chosen
    return " ".join(new_words)


def random_word_dropout(text, dropout_prob=0.1):
    """Randomly drop words with given probability."""
    words = text.split()
    if len(words) <= 5:  # Keep at least 5 words
        return text
    keep_mask = np.random.random(len(words)) > dropout_prob
    # Ensure at least 5 words remain
    if np.sum(keep_mask) < 5:
        # Force keep first 3 and last 2
        keep_mask[:3] = True
        keep_mask[-2:] = True
    new_words = [w for i, w in enumerate(words) if keep_mask[i]]
    return " ".join(new_words)


def augment_text(text):
    """Apply augmentation to a single text."""
    text = synonym_replacement(text, prob=0.3)
    text = random_word_dropout(text, dropout_prob=0.1)
    return text


# =============== TOKENIZER AND DATASET ===============
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    [str(t) for t in train_texts],
    truncation=True,
    padding=True,
    max_length=384,
    return_tensors="pt",
)
val_encodings = tokenizer(
    [str(t) for t in val_texts],
    truncation=True,
    padding=True,
    max_length=384,
    return_tensors="pt",
)
test_encodings = tokenizer(
    [str(t) for t in test_texts],
    truncation=True,
    padding=True,
    max_length=384,
    return_tensors="pt",
)


class AuthorshipDataset(Dataset):
    def __init__(self, encodings, stylo_features, raw_texts=None, labels=None, augment=False):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.stylo_features = torch.FloatTensor(stylo_features)
        self.raw_texts = raw_texts
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "stylometric_features": self.stylo_features[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]

        # Apply augmentation on-the-fly to raw texts and re-tokenize
        if self.augment and self.raw_texts is not None:
            aug_text = augment_text(str(self.raw_texts[idx]))
            aug_encoding = tokenizer(
                aug_text,
                truncation=True,
                padding="max_length",
                max_length=384,
                return_tensors="pt",
            )
            item["input_ids"] = aug_encoding["input_ids"].squeeze(0)
            item["attention_mask"] = aug_encoding["attention_mask"].squeeze(0)

        return item


# Re-tokenize with augmentation support
train_raw_texts = [str(t) for t in train_texts]

train_dataset = AuthorshipDataset(
    train_encodings, train_dense, raw_texts=train_raw_texts, labels=train_labels, augment=True
)
val_dataset = AuthorshipDataset(val_encodings, val_dense, labels=val_labels)
test_dataset = AuthorshipDataset(test_encodings, test_dense, labels=None)

batch_size = 8
gradient_accumulation_steps = 2  # Simulate effective batch size 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# =============== MODEL SETUP ===============
num_stylometric_features = train_dense.shape[1]
model = HybridAuthorshipModel(
    num_labels=3, num_stylometric_features=num_stylometric_features
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.deberta.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "lr": 1e-5,  # Reduced from 1.5e-5
        "weight_decay": 0.01,
    },
    {
        "params": [
            p
            for n, p in model.deberta.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        "lr": 1e-5,  # Reduced from 1.5e-5
        "weight_decay": 0.0,
    },
    {"params": model.stylo_proj.parameters(), "lr": 5e-4, "weight_decay": 0.01},
    {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 0.01},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, weight_decay=0.01)

num_epochs = 8
total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler = GradScaler()

# =============== TRAINING LOOP ===============
best_val_loss = float("inf")
patience = 3
no_improve_epochs = 0

print("Starting training...")
for epoch in range(num_epochs):
    model.train()

    # Apply progressive unfreezing
    model.apply_progressive_unfreezing(epoch)

    total_train_loss = 0.0
    train_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_feat = batch["stylometric_features"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            _, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylo_feat,
                labels=labels,
            )
            loss = loss / gradient_accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps
        train_batches += 1

    avg_train_loss = total_train_loss / train_batches

    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_feat = batch["stylometric_features"].to(device)
            with autocast():
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stylometric_features=stylo_feat,
                    labels=None,
                )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_preds.append(probs)
            val_true.append(batch["labels"].cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    val_logloss = log_loss(val_true, val_preds_clipped)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_logloss:.6f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        torch.save(model.state_dict(), "./working/best_model.pt")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# =============== LIGHTGBM ENSEMBLE ===============
print("Training LightGBM on sparse features for ensemble...")
np.random.seed(42)
lgb_sample_frac = 0.5
lgb_train_size = int(len(train_labels) * lgb_sample_frac)
lgb_idx = np.random.choice(len(train_labels), lgb_train_size, replace=False)

lgb_train_dense = train_features_sparse[lgb_idx].toarray()
lgb_train_labels = train_labels[lgb_idx]
lgb_val_dense = val_features_sparse.toarray()
lgb_test_dense = test_features_sparse.toarray()

lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)
lgb_model.fit(
    lgb_train_dense,
    lgb_train_labels,
    eval_set=[(lgb_val_dense, val_labels)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
lgb_val_probs = lgb_model.predict_proba(lgb_val_dense)
lgb_test_probs = lgb_model.predict_proba(lgb_test_dense)

# =============== LOAD BEST MODEL AND PREDICT ===============
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

val_preds_deberta = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_feat = batch["stylometric_features"].to(device)
        with autocast():
            logits, _ = model(input_ids, attention_mask, stylo_feat, labels=None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        val_preds_deberta.append(probs)
val_preds_deberta = np.concatenate(val_preds_deberta, axis=0)

test_preds_deberta = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_feat = batch["stylometric_features"].to(device)
        with autocast():
            logits, _ = model(input_ids, attention_mask, stylo_feat, labels=None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        test_preds_deberta.append(probs)
test_preds_deberta = np.concatenate(test_preds_deberta, axis=0)

# =============== OPTIMAL ENSEMBLE WEIGHT ===============
best_weight = 0.5
best_ensemble_loss = float("inf")
for weight in np.linspace(0, 1, 21):
    ensemble_preds = weight * val_preds_deberta + (1 - weight) * lgb_val_probs
    ensemble_preds_clipped = np.clip(ensemble_preds, 1e-15, 1 - 1e-15)
    loss = log_loss(val_labels, ensemble_preds_clipped)
    if loss < best_ensemble_loss:
        best_ensemble_loss = loss
        best_weight = weight

print(f"Best ensemble weight for DeBERTa: {best_weight:.2f}")
print(f"Ensemble val logloss: {best_ensemble_loss:.6f}")

final_val_preds = best_weight * val_preds_deberta + (1 - best_weight) * lgb_val_probs
final_val_preds = np.clip(final_val_preds, 1e-15, 1 - 1e-15)
final_val_score = log_loss(val_labels, final_val_preds)

test_preds_ensemble = (
    best_weight * test_preds_deberta + (1 - best_weight) * lgb_test_probs
)
test_preds_ensemble = np.clip(test_preds_ensemble, 1e-15, 1 - 1e-15)

# =============== CREATE SUBMISSION ===============
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_ensemble[:, 0],
        "HPL": test_preds_ensemble[:, 1],
        "MWS": test_preds_ensemble[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {final_val_score}")