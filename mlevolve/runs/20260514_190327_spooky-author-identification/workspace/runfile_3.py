import os
os.sched_setaffinity(0, {104, 41, 42, 106, 169, 170, 232, 234, 112, 240})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import math
import warnings
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author_encoded"] = le.fit_transform(train_set["author"])
val_set["author_encoded"] = le.transform(val_set["author"])

print(f"Classes: {le.classes_}")
print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
print(f"Test size: {len(test_df)}")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================


def extract_basic_features(texts):
    """Extract basic text features"""
    features = []
    for text in texts:
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s for s in sentences if s.strip()]

        n_chars = len(text_str)
        n_words = len(words)
        n_sentences = max(len(sentences), 1)
        n_unique_words = len(set(w.lower() for w in words))

        avg_word_len = n_chars / max(n_words, 1)
        avg_sentence_len = n_words / n_sentences

        n_exclamation = text_str.count("!")
        n_question = text_str.count("?")
        n_period = text_str.count(".")
        n_comma = text_str.count(",")
        n_semicolon = text_str.count(";")
        n_colon = text_str.count(":")
        n_quote = text_str.count('"') + text_str.count("'")
        n_dash = text_str.count("-") + text_str.count("—")
        n_punct = (
            n_exclamation + n_question + n_period + n_comma + n_semicolon + n_colon
        )

        n_capitalized = sum(1 for w in words if w and w[0].isupper())
        pct_capitalized = n_capitalized / max(n_words, 1)

        short_words = sum(1 for w in words if len(w) <= 3)
        medium_words = sum(1 for w in words if 4 <= len(w) <= 7)
        long_words = sum(1 for w in words if len(w) >= 8)

        lexical_diversity = n_unique_words / max(n_words, 1)

        stop_words = {
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
            "is",
            "was",
            "were",
            "be",
            "been",
            "has",
            "have",
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
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "not",
            "no",
            "nor",
            "so",
        }
        n_stop_words = sum(1 for w in words if w.lower() in stop_words)
        stop_word_ratio = n_stop_words / max(n_words, 1)

        feats = [
            n_chars,
            n_words,
            n_sentences,
            n_unique_words,
            avg_word_len,
            avg_sentence_len,
            n_exclamation,
            n_question,
            n_period,
            n_comma,
            n_semicolon,
            n_colon,
            n_quote,
            n_dash,
            n_punct,
            n_capitalized,
            pct_capitalized,
            short_words,
            medium_words,
            long_words,
            short_words / max(n_words, 1),
            medium_words / max(n_words, 1),
            long_words / max(n_words, 1),
            lexical_diversity,
            stop_word_ratio,
        ]
        features.append(feats)

    return np.array(features)


def extract_stylistic_features(texts):
    """Extract stylistic writing features"""
    features = []
    archaic_words = {
        "thou",
        "thee",
        "thy",
        "thine",
        "doth",
        "hath",
        "art",
        "wilt",
        "shalt",
        "canst",
        "didst",
        "hast",
        "whence",
        "thence",
        "hither",
        "whither",
        "ere",
        "o'er",
        "ne'er",
        "oft",
        "perchance",
        "forsooth",
        "alas",
        "aye",
        "nay",
        "yea",
        "methinks",
        "prithee",
        "anon",
    }
    horror_words = {
        "dark",
        "shadow",
        "fear",
        "dread",
        "terror",
        "horror",
        "ghost",
        "phantom",
        "spectre",
        "demon",
        "devil",
        "hell",
        "death",
        "dead",
        "corpse",
        "grave",
        "tomb",
        "scream",
        "shriek",
        "howl",
        "gloom",
        "mist",
        "fog",
        "night",
        "moon",
        "ancient",
        "eldritch",
        "cosmic",
        "abyss",
        "void",
        "chaos",
        "monster",
        "creature",
        "beast",
        "evil",
    }

    for text in texts:
        text_str = str(text).lower()
        words = text_str.split()

        n_archaic = sum(1 for w in words if w in archaic_words)
        n_horror = sum(1 for w in words if w in horror_words)

        first_person = sum(
            1
            for w in words
            if w
            in {
                "i",
                "me",
                "my",
                "mine",
                "myself",
                "we",
                "us",
                "our",
                "ours",
                "ourselves",
            }
        )
        third_person = sum(
            1
            for w in words
            if w
            in {
                "he",
                "him",
                "his",
                "himself",
                "she",
                "her",
                "hers",
                "herself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
            }
        )

        conjunctions = sum(
            1
            for w in words
            if w
            in {
                "and",
                "but",
                "or",
                "nor",
                "for",
                "yet",
                "so",
                "although",
                "because",
                "since",
                "while",
                "though",
                "unless",
            }
        )
        dialogue_markers = sum(
            1
            for w in words
            if w
            in {
                "said",
                "cried",
                "exclaimed",
                "whispered",
                "shouted",
                "asked",
                "replied",
                "answered",
                "murmured",
            }
        )
        emphatic_words = sum(
            1
            for w in words
            if w
            in {
                "very",
                "quite",
                "extremely",
                "absolutely",
                "utterly",
                "totally",
                "completely",
                "entirely",
                "highly",
            }
        )
        negation = sum(
            1
            for w in words
            if w
            in {
                "not",
                "no",
                "never",
                "nothing",
                "none",
                "neither",
                "nor",
                "nowhere",
                "nobody",
                "cannot",
                "can't",
                "don't",
                "doesn't",
            }
        )
        adjective_endings = sum(
            1
            for w in words
            if len(w) > 3
            and (
                w.endswith("ful")
                or w.endswith("ous")
                or w.endswith("ive")
                or w.endswith("able")
                or w.endswith("ible")
                or w.endswith("al")
            )
        )
        adverb_ly = sum(1 for w in words if len(w) > 4 and w.endswith("ly"))

        total_trigrams = 0
        unique_trigrams = set()
        for word in words:
            for i in range(len(word) - 2):
                trigram = word[i : i + 3]
                unique_trigrams.add(trigram)
                total_trigrams += 1
        trigram_diversity = len(unique_trigrams) / max(total_trigrams, 1)

        features.append(
            [
                n_archaic,
                n_horror,
                first_person,
                third_person,
                conjunctions,
                dialogue_markers,
                emphatic_words,
                negation,
                adjective_endings,
                adverb_ly,
                trigram_diversity,
            ]
        )

    return np.array(features)


def extract_sentiment_features(texts):
    """Extract simple positive/negative sentiment indicators based on word lists"""
    positive_words = {
        "beautiful",
        "wonderful",
        "happy",
        "joy",
        "love",
        "hope",
        "bright",
        "light",
        "peace",
        "pleasure",
        "delight",
        "glad",
        "cheerful",
        "bliss",
        "splendid",
        "magnificent",
        "excellent",
        "pleasant",
        "sweet",
        "calm",
        "gentle",
        "tender",
        "kind",
        "generous",
        "noble",
        "brave",
        "glorious",
    }
    negative_words = {
        "terrible",
        "awful",
        "horrible",
        "dreadful",
        "painful",
        "sad",
        "grief",
        "sorrow",
        "misery",
        "suffering",
        "agony",
        "despair",
        "anger",
        "rage",
        "fury",
        "hate",
        "cruel",
        "wicked",
        "vile",
        "fearful",
        "frightful",
        "ghastly",
        "hideous",
        "loathsome",
        "disgusting",
        "revolting",
        "gloomy",
        "dreary",
        "bleak",
        "somber",
    }

    features = []
    for text in texts:
        text_str = str(text).lower()
        words = set(text_str.split())
        n_pos = sum(1 for w in words if w in positive_words)
        n_neg = sum(1 for w in words if w in negative_words)
        total = n_pos + n_neg
        sentiment_ratio = (n_pos - n_neg) / total if total > 0 else 0.0
        features.append([n_pos, n_neg, sentiment_ratio])

    return np.array(features)


def extract_part_of_speech_patterns(texts):
    """Extract POS-like patterns using word endings as proxy"""
    features = []
    for text in texts:
        text_str = str(text)
        words = text_str.split()
        ing_words = sum(1 for w in words if len(w) > 4 and w.lower().endswith("ing"))
        ed_words = sum(1 for w in words if len(w) > 3 and w.lower().endswith("ed"))
        tion_words = sum(1 for w in words if len(w) > 5 and w.lower().endswith("tion"))
        ness_words = sum(1 for w in words if len(w) > 4 and w.lower().endswith("ness"))
        ment_words = sum(1 for w in words if len(w) > 4 and w.lower().endswith("ment"))
        determiners = sum(
            1
            for w in words
            if w.lower() in {"the", "a", "an", "this", "that", "these", "those"}
        )
        prepositions = sum(
            1
            for w in words
            if w.lower()
            in {
                "in",
                "on",
                "at",
                "to",
                "from",
                "by",
                "with",
                "about",
                "against",
                "between",
                "under",
                "over",
                "above",
                "below",
                "through",
                "during",
                "before",
                "after",
                "of",
                "for",
            }
        )
        articles = sum(1 for w in words if w.lower() in {"the", "a", "an"})
        features.append(
            [
                ing_words,
                ed_words,
                tion_words,
                ness_words,
                ment_words,
                determiners,
                prepositions,
                articles,
            ]
        )

    return np.array(features)


# ============================================================
# APPLY FEATURE ENGINEERING
# ============================================================

train_texts_feat = train_set["text"].values
val_texts_feat = val_set["text"].values
test_texts_feat = test_df["text"].values
print("Extracting basic features on train...")
basic_features_train = extract_basic_features(train_texts_feat)

print("Extracting stylistic features on train...")
stylistic_features_train = extract_stylistic_features(train_texts_feat)

print("Extracting sentiment features on train...")
sentiment_features_train = extract_sentiment_features(train_texts_feat)

print("Extracting POS pattern features on train...")
pos_features_train = extract_part_of_speech_patterns(train_texts_feat)

print("Extracting basic features on val...")
basic_features_val = extract_basic_features(val_texts_feat)

print("Extracting stylistic features on val...")
stylistic_features_val = extract_stylistic_features(val_texts_feat)

print("Extracting sentiment features on val...")
sentiment_features_val = extract_sentiment_features(val_texts_feat)

print("Extracting POS pattern features on val...")
pos_features_val = extract_part_of_speech_patterns(val_texts_feat)

print("Extracting basic features on test...")
basic_features_test = extract_basic_features(test_texts_feat)

print("Extracting stylistic features on test...")
stylistic_features_test = extract_stylistic_features(test_texts_feat)

print("Extracting sentiment features on test...")
sentiment_features_test = extract_sentiment_features(test_texts_feat)

print("Extracting POS pattern features on test...")
pos_features_test = extract_part_of_speech_patterns(test_texts_feat)

print("Extracting character n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
)
char_tfidf = char_vectorizer.fit_transform(train_texts_feat)
val_char_tfidf = char_vectorizer.transform(val_texts_feat)
test_char_tfidf = char_vectorizer.transform(test_texts_feat)

print("Extracting word n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=500,
    sublinear_tf=True,
    stop_words="english",
    strip_accents="unicode",
)
word_tfidf = word_vectorizer.fit_transform(train_texts_feat)
val_word_tfidf = word_vectorizer.transform(val_texts_feat)
test_word_tfidf = word_vectorizer.transform(test_texts_feat)

# Stack numerical features
num_feature_names = [
    "n_chars",
    "n_words",
    "n_sentences",
    "n_unique_words",
    "avg_word_len",
    "avg_sentence_len",
    "n_exclamation",
    "n_question",
    "n_period",
    "n_comma",
    "n_semicolon",
    "n_colon",
    "n_quote",
    "n_dash",
    "n_punct",
    "n_capitalized",
    "pct_capitalized",
    "short_words",
    "medium_words",
    "long_words",
    "pct_short_words",
    "pct_medium_words",
    "pct_long_words",
    "lexical_diversity",
    "stop_word_ratio",
    "n_archaic",
    "n_horror",
    "first_person",
    "third_person",
    "conjunctions",
    "dialogue_markers",
    "emphatic_words",
    "negation",
    "adjective_endings",
    "adverb_ly",
    "trigram_diversity",
    "n_pos",
    "n_neg",
    "sentiment_ratio",
    "ing_words",
    "ed_words",
    "tion_words",
    "ness_words",
    "ment_words",
    "determiners",
    "prepositions",
    "articles",
]

train_num = np.hstack(
    [basic_features_train, stylistic_features_train, sentiment_features_train, pos_features_train]
)
val_num = np.hstack(
    [basic_features_val, stylistic_features_val, sentiment_features_val, pos_features_val]
)
test_num = np.hstack(
    [basic_features_test, stylistic_features_test, sentiment_features_test, pos_features_test]
)
print(f"Train numerical features shape: {train_num.shape}")
print(f"Val numerical features shape: {val_num.shape}")
print(f"Test numerical features shape: {test_num.shape}")

scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num)
val_num_scaled = scaler.transform(val_num)
test_num_scaled = scaler.transform(test_num)

train_num_sparse = csr_matrix(train_num_scaled)
val_num_sparse = csr_matrix(val_num_scaled)
test_num_sparse = csr_matrix(test_num_scaled)

train_features_all = hstack([train_num_sparse, char_tfidf, word_tfidf])
val_features_all = hstack([val_num_sparse, val_char_tfidf, val_word_tfidf])
test_features_all = hstack([test_num_sparse, test_char_tfidf, test_word_tfidf])

print(f"Final train feature shape: {train_features_all.shape}")
print(f"Final val feature shape: {val_features_all.shape}")
print(f"Final test feature shape: {test_features_all.shape}")

train_targets = train_set["author_encoded"].values
val_targets = val_set["author_encoded"].values

# Save processed features
save_npz("./working/train_features.npz", train_features_all)
save_npz("./working/val_features.npz", val_features_all)
save_npz("./working/test_features.npz", test_features_all)
np.save("./working/train_targets.npy", train_targets)
np.save("./working/val_targets.npy", val_targets)
joblib.dump(scaler, "./working/scaler.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
joblib.dump(le, "./working/label_encoder.pkl")

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=3,
    output_hidden_states=False,
    output_attentions=False,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)
# Freeze first 2/3 of the encoder layers
num_layers = len(model.deberta.encoder.layer)
freeze_up_to = int(num_layers * 0.333)
for i, layer in enumerate(model.deberta.encoder.layer):
    if i < freeze_up_to:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True
# Also unfreeze the classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Separate parameters for differential learning rates
unfrozen_layers_params = []
for name, param in model.named_parameters():
    if param.requires_grad and "classifier" not in name:
        unfrozen_layers_params.append(param)
classifier_params = list(model.classifier.parameters())

optimizer = AdamW(
    [
        {"params": unfrozen_layers_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": classifier_params, "lr": 5e-5, "weight_decay": 0.01},
    ],
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

print(f"Unfrozen backbone params: {sum(p.numel() for p in unfrozen_layers_params):,}")
print(f"Classifier params: {sum(p.numel() for p in classifier_params):,}")
print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length
)
val_dataset = SpookyDataset(val_texts_final, val_labels_final, tokenizer, max_length)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
num_epochs = 25
patience = 8
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

# Proper warmup + cosine annealing scheduler (no restarts)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

            total_val_loss += loss.item()
            num_val_batches += 1
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_score = log_loss(val_true, val_probs_clipped)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model_a2e6264845a14589840ee9a5b4b0bbc8.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE - DeBERTa
# ============================================================
model.load_state_dict(torch.load("./working/best_model_a2e6264845a14589840ee9a5b4b0bbc8.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

deberta_val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)

# ============================================================
# TEST INFERENCE - DeBERTa
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

deberta_test_probs = np.concatenate(all_test_probs, axis=0)

print(f"DeBERTa val probs shape: {deberta_val_probs.shape}")
print(f"DeBERTa test probs shape: {deberta_test_probs.shape}")

# ============================================================
# LIGHTGBM MODEL ON HANDCRAFTED FEATURES
# ============================================================
print("="*60)
print("Training LightGBM on handcrafted features...")
print("="*60)

# Prepare LightGBM datasets
lgb_train_dataset = lgb.Dataset(
    train_num_scaled,
    label=train_targets,
    feature_name=[f"feat_{i}" for i in range(train_num_scaled.shape[1])]
)
lgb_val_dataset = lgb.Dataset(
    val_num_scaled,
    label=val_targets,
    feature_name=[f"feat_{i}" for i in range(val_num_scaled.shape[1])],
    reference=lgb_train_dataset
)

# LightGBM parameters
lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_child_samples': 20,
    'verbosity': -1,
    'seed': 42,
    'n_jobs': -1,
    'num_boost_round': 500,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train_dataset,
    valid_sets=[lgb_train_dataset, lgb_val_dataset],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# LightGBM predictions
lgb_val_probs = lgb_model.predict(val_num_scaled)
lgb_test_probs = lgb_model.predict(test_num_scaled)

print(f"LightGBM val probs shape: {lgb_val_probs.shape}")
print(f"LightGBM test probs shape: {lgb_test_probs.shape}")

# Evaluate LightGBM
lgb_val_probs_clipped = np.clip(lgb_val_probs, 1e-15, 1 - 1e-15)
lgb_val_probs_clipped = lgb_val_probs_clipped / lgb_val_probs_clipped.sum(axis=1, keepdims=True)
lgb_val_score = log_loss(val_targets, lgb_val_probs_clipped)
print(f"LightGBM Validation LogLoss: {lgb_val_score:.6f}")

# ============================================================
# META-CLASSIFIER
# ============================================================
print("="*60)
print("Training meta-classifier...")
print("="*60)

# Meta features: concatenated probabilities from DeBERTa and LightGBM (6 dims)
# + handcrafted dense features (47 dims) = 53 total meta features
meta_train_features = np.hstack([
    deberta_val_probs,  # 3 features
    lgb_val_probs,      # 3 features
    val_num_scaled      # 47 features
])

meta_test_features = np.hstack([
    deberta_test_probs,  # 3 features
    lgb_test_probs,      # 3 features
    test_num_scaled      # 47 features
])

print(f"Meta train features shape: {meta_train_features.shape}")
print(f"Meta test features shape: {meta_test_features.shape}")

# Train meta-classifier
meta_model = CalibratedClassifierCV(
    LogisticRegression(
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=1000,
        random_state=42
    ),
    method='sigmoid',
    cv=5
)
meta_model.fit(meta_train_features, val_targets)

# Meta predictions
meta_val_probs = meta_model.predict_proba(meta_train_features)
meta_test_probs = meta_model.predict_proba(meta_test_features)

meta_val_probs_clipped = np.clip(meta_val_probs, 1e-15, 1 - 1e-15)
meta_val_probs_clipped = meta_val_probs_clipped / meta_val_probs_clipped.sum(axis=1, keepdims=True)
meta_val_score = log_loss(val_targets, meta_val_probs_clipped)
print(f"Meta-classifier Validation LogLoss: {meta_val_score:.6f}")

# ============================================================
# FINAL EVALUATION
# ============================================================
print("="*60)
print("Final Evaluation")
print("="*60)

# DeBERTa only score
deberta_probs_clipped = np.clip(deberta_val_probs, 1e-15, 1 - 1e-15)
deberta_probs_clipped = deberta_probs_clipped / deberta_probs_clipped.sum(axis=1, keepdims=True)
deberta_score = log_loss(val_true, deberta_probs_clipped)
print(f"DeBERTa only Validation LogLoss: {deberta_score:.6f}")

print(f"LightGBM only Validation LogLoss: {lgb_val_score:.6f}")
print(f"Meta Ensemble Validation LogLoss: {meta_val_score:.6f}")

# ============================================================
# GENERATE SUBMISSION (using meta-classifier predictions)
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": meta_test_probs[:, 0],
        "HPL": meta_test_probs[:, 1],
        "MWS": meta_test_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission_a2e6264845a14589840ee9a5b4b0bbc8.csv", index=False)
print(f"Submission saved: {submission.shape}")

final_val_score = meta_val_score
print(f"Final Validation Score (Meta Ensemble): {final_val_score}")