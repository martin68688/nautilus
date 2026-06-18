import os
os.sched_setaffinity(0, {19, 58, 59, 60, 61})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import os
import joblib

# ============================================
# LOAD DATA
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
submission_df = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Create author mapping
author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
train_df["author_encoded"] = train_df["author"].map(author_mapping)

# ============================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================


def extract_basic_features(text_series, name=""):
    """Extract basic text statistics"""
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)
    features[f"{name}_char_count"] = texts.str.len()
    features[f"{name}_word_count"] = texts.str.split().str.len()
    features[f"{name}_avg_word_len"] = features[f"{name}_char_count"] / (
        features[f"{name}_word_count"] + 1
    )
    features[f"{name}_sentence_count"] = texts.str.split("[.!?]+").str.len()
    features[f"{name}_exclamation_count"] = texts.str.count("!")
    features[f"{name}_question_count"] = texts.str.count(r"\?")
    features[f"{name}_comma_count"] = texts.str.count(",")
    features[f"{name}_semicolon_count"] = texts.str.count(";")
    features[f"{name}_colon_count"] = texts.str.count(":")
    features[f"{name}_dash_count"] = texts.str.count("-")
    features[f"{name}_quote_count"] = texts.str.count('"')
    features[f"{name}_apostrophe_count"] = texts.str.count("'")
    features[f"{name}_ellipsis_count"] = texts.str.count(r"\.\.\.")
    features[f"{name}_capital_words"] = texts.str.findall(r"\b[A-Z][a-z]*\b").str.len()
    features[f"{name}_all_caps_words"] = texts.str.findall(r"\b[A-Z]{2,}\b").str.len()
    features[f"{name}_pct_capital"] = features[f"{name}_capital_words"] / (
        features[f"{name}_word_count"] + 1
    )
    features[f"{name}_unique_words"] = texts.apply(
        lambda x: len(set(x.lower().split()))
    )
    features[f"{name}_lexical_diversity"] = features[f"{name}_unique_words"] / (
        features[f"{name}_word_count"] + 1
    )
    features[f"{name}_quote_density"] = features[f"{name}_quote_count"] / (
        features[f"{name}_word_count"] + 1
    )
    return features


def extract_stylometric_features(text_series, name=""):
    """Extract author-specific stylistic features"""
    features = pd.DataFrame(index=text_series.index)
    texts = text_series.fillna("").astype(str)
    words_series = texts.str.lower().str.findall(r"\b[a-z]+\b")

    function_words = {
        "the": "the",
        "and": "and",
        "of": "of",
        "to": "to",
        "in": "in",
        "was": "was",
        "that": "that",
        "had": "had",
        "with": "with",
        "for": "for",
        "not": "not",
        "but": "but",
        "my": "my",
        "his": "his",
        "her": "her",
        "from": "from",
        "which": "which",
        "been": "been",
        "were": "were",
        "all": "all",
        "this": "this",
        "have": "have",
        "upon": "upon",
        "could": "could",
        "would": "would",
        "should": "should",
        "might": "might",
        "must": "must",
        "shall": "shall",
        "may": "may",
    }
    for word, col_name in function_words.items():
        features[f"{name}_fw_{col_name}"] = words_series.apply(
            lambda x, w=word: sum(1 for wd in x if wd == w)
        )

    lovecraft_terms = [
        "cyclopean",
        "eldritch",
        "antediluvian",
        "squamous",
        "rugose",
        "fungous",
        "ichor",
        "gibbous",
        "cryptic",
        "arcane",
        "byzantine",
        "cacophony",
        "catacomb",
        "chthonic",
        "cosmic",
        "daemonic",
        "effulgent",
        "gambrel",
        "hybrid",
        "immemorial",
        "indescribable",
        "lurker",
        "madness",
        "nameless",
        "nightmare",
        "nonhuman",
        "odd",
        "outsider",
        "phantasmal",
        "primordial",
        "prodigious",
        "prowling",
        "psychic",
        "putrid",
        "repulsive",
        "shapeless",
        "strange",
        "swarthy",
        "uncanny",
        "unmentionable",
        "vaporous",
        "vile",
        "whispered",
        "witchcraft",
        "worm",
        "accursed",
        "ancient",
        "blasphemous",
        "curse",
        "devil",
        "dread",
        "fearful",
        "ghastly",
        "grotesque",
        "hideous",
        "horrible",
        "lurking",
        "monstrous",
        "mysterious",
        "occult",
        "peculiar",
        "queer",
        "shocking",
        "supernatural",
        "terrible",
        "terrifying",
        "unearthly",
        "weird",
        "abhorrent",
        "abysmal",
        "baleful",
        "cadaverous",
        "cavern",
        "charnel",
        "corpse",
        "creeping",
        "damnation",
        "decay",
        "dire",
        "dismal",
        "doom",
        "enigmatic",
        "ethereal",
        "evil",
        "feverish",
        "forbidding",
        "forgotten",
        "foul",
        "gloomy",
        "haunted",
        "hellish",
        "howling",
        "infernal",
        "inhuman",
        "labyrinth",
        "lament",
        "macabre",
        "malevolent",
        "malignant",
        "mausoleum",
        "morbid",
        "mortal",
        "nocturnal",
        "ominous",
        "otherworldly",
        "phantom",
        "portent",
        "profane",
        "spectral",
        "stench",
        "stygian",
        "tentacle",
        "tomb",
        "torment",
        "twilight",
        "unholy",
        "vast",
        "void",
        "whisper",
        "writhing",
    ]
    poe_terms = [
        "nevermore",
        "chamber",
        "raven",
        "pallid",
        "ghastly",
        "grim",
        "dreary",
        "weary",
        "bleak",
        "dying",
        "ember",
        "sorrow",
        "terror",
        "dream",
        "nightly",
        "shadow",
        "fantasy",
        "illusion",
        "despair",
        "corpse",
        "sepulchre",
        "tomb",
        "ghoul",
        "spirit",
        "soul",
        "demon",
        "angel",
        "heaven",
        "hell",
        "fate",
        "destiny",
        "melancholy",
        "gloom",
        "horror",
        "fancy",
        "imagination",
        "reverie",
        "trance",
        "apparition",
        "spectre",
        "ominous",
        "portentous",
        "mystic",
        "magic",
        "enchant",
        "wizard",
        "witch",
        "weird",
        "doubt",
        "mystery",
        "secret",
        "strange",
        "singular",
        "unusual",
        "terror",
        "fright",
        "panic",
        "dread",
        "awful",
        "awe",
        "wonder",
        "behold",
        "gaze",
        "glance",
        "stare",
        "peer",
        "agony",
        "anguish",
        "suffering",
        "torture",
        "torment",
        "magnificent",
        "grandeur",
    ]
    shelley_terms = [
        "love",
        "heart",
        "friend",
        "spirit",
        "human",
        "nature",
        "mountain",
        "forest",
        "river",
        "sky",
        "earth",
        "sun",
        "moon",
        "star",
        "light",
        "dark",
        "night",
        "day",
        "life",
        "death",
        "soul",
        "mind",
        "thought",
        "feeling",
        "passion",
        "gentle",
        "soft",
        "tender",
        "sweet",
        "bitter",
        "joy",
        "grief",
        "hope",
        "fear",
        "weep",
        "smile",
        "kiss",
        "tear",
        "pity",
        "sympathy",
        "compassion",
        "kindness",
        "mercy",
        "virtue",
        "truth",
        "beauty",
        "sublime",
        "divine",
        "eternal",
        "infinite",
        "mortal",
        "immortal",
        "heavenly",
        "celestial",
        "wander",
        "journey",
        "travel",
        "voyage",
        "solitude",
        "desert",
        "wilderness",
        "ocean",
        "sea",
        "wave",
        "tempest",
        "storm",
        "wind",
        "breeze",
        "thunder",
        "lightning",
        "rain",
        "snow",
        "blossom",
        "flower",
        "garden",
        "meadow",
        "hill",
        "rock",
        "cave",
        "stream",
        "fountain",
        "emotion",
        "sensation",
        "perception",
        "consciousness",
        "existence",
        "beautiful",
        "romantic",
        "poetic",
        "home",
        "domestic",
        "father",
        "mother",
        "brother",
        "sister",
        "child",
        "infant",
        "knowledge",
        "science",
        "philosophy",
        "reason",
        "virtue",
        "vice",
        "good",
        "evil",
        "moral",
        "ethical",
    ]

    for word_list, author_name in [
        (lovecraft_terms, "hpl"),
        (poe_terms, "eap"),
        (shelley_terms, "mws"),
    ]:
        features[f"{name}_vocab_{author_name}"] = texts.apply(
            lambda x, wl=word_list: sum(1 for w in wl if w in x.lower())
        )

    total_vocab = features[
        [f"{name}_vocab_hpl", f"{name}_vocab_eap", f"{name}_vocab_mws"]
    ].sum(axis=1)
    for author_name in ["hpl", "eap", "mws"]:
        features[f"{name}_vocab_{author_name}_ratio"] = features[
            f"{name}_vocab_{author_name}"
        ] / (total_vocab + 1)

    transition_beginnings = [
        "and",
        "but",
        "or",
        "nor",
        "yet",
        "so",
        "for",
        "although",
        "though",
        "while",
        "whereas",
        "however",
        "moreover",
        "furthermore",
        "nevertheless",
        "nonetheless",
        "therefore",
        "thus",
        "consequently",
        "accordingly",
        "besides",
        "indeed",
        "instead",
        "meanwhile",
        "then",
    ]
    features[f"{name}_transition_starts"] = (
        texts.str.lower()
        .str.extract(r"^(" + "|".join(transition_beginnings) + r")\b", expand=False)
        .notna()
        .astype(int)
    )
    return features


def extract_ngram_features(train_texts, test_texts, name=""):
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=500,
        sublinear_tf=True,
        min_df=5,
        max_df=0.95,
    )
    char_train = char_vectorizer.fit_transform(train_texts.fillna("").astype(str))
    char_test = char_vectorizer.transform(test_texts.fillna("").astype(str))
    char_train_df = pd.DataFrame(
        char_train.toarray(),
        columns=[f"{name}_char_{i}" for i in range(char_train.shape[1])],
        index=train_texts.index,
    )
    char_test_df = pd.DataFrame(
        char_test.toarray(),
        columns=[f"{name}_char_{i}" for i in range(char_test.shape[1])],
        index=test_texts.index,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=1000,
        sublinear_tf=True,
        min_df=5,
        max_df=0.90,
        stop_words="english",
    )
    word_train = word_vectorizer.fit_transform(train_texts.fillna("").astype(str))
    word_test = word_vectorizer.transform(test_texts.fillna("").astype(str))
    word_train_df = pd.DataFrame(
        word_train.toarray(),
        columns=[f"{name}_word_{i}" for i in range(word_train.shape[1])],
        index=train_texts.index,
    )
    word_test_df = pd.DataFrame(
        word_test.toarray(),
        columns=[f"{name}_word_{i}" for i in range(word_test.shape[1])],
        index=test_texts.index,
    )
    return char_train_df, char_test_df, word_train_df, word_test_df


# ============================================
# APPLY FEATURE ENGINEERING
# ============================================
train_basic = extract_basic_features(train_df["text"], name="basic")
test_basic = extract_basic_features(test_df["text"], name="basic")
train_style = extract_stylometric_features(train_df["text"], name="style")
test_style = extract_stylometric_features(test_df["text"], name="style")

print("Extracting n-gram features...")
char_train_df, char_test_df, word_train_df, word_test_df = extract_ngram_features(
    train_df["text"], test_df["text"], name="ngram"
)

train_features = pd.concat(
    [train_basic, train_style, char_train_df, word_train_df], axis=1
)
test_features = pd.concat([test_basic, test_style, char_test_df, word_test_df], axis=1)

train_features = train_features.replace([np.inf, -np.inf], 0).fillna(0)
test_features = test_features.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Feature matrix shape: {train_features.shape}")

# ============================================
# CREATE STRATIFIED SPLITS
# ============================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_features, train_df["author_encoded"])
):
    if fold == 0:
        X_train = train_features.iloc[train_idx].copy()
        X_val = train_features.iloc[val_idx].copy()
        y_train = train_df["author_encoded"].iloc[train_idx].copy()
        y_val = train_df["author_encoded"].iloc[val_idx].copy()
        break

print(
    f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(test_features)}"
)

scaler = StandardScaler()
numerical_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
test_scaled = test_features.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
test_scaled[numerical_cols] = scaler.transform(test_features[numerical_cols])

np.save("./working/X_train.npy", X_train_scaled.values)
np.save("./working/X_val.npy", X_val_scaled.values)
np.save("./working/X_test.npy", test_scaled.values)
np.save("./working/y_train.npy", y_train.values)
np.save("./working/y_val.npy", y_val.values)
feature_names = train_features.columns.tolist()
with open("./working/feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")
np.save("./working/train_indices.npy", X_train.index.values)
np.save("./working/val_indices.npy", X_val.index.values)
np.save("./working/test_indices.npy", test_features.index.values)
np.save("./working/test_ids.npy", test_df["id"].values)
joblib.dump(scaler, "./working/feature_scaler.pkl")
print(f"Feature engineering complete. Saved {len(feature_names)} features.")

# ============================================
# MODEL SETUP - DeBERTa-v3-large
# ============================================
NUM_AUTHORS = 3
MAX_SEQ_LENGTH = 512
MODEL_NAME = "microsoft/deberta-v3-large"
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_AUTHORS,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)
model.to(device)

# Get texts for splits
train_texts = train_df["text"].iloc[X_train.index].values
val_texts = train_df["text"].iloc[X_val.index].values
test_texts = test_df["text"].values


def tokenize_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings


print("Tokenizing texts...")
train_encodings = tokenize_texts(train_texts, tokenizer, MAX_SEQ_LENGTH)
val_encodings = tokenize_texts(val_texts, tokenizer, MAX_SEQ_LENGTH)
test_encodings = tokenize_texts(test_texts, tokenizer, MAX_SEQ_LENGTH)

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(y_train.values, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(y_val.values, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
)
total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler_grad = GradScaler()

# ============================================
# TRAINING LOOP
# ============================================
best_val_score = float("inf")
epochs_no_improve = 0
os.makedirs("./working", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        scaler_grad.scale(loss).backward()
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()
            scheduler.step()
            optimizer.zero_grad()
        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_val_preds.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    eps = 1e-15
    all_val_preds = np.clip(all_val_preds, eps, 1 - eps)
    val_score = log_loss(all_val_labels, all_val_preds)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Log Loss: {val_score:.4f}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model_588f35d9aac14bcd9a7599d336bb891b.pth")
        print(f"  --> Saved best model (log loss: {val_score:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================
# LOAD BEST MODEL AND EVALUATE
# ============================================
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load("./working/best_model_588f35d9aac14bcd9a7599d336bb891b.pth"))
model.eval()

all_val_preds = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_val_preds.append(probs)
        all_val_labels.append(labels.cpu().numpy())

all_val_preds = np.concatenate(all_val_preds, axis=0)
all_val_labels = np.concatenate(all_val_labels, axis=0)
eps = 1e-15
all_val_preds = np.clip(all_val_preds, eps, 1 - eps)
final_val_score = log_loss(all_val_labels, all_val_preds)
print(f"Final Validation Log Loss: {final_val_score:.6f}")

# ============================================
# TEST INFERENCE
# ============================================
print("Performing test inference...")
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_test_preds.append(probs)

all_test_preds = np.concatenate(all_test_preds, axis=0)

# ============================================
# GENERATE SUBMISSION FILE
# ============================================
os.makedirs("./submission", exist_ok=True)
test_ids = test_df["id"].values

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": all_test_preds[:, 0],
        "HPL": all_test_preds[:, 1],
        "MWS": all_test_preds[:, 2],
    }
)

row_sums = submission_df[["EAP", "HPL", "MWS"]].sum(axis=1)
submission_df["EAP"] = submission_df["EAP"] / row_sums
submission_df["HPL"] = submission_df["HPL"] / row_sums
submission_df["MWS"] = submission_df["MWS"] / row_sums

for col in ["EAP", "HPL", "MWS"]:
    submission_df[col] = np.clip(submission_df[col], eps, 1 - eps)

row_sums = submission_df[["EAP", "HPL", "MWS"]].sum(axis=1)
submission_df["EAP"] = submission_df["EAP"] / row_sums
submission_df["HPL"] = submission_df["HPL"] / row_sums
submission_df["MWS"] = submission_df["MWS"] / row_sums

submission_df.to_csv("./submission/submission_588f35d9aac14bcd9a7599d336bb891b.csv", index=False)
print(f"Submission saved to ./submission/submission_588f35d9aac14bcd9a7599d336bb891b.csv")

score = final_val_score
print(f"Final Validation Score: {score}")
