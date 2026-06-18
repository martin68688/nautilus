import os
os.sched_setaffinity(0, {174, 175, 176, 177, 188})
import pandas as pd
import numpy as np
import re
import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# ============================================
# Load Data
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_texts = train_df["text"].values
test_texts = test_df["text"].values
all_texts = np.concatenate([train_texts, test_texts])


# ============================================
# Feature Engineering (Step 1)
# ============================================
def extract_basic_features(texts):
    features = []
    for text in texts:
        words = str(text).split()
        sentences = re.split(r"[.!?]+", str(text))
        sentences = [s.strip() for s in sentences if s.strip()]
        word_count = len(words)
        char_count = len(str(text))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        sent_count = len(sentences)
        avg_sent_length = word_count / max(sent_count, 1)
        unique_words = len(set(w.lower() for w in words))
        unique_ratio = unique_words / max(word_count, 1)
        punct_count = len(re.findall(r"[,.!?;:\'\"-]", str(text)))
        excl_count = str(text).count("!")
        quest_count = str(text).count("?")
        ellipsis_count = str(text).count("...")
        dash_count = str(text).count("--") + str(text).count("\u2014")
        semi_count = str(text).count(";")
        colon_count = str(text).count(":")
        quote_count = str(text).count('"') + str(text).count("'")
        cap_ratio = sum(1 for c in str(text) if c.isupper()) / max(char_count, 1)
        digit_count = sum(1 for c in str(text) if c.isdigit())
        special_count = len(re.findall(r"[^a-zA-Z0-9\s]", str(text)))
        punct_per_word = punct_count / max(word_count, 1)
        long_word_ratio = sum(1 for w in words if len(w) >= 6) / max(word_count, 1)
        very_long_word_ratio = sum(1 for w in words if len(w) >= 10) / max(
            word_count, 1
        )
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
        }
        stop_count = sum(1 for w in words if w.lower() in stop_words)
        stop_ratio = stop_count / max(word_count, 1)
        features.append(
            [
                word_count,
                char_count,
                avg_word_length,
                sent_count,
                avg_sent_length,
                unique_ratio,
                punct_count,
                excl_count,
                quest_count,
                ellipsis_count,
                dash_count,
                semi_count,
                colon_count,
                quote_count,
                cap_ratio,
                digit_count,
                special_count,
                punct_per_word,
                long_word_ratio,
                very_long_word_ratio,
                stop_ratio,
            ]
        )
    return np.array(features)


basic_features = extract_basic_features(all_texts)

# TF-IDF features
tfidf_word = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=5,
    max_df=0.95,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
)
word_tfidf = tfidf_word.fit_transform(all_texts).toarray()

tfidf_char = TfidfVectorizer(
    max_features=2000,
    ngram_range=(2, 5),
    sublinear_tf=True,
    min_df=5,
    max_df=0.95,
    strip_accents="unicode",
    analyzer="char",
)
char_tfidf = tfidf_char.fit_transform(all_texts).toarray()


# Readability scores
def extract_readability_features(texts):
    features = []
    for text in texts:
        words = str(text).split()
        sentences = re.split(r"[.!?]+", str(text))
        sentences = [s.strip() for s in sentences if s.strip()]
        word_count = len(words)
        sent_count = len(sentences)
        char_count = len(str(text))
        vowels = "aeiouy"
        total_syllables = 0
        for word in words:
            word = word.lower().strip('.,!?;:\u2019"-')
            if not word:
                continue
            syllable_count = 0
            prev_was_vowel = False
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            if word.endswith("e") and syllable_count > 1:
                syllable_count -= 1
            if syllable_count == 0:
                syllable_count = 1
            total_syllables += syllable_count
        flesch = (
            206.835
            - 1.015 * (word_count / max(sent_count, 1))
            - 84.6 * (total_syllables / max(word_count, 1))
            if word_count > 0 and sent_count > 0
            else 0
        )
        ari = (
            4.71 * (char_count / max(word_count, 1))
            + 0.5 * (word_count / max(sent_count, 1))
            - 21.43
            if word_count > 0 and sent_count > 0
            else 0
        )
        coleman = (
            0.0588 * (char_count / max(word_count, 1) * 100)
            - 0.296 * (sent_count / max(word_count, 1) * 100)
            - 15.8
            if word_count > 0 and sent_count > 0
            else 0
        )
        features.append([flesch, ari, coleman, total_syllables / max(word_count, 1)])
    return np.array(features)


readability_features = extract_readability_features(all_texts)


# POS-like features
def extract_pos_like_features(texts):
    features = []
    adverb_pattern = re.compile(r"\w+ly\b", re.IGNORECASE)
    verb_endings = ["ed", "ing", "s", "en"]
    noun_endings = ["tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship"]
    adj_endings = ["ous", "ive", "ful", "less", "able", "ible", "al", "ic"]
    for text in texts:
        words = str(text).split()
        word_count = max(len(words), 1)
        adverb_count = len(adverb_pattern.findall(str(text)))
        verb_count = sum(
            1 for w in words if any(w.lower().endswith(e) for e in verb_endings)
        )
        noun_count = sum(
            1 for w in words if any(w.lower().endswith(e) for e in noun_endings)
        )
        adj_count = sum(
            1 for w in words if any(w.lower().endswith(e) for e in adj_endings)
        )
        features.append(
            [
                adverb_count / word_count,
                verb_count / word_count,
                noun_count / word_count,
                adj_count / word_count,
                (adverb_count + verb_count + noun_count + adj_count) / word_count,
            ]
        )
    return np.array(features)


pos_features = extract_pos_like_features(all_texts)


# Sentiment features
def extract_sentiment_features(texts):
    features = []
    positive_words = {
        "love",
        "beautiful",
        "wonderful",
        "happy",
        "joy",
        "peace",
        "hope",
        "gentle",
        "kind",
        "sweet",
        "calm",
        "bright",
        "soft",
        "warm",
        "light",
        "delight",
        "pleasure",
        "glad",
        "cheerful",
        "bliss",
        "serene",
    }
    negative_words = {
        "death",
        "dark",
        "terrible",
        "horror",
        "fear",
        "dread",
        "pain",
        "agony",
        "gloomy",
        "shadow",
        "ghost",
        "monster",
        "evil",
        "wicked",
        "cruel",
        "sorrow",
        "misery",
        "anguish",
        "despair",
        "grief",
        "terror",
        "vile",
        "hideous",
        "frightful",
        "awful",
        "dismal",
        "gothic",
        "creepy",
        "morbid",
    }
    archaic_words = {
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "dost",
        "art",
        "wilt",
        "shalt",
        "spake",
        "methinks",
        "prithee",
        "forsooth",
        "ere",
        "thence",
        "whence",
        "hence",
        "whilst",
        "anon",
        "betwixt",
        "perchance",
        "verily",
    }
    for text in texts:
        words_lower = str(text).lower().split()
        word_count = max(len(words_lower), 1)
        pos_count = sum(
            1 for w in words_lower if w.strip('.,!?;:\u2019"-') in positive_words
        )
        neg_count = sum(
            1 for w in words_lower if w.strip('.,!?;:\u2019"-') in negative_words
        )
        arch_count = sum(
            1 for w in words_lower if w.strip('.,!?;:\u2019"-') in archaic_words
        )
        features.append(
            [
                pos_count / word_count,
                neg_count / word_count,
                arch_count / word_count,
                (pos_count + neg_count) / word_count,
            ]
        )
    return np.array(features)


sentiment_features = extract_sentiment_features(all_texts)


# Structure features
def extract_structure_features(texts):
    features = []
    for text in texts:
        sentences = re.split(r"[.!?]+", str(text))
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        words_per_sent = [len(s.split()) for s in sentences]
        chars_per_sent = [len(s) for s in sentences]
        features.append(
            [
                np.mean(words_per_sent),
                np.std(words_per_sent) if len(words_per_sent) > 1 else 0,
                np.max(words_per_sent),
                np.mean(chars_per_sent),
                np.std(chars_per_sent) if len(chars_per_sent) > 1 else 0,
            ]
        )
    return np.array(features)


structure_features = extract_structure_features(all_texts)


# Pronoun features
def extract_pronoun_features(texts):
    features = []
    first_person = {
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
    second_person = {"you", "your", "yours", "yourself", "yourselves"}
    third_person = {
        "he",
        "she",
        "it",
        "him",
        "her",
        "his",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "himself",
        "herself",
        "itself",
        "themselves",
    }
    for text in texts:
        words_lower = str(text).lower().split()
        word_count = max(len(words_lower), 1)
        first_count = sum(
            1 for w in words_lower if w.strip('.,!?;:\u2019"-') in first_person
        )
        second_count = sum(
            1 for w in words_lower if w.strip('.,!?;:\u2019"-') in second_person
        )
        third_count = sum(
            1 for w in words_lower if w.strip('.,!?;:\u2019"-') in third_person
        )
        features.append(
            [
                first_count / word_count,
                second_count / word_count,
                third_count / word_count,
                (first_count + second_count + third_count) / word_count,
            ]
        )
    return np.array(features)


pronoun_features = extract_pronoun_features(all_texts)

# Combine all features
all_features = np.concatenate(
    [
        basic_features,
        readability_features,
        pos_features,
        sentiment_features,
        structure_features,
        pronoun_features,
        word_tfidf,
        char_tfidf,
    ],
    axis=1,
)

numeric_feature_end = (
    basic_features.shape[1]
    + readability_features.shape[1]
    + pos_features.shape[1]
    + sentiment_features.shape[1]
    + structure_features.shape[1]
    + pronoun_features.shape[1]
)
numeric_features = all_features[:, :numeric_feature_end]
tfidf_features_only = all_features[:, numeric_feature_end:]

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)
all_features_scaled = np.concatenate([numeric_scaled, tfidf_features_only], axis=1)

train_features = all_features_scaled[: len(train_df)]
test_features = all_features_scaled[len(train_df) :]

# ============================================
# Setup for DeBERTa Training (Step 3)
# ============================================
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
labels = np.array([author_map[a] for a in train_df["author"]])
num_labels = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    hidden_dropout_prob=0.15,
    attention_probs_dropout_prob=0.15,
)
model.gradient_checkpointing_enable()
model.to(device)

# ============================================
# Train/Val Split
# ============================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.arange(len(train_texts)), labels)
):
    if fold == 0:
        train_texts_fold = train_texts[train_idx]
        val_texts_fold = train_texts[val_idx]
        train_labels_fold = labels[train_idx]
        val_labels_fold = labels[val_idx]
        break

print(f"Train samples: {len(train_texts_fold)}, Val samples: {len(val_texts_fold)}")

# ============================================
# Tokenization
# ============================================
MAX_LENGTH = 384
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
EPOCHS = 12
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LABEL_SMOOTHING = 0.15


def encode_texts(texts, tokenizer, max_length):
    return tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


train_encodings = encode_texts(train_texts_fold, tokenizer, MAX_LENGTH)
val_encodings = encode_texts(val_texts_fold, tokenizer, MAX_LENGTH)
test_encodings = encode_texts(test_texts, tokenizer, MAX_LENGTH)

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels_fold, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels_fold, dtype=torch.long),
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

# ============================================
# Training Components
# ============================================
no_decay = ["bias", "LayerNorm.weight"]
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

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# ============================================
# Training Loop
# ============================================
best_val_logloss = float("inf")
patience = 5
patience_counter = 0


def unfreeze_layers(epoch):
    if epoch == 0:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Epoch 0: Training classifier head only")
    elif epoch <= 3:
        for name, param in model.named_parameters():
            if (
                "layer.20" in name
                or "layer.21" in name
                or "layer.22" in name
                or "layer.23" in name
            ):
                param.requires_grad = True
            elif "classifier" in name or "pooler" in name:
                param.requires_grad = True
        print("Epoch 1-3: Unfrozen top 4 layers")
    elif epoch <= 6:
        for name, param in model.named_parameters():
            if any(f"layer.{i}" in name for i in range(16, 24)):
                param.requires_grad = True
            elif "classifier" in name or "pooler" in name:
                param.requires_grad = True
        print("Epoch 4-6: Unfrozen top 8 layers")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("Epoch 7+: Full fine-tuning")


print("\nStarting training...")
for epoch in range(EPOCHS):
    unfreeze_layers(epoch)
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, batch_labels = [
            b.to(device, non_blocking=True) for b in batch
        ]
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss = loss / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUMULATION_STEPS
        preds = torch.argmax(outputs.logits, dim=1)
        total_correct += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_acc = total_correct / total_samples

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, batch_labels = [
                b.to(device, non_blocking=True) for b in batch
            ]
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == batch_labels).sum().item()
            val_total += batch_labels.size(0)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(batch_labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_probs = np.concatenate(all_val_probs, axis=0)
    val_labels_np = np.concatenate(all_val_labels, axis=0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_logloss = 0
    for i in range(len(val_labels_np)):
        val_logloss += -np.log(val_probs_clipped[i, val_labels_np[i]])
    val_logloss /= len(val_labels_np)
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val LogLoss: {val_logloss:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        torch.save(model.state_dict(), "./working/best_model_9595f43f282a4d7189ea17711be06f8a.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# ============================================
# Load Best Model and Evaluate
# ============================================
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load("./working/best_model_9595f43f282a4d7189ea17711be06f8a.pt"))
model.eval()

all_val_probs = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, batch_labels = [
            b.to(device, non_blocking=True) for b in batch
        ]
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(batch_labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_labels_np = np.concatenate(all_val_labels, axis=0)
eps = 1e-15
val_probs_final = np.clip(val_probs, eps, 1 - eps)
val_probs_final = val_probs_final / val_probs_final.sum(axis=1, keepdims=True)
final_val_logloss = 0
for i in range(len(val_labels_np)):
    final_val_logloss += -np.log(val_probs_final[i, val_labels_np[i]])
final_val_logloss /= len(val_labels_np)

print(f"Final Validation Score: {final_val_logloss:.6f}")

# ============================================
# Generate Test Predictions
# ============================================
print("Generating test predictions...")
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)
test_probs_final = np.clip(test_probs, eps, 1 - eps)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# ============================================
# Create Submission
# ============================================
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_9595f43f282a4d7189ea17711be06f8a.csv", index=False)

print(f"Submission saved to ./submission/submission_9595f43f282a4d7189ea17711be06f8a.csv")
print(f"Submission shape: {submission.shape}")
print(f"Validation LogLoss: {final_val_logloss:.6f}")

del model, train_loader, val_loader, test_loader
gc.collect()
torch.cuda.empty_cache()

print(f"Final Validation Score: {final_val_logloss}")
