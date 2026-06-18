import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
import joblib
import os

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# =============================================================================

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
num_classes = len(le.classes_)
print(f"Classes: {le.classes_} -> {le.transform(le.classes_)}")


def engineer_features(texts):
    features_list = []
    for text in texts:
        if pd.isna(text):
            text = ""
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features = {}
        features["word_count"] = len(words)
        features["char_count"] = len(text)
        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        features["unique_words_ratio"] = len(set(w.lower() for w in words)) / (
            len(words) + 1
        )
        features["sentence_count"] = len(sentences)
        features["avg_sentence_length"] = (
            np.mean([len(s.split()) for s in sentences]) if sentences else 0
        )
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        features["comma_count"] = text.count(",")
        features["semicolon_count"] = text.count(";")
        features["colon_count"] = text.count(":")
        features["dash_count"] = text.count("--") + text.count("—")
        features["quote_count"] = text.count('"') + text.count("'")
        features["period_count"] = text.count(".")
        features["capital_ratio"] = sum(1 for c in text if c.isupper()) / (
            len(text) + 1
        )
        features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)
        features["vowel_ratio"] = sum(1 for c in text.lower() if c in "aeiou") / (
            len(text) + 1
        )
        features["consonant_ratio"] = sum(
            1 for c in text.lower() if c.isalpha() and c not in "aeiou"
        ) / (len(text) + 1)
        features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / (len(text) + 1)

        lovecraft_words = [
            "eldritch",
            "cyclopean",
            "antediluvian",
            "non",
            "euclidean",
            "unspeakable",
            "nameless",
            "unnameable",
            "crawling",
            "chaos",
            "abyss",
            "gibbering",
            "maddening",
            "blasphemous",
            "squamous",
            "rugose",
            "ichor",
            "beast",
            "daemon",
            "monstrous",
            "frightful",
            "hideous",
            "loathsome",
            "fungus",
            "pit",
            "charnel",
            "sepulchral",
        ]
        poe_words = [
            "nevermore",
            "chamber",
            "tint",
            "sable",
            "plutonian",
            "ebony",
            "sepulchre",
            "obeisance",
            "beguiling",
            "nepenthe",
            "balm",
            "gilead",
            "divan",
            "ventured",
            "scourge",
            "pestilence",
            "spectre",
            "shroud",
            "coffin",
            "tomb",
            "ghastly",
            "dreary",
            "weary",
            "dying",
            "embers",
            "ghost",
            "phantom",
            "shadow",
        ]
        shelley_words = [
            "creation",
            "creature",
            "monster",
            "frankenstein",
            "walton",
            "victor",
            "geneva",
            "ingolstadt",
            "alpine",
            "glacier",
            "cottage",
            "benevolence",
            "ardour",
            "countenance",
            "dwelling",
            "benevolent",
            "did",
            "could",
            "would",
            "shall",
            "father",
            "mother",
            "beloved",
            "sister",
            "brother",
        ]

        text_lower = text.lower()
        features["lovecraft_score"] = sum(1 for w in lovecraft_words if w in text_lower)
        features["poe_score"] = sum(1 for w in poe_words if w in text_lower)
        features["shelley_score"] = sum(1 for w in shelley_words if w in text_lower)

        archaic_words = [
            "thee",
            "thy",
            "thou",
            "thine",
            "art",
            "doth",
            "hath",
            "dost",
            "shall",
            "shalt",
            "wilt",
            "canst",
            "couldst",
            "wouldst",
            "shouldst",
            "thence",
            "whence",
            "hither",
            "thither",
            "whither",
            "anon",
            "ere",
            "betwixt",
            "unto",
            "alas",
            "forsooth",
            "perchance",
            "methinks",
        ]
        features["archaic_score"] = sum(1 for w in archaic_words if w in text_lower)

        syllable_count = sum(len(re.findall(r"[aeiouy]+", w.lower())) for w in words)
        features["syllable_per_word"] = syllable_count / (len(words) + 1)
        complex_words = sum(
            1 for w in words if len(re.findall(r"[aeiouy]+", w.lower())) > 2
        )
        features["complex_word_ratio"] = complex_words / (len(words) + 1)

        features_list.append(features)
    return pd.DataFrame(features_list)


print("Engineering features for train data...")
train_features = engineer_features(train_df["text"].tolist())
print(f"Train features shape: {train_features.shape}")

print("Engineering features for test data...")
test_features = engineer_features(test_df["text"].tolist())
print(f"Test features shape: {test_features.shape}")

print("Extracting TF-IDF features...")
tfidf_word = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)
train_tfidf_word = tfidf_word.fit_transform(train_df["text"].fillna(""))
test_tfidf_word = tfidf_word.transform(test_df["text"].fillna(""))

tfidf_char = TfidfVectorizer(
    max_features=3000,
    ngram_range=(2, 5),
    analyzer="char",
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)
train_tfidf_char = tfidf_char.fit_transform(train_df["text"].fillna(""))
test_tfidf_char = tfidf_char.transform(test_df["text"].fillna(""))

# Scale engineered features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Combine all features into sparse matrices
X_train = hstack(
    [csr_matrix(train_features_scaled), train_tfidf_word, train_tfidf_char]
)
X_test = hstack([csr_matrix(test_features_scaled), test_tfidf_word, test_tfidf_char])

y_train = train_df["author_encoded"].values

# Create stratified train/validation split
print("Creating train/validation split...")
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.15, random_state=42, stratify=y_train
)
val_texts = train_df.iloc[val_idx]["text"].tolist()
train_texts_split = train_df.iloc[train_idx]["text"].tolist()
test_texts = test_df["text"].tolist()
test_ids = test_df["id"].tolist()

print(f"Final shapes:")
print(f"X_train: {X_train_final.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train_final.shape}")
print(f"y_val: {y_val.shape}")

# =============================================================================
# STEP 2: MODEL DESIGN
# =============================================================================


class CharCNN(nn.Module):
    def __init__(
        self, num_classes=3, max_chars=2000, char_vocab_size=128, embed_dim=64
    ):
        super().__init__()
        self.max_chars = max_chars
        self.char_embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, 128, kernel_size=k) for k in [2, 3, 4, 5]]
        )
        self.fc1 = nn.Linear(128 * 4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.char_embed(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = F.relu(conv_out)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outs.append(conv_out.squeeze(2))
        x = torch.cat(conv_outs, dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiModelEnsemble(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        print("Loading DeBERTa-v3-large...")
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.deberta_model.to(device)

        print("Initializing CharCNN...")
        self.charcnn = CharCNN(num_classes=num_classes)
        self.charcnn.to(device)

        self.deberta_weight = nn.Parameter(torch.tensor(0.7))
        self.charcnn_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, input_ids, attention_mask, char_ids=None):
        deb_outputs = self.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        deb_logits = deb_outputs.logits

        if char_ids is not None:
            char_logits = self.charcnn(char_ids)
        else:
            char_logits = torch.zeros_like(deb_logits)

        weights_sum = self.deberta_weight + self.charcnn_weight

        combined_logits = (
            self.deberta_weight * deb_logits + self.charcnn_weight * char_logits
        ) / weights_sum
        return combined_logits


# MLM Augmentation function using distilroberta-base
from transformers import pipeline
import random

augment_pipeline = pipeline(
    "fill-mask",
    model="distilroberta-base",
    tokenizer="distilroberta-base",
    device=-1,
)

def augment_text_mlm(text, tokenizer=augment_pipeline.tokenizer, model=augment_pipeline.model, mask_prob=0.12, replace_prob=0.3):
    """Randomly replace words with MLM predictions with given probability."""
    if random.random() > replace_prob:
        return text
    words = text.split()
    if len(words) < 3:
        return text
    masked_text = []
    original_words = []
    for word in words:
        if random.random() < mask_prob and len(word) > 1:
            masked_text.append("<mask>")
            original_words.append(word)
        else:
            masked_text.append(word)
    if not original_words:
        return text
    masked_sentence = " ".join(masked_text)
    try:
        results = augment_pipeline(masked_sentence)
        result_list = results if isinstance(results[0], list) else [results]
        idx = 0
        for i, w in enumerate(masked_text):
            if w == "<mask>":
                if idx < len(result_list):
                    predictions = result_list[idx]
                    if isinstance(predictions, list) and len(predictions) > 0:
                        best = predictions[0]
                        if best["score"] > 0.1:
                            masked_text[i] = best["token_str"].strip()
                        else:
                            masked_text[i] = original_words[idx - sum(1 for r in masked_text[:i] if r == "<mask>")] if (idx - sum(1 for r in masked_text[:i] if r == "<mask>")) < len(original_words) else original_words[-1]
                    idx += 1
        return " ".join(masked_text)
    except:
        return text

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, encodings, char_ids, labels=None, augment=False):
        self.texts = texts
        self.encodings = encodings
        self.char_ids = char_ids
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        char_ids = self.char_ids[idx]
        if self.augment:
            text = self.texts[idx]
            augmented_text = augment_text_mlm(text)
            if augmented_text != text:
                temp_tokenizer = model.tokenizer
                new_enc = temp_tokenizer(
                    augmented_text,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                )
                input_ids = new_enc["input_ids"].squeeze(0)
                attention_mask = new_enc["attention_mask"].squeeze(0)
        if self.labels is not None:
            return input_ids, attention_mask, char_ids, self.labels[idx]
        else:
            return input_ids, attention_mask, char_ids


def char_tokenize(texts, max_length=2000, char_to_idx=None):
    if char_to_idx is None:
        char_to_idx = {chr(i): i + 1 for i in range(32, 127)}
        char_to_idx["\n"] = 96
        char_to_idx["\t"] = 97
        char_to_idx["<UNK>"] = 98

    char_ids_list = []
    for text in texts:
        text = str(text)[:max_length]
        ids = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in text]
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        char_ids_list.append(ids)
    return torch.tensor(char_ids_list, dtype=torch.long), char_to_idx


# =============================================================================
# STEP 3: TRAINING AND EVALUATION
# =============================================================================

# Initialize model
model = MultiModelEnsemble(num_classes=num_classes)
model.to(device)

# Character tokenization
print("Building character tokenizer...")
# Only use training texts to build character vocabulary (prevent data leakage)
_, char_to_idx = char_tokenize(train_texts_split[0:1])  # Initialize vocabulary from train only

# Tokenize texts for DeBERTa
print("Tokenizing texts for DeBERTa...")
train_encodings = model.tokenizer(
    train_texts_split,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
val_encodings = model.tokenizer(
    val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

# Character encodings
train_char_ids, _ = char_tokenize(train_texts_split, char_to_idx=char_to_idx)
val_char_ids, _ = char_tokenize(val_texts, char_to_idx=char_to_idx)
test_char_ids, _ = char_tokenize(test_texts, char_to_idx=char_to_idx)

# Create data loaders with augmentation
train_dataset = AugmentedDataset(
    train_texts_split,
    train_encodings,
    train_char_ids,
    labels=torch.tensor(y_train_final, dtype=torch.long),
    augment=True,
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    val_char_ids,
    torch.tensor(y_val, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True
)

# Helper function to create optimizer with discriminative learning rates
def create_optimizer(model, unfreeze_stage):
    """Create optimizer with discriminative learning rates based on unfreeze stage."""
    # Stage 0: all frozen, only train CharCNN and weights
    # Stage 1: last 2 encoder layers unfrozen
    # Stage 2: all layers unfrozen

    deberta = model.deberta_model
    base_lr = 2e-5

    param_groups = []
    # CharCNN parameters
    param_groups.append({"params": model.charcnn.parameters(), "lr": 2e-4})
    # Ensemble weights
    param_groups.append({"params": [model.deberta_weight, model.charcnn_weight], "lr": 2e-3})

    if unfreeze_stage == 0:
        # Freeze all DeBERTa layers (only classifiers trained)
        for param in deberta.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in deberta.classifier.parameters():
            param.requires_grad = True
    elif unfreeze_stage == 1:
        # Unfreeze last 2 encoder layers
        for param in deberta.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in deberta.classifier.parameters():
            param.requires_grad = True
        # Unfreeze last 2 layers
        deberta_layer = deberta.deberta.encoder.layer
        num_layers = len(deberta_layer)
        for i in range(num_layers - 2, num_layers):
            for param in deberta_layer[i].parameters():
                param.requires_grad = True
        # Add these layers with higher LR
        for i in range(num_layers - 2, num_layers):
            param_groups.append({"params": deberta_layer[i].parameters(), "lr": 2e-5})
    else:
        # Unfreeze all
        for param in deberta.parameters():
            param.requires_grad = True
        # Create parameter groups with different LRs
        deberta_layer = deberta.deberta.encoder.layer
        num_layers = len(deberta_layer)
        # Bottom layers (first 1/3)
        bottom_end = num_layers // 3
        middle_end = 2 * num_layers // 3
        if bottom_end > 0:
            param_groups.append({"params": [p for l in deberta_layer[:bottom_end] for p in l.parameters()], "lr": 1e-6})
        # Middle layers (second 1/3)
        if middle_end > bottom_end:
            param_groups.append({"params": [p for l in deberta_layer[bottom_end:middle_end] for p in l.parameters()], "lr": 5e-6})
        # Top layers (last 1/3)
        if num_layers > middle_end:
            param_groups.append({"params": [p for l in deberta_layer[middle_end:] for p in l.parameters()], "lr": 2e-5})
        # Classifier
        param_groups.append({"params": deberta.classifier.parameters(), "lr": 2e-5})
        # Pooler
        if hasattr(deberta, 'pooler'):
            param_groups.append({"params": deberta.pooler.parameters(), "lr": 2e-5})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    return optimizer

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = create_optimizer(model, unfreeze_stage=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# Training loop with progressive unfreezing
print("Starting training with progressive unfreezing...")
num_epochs = 30
best_val_loss = float("inf")
best_model_state = None
patience = 6
no_improve_count = 0
scaler_amp = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    # Determine unfreeze stage
    if epoch == 0:
        current_stage = 0  # All frozen except classifier
    elif epoch == 3:
        current_stage = 1  # Unfreeze last 2 layers
        optimizer = create_optimizer(model, unfreeze_stage=1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
    elif epoch == 6:
        current_stage = 2  # Unfreeze all
        optimizer = create_optimizer(model, unfreeze_stage=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        input_ids, attention_mask, char_ids, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask, char_ids)
            loss = criterion(logits, labels)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        train_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, char_ids, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask, char_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)

            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_val_probs.extend(probs.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_probs = np.array(all_val_probs)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_log_loss = log_loss(all_val_labels, val_probs)

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val LogLoss: {val_log_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

# Final validation evaluation
model.eval()
all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, char_ids, labels = [b.to(device) for b in batch]
        logits = model(input_ids, attention_mask, char_ids)
        probs = F.softmax(logits, dim=1)
        all_val_probs.extend(probs.cpu().numpy())
        all_val_labels.extend(labels.cpu().numpy())

val_probs = np.array(all_val_probs)
val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
final_val_score = log_loss(all_val_labels, val_probs)
print(f"Final Validation LogLoss Score: {final_val_score:.6f}")

# Test inference
print("Running test inference...")
test_encodings = model.tokenizer(
    test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], test_char_ids
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, char_ids = [b.to(device) for b in batch]
        logits = model(input_ids, attention_mask, char_ids)
        probs = F.softmax(logits, dim=1)
        all_test_probs.extend(probs.cpu().numpy())

test_probs = np.array(all_test_probs)

# Final ensemble blend (no XGBoost)
# test_probs already contains final model output, no re-blending needed
# Normalize to ensure rows sum to 1
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# Create and save submission
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Test predictions shape: {test_probs.shape}")
print(f"Final Validation Score: {final_val_score}")