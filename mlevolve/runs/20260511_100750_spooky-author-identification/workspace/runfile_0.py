import os
os.sched_setaffinity(0, {64, 59, 60, 62, 63})
import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode target
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_authors = len(label_encoder.classes_)


# Feature engineering functions
def extract_stylometric_features(text_series):
    """Extract various stylometric features from text"""
    features = pd.DataFrame(index=text_series.index)

    # Basic text statistics
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_length"] = text_series.str.split().apply(
        lambda x: np.mean([len(w) for w in x]) if len(x) > 0 else 0
    )
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.findall(r"[.!?]+", x)) if pd.notna(x) else 0
    )

    # Punctuation features
    features["exclamation_count"] = text_series.str.count(r"!")
    features["question_count"] = text_series.str.count(r"\?")
    features["comma_count"] = text_series.str.count(r",")
    features["semicolon_count"] = text_series.str.count(r";")
    features["colon_count"] = text_series.str.count(r":")
    features["quotes_count"] = text_series.str.count(r'"')
    features["dash_count"] = text_series.str.count(r"-")
    features["paren_count"] = text_series.str.count(r"[\(\)]")

    # Punctuation density
    total_chars = features["char_count"].replace(0, 1)
    features["punctuation_density"] = (
        features[
            [
                "exclamation_count",
                "question_count",
                "comma_count",
                "semicolon_count",
                "colon_count",
                "dash_count",
            ]
        ].sum(axis=1)
        / total_chars
    )

    # Word-level features
    features["unique_words_ratio"] = text_series.apply(
        lambda x: (
            len(set(x.lower().split())) / len(x.split()) if len(x.split()) > 0 else 0
        )
    )

    # Capitalization features
    features["capital_ratio"] = text_series.apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    features["proper_noun_likelihood"] = text_series.str.findall(
        r"\b[A-Z][a-z]+\b"
    ).str.len()

    # Special character patterns
    features["ellipsis_count"] = text_series.str.count(r"\.\.\.")
    features["ampersand_count"] = text_series.str.count(r"&")

    # Stopword ratio (approximate using common English stopwords)
    stopwords = set(
        [
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
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "them",
            "their",
            "his",
            "her",
            "my",
            "your",
            "our",
        ]
    )

    features["stopword_ratio"] = text_series.apply(
        lambda x: (
            sum(1 for w in x.lower().split() if w in stopwords) / len(x.split())
            if len(x.split()) > 0
            else 0
        )
    )

    # Rare/archaic word features (common in Lovecraft)
    archaic_words = set(
        [
            "thou",
            "thee",
            "thy",
            "thine",
            "hath",
            "doth",
            "art",
            "wilt",
            "dost",
            "ere",
            "ye",
            "yon",
            "yonder",
            "forsooth",
            "perchance",
            "whence",
            "thence",
            "hither",
            "thither",
            "whither",
            "anon",
            "betwixt",
            "unto",
            "wherefore",
            "therefor",
            "therein",
            "wherein",
            "herein",
            "hereafter",
            "thereafter",
            "thenceforth",
            "henceforth",
            "aught",
            "naught",
            "nay",
            "yea",
            "verily",
            "methinks",
            "prithee",
        ]
    )

    features["archaic_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in archaic_words)
    )

    # Emotional/mood words features
    horror_words = set(
        [
            "ghost",
            "horror",
            "terror",
            "fear",
            "dread",
            "gloom",
            "shadow",
            "dark",
            "night",
            "death",
            "dead",
            "corpse",
            "skeleton",
            "grave",
            "tomb",
            "coffin",
            "wraith",
            "spectre",
            "phantom",
            "demon",
            "devil",
            "hell",
            "curse",
            "evil",
            "monster",
            "creature",
            "beast",
            "wolf",
            "vampire",
            "witch",
            "sorcerer",
            "spell",
            "haunt",
            "dungeon",
            "chamber",
            "vault",
            "cellar",
            "crypt",
            "catacomb",
            "abyss",
            "void",
            "chasm",
            "labyrinth",
            "maze",
            "lurk",
            "crawl",
            "slither",
            "slime",
            "ooze",
            "mold",
            "decay",
            "rot",
            "stench",
            "howl",
            "shriek",
            "scream",
            "moan",
            "groan",
            "wail",
            "lament",
            "anguish",
            "agony",
            "torment",
            "suffering",
            "mad",
            "insane",
            "lunatic",
            "raving",
            "delirium",
            "frenzy",
            "panic",
            "supernatural",
            "eldritch",
            "uncanny",
            "weird",
            "strange",
            "mysterious",
            "secret",
            "hidden",
            "unknown",
            "forbidden",
            "ancient",
            "primordial",
            "cyclopean",
        ]
    )

    features["horror_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in horror_words)
    )

    # Sentence structure complexity (average words per sentence)
    features["avg_words_per_sentence"] = text_series.apply(
        lambda x: (
            np.mean([len(s.split()) for s in re.split(r"[.!?]+", x) if s.strip()])
            if len(re.split(r"[.!?]+", x)) > 1
            else features["word_count"].loc[x.name] if pd.notna(x) else 0
        )
    )

    return features


def extract_tfidf_features(text_series, vectorizer=None, fit=False):
    """Extract TF-IDF features for character and word n-grams"""
    if fit:
        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=2000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=2000,
            sublinear_tf=True,
            stop_words="english",
            strip_accents="unicode",
        )

        char_features = char_vectorizer.fit_transform(text_series)
        word_features = word_vectorizer.fit_transform(text_series)

        return char_features, word_features, char_vectorizer, word_vectorizer
    else:
        char_features = vectorizer[0].transform(text_series)
        word_features = vectorizer[1].transform(text_series)
        return char_features, word_features


# Create stratified folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tokenize raw text data (no leakage - tokenizer is deterministic)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
train_encodings = tokenizer(
    train_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)

print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
print(f"Classes: {label_encoder.classes_}")

# ============================================================
# 2. MODEL DESIGN - HybridAuthorshipModel
# ============================================================


class HybridAuthorshipModel(nn.Module):
    """
    Hybrid model combining DeBERTa-v3-large transformer with explicit stylometric features.
    """

    def __init__(self, num_labels=3, stylometric_dim=22, hidden_size=1024, dropout=0.3):
        super().__init__()

        # DeBERTa backbone
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1

        self.deberta = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large", config=config
        )

        # Freeze first 12 layers (out of 24)
        for i, layer in enumerate(self.deberta.encoder.layer[:12]):
            for param in layer.parameters():
                param.requires_grad = False

        # Stylometric feature encoder
        self.stylo_encoder = nn.Sequential(
            nn.Linear(stylometric_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Cross-attention fusion module
        self.cross_attention_q = nn.Linear(128, 256)
        self.cross_attention_k = nn.Linear(1024, 256)
        self.cross_attention_v = nn.Linear(1024, 256)

        self.attention_dropout = nn.Dropout(0.1)
        self.attention_norm = nn.LayerNorm(256)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 128 + 256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with scaled initialization"""
        for module in [self.stylo_encoder, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.cross_attention_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.cross_attention_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.cross_attention_v.weight, gain=0.5)

    def forward(self, input_ids, attention_mask, stylometric_features):
        # DeBERTa forward pass
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Manual mean pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(outputs.last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Encode stylometric features
        stylo_encoded = self.stylo_encoder(stylometric_features)

        # Cross-attention
        q = self.cross_attention_q(stylo_encoded)
        k = self.cross_attention_k(cls_embedding)
        v = self.cross_attention_v(cls_embedding)

        attn_scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)) / (256**0.5)
        attn_weights = F.softmax(attn_scores.squeeze(1), dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attended_features = torch.matmul(
            attn_weights.unsqueeze(1), v.unsqueeze(1)
        ).squeeze(1)
        attended_features = self.attention_norm(attended_features)

        # Concatenate all features
        combined = torch.cat([pooled_output, stylo_encoded, attended_features], dim=1)

        logits = self.classifier(combined)
        return logits


# ============================================================
# 3. TRAINING AND EVALUATION
# ============================================================


class AuthorshipDataset(Dataset):
    def __init__(self, encodings, stylo_features, labels=None):
        self.encodings = encodings
        self.stylo_features = torch.FloatTensor(stylo_features)
        self.labels = labels
        if labels is not None:
            self.labels = torch.LongTensor(labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "stylometric_features": self.stylo_features[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.stylo_features)


# No need to reload data — we keep it all in memory
# train_encodings, test_encodings, train_df, test_df, label_encoder are already defined
label_classes = label_encoder.classes_

num_labels = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 16
max_epochs = 20
learning_rate = 2e-5
weight_decay = 0.01
early_stopping_patience = 3
gradient_clip_val = 1.0

y_train = train_df["author_encoded"].values

val_predictions = np.zeros((len(train_df), num_labels))
test_predictions = np.zeros((len(test_df), num_labels))
best_val_loss = float("inf")
best_model_path = "./working/best_model_b8a8f0c09f4749d98144e9bcd290dce2.pt"

print("Starting 5-fold cross-validation training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["author"])):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*50}")

    # --- Perform feature engineering INSIDE the fold to avoid leakage ---
    train_text_fold = train_df["text"].iloc[train_idx]
    val_text_fold = train_df["text"].iloc[val_idx]

    # Stylometric features: extract separately for train and val
    train_stylo_fold = extract_stylometric_features(train_text_fold)
    val_stylo_fold = extract_stylometric_features(val_text_fold)

    # TF-IDF features: fit ONLY on train fold, transform both
    char_features_train_fold, word_features_train_fold, char_vectorizer, word_vectorizer = (
        extract_tfidf_features(train_text_fold, fit=True)
    )
    char_features_val_fold, word_features_val_fold = extract_tfidf_features(
        val_text_fold, vectorizer=(char_vectorizer, word_vectorizer)
    )

    # Convert to dense DataFrames
    char_feature_names = [f"char_ngram_{i}" for i in range(char_features_train_fold.shape[1])]
    word_feature_names = [f"word_ngram_{i}" for i in range(word_features_train_fold.shape[1])]

    char_train_dense = pd.DataFrame(
        char_features_train_fold.toarray(), columns=char_feature_names, index=train_text_fold.index
    )
    word_train_dense = pd.DataFrame(
        word_features_train_fold.toarray(), columns=word_feature_names, index=train_text_fold.index
    )
    char_val_dense = pd.DataFrame(
        char_features_val_fold.toarray(), columns=char_feature_names, index=val_text_fold.index
    )
    word_val_dense = pd.DataFrame(
        word_features_val_fold.toarray(), columns=word_feature_names, index=val_text_fold.index
    )

    # Combine
    train_features_fold = pd.concat(
        [train_stylo_fold.reset_index(drop=True),
         char_train_dense.reset_index(drop=True),
         word_train_dense.reset_index(drop=True)], axis=1
    )
    val_features_fold = pd.concat(
        [val_stylo_fold.reset_index(drop=True),
         char_val_dense.reset_index(drop=True),
         word_val_dense.reset_index(drop=True)], axis=1
    )

    # Handle NaN/inf
    train_features_fold = train_features_fold.fillna(0).replace([np.inf, -np.inf], 0)
    val_features_fold = val_features_fold.fillna(0).replace([np.inf, -np.inf], 0)

    X_train_stylo_fold = train_features_fold.values.astype(np.float32)
    X_val_stylo_fold = val_features_fold.values.astype(np.float32)

    # Split tokenized data
    train_encodings_fold = {
        "input_ids": train_encodings["input_ids"][train_idx],
        "attention_mask": train_encodings["attention_mask"][train_idx],
    }
    val_encodings_fold = {
        "input_ids": train_encodings["input_ids"][val_idx],
        "attention_mask": train_encodings["attention_mask"][val_idx],
    }

    stylometric_dim = X_train_stylo_fold.shape[1]

    train_dataset = AuthorshipDataset(
        train_encodings_fold, X_train_stylo_fold, y_train[train_idx]
    )
    val_dataset = AuthorshipDataset(
        val_encodings_fold, X_val_stylo_fold, y_train[val_idx]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model for each fold
    model = HybridAuthorshipModel(
        num_labels=num_labels,
        stylometric_dim=stylometric_dim,
        hidden_size=1024,
        dropout=0.3,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop for current fold
    best_fold_val_loss = float("inf")
    fold_patience = 0
    best_fold_model_state = None

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_features = batch["stylometric_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, stylo_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += input_ids.size(0)

        avg_train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples

        # Validation phase
        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_samples = 0
        val_all_probs = []
        val_all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                stylo_features = batch["stylometric_features"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask, stylo_features)
                    loss = criterion(logits, labels)

                val_total_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, 1)
                val_total_correct += (predicted == labels).sum().item()
                val_total_samples += input_ids.size(0)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_all_probs.append(probs)
                val_all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_total_loss / val_total_samples
        val_accuracy = val_total_correct / val_total_samples

        val_all_probs = np.concatenate(val_all_probs, axis=0)
        val_all_labels = np.concatenate(val_all_labels, axis=0)

        epsilon = 1e-15
        val_all_probs_clipped = np.clip(val_all_probs, epsilon, 1 - epsilon)
        val_all_probs_clipped = val_all_probs_clipped / val_all_probs_clipped.sum(
            axis=1, keepdims=True
        )

        val_log_loss = log_loss(val_all_labels, val_all_probs_clipped)

        print(
            f"Epoch {epoch+1:2d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val LogLoss: {val_log_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_val_loss < best_fold_val_loss:
            best_fold_val_loss = avg_val_loss
            fold_patience = 0
            best_fold_model_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_log_loss": val_log_loss,
            }
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(best_fold_model_state, best_model_path)
                print(f"  -> New best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            fold_patience += 1
            if fold_patience >= early_stopping_patience:
                print(f"  -> Early stopping triggered after epoch {epoch+1}")
                break

    # Store predictions for this fold using best model
    model.load_state_dict(best_fold_model_state["model_state_dict"])
    model.eval()

    # Validation predictions
    with torch.no_grad():
        val_loader_all = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        fold_val_probs = []
        for batch in val_loader_all:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_features = batch["stylometric_features"].to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, stylo_features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            fold_val_probs.append(probs)

        fold_val_probs = np.concatenate(fold_val_probs, axis=0)
        val_predictions[val_idx] = fold_val_probs

    # For test set, we need to transform it using the fold's vectorizers
    char_features_test_fold, word_features_test_fold = extract_tfidf_features(
        test_df["text"], vectorizer=(char_vectorizer, word_vectorizer)
    )
    char_test_dense = pd.DataFrame(
        char_features_test_fold.toarray(), columns=char_feature_names, index=test_df.index
    )
    word_test_dense = pd.DataFrame(
        word_features_test_fold.toarray(), columns=word_feature_names, index=test_df.index
    )

    test_stylo = extract_stylometric_features(test_df["text"])
    test_features_fold = pd.concat(
        [test_stylo.reset_index(drop=True),
         char_test_dense.reset_index(drop=True),
         word_test_dense.reset_index(drop=True)], axis=1
    )
    test_features_fold = test_features_fold.fillna(0).replace([np.inf, -np.inf], 0)
    X_test_stylo_fold = test_features_fold.values.astype(np.float32)

    test_dataset = AuthorshipDataset(test_encodings, X_test_stylo_fold)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_features = batch["stylometric_features"].to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, stylo_features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            fold_test_probs.append(probs)

    fold_test_probs = np.concatenate(fold_test_probs, axis=0)
    test_predictions += fold_test_probs / 5

# ============================================================
# 4. FINAL VALIDATION SCORE AND SUBMISSION
# ============================================================

epsilon = 1e-15
val_predictions_clipped = np.clip(val_predictions, epsilon, 1 - epsilon)
val_predictions_clipped = val_predictions_clipped / val_predictions_clipped.sum(
    axis=1, keepdims=True
)

final_val_log_loss = log_loss(y_train, val_predictions_clipped)
print(f"\n{'='*50}")
print(f"Final Validation Log Loss (5-fold CV): {final_val_log_loss:.6f}")

# Generate submission
test_predictions_clipped = np.clip(test_predictions, epsilon, 1 - epsilon)
test_predictions_clipped = test_predictions_clipped / test_predictions_clipped.sum(
    axis=1, keepdims=True
)

submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_predictions_clipped[:, 0],
        "HPL": test_predictions_clipped[:, 1],
        "MWS": test_predictions_clipped[:, 2],
    }
)

submission = submission[["id", "EAP", "HPL", "MWS"]]

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_b8a8f0c09f4749d98144e9bcd290dce2.csv", index=False)
print(f"Submission saved to ./submission/submission_b8a8f0c09f4749d98144e9bcd290dce2.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {final_val_log_loss:.6f}")