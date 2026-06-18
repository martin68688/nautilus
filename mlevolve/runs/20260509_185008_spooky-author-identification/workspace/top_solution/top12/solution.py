import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sentence_transformers import SentenceTransformer
import os
import pickle
import warnings
import re
from textstat import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import nltk
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")

# ─── Download WordNet data if not present ─────────────────
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# ─── Configuration ──────────────────────────────────────
NUM_CLASSES = 3
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
DROPOUT_RATE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── Load Data ──────────────────────────────────────────
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# ─── Encode Target ──────────────────────────────────────
label_encoder = LabelEncoder()
label_encoder.fit(train_df["author"])
NUM_CLASSES = len(label_encoder.classes_)
CLASS_NAMES = label_encoder.classes_

y_full = label_encoder.transform(train_df["author"].values)
X_full = train_df["text"].values

# ─── Stratified Split ──────────────────────────────────
X_train_val, X_test_internal, y_train_val, y_test_internal = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)

print(
    f"Train: {len(X_train_texts)}, Val: {len(X_val_texts)}, Test (internal): {len(X_test_internal)}"
)
print(f"Train distribution: {np.bincount(y_train)}")
print(f"Val distribution: {np.bincount(y_val)}")


# ─── Synonym Replacement Augmentation ────────────────────
STOPWORDS = set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "as", "was", "were", "had", "have", "has",
    "been", "being", "is", "are", "be", "not", "no", "so", "if", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "we", "you",
    "my", "your", "his", "her", "our", "their", "me", "him", "us", "them",
    "who", "what", "which", "where", "when", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "as",
    "until", "while", "about", "between", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "any", "nor", "not"
])


def get_synonyms(word):
    """Get synonyms for a word using WordNet, excluding the word itself."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word and synonym.isalpha():
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(text, n=2):
    """Replace up to n words in text with WordNet synonyms."""
    words = text.split()
    # Filter to words that are not stopwords, are alphabetic, and have synonyms
    candidate_indices = []
    for i, w in enumerate(words):
        w_clean = w.strip(".,;:!?\"'()").lower()
        if w_clean not in STOPWORDS and w_clean.isalpha():
            syns = get_synonyms(w_clean)
            if syns:
                candidate_indices.append((i, w, w_clean, syns))
    if not candidate_indices:
        return text
    n_replace = min(n, len(candidate_indices))
    selected = random.sample(candidate_indices, n_replace)
    words_out = words.copy()
    for idx, orig, _, syns in selected:
        replacement = random.choice(syns)
        # Preserve original capitalization if first letter was uppercase
        if orig[0].isupper():
            replacement = replacement.capitalize()
        words_out[idx] = replacement
    return " ".join(words_out)


def augment_texts(texts, labels, augment_factor=1.0):
    """
    Create augmented copies of texts by synonym replacement.
    augment_factor: fraction of the dataset to augment (1.0 = create equal number of copies).
    """
    random.seed(42)
    augmented_texts = []
    augmented_labels = []
    n_augment = int(len(texts) * augment_factor)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    for i in indices[:n_augment]:
        aug_text = synonym_replacement(texts[i], n=random.randint(1, 2))
        if aug_text != texts[i]:  # Only add if something changed
            augmented_texts.append(aug_text)
            augmented_labels.append(labels[i])
    return augmented_texts, augmented_labels


# Apply augmentation to training texts only (not validation/test)
print("Applying synonym replacement augmentation...")
aug_texts, aug_labels = augment_texts(X_train_texts.tolist(), y_train.tolist(), augment_factor=1.0)
if aug_texts:
    X_train_texts = np.concatenate([X_train_texts, np.array(aug_texts)])
    y_train = np.concatenate([y_train, np.array(aug_labels)])
    print(f"Augmented training set: {len(X_train_texts)} samples (added {len(aug_texts)} synthetic samples)")
else:
    print("No augmented samples generated.")


# ─── Feature Engineering Functions ──────────────────────
def extract_lexical_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["char_count"] = text_series.apply(len)
    features["word_count"] = text_series.apply(lambda x: len(x.split()))
    features["avg_word_len"] = text_series.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.findall(r"[.!?]+", x)) + 1
    )
    features["avg_sentence_len"] = features["char_count"] / (
        features["sentence_count"] + 1
    )
    features["unique_word_ratio"] = text_series.apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1)
    )
    features["hapax_legomena_ratio"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if x.lower().split().count(w) == 1)
        / (len(x.split()) + 1)
    )
    return features


def extract_syntactic_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    punctuation_marks = [",", ".", ";", ":", "!", "?", "-", '"', "'", "(", ")", "—"]
    features["capital_ratio"] = text_series.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    features["exclamation_density"] = text_series.apply(
        lambda x: x.count("!") / (len(x) + 1)
    )
    features["question_density"] = text_series.apply(
        lambda x: x.count("?") / (len(x) + 1)
    )
    features["ellipsis_count"] = text_series.apply(
        lambda x: len(re.findall(r"\.\.\.", x))
    )
    for punct in punctuation_marks:
        col_name = f"punct_{punct}"
        if punct == '"':
            features[col_name] = text_series.apply(
                lambda x: x.count('"') / (len(x) + 1)
            )
        elif punct == "'":
            features[col_name] = text_series.apply(
                lambda x: x.count("'") / (len(x) + 1)
            )
        elif punct == "—":
            features[col_name] = text_series.apply(
                lambda x: x.count("—") / (len(x) + 1)
            )
        else:
            features[col_name] = text_series.apply(
                lambda x: x.count(punct) / (len(x) + 1)
            )
    return features


def extract_stopword_patterns(text_series):
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
            "by",
            "with",
            "from",
            "as",
            "was",
            "were",
            "had",
            "have",
            "has",
            "been",
            "being",
            "is",
            "are",
            "be",
            "not",
            "no",
            "so",
            "if",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "we",
            "you",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "him",
            "us",
            "them",
            "who",
            "what",
            "which",
            "where",
            "when",
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
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "as",
            "until",
            "while",
            "about",
            "between",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "any",
            "nor",
            "not",
        ]
    )
    features = pd.DataFrame(index=text_series.index)
    features["stopword_ratio"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in stopwords)
        / (len(x.split()) + 1)
    )
    function_words = [
        "the",
        "and",
        "of",
        "to",
        "in",
        "that",
        "was",
        "with",
        "had",
        "have",
        "not",
        "but",
        "his",
        "her",
        "its",
        "my",
        "all",
        "very",
        "so",
        "as",
        "which",
        "what",
        "there",
        "their",
        "this",
        "our",
        "upon",
        "yet",
        "though",
        "still",
        "even",
        "also",
        "only",
        "now",
        "then",
        "here",
        "something",
        "nothing",
        "everything",
        "before",
        "after",
        "again",
    ]
    for word in function_words:
        features[f"fw_{word}"] = text_series.apply(
            lambda x, w=word: sum(1 for t in x.lower().split() if t == w)
            / (len(x.split()) + 1)
        )
    return features


def extract_structural_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["comma_per_sentence"] = text_series.apply(
        lambda x: x.count(",") / (x.count(".") + x.count("!") + x.count("?") + 1)
    )
    features["semicolon_per_sentence"] = text_series.apply(
        lambda x: x.count(";") / (x.count(".") + x.count("!") + x.count("?") + 1)
    )
    features["colon_per_sentence"] = text_series.apply(
        lambda x: x.count(":") / (x.count(".") + x.count("!") + x.count("?") + 1)
    )
    sentences = text_series.apply(lambda x: re.split(r"[.!?]+", x))
    features["avg_sentence_start_len"] = sentences.apply(
        lambda s: (
            np.mean([len(sent.split()) for sent in s if len(sent.strip()) > 0])
            if len(s) > 0
            else 0
        )
    )
    conjunctions = [
        "and",
        "but",
        "or",
        "so",
        "yet",
        "for",
        "nor",
        "although",
        "though",
        "while",
        "because",
    ]
    features["sentence_start_conjunction_ratio"] = text_series.apply(
        lambda x: sum(
            1
            for sent in re.split(r"[.!?]+", x)
            if any(sent.lower().strip().startswith(c) for c in conjunctions)
        )
        / max(len(re.split(r"[.!?]+", x)), 1)
    )
    first_person_pronouns = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
    third_person_pronouns = [
        "he",
        "she",
        "it",
        "they",
        "him",
        "her",
        "them",
        "his",
        "their",
        "its",
    ]
    features["first_person_ratio"] = text_series.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w.rstrip(".,;:!?\"'") in first_person_pronouns
        )
        / (len(x.split()) + 1)
    )
    features["third_person_ratio"] = text_series.apply(
        lambda x: sum(
            1
            for w in x.lower().split()
            if w.rstrip(".,;:!?\"'") in third_person_pronouns
        )
        / (len(x.split()) + 1)
    )
    return features


def extract_content_features(text_series, vectorizer):
    char_features = vectorizer.transform(text_series)
    char_feature_names = vectorizer.get_feature_names_out()
    char_df = pd.DataFrame(
        char_features.toarray(),
        columns=[f"char_tfidf_{name}" for name in char_feature_names],
        index=text_series.index,
    )
    return char_df


# ─── Build Feature DataFrames ───────────────────────────
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    strip_accents="unicode",
)
char_vectorizer.fit(train_df["text"])


def build_feature_df(texts, fit_scaler=False):
    text_series = pd.Series(texts)
    lexical = extract_lexical_features(text_series)
    syntactic = extract_syntactic_features(text_series)
    stopword = extract_stopword_patterns(text_series)
    structural = extract_structural_features(text_series)
    content = extract_content_features(text_series, char_vectorizer)
    combined = pd.concat([lexical, syntactic, stopword, structural, content], axis=1)
    return combined


train_features = build_feature_df(X_train_texts)
val_features = build_feature_df(X_val_texts)
test_features = build_feature_df(X_test_internal)
test_data_features = build_feature_df(test_df["text"].values)

# ─── Scale Features ────────────────────────────────────
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)
test_data_features_scaled = scaler.transform(test_data_features)

# ─── Extract Sentence-BERT Embeddings ──────────────────
sbert_model = SentenceTransformer("all-mpnet-base-v2")
sbert_model = sbert_model.to(DEVICE)
sbert_model.eval()

print("Extracting Sentence-BERT embeddings...")
train_embeddings = sbert_model.encode(
    X_train_texts.tolist(),
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=False,
)
val_embeddings = sbert_model.encode(
    X_val_texts.tolist(), convert_to_tensor=True, batch_size=64, show_progress_bar=False
)
test_embeddings = sbert_model.encode(
    X_test_internal.tolist(),
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=False,
)
test_data_embeddings = sbert_model.encode(
    test_df["text"].values.tolist(),
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=False,
)

print(f"Train embeddings shape: {train_embeddings.shape}")


# ─── Define Transformer Encoder Classifier ──────────────
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_heads=4, num_layers=2, num_classes=3, dropout_rate=0.3, feature_dim=None):
        super().__init__()
        # Project input to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # Classification head
        cls_input_dim = hidden_dim
        if feature_dim:
            cls_input_dim += feature_dim
        self.classifier = nn.Linear(cls_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feature_dim = feature_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, features=None):
        # x: (batch, 768) -> expand to (batch, seq_len=1, hidden_dim)
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, hidden_dim)
        # Add CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 2, hidden_dim)
        # Apply transformer
        x = self.transformer(x)
        # Take CLS token output
        x = x[:, 0, :]  # (batch, hidden_dim)
        x = self.layer_norm(x)
        x = self.dropout(x)
        # Optionally concatenate with engineered features
        if features is not None and self.feature_dim:
            x = torch.cat([x, features], dim=1)
        return self.classifier(x)


# Determine the dimension of engineered features (will be set after feature computation)
feature_dim = None  # Will be set after we know the shape

model = TransformerEncoderModel(
    input_dim=EMBEDDING_DIM,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
    feature_dim=feature_dim,
)
model = model.to(DEVICE)
print(f"Transformer model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─── Determine Engineered Feature Dimension ─────────────
# Rebuild features with augmentation to get correct dimension
# (features were already built, but with augmented texts we need to rebuild)
print(f"Engineered feature dimension: {train_features_scaled.shape[1]}")
feature_dim = train_features_scaled.shape[1]

# Re-initialize model with correct feature dimension
model = TransformerEncoderModel(
    input_dim=EMBEDDING_DIM,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
    feature_dim=feature_dim,
)
model = model.to(DEVICE)
print(f"Transformer model with feature_dim={feature_dim}, total params: {sum(p.numel() for p in model.parameters()):,}")

# ─── Loss, Optimizer, Scheduler ─────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20, eta_min=1e-6
)

# ─── Convert all to tensors ──────────────────────────────
train_emb_tensor = (
    train_embeddings.cpu() if train_embeddings.is_cuda else train_embeddings
)
val_emb_tensor = val_embeddings.cpu() if val_embeddings.is_cuda else val_embeddings
test_emb_tensor = test_embeddings.cpu() if test_embeddings.is_cuda else test_embeddings
test_data_emb_tensor = (
    test_data_embeddings.cpu() if test_data_embeddings.is_cuda else test_data_embeddings
)

# Convert feature arrays to tensors
train_features_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features_scaled, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
test_data_features_tensor = torch.tensor(test_data_features_scaled, dtype=torch.float32)

# Create datasets with features
train_dataset = TensorDataset(
    train_emb_tensor, train_features_tensor, torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    val_emb_tensor, val_features_tensor, torch.tensor(y_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    test_emb_tensor, test_features_tensor, torch.tensor(y_test_internal, dtype=torch.long)
)
test_data_dataset = TensorDataset(
    test_data_emb_tensor, test_data_features_tensor
)

# Create DataLoaders
batch_size = 256

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
test_data_loader = DataLoader(
    test_data_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ─── Training Loop ──────────────────────────────────────
num_epochs = 30
best_val_logloss = float("inf")
patience = 5
patience_counter = 0

os.makedirs("./working", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for embeddings, features, labels in train_loader:
        embeddings, features, labels = embeddings.to(DEVICE), features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(embeddings, features=features)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    model.eval()
    val_probs_list, val_labels_list = [], []
    with torch.no_grad():
        for embeddings, features, labels in val_loader:
            embeddings, features = embeddings.to(DEVICE), features.to(DEVICE)
            outputs = model(embeddings, features=features)
            probs = torch.softmax(outputs, dim=1)
            val_probs_list.append(probs.cpu().numpy())
            val_labels_list.append(labels.numpy())
    val_probs = np.concatenate(val_probs_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_logloss = log_loss(val_labels, val_probs_clipped)
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_logloss:.6f}"
    )
    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
        print(f"  ✓ New best model saved (Val LogLoss: {val_logloss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# ─── Load Best Model ────────────────────────────────────
print(f"\nLoading best model with Val LogLoss: {best_val_logloss:.6f}")
model.load_state_dict(torch.load("./working/best_model.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ─── Final Validation Score ─────────────────────────────
val_probs_list = []
with torch.no_grad():
    for embeddings, features, _ in val_loader:
        embeddings, features = embeddings.to(DEVICE), features.to(DEVICE)
        outputs = model(embeddings, features=features)
        probs = torch.softmax(outputs, dim=1)
        val_probs_list.append(probs.cpu().numpy())
val_probs = np.concatenate(val_probs_list, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_logloss = log_loss(y_val, val_probs_clipped)
print(f"Final Validation Score: {final_val_logloss:.6f}")

# ─── Test Inference (Internal) ─────────────────────────
test_probs_list = []
with torch.no_grad():
    for embeddings, features, _ in test_loader:
        embeddings, features = embeddings.to(DEVICE), features.to(DEVICE)
        outputs = model(embeddings, features=features)
        probs = torch.softmax(outputs, dim=1)
        test_probs_list.append(probs.cpu().numpy())
test_probs = np.concatenate(test_probs_list, axis=0)
test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs_clipped = test_probs_clipped / test_probs_clipped.sum(axis=1, keepdims=True)
test_logloss = log_loss(y_test_internal, test_probs_clipped)
print(f"Internal Test LogLoss: {test_logloss:.6f}")

# ─── Generate Submission Predictions ────────────────────
submission_probs_list = []
with torch.no_grad():
    for embeddings, features in test_data_loader:
        embeddings, features = embeddings.to(DEVICE), features.to(DEVICE)
        outputs = model(embeddings, features=features)
        probs = torch.softmax(outputs, dim=1)
        submission_probs_list.append(probs.cpu().numpy())
submission_probs = np.concatenate(submission_probs_list, axis=0)
submission_probs = np.clip(submission_probs, 1e-15, 1 - 1e-15)
submission_probs = submission_probs / submission_probs.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": submission_probs[:, 0],
        "HPL": submission_probs[:, 1],
        "MWS": submission_probs[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"First 5 rows:\n{submission_df.head()}")

# Final required print
print(f"Final Validation Score: {final_val_logloss}")
