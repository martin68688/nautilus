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

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────
NUM_CLASSES = 3
EMBEDDING_DIM = 768
HIDDEN_DIM = 512
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


# ─── Back-Translation Augmentation ──────────────────────
try:
    from transformers import MarianMTModel, MarianTokenizer

    def back_translate(texts, src_lang="en", tgt_lang="fr"):
        """Translate texts to French and back to English using MarianMT."""
        model_name_fr = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        model_name_en = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"

        tokenizer_fr = MarianTokenizer.from_pretrained(model_name_fr)
        model_fr = MarianMTModel.from_pretrained(model_name_fr).to(DEVICE)
        tokenizer_en = MarianTokenizer.from_pretrained(model_name_en)
        model_en = MarianMTModel.from_pretrained(model_name_en).to(DEVICE)

        augmented_texts = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Translate to French
            inputs = tokenizer_fr(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                translated = model_fr.generate(**inputs)
            fr_texts = tokenizer_fr.batch_decode(translated, skip_special_tokens=True)
            # Translate back to English
            inputs = tokenizer_en(fr_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                back_translated = model_en.generate(**inputs)
            en_texts = tokenizer_en.batch_decode(back_translated, skip_special_tokens=True)
            augmented_texts.extend(en_texts)
        return augmented_texts

    print("Performing back-translation augmentation...")
    aug_texts = back_translate(X_train_texts.tolist())
    X_train_aug = np.array(aug_texts)
    y_train_aug = y_train.copy()
    print(f"Augmented training set size: {len(X_train_aug)}")
except Exception as e:
    print(f"Back-translation failed: {e}. Using original data only.")
    X_train_aug = X_train_texts.copy()
    y_train_aug = y_train.copy()

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


train_features = build_feature_df(X_train_aug)
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
    X_train_aug.tolist(),
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


# ─── Define MLP Classifier ──────────────────────────────
class AuthorshipMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


model = AuthorshipMLP(
    input_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
)
model = model.to(DEVICE)

# ─── Loss, Optimizer, Scheduler ─────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# ─── Create DataLoaders ─────────────────────────────────
batch_size = 256

train_emb_tensor = (
    train_embeddings.cpu() if train_embeddings.is_cuda else train_embeddings
)
val_emb_tensor = val_embeddings.cpu() if val_embeddings.is_cuda else val_embeddings
test_emb_tensor = test_embeddings.cpu() if test_embeddings.is_cuda else test_embeddings
test_data_emb_tensor = (
    test_data_embeddings.cpu() if test_data_embeddings.is_cuda else test_data_embeddings
)

train_dataset = TensorDataset(train_emb_tensor, torch.tensor(y_train_aug, dtype=torch.long))
val_dataset = TensorDataset(val_emb_tensor, torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(
    test_emb_tensor, torch.tensor(y_test_internal, dtype=torch.long)
)
test_data_dataset = TensorDataset(test_data_emb_tensor)

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

# ─── Training Loop with Scheduler Step per Batch ────────
def train_model(model, train_loader, val_loader, num_epochs, patience, is_pseudo=False):
    best_val_logloss = float("inf")
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for i, (embeddings, labels) in enumerate(train_loader):
            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + i / len(train_loader))
            total_loss += loss.item()
            num_batches += 1
        avg_train_loss = total_loss / num_batches

        model.eval()
        val_probs_list, val_labels_list = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(DEVICE)
                outputs = model(embeddings)
                probs = torch.softmax(outputs, dim=1)
                val_probs_list.append(probs.cpu().numpy())
                val_labels_list.append(labels.numpy())
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_labels = np.concatenate(val_labels_list, axis=0)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
        val_logloss = log_loss(val_labels, val_probs_clipped)
        current_lr = optimizer.param_groups[0]["lr"]
        tag = " (pseudo)" if is_pseudo else ""
        print(
            f"[{tag}] Epoch {epoch+1:2d}/{num_epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_logloss:.6f}"
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
    return best_val_logloss

num_epochs = 30
patience = 5

os.makedirs("./working", exist_ok=True)

# Phase 1: Initial training
print("Phase 1: Initial training")
best_val_logloss = train_model(model, train_loader, val_loader, num_epochs, patience, is_pseudo=False)

# ─── Load Best Model ────────────────────────────────────
print(f"\nLoading best model with Val LogLoss: {best_val_logloss:.6f}")
model.load_state_dict(torch.load("./working/best_model.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Phase 2: Pseudo-labeling (up to 2 iterations)
print("\nPhase 2: Pseudo-labeling")
for pseudo_iter in range(2):
    # Predict on test data
    model.eval()
    test_probs_list = []
    with torch.no_grad():
        for embeddings, _ in test_loader:
            embeddings = embeddings.to(DEVICE)
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)
            test_probs_list.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs_list, axis=0)

    # Select high confidence samples (max prob > 0.9)
    max_probs = np.max(test_probs, axis=1)
    high_conf_mask = max_probs > 0.9

    if high_conf_mask.sum() == 0:
        print(f"No high-confidence samples found (threshold 0.9). Stopping pseudo-labeling.")
        break

    pseudo_texts = X_test_internal[high_conf_mask]
    pseudo_labels = np.argmax(test_probs[high_conf_mask], axis=1)
    print(f"Pseudo-iteration {pseudo_iter+1}: {high_conf_mask.sum()} high-confidence samples added")

    # Add to training set
    X_train_combined = np.concatenate([X_train_aug, pseudo_texts])
    y_train_combined = np.concatenate([y_train_aug, pseudo_labels])

    # Re-extract features and embeddings for augmented training set
    train_features = build_feature_df(X_train_combined)
    train_features_scaled = scaler.fit_transform(train_features)

    print("Extracting embeddings for pseudo-labeled data...")
    train_embeddings = sbert_model.encode(
        X_train_combined.tolist(),
        convert_to_tensor=True,
        batch_size=64,
        show_progress_bar=False,
    )
    train_emb_tensor = train_embeddings.cpu() if train_embeddings.is_cuda else train_embeddings

    train_dataset = TensorDataset(train_emb_tensor, torch.tensor(y_train_combined, dtype=torch.long))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )

    # Re-initialize model for retraining
    model = AuthorshipMLP(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    )
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_logloss = train_model(model, train_loader, val_loader, num_epochs=20, patience=patience, is_pseudo=True)
    model.load_state_dict(torch.load("./working/best_model.pt", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

# ─── Final Validation Score ─────────────────────────────
val_probs_list = []
with torch.no_grad():
    for embeddings, _ in val_loader:
        embeddings = embeddings.to(DEVICE)
        outputs = model(embeddings)
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
    for embeddings, _ in test_loader:
        embeddings = embeddings.to(DEVICE)
        outputs = model(embeddings)
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
    for (embeddings,) in test_data_loader:
        embeddings = embeddings.to(DEVICE)
        outputs = model(embeddings)
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
