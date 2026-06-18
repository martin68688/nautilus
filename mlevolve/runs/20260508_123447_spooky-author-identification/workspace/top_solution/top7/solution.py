import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import re
import string
import os
import warnings
from scipy.sparse import hstack, save_npz, load_npz

warnings.filterwarnings("ignore")

# ============================================================
# Data Processing and Feature Engineering
# ============================================================


def load_data():
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")
    return train, test


def extract_sentence_stats(text_series):
    features = pd.DataFrame()
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["unique_char_ratio"] = text_series.apply(
        lambda x: len(set(str(x).lower())) / (len(str(x)) + 1)
    )
    features["capital_ratio"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    features["first_word_caps"] = text_series.apply(
        lambda x: 1 if str(x)[0].isupper() else 0
    )
    for punct in [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "..."]:
        features[f"punct_{punct}"] = text_series.str.count(re.escape(punct))
    features["total_punct"] = features[
        [c for c in features.columns if c.startswith("punct_")]
    ].sum(axis=1)
    features["digit_count"] = text_series.str.count(r"\d")
    features["special_chars"] = text_series.apply(
        lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace())
    )
    # NEW: Character n-gram entropy (vocabulary richness)
    features["char_entropy_2"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 2)
    )
    features["char_entropy_3"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 3)
    )
    features["char_entropy_4"] = text_series.apply(
        lambda x: _ngram_entropy(str(x).lower(), 4)
    )
    # NEW: Sentence length distribution features
    features["sent_len_mean"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[0]
    )
    features["sent_len_std"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[1]
    )
    features["sent_len_skew"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[2]
    )
    features["sent_len_kurtosis"] = text_series.apply(
        lambda x: _sent_len_stats(str(x))[3]
    )
    # NEW: Punctuation bigram frequencies
    features["punct_bigram_dash_space"] = text_series.apply(
        lambda x: len(re.findall(r'—\s', str(x)))
    )
    features["punct_bigram_space_dash"] = text_series.apply(
        lambda x: len(re.findall(r'\s—', str(x)))
    )
    # NEW: Paragraph-level features (approximated by double newline counts)
    features["paragraph_count"] = text_series.str.count(r'\n\n')
    features["paragraph_ratio"] = features["paragraph_count"] / (features["word_count"] + 1)
    # NEW: Sentence count
    features["sentence_count"] = text_series.apply(
        lambda x: len(re.findall(r'[.!?]+', str(x)))
    )
    features["words_per_sentence"] = features["word_count"] / (features["sentence_count"] + 1)
    return features


def _ngram_entropy(text, n):
    """Compute entropy of character n-grams in text."""
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams:
        return 0.0
    total = len(ngrams)
    freq = {}
    for ng in ngrams:
        freq[ng] = freq.get(ng, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy


def _sent_len_stats(text):
    """Compute mean, std, skew, kurtosis of word counts per sentence."""
    sentences = re.split(r'[.!?]+', str(text))
    sent_lens = [len(s.split()) for s in sentences if len(s.strip()) > 0]
    if len(sent_lens) < 2:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.array(sent_lens, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr)) + 1e-8
    skew = float(np.mean(((arr - mean) / std) ** 3))
    kurtosis = float(np.mean(((arr - mean) / std) ** 4) - 3.0)
    return (mean, std, skew, kurtosis)


def extract_style_features(text_series):
    features = pd.DataFrame()
    features["syllable_count"] = text_series.apply(
        lambda x: len(re.findall(r"[aeiouy]+", str(x).lower()))
    )
    features["syllables_per_word"] = features["syllable_count"] / (
        text_series.str.split().str.len() + 1
    )
    words_per_sentence = text_series.str.split().str.len()
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * words_per_sentence
        - 84.6 * (features["syllable_count"] / (words_per_sentence + 1))
    )
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
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
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
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
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
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        ]
    )
    features["stopword_ratio"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).lower().split()
            if w.strip(string.punctuation) in stopwords
        )
        / (len(str(x).split()) + 1)
    )
    # NEW: Specific stopword patterns ("the" for Lovecraft, "but yet" for Poe)
    features["stopword_the_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) == "the")
        / (len(str(x).split()) + 1)
    )
    features["stopword_but_yet"] = text_series.str.contains(r'\bbut yet\b', case=False, na=False).astype(int)
    features["stopword_and_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) == "and")
        / (len(str(x).split()) + 1)
    )
    features["stopword_of_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) == "of")
        / (len(str(x).split()) + 1)
    )
    conjunctions = set(
        [
            "and",
            "but",
            "or",
            "yet",
            "so",
            "for",
            "nor",
            "because",
            "although",
            "while",
            "since",
            "unless",
            "if",
            "when",
            "where",
            "whether",
            "after",
            "before",
            "until",
            "once",
            "as",
        ]
    )
    features["conjunction_density"] = text_series.apply(
        lambda x: sum(
            1
            for w in str(x).lower().split()
            if w.strip(string.punctuation) in conjunctions
        )
        / (len(str(x).split()) + 1)
    )
    features["ttr"] = text_series.apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )
    features["long_words_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).split() if len(w) > 8)
        / (len(str(x).split()) + 1)
    )
    # NEW: Archaic/period-specific word flags
    archaic_words = ['thou', 'thee', 'thy', 'thine', 'hath', 'doth', 'art', 'wilt',
                     'canst', 'dost', 'didst', 'hast', 'shalt', 'whence', 'thence',
                     'hither', 'thither', 'wherefore', 'methinks', 'forsooth', 'prithee',
                     'ere', 'whilst', 'betwixt', 'unto', 'thrice', 'nay', 'yea']
    features["archaic_word_count"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split()
                      if w.strip(string.punctuation) in archaic_words)
    )
    word_count_col = text_series.str.split().str.len()
    features["archaic_word_ratio"] = features["archaic_word_count"] / (word_count_col + 1)
    # NEW: Pronoun usage patterns
    first_person = set(['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'])
    second_person = set(['you', 'your', 'yours', 'yourself', 'yourselves'])
    third_person = set(['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'])
    features["first_person_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) in first_person)
        / (len(str(x).split()) + 1)
    )
    features["second_person_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) in second_person)
        / (len(str(x).split()) + 1)
    )
    features["third_person_ratio"] = text_series.apply(
        lambda x: sum(1 for w in str(x).lower().split() if w.strip(string.punctuation) in third_person)
        / (len(str(x).split()) + 1)
    )
    # NEW: Repeated word patterns (stuttering/emphasis)
    features["repeated_words"] = text_series.apply(
        lambda x: len(re.findall(r'\b(\w+)\s+\1\b', str(x).lower()))
    )
    # NEW: Question and exclamation density
    features["question_count"] = text_series.str.count(r'\?')
    features["exclamation_count"] = text_series.str.count(r'!')
    features["quote_count"] = text_series.str.count(r'"')
    return features


def preprocess_data():
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Authors distribution:\n{train['author'].value_counts()}")

    # Save raw text for cross-validation (preprocessing will be done per fold)
    os.makedirs("./working", exist_ok=True)
    np.save("./working/train_texts.npy", train["text"].values)
    np.save("./working/train_authors.npy", train["author"].values)
    np.save("./working/test_texts.npy", test["text"].values)

    le = LabelEncoder()
    y_full_encoded = le.fit_transform(train["author"].values)
    np.save("./working/y_full_encoded.npy", y_full_encoded)
    np.save("./working/author_labels.npy", le.classes_)
    test_ids = test["id"].values
    test_ids_bytes = np.array([s.encode('utf-8') for s in test_ids])
    np.save("./working/test_ids.npy", test_ids_bytes)


# _save_training_state removed per improvement plan (no longer needed)

# generate_pseudo_labels removed per improvement plan


# ============================================================
# Dataset and Model Definition (Transformer backbone)
# ============================================================

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerClassifier(nn.Module):
    def __init__(self, model_name='distilroberta-base', num_labels=3, dropout=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        # Freeze first 4 layers of the transformer
        for i, layer in enumerate(self.transformer.encoder.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False
        hidden_size = self.transformer.config.hidden_size  # 768 for distilroberta-base
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        logits = self.mlp(cls_embedding)
        return logits


# ============================================================
# Training and Evaluation
# ============================================================


# train_epoch, validate, predict functions are now inlined in train_and_evaluate


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full training data and test data
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")
    test_ids_bytes = np.load("./working/test_ids.npy")
    test_ids = np.array([s.decode('utf-8') for s in test_ids_bytes])
    y_full_encoded = np.load("./working/y_full_encoded.npy")

    # Tokenizer
    model_name = 'distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare test dataset (shared across folds)
    test_texts = test["text"].values
    test_dataset = SpookyDataset(test_texts, tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 5-fold stratified cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_test_preds = []
    fold_val_loglosses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train["text"], train["author"])):
        print(f"\n{'='*40}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*40}")

        X_train_fold = train["text"].iloc[train_idx].values
        y_train_fold = y_full_encoded[train_idx]
        X_val_fold = train["text"].iloc[val_idx].values
        y_val_fold = y_full_encoded[val_idx]

        train_dataset = SpookyDataset(X_train_fold, y_train_fold, tokenizer)
        val_dataset = SpookyDataset(X_val_fold, y_val_fold, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        model = TransformerClassifier(model_name=model_name, num_labels=3, dropout=0.3).to(device)

        # Different learning rates for transformer backbone and MLP head
        transformer_params = []
        mlp_params = []
        for name, param in model.named_parameters():
            if 'mlp' in name:
                mlp_params.append(param)
            else:
                transformer_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': transformer_params, 'lr': 2e-5},
            {'params': mlp_params, 'lr': 5e-4},
        ], weight_decay=0.01)

        total_steps = 15 * len(train_loader)
        warmup_steps = int(0.05 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        best_val_logloss = float("inf")
        best_model_state = None
        patience = 5
        patience_counter = 0

        for epoch in range(15):
            model.train()
            total_train_loss = 0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / num_batches

            # Validation
            model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast():
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)

                    probs = torch.softmax(logits, dim=1)
                    all_preds.append(probs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    total_val_loss += loss.item()
                    num_val_batches += 1

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            eps = 1e-15
            all_preds = np.clip(all_preds, eps, 1 - eps)
            all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)
            n = len(all_labels)
            log_loss_val = 0.0
            for i in range(n):
                log_loss_val += np.log(all_preds[i, all_labels[i]])
            log_loss_val = -log_loss_val / n

            avg_val_loss = total_val_loss / num_val_batches

            print(f"Fold {fold+1} Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {log_loss_val:.4f}")

            if log_loss_val < best_val_logloss:
                best_val_logloss = log_loss_val
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Fold {fold+1} early stopping at epoch {epoch+1}")
                    break

        # Load best model for this fold
        model.load_state_dict(best_model_state)
        fold_val_loglosses.append(best_val_logloss)

        # Predict on test set
        model.eval()
        fold_test_pred = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                with autocast():
                    logits = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                fold_test_pred.append(probs.cpu().numpy())
        fold_test_pred = np.concatenate(fold_test_pred, axis=0)
        fold_test_preds.append(fold_test_pred)

        print(f"Fold {fold+1} Best Val LogLoss: {best_val_logloss:.4f}")

    # Average test predictions across folds
    test_preds = np.mean(fold_test_preds, axis=0)

    eps = 1e-15
    test_preds = np.clip(test_preds, eps, 1 - eps)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "EAP": test_preds[:, 0],
            "HPL": test_preds[:, 1],
            "MWS": test_preds[:, 2],
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"\nAverage 5-fold Validation LogLoss: {np.mean(fold_val_loglosses):.4f}")


if __name__ == "__main__":
    preprocess_data()
    train_and_evaluate()