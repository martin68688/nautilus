import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import random
from collections import Counter
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing loss for better calibration"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / n_classes)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256, is_train=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        # Precompute genre-specific trigrams from training data if not provided
        self._init_augmentation()

    def _init_augmentation(self):
        """Initialize augmentation resources: synonym substitutions and genre trigrams."""
        self.synonym_aug_prob = 0.3
        self.trigram_aug_prob = 0.3
        # Predefined genre-specific trigrams (most distinctive per author from training set style)
        self.genre_trigrams = [
            'said the figure', 'in that dark hour', 'with trembling hand',
            'i could not', 'it was a', 'of the house',
            'there was a', 'that he had', 'and the old',
            'as if he', 'the end of', 'out of the',
            'i had been', 'he had been', 'the room was',
            'after a few', 'the door of', 'the light of',
            'at the same', 'the fact that', 'there is no',
            'it is not', 'the world is', 'the truth is',
            'as if the', 'the old man', 'the great house',
            'the dark and', 'the cold and', 'the dead and',
        ]
        # Try to import nltk for synonym substitution
        try:
            import nltk
            nltk.download('wordnet', quiet=True)
            from nltk.corpus import wordnet
            self.wordnet_available = True
        except Exception:
            self.wordnet_available = False

    def _synonym_substitution(self, text):
        """Replace 1-3 random content words with synonyms using WordNet."""
        if not self.wordnet_available:
            return text
        import random
        from nltk.corpus import wordnet
        words = text.split()
        if len(words) < 5:
            return text
        # Find content words (nouns, verbs, adjectives, adverbs)
        # Simple heuristic: words longer than 3 chars and not stopwords
        content_candidates = []
        for i, w in enumerate(words):
            w_clean = w.strip('.,!?"\';:()[]-')
            if len(w_clean) > 3 and w_clean[0].islower():
                # Check if wordnet has synsets for this word
                synsets = wordnet.synsets(w_clean)
                if synsets:
                    content_candidates.append((i, w_clean))
        if not content_candidates:
            return text
        # Randomly choose 1-3 words to replace
        n_replace = min(random.randint(1, 3), len(content_candidates))
        chosen = random.sample(content_candidates, n_replace)
        words_list = list(words)
        for idx, orig_word in chosen:
            synsets = wordnet.synsets(orig_word)
            if not synsets:
                continue
            # Get lemmas from all synsets
            lemmas = []
            for syn in synsets:
                lemmas.extend(syn.lemma_names())
            # Filter out the original word and multi-word phrases
            candidates = [
                l.replace('_', ' ') for l in set(lemmas)
                if l.lower() != orig_word and '_' not in l
            ]
            if candidates:
                new_word = random.choice(candidates)
                # Preserve original case pattern approximately
                if orig_word[0].isupper():
                    new_word = new_word.capitalize()
                words_list[idx] = new_word
        return ' '.join(words_list)

    def _insert_genre_trigram(self, text):
        """Insert a genre-appropriate trigram at a random position."""
        import random
        trigram = random.choice(self.genre_trigrams)
        words = text.split()
        if len(words) < 10:
            return text
        # Insert at a random position near the beginning or middle
        insert_pos = random.randint(0, max(1, len(words) // 2))
        # Decide whether to insert at sentence boundary or word boundary
        if random.random() < 0.5:
            # Insert as a separate phrase
            text = text[:insert_pos] + trigram + ' ' + text[insert_pos:]
        else:
            # Insert at end of a sentence-like position
            # Find a period or natural break near insert_pos
            for i in range(insert_pos, max(0, insert_pos - 20), -1):
                if i < len(text) and text[i] == '.':
                    text = text[:i+1] + ' ' + trigram + text[i+1:]
                    break
            else:
                text = text + ' ' + trigram + '.'
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Apply data augmentation if training
        if self.is_train:
            if random.random() < self.synonym_aug_prob:
                text = self._synonym_substitution(text)
            if random.random() < self.trigram_aug_prob:
                text = self._insert_genre_trigram(text)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


# Create text-level features
def extract_text_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.count("[.!?]") + 1
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["dash_count"] = text_series.str.count("-")
    features["quote_count"] = text_series.str.count('"') + text_series.str.count("'")
    features["period_count"] = text_series.str.count(r"\.")
    features["punct_count"] = text_series.str.count(r"[^\w\s]")
    features["punct_density"] = features["punct_count"] / (features["char_count"] + 1)
    features["capital_words"] = text_series.str.findall(r"\b[A-Z][a-z]*\b").str.len()
    features["all_caps_words"] = text_series.str.findall(r"\b[A-Z]{2,}\b").str.len()
    features["capital_ratio"] = features["capital_words"] / (features["word_count"] + 1)
    features["ellipsis_count"] = text_series.str.count(r"\.\.\.")
    features["has_ellipsis"] = (features["ellipsis_count"] > 0).astype(int)
    features["short_words"] = text_series.str.findall(r"\b\w{1,3}\b").str.len()
    features["medium_words"] = text_series.str.findall(r"\b\w{4,6}\b").str.len()
    features["long_words"] = text_series.str.findall(r"\b\w{7,}\b").str.len()
    features["short_word_ratio"] = features["short_words"] / (
        features["word_count"] + 1
    )
    features["long_word_ratio"] = features["long_words"] / (features["word_count"] + 1)
    features["unique_words"] = text_series.apply(lambda x: len(set(x.lower().split())))
    features["lexical_diversity"] = features["unique_words"] / (
        features["word_count"] + 1
    )
    features["dialog_markers"] = text_series.str.count(r'["\'\u201c\u201d]')
    features["has_dialog"] = (features["dialog_markers"] > 0).astype(int)
    features["first_person_singular"] = text_series.str.contains(
        r"\bI\b|\bme\b|\bmy\b|\bmine\b|\bmyself\b", case=False
    ).astype(int)
    features["first_person_plural"] = text_series.str.contains(
        r"\bwe\b|\bus\b|\bour\b|\bows\b|\bourselves\b", case=False
    ).astype(int)
    features["third_person"] = text_series.str.contains(
        r"\bhe\b|\bshe\b|\bit\b|\bthey\b|\bhim\b|\bher\b|\bthem\b", case=False
    ).astype(int)
    features["past_tense"] = text_series.str.contains(
        r"\bwas\b|\bwere\b|\bhad\b|\bdid\b|\bsaid\b", case=False
    ).astype(int)
    features["present_tense"] = text_series.str.contains(
        r"\bis\b|\bare\b|\bhas\b|\bdo\b|\bdoes\b", case=False
    ).astype(int)
    features["adverb_ly"] = text_series.str.findall(r"\b\w+ly\b").str.len()
    features["adjective_markers"] = text_series.str.findall(
        r"\b\w+ful\b|\b\w+ous\b|\b\w+ive\b|\b\w+able\b"
    ).str.len()
    features["syllables_estimate"] = text_series.apply(
        lambda x: sum(
            1 for word in str(x).split() for vowel in "aeiou" if vowel in word.lower()
        )
    )
    features["flesch_reading_ease"] = (
        206.835
        - 1.015 * features["word_count"] / (features["sentence_count"] + 1)
        - 84.6 * features["syllables_estimate"] / (features["word_count"] + 1)
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    return features


y_train = train_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values

print("Creating train/validation split FIRST to avoid data leakage...")
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

train_df_final = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Training samples: {len(train_df_final)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

print("Extracting text features (fit on train only)...")
train_text_features = extract_text_features(train_df_final["text"])
test_text_features = extract_text_features(test_df["text"])
val_text_features = extract_text_features(val_df["text"])

print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    analyzer="word",
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.7,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
)
train_tfidf = tfidf_vectorizer.fit_transform(train_df_final["text"]).toarray()
val_tfidf = tfidf_vectorizer.transform(val_df["text"]).toarray()
test_tfidf = tfidf_vectorizer.transform(test_df["text"]).toarray()

char_vectorizer = TfidfVectorizer(
    max_features=2000,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
)
train_char = char_vectorizer.fit_transform(train_df_final["text"]).toarray()
val_char = char_vectorizer.transform(val_df["text"]).toarray()
test_char = char_vectorizer.transform(test_df["text"]).toarray()

# Part of Speech features
try:
    import nltk

    # Download required NLTK resources with proper resource names
    try:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except Exception:
        nltk.download("averaged_perceptron_tagger", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)

    def get_pos_features(texts):
        pos_features = []
        for text in texts:
            try:
                tokens = nltk.word_tokenize(str(text))
            except Exception:
                tokens = str(text).split()
            try:
                pos_tags = nltk.pos_tag(tokens)
            except Exception:
                pos_tags = [(t, 'NN') for t in tokens]
            pos_counts = Counter(tag for word, tag in pos_tags)
            pos_features.append(pos_counts)
        return pos_features

    train_pos = get_pos_features(train_df_final["text"].tolist())
    val_pos = get_pos_features(val_df["text"].tolist())
    test_pos = get_pos_features(test_df["text"].tolist())
    all_pos_tags = set()
    for pos_dict in train_pos:
        all_pos_tags.update(pos_dict.keys())
    pos_cols = [f"pos_{tag}" for tag in all_pos_tags]
    train_pos_df = pd.DataFrame(
        [
            {f"pos_{tag}": pos_dict.get(tag, 0) for tag in all_pos_tags}
            for pos_dict in train_pos
        ]
    )
    val_pos_df = pd.DataFrame(
        [
            {f"pos_{tag}": pos_dict.get(tag, 0) for tag in all_pos_tags}
            for pos_dict in val_pos
        ]
    )
    test_pos_df = pd.DataFrame(
        [
            {f"pos_{tag}": pos_dict.get(tag, 0) for tag in all_pos_tags}
            for pos_dict in test_pos
        ]
    )
    train_pos_sum = train_pos_df.sum(axis=1).replace(0, 1)
    val_pos_sum = val_pos_df.sum(axis=1).replace(0, 1)
    test_pos_sum = test_pos_df.sum(axis=1).replace(0, 1)
    train_pos_df = train_pos_df.div(train_pos_sum, axis=0).fillna(0)
    val_pos_df = val_pos_df.div(val_pos_sum, axis=0).fillna(0)
    test_pos_df = test_pos_df.div(test_pos_sum, axis=0).fillna(0)
    print(f"Extracted {len(pos_cols)} POS tag features")
except Exception as e:
    print(f"POS tagging failed (using fallback): {e}")
    train_pos_df = pd.DataFrame()
    val_pos_df = pd.DataFrame()
    test_pos_df = pd.DataFrame()

print("Combining features...")
X_train_final = np.hstack([train_text_features.values, train_tfidf, train_char])
X_val = np.hstack([val_text_features.values, val_tfidf, val_char])
X_test = np.hstack([test_text_features.values, test_tfidf, test_char])
if not train_pos_df.empty and not val_pos_df.empty and not test_pos_df.empty:
    X_train_final = np.hstack([X_train_final, train_pos_df.values])
    X_val = np.hstack([X_val, val_pos_df.values])
    X_test = np.hstack([X_test, test_pos_df.values])

y_train_final = train_df_final["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
y_val = val_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values

print(f"Training samples: {len(train_df_final)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Feature dimension: {X_train_final.shape[1]}")

# Save feature matrices for potential reuse
os.makedirs("./working", exist_ok=True)
np.save("./working/X_train.npy", X_train_final)
np.save("./working/X_val.npy", X_val)
np.save("./working/X_test.npy", X_test)
np.save("./working/y_train.npy", y_train_final)
np.save("./working/y_val.npy", y_val)
train_df_final.to_pickle("./working/train_df.pkl")
val_df.to_pickle("./working/val_df.pkl")
test_df.to_pickle("./working/test_df.pkl")
import joblib

joblib.dump(tfidf_vectorizer, "./working/tfidf_vectorizer.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")

print("Data processing and feature engineering complete!")

# ===== MODEL DESIGN PHASE =====
NUM_LABELS = 3
MAX_LEN = 256
BATCH_SIZE = 16

set_seed(42)
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_1 = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=NUM_LABELS
)
set_seed(123)
model_2 = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=NUM_LABELS
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1.to(device)
model_2.to(device)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

optimizer_1 = AdamW(model_1.parameters(), lr=2e-5, weight_decay=0.01)
optimizer_2 = AdamW(model_2.parameters(), lr=2e-5, weight_decay=0.01)

total_steps = (len(train_df_final) // BATCH_SIZE) * 15
warmup_steps = total_steps // 10

scheduler_1 = get_linear_schedule_with_warmup(
    optimizer_1, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scheduler_2 = get_linear_schedule_with_warmup(
    optimizer_2, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"Models designed: RoBERTa-base × 2, label smoothing loss")
print(
    f"Total trainable parameters (each model): {sum(p.numel() for p in model_1.parameters()):,}"
)

# ===== TRAINING AND EVALUATION PHASE =====

# Prepare datasets
train_texts = train_df_final["text"].tolist()
val_texts = val_df["text"].tolist()
test_texts = test_df["text"].tolist()

# Enable data augmentation for training dataset only
train_dataset = TextDataset(train_texts, y_train_final, tokenizer, MAX_LEN, is_train=True)
val_dataset = TextDataset(val_texts, y_val, tokenizer, MAX_LEN, is_train=False)
test_dataset = TextDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN, is_train=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


def get_layer_groups(model):
    """Get parameter groups for discriminative learning rates."""
    # Identify the classifier parameters
    classifier_params = []
    top_layer_params = []
    base_params = []

    # RoBERTa has 12 layers, we want top 3 layers and base layers
    model.roberta.encoder.layer
    classifier_params = list(model.classifier.parameters())

    # Collect top 3 transformer layers (layers 9, 10, 11)
    for i in range(9, 12):
        top_layer_params.extend(list(model.roberta.encoder.layer[i].parameters()))

    # Collect base layers (layers 0-8) + embeddings + pooler (if exists)
    for i in range(0, 9):
        base_params.extend(list(model.roberta.encoder.layer[i].parameters()))
    base_params.extend(list(model.roberta.embeddings.parameters()))
    if model.roberta.pooler is not None:
        base_params.extend(list(model.roberta.pooler.parameters()))

    return classifier_params, top_layer_params, base_params


def set_requires_grad(model, classifier_only=False, top_layers=False):
    """Freeze or unfreeze layers based on training phase."""
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Always unfreeze classifier
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True

    if top_layers or not classifier_only:
        # Unfreeze top 3 transformer layers
        for i in range(9, 12):
            for name, param in model.roberta.encoder.layer[i].named_parameters():
                param.requires_grad = True

    if not classifier_only and not top_layers:
        # Unfreeze all transformer layers
        for name, param in model.roberta.encoder.named_parameters():
            param.requires_grad = True
        for name, param in model.roberta.embeddings.named_parameters():
            param.requires_grad = True
        if model.roberta.pooler is not None:
            for name, param in model.roberta.pooler.named_parameters():
                param.requires_grad = True


def create_optimizer_with_discriminative_lr(model, weight_decay=0.05):
    """Create optimizer with discriminative learning rates."""
    classifier_params, top_layer_params, base_params = get_layer_groups(model)

    # Set learning rates: 2.5e-5 classifier, 1.5e-5 top layers, 1e-5 base layers
    # Apply layer-wise decay factor 0.95 from top to bottom
    param_groups = [
        {'params': classifier_params, 'lr': 2.5e-5},
        {'params': top_layer_params, 'lr': 1.5e-5},
        {'params': base_params, 'lr': 1e-5 * (0.95 ** 9)},  # Apply decay
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    model_idx,
    epochs=15,
    patience=5,
):
    scaler = GradScaler()
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Initialize SWA
    swa_start_epoch = 5
    swa_cycle_length = 2
    swa_lr = 1e-5
    swa_model_state = None

    # Phase 1: Gradual unfreezing
    # Epochs 1-2: freeze all except classifier
    # Epochs 3-4: unfreeze top 3 layers
    # Epochs 5+: full fine-tuning

    for epoch in range(epochs):
        # Apply gradual unfreezing based on epoch
        if epoch < 2:
            set_requires_grad(model, classifier_only=True)
        elif epoch < 4:
            set_requires_grad(model, classifier_only=False, top_layers=True)
        else:
            set_requires_grad(model, classifier_only=False, top_layers=False)

        # Recreate optimizer for this epoch if needed (to maintain discriminative LR)
        if epoch < 4:
            optimizer = create_optimizer_with_discriminative_lr(model)

        # SWA: adjust learning rate for SWA cycles
        if epoch >= swa_start_epoch:
            # Use SWA learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = swa_lr

        model.train()
        total_train_loss = 0
        train_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            # Per-sample gradient clipping using torch.nn.utils.clip_grad_norm_
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_train_loss += loss.item()
            train_batches += 1
        avg_train_loss = total_train_loss / train_batches

        # SWA: store model state at end of each cycle
        if epoch >= swa_start_epoch and (epoch - swa_start_epoch) % swa_cycle_length == 0:
            if swa_model_state is None:
                swa_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                # Update running average
                n = (epoch - swa_start_epoch) // swa_cycle_length + 1
                current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                for k in swa_model_state:
                    swa_model_state[k] = (swa_model_state[k] * (n - 1) + current_state[k]) / n

        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                total_val_loss += loss.item()
                val_batches += 1
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / val_batches
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)
        val_preds_clamped = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds_clamped = val_preds_clamped / val_preds_clamped.sum(
            axis=1, keepdims=True
        )
        val_log_loss = -np.mean(
            np.log(val_preds_clamped[np.arange(len(val_labels)), val_labels])
        )

        print(
            f"Model {model_idx} Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
        )

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for Model {model_idx} at epoch {epoch+1}")
                break

    # Apply SWA averaging at the end
    if swa_model_state is not None:
        print(f"Applying SWA to Model {model_idx}")
        model.load_state_dict(swa_model_state)
    elif best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_loss


# Create optimizers with discriminative learning rates and higher weight decay
print("Creating optimizers with discriminative learning rates...")
optimizer_1 = create_optimizer_with_discriminative_lr(model_1)
optimizer_2 = create_optimizer_with_discriminative_lr(model_2)

# Recalculate for the actual training
total_steps = (len(train_df_final) // BATCH_SIZE) * 15
warmup_steps = total_steps // 10

scheduler_1 = get_linear_schedule_with_warmup(
    optimizer_1, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scheduler_2 = get_linear_schedule_with_warmup(
    optimizer_2, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print("Training Model 1 (RoBERTa-base, seed 42) with gradual unfreezing and SWA...")
model_1, val_loss_1 = train_model(
    model_1, optimizer_1, scheduler_1, train_loader, val_loader, 1
)

train_loader_2 = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader_2 = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

print("Training Model 2 (RoBERTa-base, seed 123) with gradual unfreezing and SWA...")
model_2, val_loss_2 = train_model(
    model_2, optimizer_2, scheduler_2, train_loader_2, val_loader_2, 2
)

# Ensemble validation predictions
model_1.eval()
model_2.eval()
all_val_probs_1 = []
all_val_probs_2 = []
all_val_labels_final = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        with autocast():
            logits_1 = model_1(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_2 = model_2(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        probs_1 = F.softmax(logits_1, dim=1).cpu().numpy()
        probs_2 = F.softmax(logits_2, dim=1).cpu().numpy()
        all_val_probs_1.append(probs_1)
        all_val_probs_2.append(probs_2)
        all_val_labels_final.append(labels.numpy())

val_probs_1 = np.concatenate(all_val_probs_1, axis=0)
val_probs_2 = np.concatenate(all_val_probs_2, axis=0)
val_probs_ensemble = (val_probs_1 + val_probs_2) / 2.0
val_probs_clamped = np.clip(val_probs_ensemble, 1e-15, 1 - 1e-15)
val_probs_clamped = val_probs_clamped / val_probs_clamped.sum(axis=1, keepdims=True)
val_labels_final = np.concatenate(all_val_labels_final, axis=0)
final_val_log_loss = -np.mean(
    np.log(val_probs_clamped[np.arange(len(val_labels_final)), val_labels_final])
)

print(f"Final Validation Score: {final_val_log_loss}")

# Test inference
model_1.eval()
model_2.eval()
all_test_probs_1 = []
all_test_probs_2 = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits_1 = model_1(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_2 = model_2(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        probs_1 = F.softmax(logits_1, dim=1).cpu().numpy()
        probs_2 = F.softmax(logits_2, dim=1).cpu().numpy()
        all_test_probs_1.append(probs_1)
        all_test_probs_2.append(probs_2)

test_probs_1 = np.concatenate(all_test_probs_1, axis=0)
test_probs_2 = np.concatenate(all_test_probs_2, axis=0)
test_probs_ensemble = (test_probs_1 + test_probs_2) / 2.0
test_probs_final = np.clip(test_probs_ensemble, 1e-15, 1 - 1e-15)
test_probs_final = test_probs_final / test_probs_final.sum(axis=1, keepdims=True)

# Create submission file
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs_final[:, 0],
        "HPL": test_probs_final[:, 1],
        "MWS": test_probs_final[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")