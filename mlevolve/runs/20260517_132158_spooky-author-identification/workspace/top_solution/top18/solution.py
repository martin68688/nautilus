import pandas as pd
import numpy as np
import re
import string
import pickle
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, ModernBertForSequenceClassification
import xgboost as xgb

warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"] = test_df["text"].apply(clean_text)

# ============================================================
# 3. LEXICAL FEATURES
# ============================================================
def extract_lexical_features(text):
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    n_words = len(words)
    n_chars = len(text)
    n_sentences = len(sentences)
    n_unique_words = len(set(words))
    avg_word_len = np.mean([len(w) for w in words]) if n_words > 0 else 0
    long_words_ratio = sum(1 for w in words if len(w) > 6) / max(n_words, 1)
    very_long_words_ratio = sum(1 for w in words if len(w) > 10) / max(n_words, 1)
    type_token_ratio = n_unique_words / max(n_words, 1)
    hapax_legomena_ratio = sum(1 for w in words if words.count(w) == 1) / max(
        n_words, 1
    )
    char_per_word = n_chars / max(n_words, 1)
    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "n_sentences": n_sentences,
        "n_unique_words": n_unique_words,
        "avg_word_len": avg_word_len,
        "long_words_ratio": long_words_ratio,
        "very_long_words_ratio": very_long_words_ratio,
        "type_token_ratio": type_token_ratio,
        "hapax_legomena_ratio": hapax_legomena_ratio,
        "char_per_word": char_per_word,
    }

lex_train = train_df["text_clean"].apply(
    lambda x: pd.Series(extract_lexical_features(x))
)
lex_test = test_df["text_clean"].apply(lambda x: pd.Series(extract_lexical_features(x)))

# ============================================================
# 4. SYNTACTIC FEATURES
# ============================================================
def extract_syntactic_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    std_sent_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    max_sent_len = max(sent_lengths) if sent_lengths else 0
    min_sent_len = min(sent_lengths) if sent_lengths else 0
    pos_tags = pos_tag(words)
    pos_counts = {}
    for _, tag in pos_tags:
        coarse_tag = tag[:2]
        pos_counts[coarse_tag] = pos_counts.get(coarse_tag, 0) + 1
    n_words = len(words)
    noun_ratio = sum(
        pos_counts.get(t, 0) for t in ["NN", "NN", "NN", "NN", "NN"]
    ) / max(n_words, 1)
    verb_ratio = sum(
        pos_counts.get(t, 0) for t in ["VB", "VB", "VB", "VB", "VB"]
    ) / max(n_words, 1)
    adj_ratio = sum(pos_counts.get(t, 0) for t in ["JJ", "JJ", "JJ"]) / max(n_words, 1)
    adv_ratio = sum(pos_counts.get(t, 0) for t in ["RB", "RB", "RB", "RB"]) / max(
        n_words, 1
    )
    pronoun_count = sum(1 for _, tag in pos_tags if tag == "PRP" or tag == "PRP$")
    pronoun_ratio = pronoun_count / max(n_words, 1)
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_ratio = punct_count / max(len(text), 1)
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    exclamation_count = text.count("!")
    question_count = text.count("?")
    quote_count = text.count('"') + text.count("'")
    dash_count = text.count("-")
    n_chars = len(text)
    punct_features = {
        "comma_ratio": comma_count / max(n_chars, 1),
        "semicolon_ratio": semicolon_count / max(n_chars, 1),
        "colon_ratio": colon_count / max(n_chars, 1),
        "exclamation_ratio": exclamation_count / max(n_chars, 1),
        "question_ratio": question_count / max(n_chars, 1),
        "quote_ratio": quote_count / max(n_chars, 1),
        "dash_ratio": dash_count / max(n_chars, 1),
    }
    return {
        "avg_sent_len": avg_sent_len,
        "std_sent_len": std_sent_len,
        "max_sent_len": max_sent_len,
        "min_sent_len": min_sent_len,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio,
        "pronoun_ratio": pronoun_ratio,
        "punct_density": punct_ratio,
        **punct_features,
    }

syn_train = train_df["text_clean"].apply(
    lambda x: pd.Series(extract_syntactic_features(x))
)
syn_test = test_df["text_clean"].apply(
    lambda x: pd.Series(extract_syntactic_features(x))
)

# ============================================================
# 5. READABILITY & COMPLEXITY FEATURES
# ============================================================
def extract_readability_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    n_words = len(words)
    n_sentences = len(sentences)
    n_chars = len(text)

    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    total_syllables = sum(count_syllables(w) for w in words)
    if n_sentences > 0 and n_words > 0:
        avg_words_per_sent = n_words / n_sentences
        avg_syllables_per_word = total_syllables / n_words
        flesch = 206.835 - 1.015 * avg_words_per_sent - 84.6 * avg_syllables_per_word
    else:
        flesch = 0
    if n_sentences > 0 and n_words > 0:
        avg_chars_per_word = n_chars / n_words
        ari = 4.71 * avg_chars_per_word + 0.5 * (n_words / n_sentences) - 21.43
    else:
        ari = 0
    if n_sentences > 0 and n_words > 0:
        L = (n_chars / n_words) * 100
        S = (n_sentences / n_words) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
    else:
        coleman_liau = 0
    poly_syllable_words = sum(1 for w in words if count_syllables(w) >= 3)
    if n_sentences >= 3:
        smog = 1.043 * np.sqrt(poly_syllable_words * (30 / n_sentences)) + 3.1291
    else:
        smog = 0
    return {
        "flesch_reading_ease": flesch,
        "automated_readability_index": ari,
        "coleman_liau_index": coleman_liau,
        "smog_index": smog,
        "avg_words_per_sentence": avg_words_per_sent if n_sentences > 0 else 0,
        "poly_syllable_word_ratio": poly_syllable_words / max(n_words, 1),
        "syllables_per_word": total_syllables / max(n_words, 1),
    }

read_train = train_df["text_clean"].apply(
    lambda x: pd.Series(extract_readability_features(x))
)
read_test = test_df["text_clean"].apply(
    lambda x: pd.Series(extract_readability_features(x))
)

# ============================================================
# 6. SENTIMENT FEATURES
# ============================================================
def extract_sentiment_features(text):
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        return {
            "vader_neg": sentiment["neg"],
            "vader_neu": sentiment["neu"],
            "vader_pos": sentiment["pos"],
            "vader_compound": sentiment["compound"],
        }
    except:
        return {"vader_neg": 0, "vader_neu": 1, "vader_pos": 0, "vader_compound": 0}

sent_train = train_df["text_clean"].apply(
    lambda x: pd.Series(extract_sentiment_features(x))
)
sent_test = test_df["text_clean"].apply(
    lambda x: pd.Series(extract_sentiment_features(x))
)

# ============================================================
# 7. TF-IDF FEATURES
# ============================================================
tfidf_char = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=5000, sublinear_tf=True
)
char_train = tfidf_char.fit_transform(train_df["text_clean"].values)
char_test = tfidf_char.transform(test_df["text_clean"].values)

char_train_df = pd.DataFrame(
    char_train.toarray(),
    columns=[f"char_tfidf_{i}" for i in range(char_train.shape[1])],
)
char_test_df = pd.DataFrame(
    char_test.toarray(), columns=[f"char_tfidf_{i}" for i in range(char_test.shape[1])]
)

tfidf_word = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 3), max_features=3000, sublinear_tf=True, min_df=5
)
word_train = tfidf_word.fit_transform(train_df["text_clean"].values)
word_test = tfidf_word.transform(test_df["text_clean"].values)

word_train_df = pd.DataFrame(
    word_train.toarray(),
    columns=[f"word_tfidf_{i}" for i in range(word_train.shape[1])],
)
word_test_df = pd.DataFrame(
    word_test.toarray(), columns=[f"word_tfidf_{i}" for i in range(word_test.shape[1])]
)

# ============================================================
# 8. COMBINE ALL FEATURES
# ============================================================
X_train_feats = pd.concat(
    [
        lex_train.reset_index(drop=True),
        syn_train.reset_index(drop=True),
        read_train.reset_index(drop=True),
        sent_train.reset_index(drop=True),
        char_train_df.reset_index(drop=True),
        word_train_df.reset_index(drop=True),
    ],
    axis=1,
)

X_test_feats = pd.concat(
    [
        lex_test.reset_index(drop=True),
        syn_test.reset_index(drop=True),
        read_test.reset_index(drop=True),
        sent_test.reset_index(drop=True),
        char_test_df.reset_index(drop=True),
        word_test_df.reset_index(drop=True),
    ],
    axis=1,
)

le = LabelEncoder()
y_train_encoded = le.fit_transform(train_df["author"])
class_names = le.classes_
print(f"Training features shape: {X_train_feats.shape}")
print(f"Test features shape: {X_test_feats.shape}")
print(f"Class labels: {class_names}")

# ============================================================
# 9. FEATURE SCALING
# ============================================================
numeric_cols = [
    "n_words",
    "n_chars",
    "n_sentences",
    "n_unique_words",
    "avg_word_len",
    "long_words_ratio",
    "type_token_ratio",
    "hapax_legomena_ratio",
    "char_per_word",
    "avg_sent_len",
    "std_sent_len",
    "max_sent_len",
    "min_sent_len",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "pronoun_ratio",
    "punct_density",
    "comma_ratio",
    "semicolon_ratio",
    "colon_ratio",
    "exclamation_ratio",
    "question_ratio",
    "quote_ratio",
    "dash_ratio",
    "flesch_reading_ease",
    "automated_readability_index",
    "coleman_liau_index",
    "smog_index",
    "avg_words_per_sentence",
    "poly_syllable_word_ratio",
    "syllables_per_word",
    "vader_neg",
    "vader_neu",
    "vader_pos",
    "vader_compound",
]
cols_to_scale = [col for col in numeric_cols if col in X_train_feats.columns]
scaler = StandardScaler()
X_train_scaled = X_train_feats.copy()
X_test_scaled = X_test_feats.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_feats[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test_feats[cols_to_scale])

# ============================================================
# 10. CREATE TRAIN/VALIDATION SPLITS (Stratified) - CORRECT INDEX HANDLING
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train_scaled, y_train_encoded))

# CORRECT: Use split indices directly for original data
train_texts = train_df["text"].values[train_idx]
train_labels = y_train_encoded[train_idx]
val_texts = train_df["text"].values[val_idx]
val_labels = y_train_encoded[val_idx]

# Also split engineered features using the same original indices - no reset_index
X_train_feats_split = X_train_scaled.iloc[train_idx].values
X_val_feats_split = X_train_scaled.iloc[val_idx].values

assert len(set(train_idx) & set(val_idx)) == 0, "DATA LEAKAGE DETECTED!"
print(f"Train size: {len(X_train_feats_split)}, Val size: {len(X_val_feats_split)}")

# ============================================================
# 11. PREPARE MODERNBERT DATASET & DATALOADERS
# ============================================================
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
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
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_df["text"].values, None, tokenizer)

batch_size = 16
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# ============================================================
# 12. FINE-TUNE MODERNBERT FOR SEQUENCE CLASSIFICATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ModernBertForSequenceClassification - with increased dropout & layer-wise decay
model = ModernBertForSequenceClassification.from_pretrained(model_id, num_labels=3)

# Increase dropout probabilities from default 0.1 to 0.2
model.config.hidden_dropout_prob = 0.2
model.config.attention_probs_dropout_prob = 0.2

model = model.to(device)

# Layer-wise learning rate decay: 0.95 per layer from top to bottom
# Group parameters: classifier head (top), then encoder layers (last to first)
decay_factor = 0.95
base_lr = 5e-5

# Collect parameter groups with their layer indices for naming
# ModernBERT uses "model.layers" for encoder layers
param_groups = []
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

# Classifier head (top layer) - gets full base_lr
classifier_params = []
for n, p in model.named_parameters():
    if "classifier" in n:
        classifier_params.append(p)
param_groups.append({"params": classifier_params, "lr": base_lr, "weight_decay": 0.01})

# Encoder layers: group by layer index, apply decay
# ModernBERT encoder layers are named as "model.layers.{i}"
import re
encoder_layers = {}
for n, p in model.named_parameters():
    if "model.layers" in n:
        layer_match = re.search(r'model\.layers\.(\d+)', n)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            if layer_idx not in encoder_layers:
                encoder_layers[layer_idx] = {"params": [], "decay": [], "no_decay": []}
            if any(nd in n for nd in no_decay):
                encoder_layers[layer_idx]["no_decay"].append(p)
            else:
                encoder_layers[layer_idx]["decay"].append(p)
    elif "model." in n and "layers" not in n and "classifier" not in n:
        # Other base model parameters (embeddings, pooler etc.) - lowest layers
        if any(nd in n for nd in no_decay):
            param_groups.append({"params": [p], "lr": base_lr * (decay_factor ** 10), "weight_decay": 0.0})
        else:
            param_groups.append({"params": [p], "lr": base_lr * (decay_factor ** 10), "weight_decay": 0.01})

# Add encoder layers with decay: top layer gets least decay (factor^1), bottom gets most (factor^num_layers)
num_layers = len(encoder_layers)
for layer_idx in sorted(encoder_layers.keys(), reverse=True):
    reverse_depth = num_layers - layer_idx  # top layers have smaller reverse_depth
    layer_lr = base_lr * (decay_factor ** (reverse_depth + 1))
    group_decay = encoder_layers[layer_idx]["decay"]
    group_no_decay = encoder_layers[layer_idx]["no_decay"]
    if group_decay:
        param_groups.append({"params": group_decay, "lr": layer_lr, "weight_decay": 0.01})
    if group_no_decay:
        param_groups.append({"params": group_no_decay, "lr": layer_lr, "weight_decay": 0.0})

optimizer = AdamW(param_groups, lr=base_lr, weight_decay=0.01)

n_epochs = 3
accumulation_steps = 4
total_steps = (len(train_loader) * n_epochs) // accumulation_steps
warmup_steps = int(total_steps * 0.1)

# Linear warmup + cosine decay scheduler
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

best_val_loss = float("inf")
best_model_state = None
global_step = 0

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
    avg_train_loss = total_loss / n_batches

    model.eval()
    val_loss = 0
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            val_preds.append(probs.cpu().numpy())
            val_true.append(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_log_loss = log_loss(val_true, val_preds_clipped)
    print(
        f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_log_loss:.4f}"
    )
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()

model.load_state_dict(best_model_state)
print(f"Best validation loss: {best_val_loss:.4f}")

# ============================================================
# 13. EXTRACT MODERNBERT EMBEDDINGS
# ============================================================
model.eval()
print("Extracting ModernBERT embeddings...")

def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # For ModernBertForSequenceClassification, hidden states are returned via outputs.hidden_states
            if outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]
                cls_emb = last_hidden[:, 0, :].cpu().numpy()
            else:
                # Fallback: use the encoder's output by re-running the base model
                # Actually for ModernBERT, we can use the model's underlying bert encoder
                # Get the underlying encoder model
                base_model = model.bert if hasattr(model, 'bert') else model.model
                # Forward through base model only
                base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden = base_outputs.last_hidden_state
                cls_emb = last_hidden[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
    return np.concatenate(all_embeddings, axis=0)

# Extract embeddings directly from model using the correctly split data
train_bert_emb = extract_embeddings(model, train_loader, device)
val_bert_emb = extract_embeddings(model, val_loader, device)
test_bert_emb = extract_embeddings(model, test_loader, device)
print(f"BERT embeddings shape: {train_bert_emb.shape}")

# ============================================================
# 14. COMBINE WITH ENGINEERED FEATURES & TRAIN XGBOOST
# ============================================================
X_train_combined = np.concatenate([train_bert_emb, X_train_feats_split], axis=1)
X_val_combined = np.concatenate([val_bert_emb, X_val_feats_split], axis=1)
# For test, use the full scaled features (test set has no split)
X_test_combined = np.concatenate([test_bert_emb, X_test_scaled.values], axis=1)
print(f"Combined feature shape: {X_train_combined.shape}")

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_weight=3,
    gamma=0.1,
    seed=42,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    verbosity=0,
)
xgb_model.fit(
    X_train_combined,
    train_labels,
    eval_set=[(X_val_combined, val_labels)],
    verbose=False,
)

val_preds_xgb = xgb_model.predict_proba(X_val_combined)
val_preds_xgb_clipped = np.clip(val_preds_xgb, 1e-15, 1 - 1e-15)
val_log_loss_xgb = log_loss(val_labels, val_preds_xgb_clipped)
print(f"XGBoost Validation LogLoss: {val_log_loss_xgb:.4f}")

# ============================================================
# 15. GENERATE SUBMISSION
# ============================================================
test_preds = xgb_model.predict_proba(X_test_combined)
test_preds_clipped = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_preds_normalized = test_preds_clipped / test_preds_clipped.sum(
    axis=1, keepdims=True
)

test_ids = test_df["id"].values
submission = pd.DataFrame(
    {
        "id": test_ids,
        class_names[0]: test_preds_normalized[:, 0],
        class_names[1]: test_preds_normalized[:, 1],
        class_names[2]: test_preds_normalized[:, 2],
    }
)
submission = submission[["id", "EAP", "HPL", "MWS"]]

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(submission.head())

final_score = val_log_loss_xgb
print(f"Final Validation Score: {final_score}")
