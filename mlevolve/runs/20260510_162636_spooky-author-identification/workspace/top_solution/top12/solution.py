import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix, isspmatrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    get_cosine_schedule_with_warmup,
    AutoConfig,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import re
import os
import warnings
import joblib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")


# Basic text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Feature engineering function
def extract_features(df, is_train=True):
    features = pd.DataFrame()
    features["id"] = df["id"]
    text_series = df["text"].astype(str)
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.count("[.!?]") + 1
    features["avg_sentence_len"] = features["word_count"] / (
        features["sentence_count"] + 1
    )
    features["unique_words"] = text_series.apply(lambda x: len(set(x.lower().split())))
    features["lexical_diversity"] = features["unique_words"] / (
        features["word_count"] + 1
    )
    features["exclamation_count"] = text_series.str.count("!")
    features["question_count"] = text_series.str.count(r"\?")
    features["period_count"] = text_series.str.count(r"\.")
    features["comma_count"] = text_series.str.count(",")
    features["semicolon_count"] = text_series.str.count(";")
    features["colon_count"] = text_series.str.count(":")
    features["quote_count"] = text_series.str.count('"')
    features["apostrophe_count"] = text_series.str.count("'")
    features["dash_count"] = text_series.str.count("-")
    features["paren_count"] = text_series.str.count("\(|\)")
    features["punctuation_ratio"] = (
        features["exclamation_count"]
        + features["question_count"]
        + features["period_count"]
        + features["comma_count"]
        + features["semicolon_count"]
        + features["colon_count"]
        + features["quote_count"]
        + features["apostrophe_count"]
        + features["dash_count"]
        + features["paren_count"]
    ) / (features["word_count"] + 1)
    features["capital_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in x.split() if w[0].isupper()) / (len(x.split()) + 1)
    )
    features["all_caps_words"] = text_series.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
    )
    features["syllable_count"] = text_series.apply(
        lambda x: sum(1 for char in x.lower() if char in "aeiou")
    )
    features["complex_word_ratio"] = features["syllable_count"] / (
        features["word_count"] + 1
    )
    stop_words = set(
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
        ]
    )
    features["stop_word_ratio"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in stop_words)
        / (len(x.split()) + 1)
    )
    eap_words = {
        "nevermore",
        "raven",
        "chamber",
        "dreary",
        "quoth",
        "charnel",
        "sepulchre",
    }
    hpl_words = {
        "cthulhu",
        "eldritch",
        "shoggoth",
        "necronomicon",
        "yog",
        "sothoth",
        "rlyeh",
        "non",
        "euclidean",
    }
    mws_words = {
        "frankenstein",
        "creature",
        "monster",
        "victor",
        "elizabeth",
        "walton",
        "destiny",
        "sublime",
    }
    features["eap_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in eap_words)
    )
    features["hpl_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in hpl_words)
    )
    features["mws_word_count"] = text_series.apply(
        lambda x: sum(1 for w in x.lower().split() if w in mws_words)
    )
    features["verb_like"] = text_series.apply(
        lambda x: len(re.findall(r"\b\w+ed\b", x.lower()))
    )
    features["ing_words"] = text_series.apply(
        lambda x: len(re.findall(r"\b\w+ing\b", x.lower()))
    )
    features["ly_words"] = text_series.apply(
        lambda x: len(re.findall(r"\b\w+ly\b", x.lower()))
    )
    features["article_count"] = text_series.apply(
        lambda x: len(re.findall(r"\b(a|an|the)\b", x.lower()))
    )
    features["pronoun_count"] = text_series.apply(
        lambda x: len(
            re.findall(r"\b(i|you|he|she|it|we|they|me|him|her|us|them)\b", x.lower())
        )
    )
    features["processed_text"] = text_series.apply(clean_text)
    return features


# Author encoding
label_encoder = LabelEncoder()
if "author" in train_df.columns:
    y_all = label_encoder.fit_transform(train_df["author"])

# Create validation split FIRST
indices = np.arange(len(train_df))
train_idx, val_idx = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y_all
)

# Split data before any vectorizer/scaler fitting
train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
val_df_split = train_df.iloc[val_idx].reset_index(drop=True)
test_df_final = test_df

# Extract features on split data
train_features_split = extract_features(train_df_split, is_train=True)
val_features_split = extract_features(val_df_split, is_train=True)
test_features_final = extract_features(test_df_final, is_train=False)

# Author labels for split
y_train_final = y_all[train_idx]
y_val = y_all[val_idx]

# Numeric columns for scaling
numeric_cols = [
    "char_count",
    "word_count",
    "avg_word_len",
    "sentence_count",
    "avg_sentence_len",
    "unique_words",
    "lexical_diversity",
    "exclamation_count",
    "question_count",
    "period_count",
    "comma_count",
    "semicolon_count",
    "colon_count",
    "quote_count",
    "apostrophe_count",
    "dash_count",
    "paren_count",
    "punctuation_ratio",
    "capital_word_ratio",
    "all_caps_words",
    "syllable_count",
    "complex_word_ratio",
    "stop_word_ratio",
    "eap_word_count",
    "hpl_word_count",
    "mws_word_count",
    "verb_like",
    "ing_words",
    "ly_words",
    "article_count",
    "pronoun_count",
]

# Fit vectorizers on TRAIN only
vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 3), min_df=3, max_df=0.7, sublinear_tf=True
)
train_text_features = vectorizer.fit_transform(train_df_split["text"].astype(str))
val_text_features = vectorizer.transform(val_df_split["text"].astype(str))
test_text_features = vectorizer.transform(test_df_final["text"].astype(str))

char_vectorizer = TfidfVectorizer(
    analyzer="char", max_features=2000, ngram_range=(2, 5), min_df=3, max_df=0.7
)
train_char_features = char_vectorizer.fit_transform(train_df_split["text"].astype(str))
val_char_features = char_vectorizer.transform(val_df_split["text"].astype(str))
test_char_features = char_vectorizer.transform(test_df_final["text"].astype(str))

# Fit scaler on TRAIN only
scaler = StandardScaler()
train_numeric_scaled = scaler.fit_transform(train_features_split[numeric_cols])
val_numeric_scaled = scaler.transform(val_features_split[numeric_cols])
test_numeric_scaled = scaler.transform(test_features_final[numeric_cols])

# Combine all features for train
train_numeric_sparse = csr_matrix(train_numeric_scaled)
val_numeric_sparse = csr_matrix(val_numeric_scaled)
test_numeric_sparse = csr_matrix(test_numeric_scaled)

X_train_final = hstack([train_numeric_sparse, train_text_features, train_char_features])
X_val = hstack([val_numeric_sparse, val_text_features, val_char_features])
X_test = hstack([test_numeric_sparse, test_text_features, test_char_features])

# Save feature dimensions
num_numeric = len(numeric_cols)
num_text = vectorizer.get_feature_names_out().shape[0]
num_char = char_vectorizer.get_feature_names_out().shape[0]

# Save feature dimensions
num_numeric = len(numeric_cols)
num_text = vectorizer.get_feature_names_out().shape[0]
num_char = char_vectorizer.get_feature_names_out().shape[0]


# Model Architecture: Hybrid ModernBERT + Stylometric Features
class HybridSpookyAuthorClassifier(nn.Module):
    def __init__(
        self,
        num_classes=3,
        modernbert_hidden_dim=1024,
        manual_feature_dim=num_numeric,
        text_feature_dim=num_text,
        char_feature_dim=num_char,
        dropout=0.3,
    ):
        super().__init__()
        self.modernbert_config = AutoConfig.from_pretrained(
            "answerdotai/ModernBERT-large",
            num_labels=num_classes,
            output_hidden_states=True,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.modernbert = ModernBertForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-large", config=self.modernbert_config
        )
        for name, param in self.modernbert.named_parameters():
            if "layer" in name:
                layer_num = (
                    int(name.split(".")[2].split("_")[0]) if "layer" in name else 0
                )
                if isinstance(layer_num, int) and layer_num < 20:
                    param.requires_grad = False
            elif "embedding" in name or "pooler" in name:
                param.requires_grad = False
        self.manual_feature_encoder = nn.Sequential(
            nn.Linear(manual_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_feature_encoder = nn.Sequential(
            nn.Linear(text_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.char_feature_encoder = nn.Sequential(
            nn.Linear(char_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_dim = modernbert_hidden_dim + 128 + 128 + 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [
            self.manual_feature_encoder,
            self.text_feature_encoder,
            self.char_feature_encoder,
            self.fusion_layer,
        ]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(
        self, input_ids, attention_mask, manual_features, text_features, char_features
    ):
        modernbert_outputs = self.modernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = modernbert_outputs.hidden_states[-1]
        cls_embedding = hidden_states[:, 0, :]
        manual_encoded = self.manual_feature_encoder(manual_features.float())
        text_encoded = self.text_feature_encoder(text_features.float())
        char_encoded = self.char_feature_encoder(char_features.float())
        combined = torch.cat(
            [cls_embedding, manual_encoded, text_encoded, char_encoded], dim=1
        )
        logits = self.fusion_layer(combined)
        return logits


# Loss function with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=3):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / (self.num_classes - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss


# No stochastic depth wrapper needed - using built-in dropout from config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridSpookyAuthorClassifier().to(device)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
criterion = LabelSmoothingCrossEntropy(smoothing=0.15, num_classes=3)

# Optimizer with differential learning rates (NAdamW with gradient centralization)
class NAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, momentum_decay=0.004):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super(NAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum_decay = group['momentum_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['mu_product'] = 1.0
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                mu_product = state['mu_product']
                state['step'] += 1
                t = state['step']
                beta1_t = beta1 * (1 - 0.5 * (0.96 ** (t * momentum_decay)))
                exp_avg.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                mu_product = mu_product * beta1_t
                mu_product_next = mu_product * (beta1 * (1 - 0.5 * (0.96 ** ((t + 1) * momentum_decay))))
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                bias_correction1 = 1 - mu_product_next / (1 - beta1_t)
                bias_correction2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                state['mu_product'] = mu_product
        return loss

def apply_gradient_centralization(model):
    for param in model.parameters():
        if param.grad is not None and param.dim() > 1:
            param.grad.data = param.grad.data - param.grad.data.mean(dim=tuple(range(1, param.dim())), keepdim=True)

optimizer = torch.optim.AdamW(
    [
        {
            "params": [
                p for n, p in model.modernbert.named_parameters() if p.requires_grad
            ],
            "lr": 1e-5,
            "weight_decay": 0.01,
        },
        {
            "params": model.manual_feature_encoder.parameters(),
            "lr": 5e-4,
            "weight_decay": 0.01,
        },
        {
            "params": model.text_feature_encoder.parameters(),
            "lr": 5e-4,
            "weight_decay": 0.01,
        },
        {
            "params": model.char_feature_encoder.parameters(),
            "lr": 5e-4,
            "weight_decay": 0.01,
        },
        {"params": model.fusion_layer.parameters(), "lr": 5e-4, "weight_decay": 0.01},
    ]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=3, T_mult=2, eta_min=1e-6
)

num_epochs = 10
steps_per_epoch = X_train_final.shape[0] // 32 + 1
num_warmup_steps = 5 * steps_per_epoch
restart_interval = 3  # hard restarts every 3 epochs
scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Get texts for ModernBERT (from split data)
train_texts = train_df_split["text"].astype(str).values
val_texts = val_df_split["text"].astype(str).values
test_texts = test_df["text"].astype(str).values

# Extract feature components
X_train_manual = (
    X_train_final[:, :num_numeric].toarray()
    if isspmatrix(X_train_final)
    else X_train_final[:, :num_numeric]
)
X_train_text = (
    X_train_final[:, num_numeric : num_numeric + num_text].toarray()
    if isspmatrix(X_train_final)
    else X_train_final[:, num_numeric : num_numeric + num_text]
)
X_train_char = (
    X_train_final[
        :, num_numeric + num_text : num_numeric + num_text + num_char
    ].toarray()
    if isspmatrix(X_train_final)
    else X_train_final[:, num_numeric + num_text : num_numeric + num_text + num_char]
)

X_val_manual = (
    X_val[:, :num_numeric].toarray() if isspmatrix(X_val) else X_val[:, :num_numeric]
)
X_val_text = (
    X_val[:, num_numeric : num_numeric + num_text].toarray()
    if isspmatrix(X_val)
    else X_val[:, num_numeric : num_numeric + num_text]
)
X_val_char = (
    X_val[:, num_numeric + num_text : num_numeric + num_text + num_char].toarray()
    if isspmatrix(X_val)
    else X_val[:, num_numeric + num_text : num_numeric + num_text + num_char]
)

X_test_manual = (
    X_test[:, :num_numeric].toarray() if isspmatrix(X_test) else X_test[:, :num_numeric]
)
X_test_text = (
    X_test[:, num_numeric : num_numeric + num_text].toarray()
    if isspmatrix(X_test)
    else X_test[:, num_numeric : num_numeric + num_text]
)
X_test_char = (
    X_test[:, num_numeric + num_text : num_numeric + num_text + num_char].toarray()
    if isspmatrix(X_test)
    else X_test[:, num_numeric + num_text : num_numeric + num_text + num_char]
)

# Tokenize texts
max_length = 512


def tokenize_texts(texts):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


train_input_ids, train_attention_mask = tokenize_texts(train_texts)
val_input_ids, val_attention_mask = tokenize_texts(val_texts)
test_input_ids, test_attention_mask = tokenize_texts(test_texts)

# Create datasets
train_dataset = TensorDataset(
    train_input_ids,
    train_attention_mask,
    torch.tensor(X_train_manual, dtype=torch.float32),
    torch.tensor(X_train_text, dtype=torch.float32),
    torch.tensor(X_train_char, dtype=torch.float32),
    torch.tensor(y_train_final, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_input_ids,
    val_attention_mask,
    torch.tensor(X_val_manual, dtype=torch.float32),
    torch.tensor(X_val_text, dtype=torch.float32),
    torch.tensor(X_val_char, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_input_ids,
    test_attention_mask,
    torch.tensor(X_test_manual, dtype=torch.float32),
    torch.tensor(X_test_text, dtype=torch.float32),
    torch.tensor(X_test_char, dtype=torch.float32),
)

batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# Training loop
best_val_loss = float("inf")
patience = 5
patience_counter = 0
eps = 1e-15

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    train_batches = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, manual_feat, text_feat, char_feat, labels = [
            b.to(device) for b in batch
        ]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, manual_feat, text_feat, char_feat)
            loss = criterion(logits, labels)
        scaler_amp.scale(loss).backward()
        if torch.cuda.is_available():
            scaler_amp.unscale_(optimizer)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_train_loss += loss.item()
        train_batches += 1
    avg_train_loss = total_train_loss / train_batches
    scheduler.step()

    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, manual_feat, text_feat, char_feat, labels = [
                b.to(device) for b in batch
            ]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(
                    input_ids, attention_mask, manual_feat, text_feat, char_feat
                )
                loss = criterion(logits, labels)
            total_val_loss += loss.item()
            val_batches += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_val_preds.append(probs)
            all_val_labels.append(labels.cpu().numpy())
    avg_val_loss = total_val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )
    val_score = log_loss(val_labels, val_preds_normalized)
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.6f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

# Final validation inference
all_val_preds = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, manual_feat, text_feat, char_feat, labels = [
            b.to(device) for b in batch
        ]
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, manual_feat, text_feat, char_feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_val_preds.append(probs)
        all_val_labels.append(labels.cpu().numpy())
val_preds = np.concatenate(all_val_preds, axis=0)
val_labels = np.concatenate(all_val_labels, axis=0)
val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_labels, val_preds_normalized)

# Test inference
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, manual_feat, text_feat, char_feat = [
            b.to(device) for b in batch
        ]
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, manual_feat, text_feat, char_feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_test_preds.append(probs)
test_preds = np.concatenate(all_test_preds, axis=0)
test_preds_clipped = np.clip(test_preds, eps, 1 - eps)
test_preds_normalized = test_preds_clipped / test_preds_clipped.sum(
    axis=1, keepdims=True
)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_preds_normalized[:, 0],
        "HPL": test_preds_normalized[:, 1],
        "MWS": test_preds_normalized[:, 2],
    }
)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
print("Submission saved to ./submission/submission.csv")