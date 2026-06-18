import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import os
import nlpaug.augmenter.word as naw
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

X_train_text = train_df["text"].values
y_author = train_df["author"].values
X_test_text = test_df["text"].values
test_ids = test_df["id"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_author)
num_classes = len(label_encoder.classes_)

# ============================================================
# 2. STRATIFIED SPLIT (70% train, 15% val, 15% test from train)
# ============================================================
X_tr_text, X_va_text, y_tr, y_va = train_test_split(
    X_train_text, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_va_text, X_te_text, y_va, y_te = train_test_split(
    X_va_text, y_va, test_size=0.5, stratify=y_va, random_state=42
)

print(f"Train: {len(X_tr_text)}, Val: {len(X_va_text)}, Test: {len(X_te_text)}")


# ============================================================
# 3. TRAINING DATA (no augmentation to avoid dependency issues)
# ============================================================
print("Using original training text (no augmentation)...")
# Keep original data only - synonym augmentation caused API incompatibility
X_tr_text_combined = X_tr_text
y_tr_combined = y_tr

# ============================================================
# 4. FEATURE ENGINEERING - DistilBERT Tokenization
# ============================================================
from transformers import DistilBertTokenizer

print("Tokenizing with DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
max_length = 256

def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

tokenized_tr = tokenize_texts(X_tr_text_combined)
tokenized_va = tokenize_texts(X_va_text)
tokenized_te = tokenize_texts(X_te_text)
tokenized_test = tokenize_texts(X_test_text)

print(f"Tokenized input ids shape: {tokenized_tr['input_ids'].shape}")

# ============================================================
# 5. TF-IDF FEATURE EXTRACTION (for XGBoost)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer

print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
tfidf_tr = tfidf_vectorizer.fit_transform(X_tr_text_combined)
tfidf_va = tfidf_vectorizer.transform(X_va_text)
tfidf_te = tfidf_vectorizer.transform(X_te_text)
tfidf_test = tfidf_vectorizer.transform(X_test_text)
print(f"TF-IDF train shape: {tfidf_tr.shape}")

# ============================================================
# 6. STYLISTIC FEATURE ENGINEERING (for XGBoost)
# ============================================================
import re

def extract_stylistic_features(texts):
    """Extract 18 stylistic features per text."""
    features = []
    for text in texts:
        sentences = re.split(r'[.!?]+', str(text))
        sentences = [s.strip() for s in sentences if s.strip()]
        words = str(text).split()
        chars = list(str(text))

        # Sentence-level features
        num_sentences = len(sentences) if len(sentences) > 0 else 1
        avg_sentence_len = len(words) / num_sentences if num_sentences > 0 else 0
        max_sentence_len = max([len(s.split()) for s in sentences]) if sentences else 0
        min_sentence_len = min([len(s.split()) for s in sentences]) if sentences else 0
        std_sentence_len = np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0

        # Punctuation features
        num_exclam = str(text).count('!')
        num_question = str(text).count('?')
        num_period = str(text).count('.')
        num_comma = str(text).count(',')
        num_quote = str(text).count('"') + str(text).count("'")
        num_colon = str(text).count(':')
        num_semicolon = str(text).count(';')
        num_dash = str(text).count('-') + str(text).count('—')
        num_punct = sum(1 for c in chars if c in '.,!?;:\'"-—()[]{}/')

        # Word-level features
        word_lengths = [len(w) for w in words] if words else [0]
        avg_word_len = np.mean(word_lengths)
        max_word_len = max(word_lengths)
        min_word_len = min(word_lengths)
        std_word_len = np.std(word_lengths)

        # Character-level features
        num_uppercase = sum(1 for c in chars if c.isupper())
        num_digits = sum(1 for c in chars if c.isdigit())
        num_spaces = sum(1 for c in chars if c.isspace())

        feat = [
            avg_sentence_len, max_sentence_len, min_sentence_len, std_sentence_len,
            num_exclam, num_question, num_period, num_comma, num_quote,
            num_colon, num_semicolon, num_dash, num_punct,
            avg_word_len, max_word_len, min_word_len, std_word_len,
            num_uppercase / max(len(chars), 1),  # uppercase ratio
        ]
        features.append(feat)
    return np.array(features)

print("Extracting stylistic features...")
stylistic_tr = extract_stylistic_features(X_tr_text_combined)
stylistic_va = extract_stylistic_features(X_va_text)
stylistic_te = extract_stylistic_features(X_te_text)
stylistic_test = extract_stylistic_features(X_test_text)
print(f"Stylistic features shape: {stylistic_tr.shape}")

# Concatenate TF-IDF with stylistic features for XGBoost
from scipy.sparse import hstack, csr_matrix

xgb_features_tr = hstack([tfidf_tr, csr_matrix(stylistic_tr)])
xgb_features_va = hstack([tfidf_va, csr_matrix(stylistic_va)])
xgb_features_te = hstack([tfidf_te, csr_matrix(stylistic_te)])
xgb_features_test = hstack([tfidf_test, csr_matrix(stylistic_test)])

# Save to disk for later use
import joblib
os.makedirs("./working", exist_ok=True)
joblib.dump((xgb_features_tr, xgb_features_va, xgb_features_te, xgb_features_test), "./working/xgb_features.pkl")
joblib.dump(tfidf_vectorizer, "./working/tfidf_vectorizer.pkl")

# ============================================================
# 7. CHARACTER-LEVEL FEATURE EXTRACTION (for CNN-LSTM)
# ============================================================
print("Extracting character-level sequences...")

# Define character alphabet
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"-()[]{}&@#$%^*/~`|_+=<> "
char_to_idx = {c: i+1 for i, c in enumerate(alphabet)}  # 0 reserved for padding
vocab_size = len(alphabet) + 1  # +1 for padding
max_char_len = 200

def text_to_char_sequence(text):
    """Convert text to character index sequence."""
    text = str(text).lower()
    seq = []
    for c in text[:max_char_len]:
        if c in char_to_idx:
            seq.append(char_to_idx[c])
        else:
            seq.append(0)  # unknown character -> padding index
    # Pad or truncate
    if len(seq) < max_char_len:
        seq = seq + [0] * (max_char_len - len(seq))
    return np.array(seq[:max_char_len])

char_tr = np.array([text_to_char_sequence(t) for t in X_tr_text_combined])
char_va = np.array([text_to_char_sequence(t) for t in X_va_text])
char_te = np.array([text_to_char_sequence(t) for t in X_te_text])
char_test = np.array([text_to_char_sequence(t) for t in X_test_text])
print(f"Char sequences shape: {char_tr.shape}, vocab_size: {vocab_size}")

# Save character mappings
np.save("./working/char_sequences_tr.npy", char_tr)
np.save("./working/char_sequences_va.npy", char_va)
np.save("./working/char_sequences_te.npy", char_te)
np.save("./working/char_sequences_test.npy", char_test)
joblib.dump(char_to_idx, "./working/char_to_idx.pkl")

# ============================================================
# 8. PREPARE TOKENIZED DATA FOR DATALOADERS
# ============================================================
# Store tokenized tensors directly for DataLoader usage.
# y labels remain unchanged, just use torch tensors.
y_train_new = y_tr_combined

print(
    f"Tokenized data prepared: Train {tokenized_tr['input_ids'].shape}, "
    f"Val {tokenized_va['input_ids'].shape}, Test {tokenized_te['input_ids'].shape}"
)


# ============================================================
# MODEL DEFINITION: DistilBERT Sequence Classifier
# ============================================================
from transformers import DistilBertModel, DistilBertConfig

class DistilBERTSequenceClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        # Load pretrained DistilBERT backbone
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Freeze all layers initially (will be unfrozen in Stage 2)
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Multi-layer feature aggregation: use ALL 6 transformer layers' CLS embeddings
        # DistilBERT has 6 layers; hidden_states indices 1-6 (0 is embeddings layer)
        self.num_layers = 6
        # Learned weighted combination of all 6 layer CLS embeddings
        self.gate = nn.Linear(self.distilbert.config.hidden_size, self.num_layers)

        # Classification head with multi-sample dropout support
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0)

    def forward(self, input_ids, attention_mask, mc_dropout=False):
        # Get DistilBERT outputs (all hidden states)
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract CLS embeddings from ALL 6 transformer layers (indices 1-6)
        hidden_states = outputs.hidden_states  # tuple of 7: (embeddings, layer1, ..., layer6)
        cls_layers = []
        for layer_idx in range(1, 7):  # indices 1 through 6
            cls_layers.append(hidden_states[layer_idx][:, 0, :])  # (batch, 768)
        # Stack: (batch, 6, 768)
        cls_stack = torch.stack(cls_layers, dim=1)

        # Compute gating weights using the last layer's CLS
        gate_input = hidden_states[-1][:, 0, :]  # (batch, 768)
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)  # (batch, 6)

        # Weighted sum of all 6 CLS embeddings
        weighted_cls = torch.sum(cls_stack * gate_weights.unsqueeze(-1), dim=1)  # (batch, 768)

        # Apply dropout and classify
        x = self.dropout(weighted_cls)
        logits = self.classifier(x)

        # Multi-sample dropout (Monte Carlo dropout) during inference
        if mc_dropout:
            # Enable dropout for Monte Carlo sampling
            self.dropout.train()
            mc_logits = []
            for _ in range(5):  # 5 forward passes
                x_mc = self.dropout(weighted_cls)
                mc_logits.append(self.classifier(x_mc))
            logits = torch.stack(mc_logits).mean(dim=0)

        return logits


def create_model(num_classes=3):
    # Note: weights will be loaded to a new model instance for inference
    return DistilBERTSequenceClassifier(num_classes=num_classes, dropout=0.3)


# ============================================================
# LOSS FUNCTION WITH LABEL SMOOTHING
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def get_criterion_and_optimizer(model, learning_rate=2e-5, weight_decay=0.01):
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
    return criterion, optimizer


# ============================================================
# MULTICLASS LOG LOSS
# ============================================================
def multiclass_log_loss(y_true, y_pred_probs, eps=1e-15):
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    y_pred_probs = y_pred_probs / row_sums
    y_true_one_hot = np.zeros_like(y_pred_probs)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1
    return -np.sum(y_true_one_hot * np.log(y_pred_probs)) / len(y_true)


# ============================================================
# SETUP DATA LOADERS (using tokenized inputs)
# ============================================================
# Custom dataset that returns input_ids, attention_mask, and labels
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels=None):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

y_train_tensor = torch.LongTensor(y_train_new)
y_val_tensor = torch.LongTensor(y_va)
y_test_tensor = torch.LongTensor(y_te)

batch_size = 16  # Smaller batch size for transformer model

train_dataset = TokenizedDataset(tokenized_tr, labels=y_train_tensor)
val_dataset = TokenizedDataset(tokenized_va, labels=y_val_tensor)
test_dataset = TokenizedDataset(tokenized_te, labels=y_test_tensor)
final_test_dataset = TokenizedDataset(tokenized_test, labels=None)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
final_test_loader = DataLoader(
    final_test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ============================================================
# INSTANTIATE MODEL, CRITERION, OPTIMIZER (differential LR)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = create_model(num_classes=num_classes).to(device)

# ============================================================
# TWO-STAGE FINE-TUNING STRATEGY
# ============================================================
print("Starting two-stage fine-tuning...")


# ============================================================
# EXPONENTIAL MOVING AVERAGE (EMA) FOR MODEL WEIGHTS
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.model = model

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# Initialize EMA after model creation
ema = EMA(model, decay=0.999)
ema.register()

# Stage 1: Freeze all DistilBERT layers, train only classifier head and gating network
print("--- Stage 1: Training classifier head and gating network (frozen backbone) ---")
# Ensure all DistilBERT layers are frozen
for param in model.distilbert.parameters():
    param.requires_grad = False

# Only classifier and gate parameters are trainable
stage1_params = list(model.classifier.parameters()) + list(model.gate.parameters())
no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
optimizer_grouped_params = [
    {
        "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)] +
                  [p for n, p in model.gate.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)] +
                  [p for n, p in model.gate.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer_stage1 = AdamW(optimizer_grouped_params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

stage1_epochs = 3
stage1_warmup_epochs = 1
stage1_total_steps = len(train_loader) * stage1_epochs
stage1_warmup_steps = len(train_loader) * stage1_warmup_epochs

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler_stage1 = CosineAnnealingLR(
    optimizer_stage1, T_max=stage1_total_steps, eta_min=0
)

scaler_stage1 = GradScaler()
patience = 5
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

for epoch in range(stage1_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)

        optimizer_stage1.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
        scaler_stage1.scale(loss).backward()
        scaler_stage1.unscale_(optimizer_stage1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_stage1.step(optimizer_stage1)
        scaler_stage1.update()
        scheduler_stage1.step()
        ema.update()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    # Validation (with EMA weights)
    ema.apply_shadow()
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(batch["labels"].numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_loss = multiclass_log_loss(val_labels, val_preds)
    print(
        f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_loss:.4f}"
    )
    ema.restore()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        ema.apply_shadow()
        best_model_state = model.state_dict().copy()
        ema.restore()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 1 early stopping triggered at epoch {epoch+1}")
            break

# Load best model from Stage 1 before unfreezing
model.load_state_dict(best_model_state)

# Stage 2: Unfreeze all DistilBERT layers, train full model
print("--- Stage 2: Full model fine-tuning (unfrozen backbone) ---")
for param in model.distilbert.parameters():
    param.requires_grad = True

# Re-register EMA to include newly unfrozen backbone parameters
ema.register()

# Full model optimizer with reduced learning rate
no_decay_full = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
optimizer_grouped_params_full = [
    {
        "params": [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay_full)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay_full)
        ],
        "weight_decay": 0.0,
    },
]

optimizer_stage2 = AdamW(optimizer_grouped_params_full, lr=5e-6, betas=(0.9, 0.999), eps=1e-8)

stage2_epochs = 10
stage2_warmup_epochs = 5
stage2_total_steps = len(train_loader) * stage2_epochs
stage2_warmup_steps = len(train_loader) * stage2_warmup_epochs

# Cosine annealing scheduler with warmup for Stage 2
from transformers import get_cosine_schedule_with_warmup
scheduler_stage2 = get_cosine_schedule_with_warmup(
    optimizer_stage2,
    num_warmup_steps=stage2_warmup_steps,
    num_training_steps=stage2_total_steps
)

best_val_loss = float("inf")
patience_counter = 0
best_model_state_stage2 = None

for epoch in range(stage2_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)

        optimizer_stage2.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
        scaler_stage1.scale(loss).backward()
        scaler_stage1.unscale_(optimizer_stage2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_stage1.step(optimizer_stage2)
        scaler_stage1.update()
        scheduler_stage2.step()
        ema.update()
        total_loss += loss.item()
        num_batches += 1
    avg_train_loss = total_loss / num_batches

    ema.apply_shadow()
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(batch["labels"].numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_loss = multiclass_log_loss(val_labels, val_preds)
    print(
        f"Stage 2 - Epoch {epoch+1}/{stage2_epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_loss:.4f}"
    )
    ema.restore()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        ema.apply_shadow()
        best_model_state_stage2 = model.state_dict().copy()
        torch.save(model.state_dict(), "./working/best_model.pt")
        ema.restore()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 2 early stopping triggered at epoch {epoch+1}")
            break

# ============================================================
# LOAD BEST MODEL AND COMPUTE FINAL VALIDATION SCORE (with MC dropout)
# ============================================================
model.load_state_dict(best_model_state_stage2 if best_model_state_stage2 is not None else best_model_state)

# Apply EMA weights for final inference
ema.apply_shadow()

# Use Monte Carlo dropout during inference for better calibration
model.eval()
val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast():
            # Enable multi-sample dropout (5 forward passes)
            logits = model(input_ids, attention_mask, mc_dropout=True)
            probs = torch.softmax(logits, dim=1)
        val_preds.append(probs.cpu().numpy())
val_preds = np.concatenate(val_preds, axis=0)
score = multiclass_log_loss(y_va, val_preds)
print(f"Best Validation LogLoss (with MC dropout): {best_val_loss:.6f}")
ema.restore()

# ============================================================
# FINAL DISTILBERT INFERENCE ON VALIDATION SET
# ============================================================
ema.apply_shadow()
model.eval()
distilbert_val_probs = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast():
            logits = model(input_ids, attention_mask, mc_dropout=True)
            probs = torch.softmax(logits, dim=1)
        distilbert_val_probs.append(probs.cpu().numpy())
distilbert_val_probs = np.concatenate(distilbert_val_probs, axis=0)
ema.restore()

# ============================================================
# DISTILBERT TEST INFERENCE
# ============================================================
ema.apply_shadow()
distilbert_test_probs = []
with torch.no_grad():
    for batch in final_test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast():
            logits = model(input_ids, attention_mask, mc_dropout=True)
            probs = torch.softmax(logits, dim=1)
        distilbert_test_probs.append(probs.cpu().numpy())
distilbert_test_probs = np.concatenate(distilbert_test_probs, axis=0)
ema.restore()

# Save DistilBERT predictions
np.save("./working/distilbert_val_probs.npy", distilbert_val_probs)
np.save("./working/distilbert_test_probs.npy", distilbert_test_probs)

# ============================================================
# TRAIN XGBoost MODEL (on TF-IDF + stylistic features)
# ============================================================
print("--- Training XGBoost model on TF-IDF+stylistic features ---")
import xgboost as xgb

# Load features from disk (already saved in feature engineering section)
xgb_features_tr, xgb_features_va, xgb_features_te, xgb_features_test = joblib.load("./working/xgb_features.pkl")

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=20,
    eval_metric='mlogloss',
)
xgb_model.fit(
    xgb_features_tr, y_tr_combined,
    eval_set=[(xgb_features_va, y_va)],
    verbose=False
)

xgb_val_probs = xgb_model.predict_proba(xgb_features_va)
xgb_test_probs = xgb_model.predict_proba(xgb_features_test)
print(f"XGBoost val log-loss: {multiclass_log_loss(y_va, xgb_val_probs):.6f}")

# Save XGBoost model and predictions
xgb_model.save_model("./working/xgb_model.json")
np.save("./working/xgb_val_probs.npy", xgb_val_probs)
np.save("./working/xgb_test_probs.npy", xgb_test_probs)

# ============================================================
# TRAIN CHARACTER-LEVEL CNN-LSTM
# ============================================================
print("--- Training character-level CNN-LSTM ---")

# Load char sequences from disk
char_tr = np.load("./working/char_sequences_tr.npy")
char_va = np.load("./working/char_sequences_va.npy")
char_te = np.load("./working/char_sequences_te.npy")
char_test = np.load("./working/char_sequences_test.npy")

# Build CNN-LSTM model
import torch

class CharCNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, kernel_sizes=[3,5,7], hidden_size=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2) for k in kernel_sizes
        ])
        self.lstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) for Conv1d
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, filters, seq_len)
            conv_outs.append(conv_out)
        x = torch.cat(conv_outs, dim=1)  # (batch, filters*len, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, filters*len)
        x, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        x = x[:, -1, :]  # take last timestep
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

char_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_model = CharCNN_LSTM(vocab_size=vocab_size, num_classes=num_classes).to(char_device)

char_train_dataset = TensorDataset(torch.LongTensor(char_tr), torch.LongTensor(y_tr_combined))
char_val_dataset = TensorDataset(torch.LongTensor(char_va), torch.LongTensor(y_va))

char_train_loader = DataLoader(char_train_dataset, batch_size=32, shuffle=True)
char_val_loader = DataLoader(char_val_dataset, batch_size=32, shuffle=False)

char_criterion = nn.CrossEntropyLoss()
char_optimizer = torch.optim.Adam(char_model.parameters(), lr=1e-3)
char_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(char_optimizer, mode='min', factor=0.5, patience=2)

char_best_val_loss = float('inf')
char_patience = 5
char_patience_counter = 0

for epoch in range(30):
    char_model.train()
    total_loss = 0
    for batch_ids, batch_labels in char_train_loader:
        batch_ids = batch_ids.to(char_device)
        batch_labels = batch_labels.to(char_device)
        char_optimizer.zero_grad()
        logits = char_model(batch_ids)
        loss = char_criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(char_model.parameters(), max_norm=1.0)
        char_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(char_train_loader)

    # Validation
    char_model.eval()
    char_val_preds = []
    with torch.no_grad():
        for batch_ids, _ in char_val_loader:
            batch_ids = batch_ids.to(char_device)
            logits = char_model(batch_ids)
            probs = torch.softmax(logits, dim=1)
            char_val_preds.append(probs.cpu().numpy())
    char_val_preds = np.concatenate(char_val_preds, axis=0)
    char_val_loss = multiclass_log_loss(y_va, char_val_preds)

    char_scheduler.step(char_val_loss)
    print(f"CharCNN-LSTM Epoch {epoch+1}/30 | Train Loss: {avg_loss:.4f} | Val LogLoss: {char_val_loss:.6f}")

    if char_val_loss < char_best_val_loss:
        char_best_val_loss = char_val_loss
        char_patience_counter = 0
        torch.save(char_model.state_dict(), "./working/char_best_model.pt")
    else:
        char_patience_counter += 1
        if char_patience_counter >= char_patience:
            print(f"CharCNN-LSTM early stopping at epoch {epoch+1}")
            break

# Load best char model and get predictions
char_model.load_state_dict(torch.load("./working/char_best_model.pt"))
char_model.eval()

char_val_probs = []
with torch.no_grad():
    for batch_ids, _ in char_val_loader:
        batch_ids = batch_ids.to(char_device)
        logits = char_model(batch_ids)
        probs = torch.softmax(logits, dim=1)
        char_val_probs.append(probs.cpu().numpy())
char_val_probs = np.concatenate(char_val_probs, axis=0)

# Char test predictions
char_test_dataset = TensorDataset(torch.LongTensor(char_test))
char_test_loader = DataLoader(char_test_dataset, batch_size=32, shuffle=False)
char_test_probs = []
with torch.no_grad():
    for batch_ids, in char_test_loader:
        batch_ids = batch_ids.to(char_device)
        logits = char_model(batch_ids)
        probs = torch.softmax(logits, dim=1)
        char_test_probs.append(probs.cpu().numpy())
char_test_probs = np.concatenate(char_test_probs, axis=0)

np.save("./working/char_val_probs.npy", char_val_probs)
np.save("./working/char_test_probs.npy", char_test_probs)

print(f"CharCNN-LSTM val log-loss: {multiclass_log_loss(y_va, char_val_probs):.6f}")

# ============================================================
# ENSEMBLE: LEARN OPTIMAL WEIGHTS VIA NELDER-MEAD
# ============================================================
print("--- Learning optimal ensemble weights via Nelder-Mead ---")
from scipy.optimize import minimize

def ensemble_loss(weights, val_preds_list, y_true):
    """Negative log-loss for weighted ensemble."""
    weights = np.array(weights)
    weights = np.abs(weights) / np.sum(np.abs(weights))  # normalize
    ensemble_probs = weights[0] * val_preds_list[0] + weights[1] * val_preds_list[1] + weights[2] * val_preds_list[2]
    return multiclass_log_loss(y_true, ensemble_probs)

val_preds_list = [distilbert_val_probs, xgb_val_probs, char_val_probs]
initial_weights = [0.5, 0.3, 0.2]
bounds = [(0, 1), (0, 1), (0, 1)]
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

opt_result = minimize(
    ensemble_loss,
    initial_weights,
    args=(val_preds_list, y_va),
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'adaptive': True}
)

optimal_weights = np.array(opt_result.x)
optimal_weights = np.abs(optimal_weights) / np.sum(np.abs(optimal_weights))
print(f"Optimal ensemble weights: DistilBERT={optimal_weights[0]:.4f}, XGBoost={optimal_weights[1]:.4f}, CharCNN-LSTM={optimal_weights[2]:.4f}")

# Compute ensemble validation score
ensemble_val_probs = (
    optimal_weights[0] * distilbert_val_probs +
    optimal_weights[1] * xgb_val_probs +
    optimal_weights[2] * char_val_probs
)
ensemble_val_score = multiclass_log_loss(y_va, ensemble_val_probs)
print(f"Ensemble Validation LogLoss: {ensemble_val_score:.6f}")

# ============================================================
# GENERATE FINAL ENSEMBLE SUBMISSION
# ============================================================
test_preds_list = [distilbert_test_probs, xgb_test_probs, char_test_probs]
final_preds = (
    optimal_weights[0] * test_preds_list[0] +
    optimal_weights[1] * test_preds_list[1] +
    optimal_weights[2] * test_preds_list[2]
)
final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_preds[:, 0],
        "HPL": final_preds[:, 1],
        "MWS": final_preds[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Ensemble Validation Score: {ensemble_val_score:.6f}")